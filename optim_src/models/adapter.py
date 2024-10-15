import torch
from torch.nn import (
    Module,
    Conv1d,
    Linear,
    BatchNorm1d,
    Sequential,
    LeakyReLU,
    AvgPool1d,
)


class Adapter(Module):
    def __init__(
        self,
        input_size: int = 2,
        output_size: int = 32,
        num_layers: int = 4,
        hidden_size=128,
    ) -> None:
        super(Adapter, self).__init__()

        modules = []
        in_channels = input_size
        out_channels = output_size
        for i in range(num_layers):
            if i == num_layers - 1:
                out_channels = 1

            # print("in: ", in_channels, "out: ", out_channels)

            modules.append(
                Sequential(
                    Conv1d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    BatchNorm1d(out_channels),
                    LeakyReLU(0.2),
                ).to(
                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                )
            )
            in_channels = out_channels
            out_channels *= 2
        self.layers = Sequential(*modules)
        self.pool = AvgPool1d(kernel_size=2)
        self.fc = Linear(hidden_size, output_size)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x
