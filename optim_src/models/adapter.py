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
from torch import nn


class Adapter(Module):
    def __init__(
        self,
        input_size: int = 2,
        output_size: int = 32,
        num_layers: int = 2,
        hidden_size=768,
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
                # Sequential(
                Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            modules.append(BatchNorm1d(out_channels))
            modules.append(LeakyReLU(0.2))
            in_channels = out_channels
            out_channels *= 2
        self.layers = Sequential(*modules).to(
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.pool = AvgPool1d(kernel_size=1)
        self.fc = Linear(hidden_size, hidden_size)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # print(x.shape)
        # print(x.shape)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.fc(x)
        # print(x.shape)
        return x

    # class Adapter(nn.Module):
    #     def __init__(self, input_size: int = 2, output_size: int= 32, num_conv_layers: int = 2):
    #         super(Adapter, self).__init__()

    #         layers = []
    #         in_channels = input_size

    #         for _ in range(num_conv_layers):
    #             layers.append(
    #                 nn.Conv1d(in_channels, in_channels * 2, kernel_size=3, padding=1)
    #             )
    #             layers.append(nn.BatchNorm1d(in_channels * 2))
    #             layers.append(nn.LeakyReLU(0.2))
    #             in_channels *= 2

    #         self.conv_layers = nn.Sequential(*layers)
    #         self.pool = nn.AvgPool1d(kernel_size=2)  # Downsample to reduce dimensionality
    #         self.fc = nn.Linear(
    #             in_channels // 2, output_size
    #         )  # Adjust in_channels based on pooling

    #     def forward(self, x):
    #         x = self.conv_layers(x)
    #         x = self.pool(x)
    #         x = x.view(x.size(0), -1)  # Flatten for the fully connected layer
    #         x = self.fc(x)
    #         return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
