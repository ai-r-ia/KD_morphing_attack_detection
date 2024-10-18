from torchvision.models import vit_b_16, ViT_B_16_Weights
# resnet34
import torch
from torch.nn import (
    Module,
    Linear,
    Sequential,
    Softmax,
)


class ViTEmbeddings(Module):
    def __init__(self, classes: int = 2, only_embeddings: bool = False) -> None:
        super(ViTEmbeddings, self).__init__()
        self.model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        self.only_embeddings = only_embeddings
        self.model.heads.head = Sequential(
            Linear(
                self.model.representation_size
                if self.model.representation_size
                else self.model.hidden_dim,
                classes,
            ),
            Softmax(dim=1),
        )

    def forward(self, x):
        x = self.model._process_input(x)
        n = x.shape[0]

        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.model.encoder(x)

        x = x[:, 0]
        embds = x
        if self.only_embeddings:
            return x

        x = self.model.heads(x)
        return x, embds


class ViTEmbeddingsAdaptive(Module):
    def __init__(self, classes: int = 2, only_embeddings: bool = False) -> None:
        super(ViTEmbeddingsAdaptive, self).__init__()
        self.model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        self.only_embeddings = only_embeddings
        self.model.heads.head = Sequential(
            Linear(
                self.model.representation_size
                if self.model.representation_size
                else self.model.hidden_dim,
                classes,
            ),
            Softmax(dim=1),
        )

    def forward(self, x, adapter_embds):
        x = self.model._process_input(x)
        n = x.shape[0]

        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.model.encoder(x)

        x = x[:, 0]
        embds = x
        if self.only_embeddings:
            return x

        x = self.model.heads(x)
        adapter_preds = self.model.heads(adapter_embds)
        return x, adapter_preds, embds
