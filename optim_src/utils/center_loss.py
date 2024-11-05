import torch
from torch.nn import Module, functional


class Center_loss(Module):
    def __init__(
        self,
        batch_size=1,
        lambda_c=1.0,
        lambda_s=0.1,
        max_separation_loss=10.0,
        embedding_dim=768,
    ) -> None:
        super(Center_loss, self).__init__()
        self.embedding_dim = embedding_dim
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        self.max_separation_loss = max_separation_loss

        self.bonafide_center = torch.nn.Parameter(
            torch.randn(1, embedding_dim).cuda(), requires_grad=True
        )
        self.morph_center = torch.nn.Parameter(
            torch.randn(1, embedding_dim).cuda(), requires_grad=True
        )
        # self.bonafide_center = torch.randn(1, embedding_dim).cuda()
        # self.morph_center = torch.randn(1, embedding_dim).cuda()

    def forward(self, embeddings, labels):
        embeddings = functional.normalize(embeddings, p=2, dim=1)

        labels = torch.argmax(labels, dim=1)
        bonafide_mask = labels == 0
        morph_mask = labels == 1

        bonafide_embeddings = embeddings[bonafide_mask]
        morph_embeddings = embeddings[morph_mask]

        bonafide_loss = 0
        if bonafide_embeddings.size(0) > 0:
            bonafide_loss = functional.mse_loss(
                bonafide_embeddings, self.bonafide_center.expand_as(bonafide_embeddings)
            )

        morph_loss = 0
        if morph_embeddings.size(0) > 0:
            morph_loss = functional.mse_loss(
                morph_embeddings, self.morph_center.expand_as(morph_embeddings)
            )

        separation_loss = torch.norm(self.bonafide_center - self.morph_center, p=2)
        separation_loss = torch.clamp(separation_loss, max=self.max_separation_loss)

        # minimize center loss + maximize separation loss
        total_loss = (
            self.lambda_c * (bonafide_loss + morph_loss)
            - self.lambda_s * separation_loss
        )

        # Prevents the total loss from going negative
        total_loss = torch.clamp(total_loss, min=1e-8)

        return total_loss
