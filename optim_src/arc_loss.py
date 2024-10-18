import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcLoss(nn.Module):
    def __init__(self, feature_num=768, margin=0.1, scale=64):
        super(ArcLoss, self).__init__()
        self.margin = margin  # Margin to add to the correct class
        self.scale = scale  # Scaling factor
        self.weights = nn.Parameter(
            torch.randn(feature_num, 2).to(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        )

    def forward(self, features, labels):
        labels = labels.long()

        normalized_weights = F.normalize(self.weights, dim=0)
        normalized_features = F.normalize(features, dim=1)

        cos_theta = torch.matmul(
            normalized_features, normalized_weights
        )  # Shape: (batch_size, 2)

        margin_tensor = torch.tensor(self.margin).to(cos_theta.device)

        target_cosine = cos_theta * torch.cos(margin_tensor) - torch.sqrt(
            1 - cos_theta**2
        ) * torch.sin(margin_tensor)

        weighted_target_cosine = (
            target_cosine * labels[:, 1].unsqueeze(1) * self.weights[1]
        )  # Weight for morph class
        weighted_cos_theta = (
            cos_theta * labels[:, 0].unsqueeze(1) * self.weights[0]
        )  # Weight for bonafide class

        logits = weighted_target_cosine + weighted_cos_theta

        logits *= self.scale
        return logits
