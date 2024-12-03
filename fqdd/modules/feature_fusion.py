import torch
import torch.nn as nn

from fqdd.nnets.base_utils import FQDD_ACTIVATIONS


class AdaptiveFeatureFusion(nn.Module):
    def __init__(self,
                 local_dim,
                 global_dim,
                 fusion_dim,
                 gate_activation: str,
                 dropout_rate: float):
        super(AdaptiveFeatureFusion, self).__init__()
        self.drop = nn.Dropout(dropout_rate)
        self.local_fc = nn.Linear(local_dim, fusion_dim)
        self.global_fc = nn.Linear(global_dim, fusion_dim)
        self.attention = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            FQDD_ACTIVATIONS[gate_activation](),
            self.drop,
            nn.Linear(fusion_dim, 2),
            nn.Softmax(dim=1)
        )
        self.after_norm = nn.LayerNorm(local_dim)
        self.catline = nn.Linear(global_dim, global_dim)

    def forward(self, local_features, global_features):
        local_features = self.local_fc(local_features)
        global_features = self.global_fc(global_features)
        concatenated_features = torch.cat((local_features, global_features), dim=-1)
        attention_weights = self.attention(concatenated_features)
        fused_features = attention_weights[:, :, 1:] * local_features + attention_weights[:, :, 1:] * global_features
        fused_features = self.catline(self.after_norm(fused_features))
        return self.drop(fused_features)
