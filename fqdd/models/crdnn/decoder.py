import torch
import torch.nn as nn
from torch.nn import Embedding

from fqdd.models.crdnn.decoder_layer import Decoder_layer


class Decoder(nn.Module):
    def __init__(
            self,
            num_classifies,
            consider_as_one_hot=False,
            embedding_dim=1024,
            output_size=1024,
            num_block=2,
            bidirectional=False,
            dropout=0.15,
            blank_id=0,
    ):
        super(Decoder, self).__init__()
        self.num_block = num_block
        self.embedding = Embedding(num_embeddings=num_classifies, embedding_dim=embedding_dim,
                                   consider_as_one_hot=False)
        self.l1 = nn.Sequential(
            nn.Linear(embedding_dim, output_size, bias=True),
            nn.LayerNorm(output_size),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout)
        )
        self.blocks = nn.ModuleList()
        for _ in range(self.num_block):
            self.blocks.append(
                Decoder_layer(
                    output_size=output_size,
                    bidirectional=False,
                    consider_as_one_hot=False,
                    dropout=0.15,
                    blank_id=0,
                )
            )

    def forward(self, x, x_en):
        x = self.embedding(x)
        x = self.l1(x)  # (B, T, D)
        for layer in self.blocks:
            x = layer(x, x_en)
        return x
