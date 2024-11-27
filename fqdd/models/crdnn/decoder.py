import torch
import torch.nn as nn
from torch.nn import Embedding

from fqdd.models.crdnn.decoder_layer import Decoder_layer


class CrdnnDecoder(nn.Module):
    def __init__(
            self,
            vocab_size,
            decoder_conf
    ):
        super().__init__()
        encoder_output_size = decoder_conf.get("encoder_output_size", 256)
        num_block = decoder_conf.get("num_block", 2)
        embedding_dim = decoder_conf.get("embedding_dim", 256)
        bidirectional = decoder_conf.get("bidirectional", False)
        dropout_rate = decoder_conf.get("dropout_rate", 0.1)
        consider_as_one_hot = decoder_conf.get("consider_as_one_hot", False)
        blank_id = decoder_conf.get("consider_as_one_hot", 0)

        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim,
                                   consider_as_one_hot=False)
        self.l1 = nn.Sequential(
            nn.Linear(embedding_dim, encoder_output_size, bias=True),
            nn.LayerNorm(encoder_output_size),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_rate)
        )
        self.blocks = nn.ModuleList()
        for _ in range(num_block):
            self.blocks.append(
                Decoder_layer(
                    output_size=encoder_output_size,
                    bidirectional=bidirectional,
                    consider_as_one_hot=consider_as_one_hot,
                    dropout=0.15,
                    blank_id=blank_id,
                )
            )

    def forward(self, x, x_en):
        x = self.embedding(x)
        x = self.l1(x)  # (B, T, D)
        for layer in self.blocks:
            x = layer(x, x_en)
        return x
