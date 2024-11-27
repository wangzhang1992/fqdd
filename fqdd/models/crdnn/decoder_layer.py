import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(
            self,
            output_size=1024,
            dropout=0.01,
    ):
        super(Attention, self).__init__()
        self.output_size = output_size
        self.q = nn.Linear(output_size, output_size)
        self.k = nn.Linear(output_size, output_size)
        self.v = nn.Linear(output_size, output_size)
        self.out = nn.Sequential(
            nn.Linear(output_size, output_size),
            nn.LayerNorm(output_size),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

    def forward(self, q, k, v):
        q = self.q(q)  # [4, 10, 1024]
        k = self.k(k)  # [4, 1000, 1024]
        v = self.v(v)  # [4, 1000, 1024]

        score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.output_size)
        # clip
        # score = torch.clip(score, 0) # >0 正相关保留
        score = F.softmax(score, dim=-1)  # [4, 10, 1000]
        # print(q.shape, k.shape, v.shape, score.shape)
        x = torch.matmul(score, v)  # [4, 10, 1000] * [4, 1000, 1024] --> [4, 10, 1024]
        x = self.out(x)
        # print(x.shape)
        return x


class Decoder_layer(nn.Module):
    def __init__(
            self,
            output_size=1024,
            bidirectional=False,
            consider_as_one_hot=False,
            dropout=0.15,
            blank_id=0
    ):
        super(Decoder_layer, self).__init__()
        self.rnn = nn.LSTM(input_size=output_size, hidden_size=output_size, num_layers=2, batch_first=True,
                           dropout=0.15, bidirectional=bidirectional)
        self.l1 = nn.Sequential(
            nn.Linear(output_size, output_size, bias=True),
            nn.LayerNorm(output_size),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout)
        )

        self.att = Attention(output_size=output_size, dropout=dropout)
        self.ln = nn.LayerNorm(output_size)

    def forward(self, x, en_x):
        x, _ = self.rnn(x)
        x = self.l1(x)
        x = self.att(x, en_x, en_x)
        x = self.ln(x)
        return x
