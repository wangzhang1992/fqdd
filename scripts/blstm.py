import torch
import torch.nn as nn
import torch.nn.functional as F


class BlstmNet(nn.Module):
    def __init__(self, input_dim, output_dim, hiddle, layers, drop, bi):
        super(BlstmNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hiddle = hiddle
        self.layers = layers
        self.drop = drop
        self.bi = bi
        self.blstm = nn.LSTM(self.input_dim, self.hiddle, self.layers, batch_first=True, dropout=self.drop,
                             bidirectional=self.bi)
        self.l1_dim = self.hiddle
        if self.bi:
            self.l1_dim = self.hiddle * 2
        self.line1 = nn.Sequential(nn.Linear(self.l1_dim, self.output_dim))
        # self.line2 = nn.Sequential(nn.Linear(512, 512))
        # self.line3 = nn.Sequential(nn.Linear(self.l1_dim, self.output_dim))
        # self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        x, _ = self.blstm(x)
        x = self.line1(x)
        # x = self.line2(x)
        # x = self.line3(x)
        # return self.softmax(x)

        return x
