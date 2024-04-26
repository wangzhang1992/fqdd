import torch
import math
import copy
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # batch*768*40
        self.l1 = nn.Sequential(nn.Linear(self.input_dim, 128), nn.ReLU(True))
        self.l2 = nn.Sequential(nn.Linear(128, 512), nn.ReLU(True))
        self.l3 = nn.Sequential(nn.Linear(512, 512), nn.ReLU(True))
        self.l4 = nn.Sequential(nn.Linear(512, 512), nn.ReLU(True))
        self.l5 = nn.Sequential(nn.Linear(512, 1024), nn.ReLU(True))
        self.l6 = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU(True))
        self.l7 = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU(True))
        self.l8 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(True))
        self.l9 = nn.Sequential(nn.Linear(512, 512), nn.ReLU(True))
        self.l10 = nn.Sequential(nn.Linear(512, self.output_dim))

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        return self.l10(x)
