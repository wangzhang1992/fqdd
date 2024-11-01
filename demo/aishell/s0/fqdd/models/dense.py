#!python
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class InferenceBatchSoftmax(nn.Module):

    def forward(self, input_):
        if not self.training:
            return F.softmax(input_, dim=-1)
        else:
            return input_

class Swish(nn.Module):

    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.mul_(torch.sigmoid(x))
            return x
        else:
            return x * torch.sigmoid(x)

class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super().__init__()
        self.add_module("norm1", nn.BatchNorm2d(num_input_features))
        self.add_module("relu1", Swish(inplace=True))
        self.add_module("conv1", nn.Conv2d(num_input_features, bn_size * growth_rate,
                                            kernel_size=1, stride=1, bias=False))
        self.add_module("norm2", nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module("relu2", Swish(inplace=True))
        self.add_module("conv2", nn.Conv2d(bn_size * growth_rate, growth_rate,
                                            kernel_size=3, stride=1, padding=1, bias=False))
        if drop_rate > 0:
            self.add_module("do", nn.Dropout(p=drop_rate, inplace=True))

    def forward(self, x):
        new_features = super().forward(x)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module(f"denselayer{i+1}", layer)


class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.add_module("norm_tr", nn.BatchNorm2d(num_input_features))
        self.add_module("swish_tr", Swish(inplace=True))
        self.add_module("conv_tr", nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module("pool_tr", nn.AvgPool2d(kernel_size=3, stride=(1, 1), padding=1))


class DenseNet(nn.Module):
    """ Densenet-BC model class, based on
        `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

        Args:
            growth_rate (int) - how many filters to add each layer (`k` in paper)
            block_config (list of 4 ints) - how many layers in each pooling block
            num_init_features (int) - the number of filters to learn in the first convolution layer
            bn_size (int) - multiplicative factor for number of bottle neck layers
              (i.e. bn_size * k features in the bottleneck layer)
            drop_rate (float) - dropout rate after each dense layer
    """
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0.5, num_classes=1000):
        super().__init__()

        # First convolution
        #self.hidden = nn.Sequential([
        #    #("view_i", View(dim=(-1, 2, 129, 21))),
        #    nn.Conv2d(2, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
        #    nn.BatchNorm2d(num_init_features),
        #    Swish(inplace=True),
        #    nn.MaxPool2d(kernel_size=3, stride=(2, 1), padding=1),
        #])
        self.hidden = nn.Sequential(
            nn.Conv2d(1, num_init_features, kernel_size=(11, 1), stride=(3, 2), padding=(5, 0)),
            nn.BatchNorm2d(num_init_features),
            #nn.ReLU(inplace=True),
            #nn.Hardtanh(0, 20, inplace=True),
            Swish(inplace=True),
            nn.Conv2d(num_init_features, num_init_features, kernel_size=(11, 1), stride=(3, 2), padding=(5, 0)),
            nn.BatchNorm2d(num_init_features),
            #nn.ReLU(inplace=True),
            #nn.Hardtanh(0, 20, inplace=True)
            Swish(inplace=True),
        )
        # [4, 32, 500, 20]
        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.hidden.add_module(f"denseblock{i+1}", block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.hidden.add_module(f"transition{i+1}", trans)
                num_features = num_features // 2
        # [4, 131, 63, 20]
        # Final layer
        self.hidden.add_module("norm_f", nn.BatchNorm2d(num_features))
        self.hidden.add_module("relu_f", Swish())
        #self.hidden.add_module("pool_f", nn.AvgPool2d(kernel_size=3, stride=(2, 1), padding=1))
        # (4, 20, 131, 8)
        self.fc = nn.Sequential(
            nn.Linear(131 * 9, 1024),
            #nn.ReLU(),
            #nn.BatchNorm1d(174),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            #nn.BatchNorm1d(174),
            nn.ReLU(),
            nn.Dropout(p=0.15, inplace=True),
            nn.Linear(1024, num_classes),
        )
        #self.softmax = InferenceBatchSoftmax()
        self.softmax = nn.LogSoftmax(dim=1)
        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)
        x = self.hidden(x)
        # BxCxWxH -> BxHxCxW -> BxTxH
        x = x.transpose(2, 3).transpose(1, 2)  #4, 131, 20, 20 ) (4, 131, 20, 20)  (4, 20 ,131, 20)
        x = self.fc(x.view(x.size(0), x.size(1), -1))
        x = self.softmax(x)
        return x


def densenet_custom(**kwargs):
    return DenseNet(growth_rate=4, block_config=(6, 12, 24, 16), num_init_features=32, **kwargs)


def densenet121(**kwargs):
    return DenseNet(growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, **kwargs)


def densenet169(**kwargs):
    return DenseNet(growth_rate=32, block_config=(6, 12, 32, 32), num_init_features=64, **kwargs)


def densenet201(**kwargs):
    return DenseNet(growth_rate=32, block_config=(6, 12, 48, 32), num_init_features=64, **kwargs)


def densenet161(**kwargs):
    return DenseNet(growth_rate=48, block_config=(6, 12, 36, 24), num_init_features=96, **kwargs)

'''
if __name__ == "__main__":
    print("densenet")
    
    test_data = torch.randn(4, 398, 120)
    net = densenet_custom(num_classes=2048)
    print(net)
    print(net(test_data).shape)
'''
