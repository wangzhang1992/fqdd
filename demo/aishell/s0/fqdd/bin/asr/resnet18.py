import torch
import torch.nn as nn
import torch.autograd.variable as variable
import torch.nn.functional as F
from thop import profile


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Base_Block(nn.Module):
    expansion = 1

    def __init__(self, in_Channel, out_Channel, stride=1, down_Sample=None):
        super(Base_Block, self).__init__()
        self.in_Channel = in_Channel
        self.out_Channel = out_Channel
        self.stride = stride
        self.down_Sample = down_Sample
        self.conv1 = conv3x3(self.in_Channel, self.out_Channel, stride)
        self.bachN1 = nn.BatchNorm2d(self.out_Channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(self.out_Channel, self.out_Channel)
        self.bachN2 = nn.BatchNorm2d(self.out_Channel)

    def forward(self, x):
        identity = x
        cov1 = self.conv1(x)
        bN1 = self.bachN1(cov1)
        relu1 = self.relu(bN1)
        cov2 = self.conv2(relu1)
        bN2 = self.bachN2(cov2)

        if self.down_Sample is not None:
            identity = self.down_Sample(x)
        out = bN2 + identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, out_Classify, block, n_block, batch_size):
        super(ResNet, self).__init__()
        self.out_Classify = out_Classify
        self.n_block = n_block
        self.inplanes = 64
        self.block = block
        self.batch_size = batch_size
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._one_layer(self.block, 64, self.n_block[0])
        self.layer2 = self._one_layer(self.block, 128, self.n_block[0], stride=2)
        self.layer3 = self._one_layer(self.block, 256, self.n_block[0], stride=2)
        self.layer4 = self._one_layer(self.block, 512, self.n_block[0], stride=2)
        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv2 = nn.Conv2d(512 * self.block.expansion, 2200, kernel_size=2, stride=2, padding=3, bias=False)
        self.bn2 = nn.BatchNorm2d(2200)

        self.line = nn.Linear(185, self.out_Classify)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        cov1 = self.conv1(x)
        bN = self.bn1(cov1)
        relu1 = self.relu(bN)
        maxp = self.maxpool(relu1)
        out = self.layer1(maxp)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        # out = self.avgpool(out)
        out = out.reshape((self.batch_size, 2200, -1))
        out = self.line(out)
        return out

    def _one_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


# batch=1 flops:160574164992.0	params:12299846.0
# torch.manual_seed(2021)
# x_train = [torch.randn(16, 2200, 120) for i in range(10)]
# net = model = ResNet(4078, Base_Block, [2, 2, 2, 2], 16)
# input = torch.randn(16, 2200, 120)
# flops, params = profile(net, inputs=(input,))
# print("flops:{}\tparams:{}".format(flops, params))
# for batch_x in x_train:
#     res = net(batch_x)
#     print(res.size())
