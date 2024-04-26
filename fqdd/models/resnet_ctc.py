#!python
import os, sys
sys.path.insert(0, "./")
import torch
import torch.nn as nn
from torch.autograd import Variable
from thop import profile
from script.utils.misc import Swish, InferenceBatchSoftmax


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        #self.relu1 = nn.ReLU(inplace=True)
        self.relu1 = Swish(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        #self.relu2 = nn.ReLU(inplace=True)
        self.relu2 = Swish(inplace=True)
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu2(out)
        return out


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        #self.relu1 = nn.ReLU(inplace=True)
        self.relu1 = Swish(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        #self.relu2 = nn.ReLU(inplace=True)
        self.relu2 = Swish(inplace=True)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.downsample = downsample
        #self.relu3 = nn.ReLU(inplace=True)
        self.relu3 = Swish(inplace=True)
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu3(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 32
        super(ResNet, self).__init__()

        #self.conv1 = nn.Conv2d(2, self.inplanes, kernel_size=7, stride=(2, 1), padding=3, bias=False)
        #self.bn1 = nn.BatchNorm2d(self.inplanes)
        ##self.relu = nn.ReLU(inplace=True)
        #self.relu = Swish(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv = nn.Sequential(
            nn.Conv2d(1, self.inplanes, kernel_size=(41, 11), stride=(2, 2), padding=(0, 5)),
            nn.BatchNorm2d(self.inplanes),
            #nn.ReLU(inplace=True),
            #nn.Hardtanh(0, 20, inplace=True),
            Swish(inplace=True),
            nn.Conv2d(self.inplanes, self.inplanes, kernel_size=(21, 11), stride=(2, 1), padding=(0, 5)),
            nn.BatchNorm2d(self.inplanes),
            #nn.ReLU(inplace=True),
            #nn.Hardtanh(0, 20, inplace=True)
            Swish(inplace=True),
        )

        # Based on the conv formula (W - F + 2P) // S + 1
        freq_size = 1500
        freq_size = (freq_size - 41) // 2 + 1
        freq_size = (freq_size - 21) // 2 + 1

        self.layer1 = self._make_layer(block, self.inplanes, layers[0])
        self.layer2 = self._make_layer(block, self.inplanes, layers[1], stride=(2, 1))
        self.layer3 = self._make_layer(block, self.inplanes, layers[2], stride=(2, 1))
        self.layer4 = self._make_layer(block, self.inplanes, layers[3], stride=(2, 1))

        #self.avgpool = nn.AvgPool2d(3, stride=1, padding=(1, 1))

        freq_size = (freq_size - 3 + 2) // 2 + 1
        freq_size = (freq_size - 3 + 2) // 2 + 1
        freq_size = (freq_size - 3 + 2) // 2 + 1
        print(freq_size)
        print(self.inplanes)
        self.fc1 = nn.Linear(self.inplanes * freq_size, 1024)
        self.do1 = nn.Dropout(p=0.5, inplace=True)
        self.fc2 = nn.Linear(1024, num_classes)

        self.softmax = InferenceBatchSoftmax()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        if len(x.shape) !=4:
            x = x.unsqueeze(1)
        x = self.conv(x)
        #x = self.bn1(x)
        #x = self.relu(x)
        #x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #x = self.avgpool(x)
        # BxCxWxH -> BxHxCxW -> BxTxH
        x = x.transpose(2, 3).transpose(1, 2)
        print(x.shape)
        print(x.view(x.size(0), x.size(1), -1).shape)
        x = self.fc1(x.view(x.size(0), x.size(1), -1))
        x = self.do1(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet152(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)


if __name__ == "__main__":
   
    # btch=16: flops:1048655872000.0       params:21151534.0
    torch.manual_seed(2021)
    x_train = [torch.randn(4, 1500, 120) for i in range(10)]
    inputd = torch.randn(4, 1500, 120)
    print("input.shape:{}".format(inputd.shape))
    net = resnet152(num_classes=4078)
    #for name, layer in net.named_parameters():
    #    print(name, layer)
    print(net)
    flops, params = profile(net, inputs=(inputd,))
    print("flops:{}\tparams:{}".format(flops, params))
    for batch_x in x_train:
        res = net(batch_x)
        print(res.size())
    print("resnet")
    net = resnet152(num_classes=4078)
    print(net)
