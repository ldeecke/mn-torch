import functools
import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt
from nn.ops import ModeNorm
from torch.nn import BatchNorm2d

resnet20 = lambda config: ResNet(BasicBlock, [3, 3, 3], config)
resnet56 = lambda config: ResNet(BasicBlock, [9, 9, 9], config)
resnet110 = lambda config: ResNet(BasicBlock, [18, 18, 18], config)


class ResNet(nn.Module):
    def __init__(self, block, layers, config):
        super(ResNet, self).__init__()
        self.mn = config.mn

        if config.mn == "full":
            Norm = functools.partial(ModeNorm, momentum=config.momentum, n_components=config.num_components)
        elif config.mn == "init":
            InitNorm = functools.partial(ModeNorm, momentum=config.momentum, n_components=config.num_components)
            Norm = functools.partial(BatchNorm2d, momentum=config.momentum)

        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)

        self.norm1 = InitNorm(16) if config.mn == "init" else Norm(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 16, layers[0], Norm)
        self.layer2 = self._make_layer(block, 32, layers[1], Norm, stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], Norm, stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64 * block.expansion, config.num_classes)

        self._init_weights()


    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2./n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, ModeNorm):
                m.alpha.data.fill_(1)
                m.beta.data.zero_()


    def _make_layer(self, block, planes, blocks, norm, stride=1):
        downsample = None
        if (stride != 1) or (self.inplanes != planes * block.expansion):
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm(planes * block.expansion)
                )

        layers = []
        layers.append(block(self.inplanes, planes, norm, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm))

        return nn.Sequential(*layers)


class BasicBlock(nn.Module):
    expansion=1

    def __init__(self, inplanes, planes, norm, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = self._conv3x3(inplanes, planes, stride)
        self.norm1 = norm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = self._conv3x3(planes, planes)
        self.norm2 = norm(planes)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


    def _conv3x3(self, in_planes, out_planes, stride=1):
        '''3x3 convolution with padding'''
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, norm, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.norm1 = norm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm2 = norm(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.norm3 = norm(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                norm(self.expansion*planes)
            )


    def forward(self, x):
        out = F.relu(self.norm1(self.conv1(x)))
        out = F.relu(self.norm2(self.conv2(out)))
        out = self.norm3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
