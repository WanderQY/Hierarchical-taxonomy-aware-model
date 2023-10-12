import math
import torch
from torch import nn
from torch.nn import functional as F


#### attention block ####

class PositionAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.convB = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False)
        self.convC = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False)
        self.convD = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False)
        self.gamma = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.gamma.data.fill_(0.)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        b, c, h, w = x.size()
        B = self.convB(x).view(b, c, h * w)  # (B, C, H*W)
        C = self.convC(x).view(b, c, h * w)
        D = self.convD(x).view(b, c, h * w)
        S = self.softmax(torch.matmul(B.transpose(1, 2), C))
        attention_maps = torch.matmul(D, S.transpose(1, 2)).view(b, c, h, w)
        # gamma is a parameter which can be training and iter
        feature_matrix = self.gamma * attention_maps + x

        return feature_matrix


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True, use_attention=False):
        super(Block, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

        if use_attention:
            self.ca = PositionAttention(out_filters)
        self.use_attention = use_attention

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        if self.use_attention:
            x = self.ca(x)
        x += skip
        return x

class Xception(nn.Module):
    def __init__(self, num_classes=145, use_attention=False):   # 128x938x1
        """ Constructor
        Args:
            num_classes: number of classes
            use_attention: if use attentions?
        """
        super(Xception, self).__init__()
        self.num_classes = num_classes
        self.use_attention = use_attention

        # Entry flow
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)  # (128+2*0-3)/2(向下取整)+1, 63x468x32
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)  # 31x233x64
        self.bn2 = nn.BatchNorm2d(64)
        # do relu here
        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True, use_attention=False)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True, use_attention=False)
        self.block3 = Block(256, 728, 2, 2, start_with_relu=True, grow_first=True, use_attention=False)

        # Middle flow
        self.block4 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True, use_attention=False)
        self.block5 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True, use_attention=False)
        self.block6 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True, use_attention=False)
        self.block7 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True, use_attention=False)
        self.block8 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True, use_attention=False)
        self.block9 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True, use_attention=False)
        self.block10 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True, use_attention=False)
        self.block11 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True, use_attention=False)

        # Exit flow
        self.block12 = Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False, use_attention=self.use_attention)
        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)
        # do relu here
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)
        self.fc = nn.Linear(2048, num_classes)

        # ------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        # -----------------------------

    def forward(self, x):
        # Entry flow

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        # Middle flow
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)

        # Exit flow
        x = self.block12(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x



if __name__ == '__main__':
    model = Xception(num_classes=150, use_attention=True)  # 38.82M
    #model = Xception(num_classes=1500, use_attention=True) 

    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    input = torch.randn(16, 3, 256, 431)
    out = model(input)
