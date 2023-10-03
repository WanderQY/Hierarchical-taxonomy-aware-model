import math
import torch
from torch import nn
from torch.nn import functional as F


#### attention block ####
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, groups=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // groups)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.relu = h_swish()

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # 维度换位

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()
        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)

        y = identity * x_w * x_h

        return y


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.convB = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False)
        self.convC = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False)
        self.convD = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False)
        # 创建一个可学习参数a作为权重,并初始化为0.
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
            # self.ca = CoordAtt(out_filters, out_filters)
            self.ca = SelfAttention(out_filters)
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
    #model = Xception(num_classes=1500, use_attention=True)  # 24.44M
    # 模型参数量
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    input = torch.randn(16, 3, 256, 431)
    out = model(input)
