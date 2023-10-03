import torch
from torch import nn

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
