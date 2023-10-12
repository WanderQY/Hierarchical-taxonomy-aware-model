import torch
from torch import nn

class PositonAttention(nn.Module):
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
