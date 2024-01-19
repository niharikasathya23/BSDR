import torch
import torch.nn as nn

class ConvBNReLU(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,bias=True):
        super(ConvBNReLU, self).__init__()

        self.conv = torch.nn.Conv2d(in_channels,out_channels, kernel_size, stride = stride, padding = padding, dilation = dilation,bias = bias)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ConvReLU(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,bias=True):
        super(ConvReLU, self).__init__()

        self.conv = torch.nn.Conv2d(in_channels,out_channels, kernel_size, stride = stride, padding = padding, dilation = dilation,bias = bias)
        self.relu = torch.nn.ReLU()

    def forward(self,x):
        x = self.conv(x)
        x = self.relu(x)
        return x

class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)

class ConvSine(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,bias=True):
        super(ConvSine, self).__init__()

        self.conv = torch.nn.Conv1d(in_channels,out_channels, kernel_size, stride = stride, padding = padding, dilation = dilation,bias = bias)
        self.sine = Sine()

    def forward(self,x):
        x = self.conv(x)
        x = self.sine(x)
        return x

class ConvTanh(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,bias=True):
        super(ConvTanh, self).__init__()

        self.conv = torch.nn.Conv1d(in_channels,out_channels, kernel_size, stride = stride, padding = padding, dilation = dilation,bias = bias)
        self.tanh = torch.nn.Tanh()

    def forward(self,x):
        x = self.conv(x)
        x = self.tanh(x)
        return x

class MLPClassificationLite(nn.Module):

    def __init__(self, max_disp):
        super(MLPClassificationLite, self).__init__()

        self.conv0 = ConvTanh(96, max_disp, 1)

    def forward(self, x):

        return self.conv0(x)


class MLPOffsetLite(nn.Module):

    def __init__(self):
        super(MLPOffsetLite, self).__init__()

        self.conv0 = ConvTanh(97, 1, 1) # TODO: original implementation does not have

    def forward(self, x, argmax_conv3):
        # x must be concatenated upsample3, upconv13

        # also edited this to concat argmax_conv3 to every input
        # since the denoted concatenation was off by 1
        x = torch.cat([x, argmax_conv3], axis = 1)
        conv0 = self.conv0(x)

        return conv0
