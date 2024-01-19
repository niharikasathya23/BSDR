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

class ConvModule(nn.Sequential):
    def __init__(self, c1, c2, k, s=1, p=0, d=1, g=1):
        super().__init__(
            nn.Conv2d(c1, c2, k, s, p, d, g, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(True)
        )

class MLPClassification(nn.Module):

    def __init__(self, max_disp):
        super(MLPClassification, self).__init__()

        # self.conv0 = ConvSine(96, 512, 1)
        # self.conv1 = ConvSine(608, 256, 1)
        # self.conv2 = ConvSine(352, 128, 1)
        # self.conv3 = nn.Conv1d(224, max_disp, 1)

        self.conv0 = ConvTanh(96, 512, 1)
        self.conv1 = ConvTanh(608, 256, 1)
        self.conv2 = ConvTanh(352, 128, 1)
        self.conv3 = nn.Conv1d(224, max_disp, 1)


    def forward(self, x):
        # x must be concatenated upsample3, upconv13
        conv0 = self.conv0(x)

        x_in = torch.cat([conv0, x], axis = 1)
        conv1 = self.conv1(x_in)

        x_in = torch.cat([conv1, x], axis = 1)
        conv2 = self.conv2(x_in)

        x_in = torch.cat([conv2, x], axis = 1)
        conv3 = self.conv3(x_in)

        return conv3


class MLPOffset(nn.Module):

    def __init__(self):
        super(MLPOffset, self).__init__()

        # self.conv0 = ConvSine(97, 128, 1)
        # self.conv1 = ConvSine(225, 64, 1)
        # self.conv2 = ConvTanh(161, 1, 1)

        self.conv0 = ConvTanh(97, 128, 1) # TODO: original implementation does not have
        self.conv1 = ConvTanh(225, 64, 1)
        self.conv2 = ConvTanh(161, 1, 1)

    def forward(self, x, argmax_conv3):
        # x must be concatenated upsample3, upconv13

        # also edited this to concat argmax_conv3 to every input
        # since the denoted concatenation was off by 1
        x = torch.cat([x, argmax_conv3], axis = 1)
        conv0 = self.conv0(x)

        x_in = torch.cat([conv0, x], axis = 1)
        conv1 = self.conv1(x_in)

        x_in = torch.cat([conv1, x], axis = 1)
        conv2 = self.conv2(x_in)

        return conv2

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

class DispHead(nn.Module):
    def __init__(self, c1, n_classes, upscale_factor, is_aux=False) -> None:
        super().__init__()
        ch = 256 if is_aux else 64
        # c2 = n_classes * upscale_factor * upscale_factor
        c2 = n_classes
        self.conv_3x3 = ConvModule(c1, ch, 3, 1, 1)
        self.conv_1x1 = nn.Conv2d(ch, c2, 1, 1, 0)
        # self.upscale = nn.PixelShuffle(upscale_factor)
        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.conv_3x3(x)
        x = self.upsample(x)
        x = self.conv_1x1(x)
        return x
