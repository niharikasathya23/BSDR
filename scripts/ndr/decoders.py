import torch
import torch.nn as nn

from ndr.layers import ConvBNReLU, ConvReLU


class Decoder(nn.Module):

    def __init__(self, use_bn=False):
        super(Decoder, self).__init__()

        conv_module = ConvBNReLU if use_bn else ConvReLU

        self.upconv0 = conv_module(512, 512, 3, padding = 1)
        self.upconv1 = conv_module(512, 512, 3, padding = 1)
        self.upconv2 = conv_module(512, 512, 3, padding = 1)
        self.upsample0 = nn.Upsample(scale_factor=2)

        self.upconv3 = conv_module(512, 512, 3, padding = 1)
        self.upconv4 = conv_module(512, 256, 3, padding = 1)
        self.upconv5 = conv_module(256, 256, 3, padding = 1)
        self.upsample1 = nn.Upsample(scale_factor=2)

        self.upconv6 = conv_module(256, 256, 3, padding = 1)
        self.upconv7 = conv_module(256, 256, 3, padding = 1)
        self.upconv8 = conv_module(256, 128, 3, padding = 1)
        self.upsample2 = nn.Upsample(scale_factor=2)

        self.upconv9 = conv_module(128, 128, 3, padding = 1)
        self.upconv10 = conv_module(128, 128, 3, padding = 1)
        self.upconv11 = conv_module(128, 64, 3, padding = 1)
        self.upsample3 = nn.Upsample(scale_factor=2)

        self.upconv12 = conv_module(64, 64, 3, padding = 1)
        self.upsample4 = nn.Upsample(scale_factor=2)

        self.upconv13 = nn.Conv2d(64, 32, 3, padding = 1)


    def forward(self, conv3, conv5, conv7, conv9):
        x = self.upconv0(conv9)
        x = self.upconv1(x)
        x = self.upconv2(x)
        x = self.upsample0(x)

        x = x + conv7
        x = self.upconv3(x)
        x = self.upconv4(x)
        x = self.upconv5(x)
        x = self.upsample1(x)

        x = x + conv5
        x = self.upconv6(x)
        x = self.upconv7(x)
        x = self.upconv8(x)
        x = self.upsample2(x)

        x = x + conv3
        x = self.upconv9(x)
        x = self.upconv10(x)
        x = self.upconv11(x)
        upsample3 = self.upsample3(x)

        x = self.upconv12(upsample3)
        x = self.upsample4(x)
        x = self.upconv13(x)

        return upsample3, x
