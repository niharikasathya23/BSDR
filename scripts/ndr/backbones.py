import torch
import torch.nn as nn
import timm

from ndr.layers import ConvBNReLU, ConvReLU

class VGG13_adopted(nn.Module):

    def __init__(self, in_chanels, use_bn = False):
        super(VGG13_adopted, self).__init__()

        conv_module = ConvBNReLU if use_bn else ConvReLU

        self.conv0 = conv_module(in_chanels, 64, 3, padding = 1)
        self.pool0 = nn.MaxPool2d(2)

        self.conv1 = conv_module(64, 64, 3, padding = 1)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = conv_module(64, 128, 3, padding = 1)
        self.conv3 = conv_module(128, 128, 3, padding = 1)
        self.pool2 = nn.MaxPool2d(2)

        self.conv4 = conv_module(128, 256, 3, padding = 1)
        self.conv5 = conv_module(256, 256, 3, padding = 1)
        self.pool3 = nn.MaxPool2d(2)

        self.conv6 = conv_module(256, 512, 3, padding = 1)
        self.conv7 = conv_module(512, 512, 3, padding = 1)
        self.pool4 = nn.MaxPool2d(2)

        self.conv8 = conv_module(512, 512, 3, padding = 1)
        self.conv9 = conv_module(512, 512, 3, padding = 1)

    def forward(self, x):

        x = self.conv0(x)
        x = self.pool0(x)

        x = self.conv1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        conv3 = self.conv3(x)
        x = self.pool2(conv3)

        x = self.conv4(x)
        conv5 = self.conv5(x)
        x = self.pool3(conv5)

        x = self.conv6(x)
        conv7 = self.conv7(x)
        x = self.pool4(conv7)

        x = self.conv8(x)
        x = self.conv9(x)

        # print(conv3.shape)
        # print(conv5.shape)
        # print(conv7.shape)
        # print(x.shape)
        return conv3, conv5, conv7, x

class ResNet18(nn.Module):
    """ ResNet18 class from timm """

    def __init__(self, n_channels):
        super(ResNet18, self).__init__()
        """
        Model output: 4 feature maps of (N, C, H, W)
            (N, 64, 96, 96),
            (N, 128, 48, 48),
            (N, 256, 24, 24),
            (N, 512, 12, 12)
        These resolutions match the VGG13 encoder but the number of channels are different
        """
        self.model = timm.create_model('resnet18', features_only=True, out_indices=(1, 2, 3, 4), pretrained=True)
        # set the input layer to take a variable number of channels
        self.model.conv1 = torch.nn.Conv2d(n_channels, 64, 7, stride=2, padding=3, bias=False)

    def forward(self, x):
        conv3, conv5, conv7, conv9 = self.model(x)
        # print(conv3.shape)
        # print(conv5.shape)
        # print(conv7.shape)
        # print(conv9.shape)
        return conv3, conv5, conv7, conv9
