import torch
import torch.nn as nn

from ndr.backbones import VGG13_adopted
from ndr.layers import *
from ndr.decoders import Decoder

class Model(nn.Module):

    def __init__(self, max_disp=256, distillation=False):
        super(Model, self).__init__()

        self.max_disp = max_disp
        self.distillation = distillation

        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.backbone_rgb = VGG13_adopted(3, use_bn = False).apply(weights_init)
        self.backbone_disp = VGG13_adopted(1, use_bn = False).apply(weights_init)

        self.upsampler = nn.UpsamplingBilinear2d(scale_factor=2)

        self.mlp_c = MLPClassification(self.max_disp).apply(weights_init)
        self.mlp_o = MLPOffset().apply(weights_init)

        self.decoder = Decoder(use_bn = False).apply(weights_init)

    def forward(self, x_rgb, x_disp):

        conv3_rgb, conv5_rgb, conv7_rgb, conv9_rgb = self.backbone_rgb(x_rgb)
        conv3_d, conv5_d, conv7_d, conv9_d = self.backbone_disp(x_disp)

        conv3 = conv3_rgb + conv3_d
        conv5 = conv5_rgb + conv5_d
        conv7 = conv7_rgb + conv7_d
        conv9 = conv9_rgb + conv9_d

        upsample3, upconv13 = self.decoder(conv3, conv5, conv7, conv9)
        # this is wrong, because bilinear grid sample is not supported, TODO: replace that with grid sample equivalent
        upsample3 = self.upsampler(upsample3)

        x = torch.cat([upsample3, upconv13], axis = 1)
        if self.distillation: return x
        x = torch.flatten(x, start_dim=2, end_dim=3) # FOR CONV1D

        conv3_c = self.mlp_c(x)

        probabilities = nn.functional.softmax(conv3_c, dim=1)
        # argmax_conv3 = torch.argmax(conv3_c, axis = 1, keepdim = True).float() # cast to float for blob conversion
        argmax_conv3 = torch.argmax(probabilities, axis = 1, keepdim = True).float() # cast to float for blob conversion

        conv2_o = self.mlp_o(x, argmax_conv3)

        D_tilda = argmax_conv3 + conv2_o

        return D_tilda, probabilities

class Teacher(nn.Module):

    def __init__(self, max_disp=256):
        super(Teacher, self).__init__()

        self.max_disp = max_disp

        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.backbone_rgb = VGG13_adopted(3, use_bn = False).apply(weights_init)
        self.backbone_disp = VGG13_adopted(1, use_bn = False).apply(weights_init)

        self.upsampler = nn.UpsamplingBilinear2d(scale_factor=2)
        self.downsampler = nn.UpsamplingBilinear2d(scale_factor=1/8)
        #
        # self.mlp_c = MLPClassification(self.max_disp).apply(weights_init)
        # self.mlp_o = MLPOffset().apply(weights_init)

        self.head = DispHead(96, 1, upscale_factor=8, is_aux=False)

        self.decoder = Decoder(use_bn = False).apply(weights_init)

    def forward(self, x_rgb, x_disp):

        conv3_rgb, conv5_rgb, conv7_rgb, conv9_rgb = self.backbone_rgb(x_rgb)
        conv3_d, conv5_d, conv7_d, conv9_d = self.backbone_disp(x_disp)

        conv3 = conv3_rgb + conv3_d
        conv5 = conv5_rgb + conv5_d
        conv7 = conv7_rgb + conv7_d
        conv9 = conv9_rgb + conv9_d

        upsample3, upconv13 = self.decoder(conv3, conv5, conv7, conv9)
        # this is wrong, because bilinear grid sample is not supported, TODO: replace that with grid sample equivalent
        upsample3 = self.upsampler(upsample3)

        # print('feat list?', upsample3.shape, upconv13.shape)

        x = torch.cat([upsample3, upconv13], axis = 1)

        x = self.downsampler(x)

        x = self.head(x)

        return x

class ModelLite(nn.Module):

    def __init__(self, max_disp=256):
        super(ModelLite, self).__init__()

        self.max_disp = max_disp

        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.backbone_rgb = VGG13_adopted(3, use_bn = False).apply(weights_init)
        self.backbone_disp = VGG13_adopted(1, use_bn = False).apply(weights_init)

        self.upsampler = nn.UpsamplingBilinear2d(scale_factor=2)
        self.downsampler = nn.UpsamplingBilinear2d(scale_factor=1/8)
        self.upsample2 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        self.mlp_c = MLPClassification(self.max_disp).apply(weights_init)
        self.mlp_o = MLPOffset().apply(weights_init)

        self.decoder = Decoder(use_bn = False).apply(weights_init)

    def upsample_1d(self, x, d1=48, d2=80):
        x = x.reshape(1, 1, d1, d2)
        x = self.upsample2(x)
        x = x.reshape(1, 1, d1*d2*8*8)
        return x

    def forward(self, x_rgb, x_disp):

        conv3_rgb, conv5_rgb, conv7_rgb, conv9_rgb = self.backbone_rgb(x_rgb)
        conv3_d, conv5_d, conv7_d, conv9_d = self.backbone_disp(x_disp)

        conv3 = conv3_rgb + conv3_d
        conv5 = conv5_rgb + conv5_d
        conv7 = conv7_rgb + conv7_d
        conv9 = conv9_rgb + conv9_d

        upsample3, upconv13 = self.decoder(conv3, conv5, conv7, conv9)
        upsample3 = self.upsampler(upsample3)

        x = torch.cat([upsample3, upconv13], axis = 1)
        x = self.downsampler(x)

        x = torch.flatten(x, start_dim=2, end_dim=3) # FOR CONV1D

        conv3_c = self.mlp_c(x)

        probabilities = nn.functional.softmax(conv3_c, dim=1)
        # argmax_conv3 = torch.argmax(probabilities, axis = 1, keepdim = True).float() # cast to float for blob conversion
        argmax_conv3 = torch.argmax(conv3_c, axis = 1, keepdim = True).float() # cast to float for blob conversion

        conv2_o = self.mlp_o(x, argmax_conv3)

        argmax_conv3 = self.upsample_1d(argmax_conv3)
        conv2_o = self.upsample_1d(conv2_o)

        D_tilda = argmax_conv3 + conv2_o

        return D_tilda, probabilities
