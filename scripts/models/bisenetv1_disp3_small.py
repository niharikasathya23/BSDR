import torch
import math
from torch import nn, Tensor
from torch.nn import functional as F
from models.backbones import MicroNet
from .modules.common import ConvModule
from ndr.layers import *

class SpatialPath(nn.Module):
    def __init__(self, c1, c2) -> None:
        super().__init__()
        ch = 64
        self.conv_7x7 = ConvModule(c1, ch, 7, 2, 3)
        self.conv_3x3_1 = ConvModule(ch, ch, 3, 2, 1)
        self.conv_3x3_2 = ConvModule(ch, ch, 3, 2, 1)
        self.conv_1x1 = ConvModule(ch, c2, 1, 1, 0)
        # self.conv_7x7 = ConvModule(c1, ch, 7, 1, 3)
        # self.conv_3x3_1 = ConvModule(ch, ch, 3, 1, 1)
        # self.conv_3x3_2 = ConvModule(ch, ch, 3, 1, 1)
        # self.conv_1x1 = ConvModule(ch, c2, 1, 1, 0)

    def forward(self, x):
        x = self.conv_7x7(x)
        x = self.conv_3x3_1(x)
        x = self.conv_3x3_2(x)
        return self.conv_1x1(x)


class ContextPath(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        c3, c4 = self.backbone.channels[-2:]

        self.arm16 = AttentionRefinmentModule(c3, 128)
        self.arm32 = AttentionRefinmentModule(c4, 128)

        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(c4, 128, 1, 1, 0)
        )

        self.up16 = nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=True)
        self.up32 = nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=True)

        self.refine16 = ConvModule(128, 128, 3, 1, 1)
        self.refine32 = ConvModule(128, 128, 3, 1, 1)


    def forward(self, x):
        _, _, down16, down32 = self.backbone(x)                 # 4x256x64x128, 4x512x32x64

        print(f"Initial down16 size: {down16.size()}")
        print(f"Initial down32 size: {down32.size()}")
        arm_down16 = self.arm16(down16)                 # 4x128x64x128
        arm_down32 = self.arm32(down32)                 # 4x128x32x64
        print(f"Initial down16 size: {arm_down16.size()}")
        print(f"Initial down32 size: {arm_down32.size()}")

        global_down32 = self.global_context(down32)     # 4x128x1x1
        global_down32 = F.interpolate(global_down32, size=down32.size()[2:], mode='bilinear', align_corners=True)   # 4x128x32x64

        arm_down32 = arm_down32 + global_down32             # 4x128x32x64
        arm_down32 = self.up32(arm_down32)                  # 4x128x64x128
        arm_down32 = self.refine32(arm_down32)              # 4x128x64x128

        # Check if arm_down16 and arm_down32 have the same spatial dimensions
        if arm_down16.size()[2:] != arm_down32.size()[2:]:
            # Resize arm_down32 to match arm_down16's dimensions
            arm_down32 = F.interpolate(arm_down32, size=arm_down16.size()[2:], mode='bilinear', align_corners=True)

        print(f"arm_down16 size: {arm_down16.size()}")
        print(f"arm_down32 size: {arm_down32.size()}")

        arm_down16 = arm_down16 + arm_down32                            # 4x128x64x128
        arm_down16 = self.up16(arm_down16)                  # 4x128x128x256
        arm_down16 = self.refine16(arm_down16)              # 4x128x128x256

        return arm_down16, arm_down32


class AttentionRefinmentModule(nn.Module):
    def __init__(self, c1, c2) -> None:
        super().__init__()
        self.conv_3x3 = ConvModule(c1, c2, 3, 1, 1)

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.Sigmoid()
        )

    def forward(self, x):
        fm = self.conv_3x3(x)
        fm_se = self.attention(fm)
        return fm * fm_se


class FeatureFusionModule(nn.Module):
    def __init__(self, c1, c2, reduction=1) -> None:
        super().__init__()
        self.conv_1x1 = ConvModule(c1, c2, 1, 1, 0)
        # self.upsample = nn.Upsample(scale_factor=8, mode='bilinear')

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, c2 // reduction, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(c2 // reduction, c2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        # x2 = self.upsample(x2)
        fm = torch.cat([x1, x2], dim=1)
        fm = self.conv_1x1(fm)
        fm_se = self.attention(fm)
        return fm + fm * fm_se


class Head(nn.Module):
    def __init__(self, c1, n_classes, upscale_factor, is_aux=False) -> None:
        super().__init__()
        ch = 256 if is_aux else 64
        c2 = n_classes * upscale_factor * upscale_factor
        self.conv_3x3 = ConvModule(c1, ch, 3, 1, 1)
        self.conv_1x1 = nn.Conv2d(ch, c2, 1, 1, 0)
        self.upscale = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv_1x1(self.conv_3x3(x))
        return self.upscale(x)

class DispHead(nn.Module):
    def __init__(self, c1, n_classes, upscale_factor, is_aux=False) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.mlp_c = MLPClassification(max_disp=256)
        self.mlp_o = MLPOffset()

    def upsample_1d(self, x, d2=80):
        batch_size, channels, total_size = x.size()
        # Assuming the total size is the product of the height, width, and possibly other factors (like upsampling)
        # and given d2, calculate d1 accordingly
        d1 = total_size // d2  # This assumes the total size is directly divisible by d2

        if total_size % d2 != 0:
            raise ValueError(f"Total size {total_size} is not directly divisible by width {d2}, got total size: {total_size}")

        x = x.reshape(batch_size, channels, d1, d2)
        x = self.upsample(x)
        # Flatten the tensor back to 3D, if necessary, or handle accordingly
        return x


    def forward(self, x):
        print(f"Input tensor shape: {x.shape}")
        x = torch.flatten(x, start_dim=2, end_dim=3)
        print(f"Flattened tensor shape: {x.shape}")

        conv3_c = self.mlp_c(x)
        print(f"Output shape after MLP_c: {conv3_c.shape}")

        argmax_conv3 = torch.argmax(conv3_c, axis=1, keepdim=True).float()
        print(f"Argmax tensor shape: {argmax_conv3.shape}")

        conv2_o = self.mlp_o(x, argmax_conv3)
        print(f"Output shape after MLP_o: {conv2_o.shape}")

        argmax_conv3 = self.upsample_1d(argmax_conv3)
        print(f"Upsampled argmax tensor shape: {argmax_conv3.shape}")
        conv2_o = self.upsample_1d(conv2_o)
        print(f"Upsampled conv2_o tensor shape: {conv2_o.shape}")

        if self.training:
            conv3_c = self.upsample_1d(conv3_c)
            print(f"Upsampled conv3_c tensor shape: {conv3_c.shape}")
            probabilities = nn.functional.softmax(conv3_c, dim=1)
            print(f"Probabilities tensor shape: {probabilities.shape}")

        D_tilda = argmax_conv3 + conv2_o
        return D_tilda, probabilities if self.training else D_tilda


class BiSeNetv1Disp3(nn.Module):
    def __init__(self, backbone: str = 'MicroNet-M1', num_classes: int = 2, using_separation_loss=False, distillation=False) -> None:
        super().__init__()
        backbone, variant = backbone.split('-')

        self.distillation = distillation

        self.context_path = ContextPath(eval(backbone)(variant))
        self.spatial_path = SpatialPath(2, 128)
        self.ffm = FeatureFusionModule(256, 96)

        self.logits_head = Head(96, num_classes, upscale_factor=8, is_aux=False)
        self.disp_head = DispHead(96, 1, upscale_factor=8, is_aux=False)
        self.context16_head = Head(128, num_classes, upscale_factor=8, is_aux=True)
        self.context32_head = Head(128, num_classes, upscale_factor=16, is_aux=True)

        self.using_separation_loss = using_separation_loss
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def init_pretrained(self, pretrained: str = None) -> None:
        if pretrained:
            self.context_path.backbone.load_state_dict(torch.load(pretrained, map_location='cpu'), strict=False)

    def forward(self, x):                                       # 4x3x1024x2048
        spatial_out = self.spatial_path(x)                      # 4x128x128x256
        context16, context32 = self.context_path(x)             # 4x128x128x256, 4x128x64x128

        fm_fuse = self.ffm(spatial_out, context16)                      # 4x256x128x256
        if self.distillation:
            return fm_fuse
        logits = self.logits_head(fm_fuse)                      # 4xn_classesx1024x2048
        if self.training:
            D_tilda, probabilities = self.disp_head(fm_fuse)
        else:
            D_tilda = self.disp_head(fm_fuse)

        if self.training:
            context_out16 = self.context16_head(context16)      # 4xn_classesx1024x2048
            context_out32 = self.context32_head(context32)      # 4xn_classesx1024x2048

            if self.using_separation_loss:
                return logits, context32, context_out32, D_tilda, probabilities
            else:
                return logits, context_out16, context_out32, D_tilda, probabilities

        return logits, D_tilda


if __name__ == '__main__':
    model = BiSeNetv1Disp3('MicroNet-M1', 19)
    # model.init_pretrained('checkpoints/backbones/resnet/resnet18.pth')
    model.eval()
    image = torch.randn(1, 2, 384, 640)
    output = model(image)
    print(output[0].shape)
