import torch.nn as nn
import torch.nn.functional as F
import torch


# def weights_init_normal(m):
#     classname = m.__class__.__name__
#     if classname.find("Conv") != -1:
#         torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
#         if hasattr(m, "bias") and m.bias is not None:
#             torch.nn.init.constant_(m.bias.data, 0.0)
#     elif classname.find("BatchNorm2d") != -1:
#         torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
#         torch.nn.init.constant_(m.bias.data, 0.0)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif hasattr(m, 'bias') and m.bias is not None:
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           RESNET
##############################


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)

class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
        super(GeneratorResNet, self).__init__()

        channels = input_shape[0]

        # Initial convolution block
        self.reflection_pad1 = nn.ReflectionPad2d(channels)
        self.conv1 = nn.Conv2d(channels, 64, 7)
        self.norm1 = nn.InstanceNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        # Downsampling
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.norm2 = nn.InstanceNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.norm3 = nn.InstanceNorm2d(256)
        self.relu3 = nn.ReLU(inplace=True)

        # Residual blocks
        self.res_blocks = nn.ModuleList([ResidualBlock(256) for _ in range(num_residual_blocks)])

        # Upsampling
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.conv4 = nn.Conv2d(256, 128, 3, stride=1, padding=1)
        self.norm4 = nn.InstanceNorm2d(128)
        self.relu4 = nn.ReLU(inplace=True)

        self.upsample2 = nn.Upsample(scale_factor=2)
        self.conv5 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.norm5 = nn.InstanceNorm2d(64)
        self.relu5 = nn.ReLU(inplace=True)

        # Output layer
        self.reflection_pad2 = nn.ReflectionPad2d(channels)
        self.conv6 = nn.Conv2d(64, channels, 7)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Initial convolution block
        x = self.reflection_pad1(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        # Downsampling
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)

        # Residual blocks
        for res_block in self.res_blocks:
            x = res_block(x)

        # Upsampling
        x = self.upsample1(x)
        x = self.conv4(x)
        x = self.norm4(x)
        x = self.relu4(x)

        x = self.upsample2(x)
        x = self.conv5(x)
        x = self.norm5(x)
        x = self.relu5(x)

        # Output layer
        x = self.reflection_pad2(x)
        x = self.conv6(x)
        x = self.tanh(x)
        return x

##############################
#        UNET GENERATOR
##############################

class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)
        ]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.model(x)
        # Concatenate skip connection
        x = torch.cat((x, skip), dim=1)
        return x


class GeneratorUNet(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        channels = input_shape[0]

        # -------- Encoder --------
        self.down1 = UNetDown(channels, 64, normalize=False)   # 256 → 128
        self.down2 = UNetDown(64, 128)                         # 128 → 64
        self.down3 = UNetDown(128, 256)                        # 64 → 32
        self.down4 = UNetDown(256, 512)                        # 32 → 16

        # -------- Bottleneck --------
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 4, stride=2, padding=1),       # 16 → 8
            nn.ReLU(inplace=True)
        )

        # -------- Decoder --------
        self.up1 = UNetUp(512, 512)     # 8 → 16
        self.up2 = UNetUp(1024, 256)    # 16 → 32
        self.up3 = UNetUp(512, 128)     # 32 → 64
        self.up4 = UNetUp(256, 64)      # 64 → 128

        # -------- Output --------
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),                      # 128 → 256
            nn.ReflectionPad2d(3),
            nn.Conv2d(128, channels, 7),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        # Bottleneck
        bn = self.bottleneck(d4)

        # Decoder + skip connections
        u1 = self.up1(bn, d4)
        u2 = self.up2(u1, d3)
        u3 = self.up3(u2, d2)
        u4 = self.up4(u3, d1)

        return self.final(u4)
    

##############################
#        LoRA
##############################

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRAConv2d(nn.Module):
    def __init__(self, conv: nn.Conv2d, rank=4, alpha=1.0):
        super().__init__()

        self.conv = conv
        self.rank = rank
        self.alpha = alpha

        # Freeze original weights
        self.conv.weight.requires_grad = False
        if self.conv.bias is not None:
            self.conv.bias.requires_grad = False

        out_ch, in_ch, kH, kW = conv.weight.shape

        # LoRA parameters (low-rank)
        self.lora_A = nn.Parameter(
            torch.zeros(rank, in_ch * kH * kW)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_ch, rank)
        )

        # Good initialisation practice
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # Original convolution
        out = self.conv(x)

        # ---- LoRA update ----
        # (out_ch, rank) @ (rank, in_ch*kH*kW)
        delta_w = self.lora_B @ self.lora_A

        # Reshape back to Conv2d kernel
        delta_w = delta_w.view(self.conv.weight.shape)

        # Apply delta convolution
        out += self.alpha * F.conv2d(
            x,
            delta_w,
            bias=None,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )

        return out


def apply_lora_to_unet(unet_model, rank=4, alpha=1.0):
    """
    Recursively finds all Conv2d layers and wraps them with LoRA.
    
    unet_model: Description
    rank: Description
    alpha: Description
    """
    for name, module in unet_model.named_modules():
        if isinstance(module, nn.Conv2d):
            parent = unet_model
            names = name.split(".")
            for n in names[:-1]:
                parent = getattr(parent, n)
            setattr(parent, names[-1], LoRAConv2d(module, rank=rank, alpha=alpha))
    return unet_model

##############################
#        Discriminator
##############################


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)
