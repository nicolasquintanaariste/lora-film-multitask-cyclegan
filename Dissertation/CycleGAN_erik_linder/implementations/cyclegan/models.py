import torch.nn as nn
import torch.nn.functional as F
import torch


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
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
