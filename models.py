import torch
import torch.nn as nn
from utils import check_tensor_size

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.instance_norm = nn.InstanceNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.instance_norm(self.conv1(x)))
        out = self.instance_norm(self.conv2(out))
        out += residual
        return self.relu(out)

class Generator(nn.Module):
    def __init__(self, input_channels, output_channels, num_residual_blocks=6):
        super(Generator, self).__init__()
        
        model = [
            nn.Conv3d(input_channels, 32, kernel_size=7, padding=3),
            nn.InstanceNorm3d(32),
            nn.ReLU(inplace=True)
        ]
        
        # Downsampling
        in_features = 32
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv3d(in_features, out_features, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm3d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose3d(in_features, out_features, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm3d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2

        model += [nn.Conv3d(32, output_channels, kernel_size=7, padding=3)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        check_tensor_size(x, (x.size(0), 1, 128, 128, 128), "Generator input")
        output = self.model(x)
        check_tensor_size(output, (x.size(0), 1, 128, 128, 128), "Generator output")
        return output

class Discriminator(nn.Module):
    def __init__(self, input_channels):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv3d(in_filters, out_filters, kernel_size=4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm3d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(input_channels, 32, normalize=False),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            nn.Conv3d(256, 1, kernel_size=4, padding=1)
        )

    def forward(self, x):
        check_tensor_size(x, (x.size(0), 1, 128, 128, 128), "Discriminator input")
        output = self.model(x)
        check_tensor_size(output, (x.size(0), 1, 8, 8, 8), "Discriminator output")
        return output