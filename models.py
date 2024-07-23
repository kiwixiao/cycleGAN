import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models
import numpy as np
from utils import check_tensor_size
from torchviz import make_dot

# Residual Block with Self-Attention
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv3d(in_features, in_features, kernel_size=3, padding=1),
            nn.InstanceNorm3d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_features, in_features, kernel_size=3, padding=1),
            nn.InstanceNorm3d(in_features)
        )
        self.attention = SelfAttention(in_features)

    def forward(self, x):
        check_tensor_size(x, (x.size(0), x.size(1), x.size(2), x.size(4)), "ResidualBlock input") # add dimensional check
        out = x + self.conv_block(x)
        out, _ = self.attention(out)
        check_tensor_size(x, (x.size(0), x.size(1), x.size(2), x.size(4)), "ResidualBlock output") # add dimensional check
        return out

# self attention mechanism
class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv3d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv3d(in_dim, in_dim // 8, kernel_size=1)
        self,value_conv = nn.Conv3d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, width, height, depth = x.size()
        proj_query =  self.query_conv(x).view(m_batchsize, -1, width * height * depth).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height * depth)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height * depth)
        out = torch.bmm(proj_value, attention.permute(0,2,1))
        out = out.view(m_batchsize, C, width, height, depth)
        out = self.gamma * out + x
        check_tensor_size(out, (m_batchsize, C, width, height, depth), "SelfAttention Output")
        return out, attention
# Positional Attention Mechanism
class PositionalAttention(nn.Module):
    def __init__(self, in_dim, img_size, focus_radius):
        super(PositionalAttention, self).__init__()
        self.in_dim = in_dim
        self.img_size = img_size
        self.focus_radius =  focus_radius
        self.position = self.create_position_mask()
    
    def create_position_mask(self):
        mask = np.zeros(self.img_size)
        center = [s // 2 for s in self.img_size]
        for x in range(self.img_size[0]):
            for y in range(self.img_size[1]):
                for z in range(self.img_size[2]):
                    if np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2) <= self.focus_radius:
                        mask[x, y, z] = 1
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        return nn.Parameter(mask, requires_grad=False)
    def forward(self, x):
        check_tensor_size(x, (x.size(0), self.in_dim, *self.img_size), "PositionalAttention input")  # Added dimension check
        out = x * self.position.to(x.device)
        check_tensor_size(out, (x.size(0), self.in_dim, *self.img_size), "PositionalAttention output")  # Added dimension check
        return out

class Generator(nn.Module):
    def __init__(self, input_channels, output_channels, num_residual_blocks=9, img_size=(128, 128, 128), focus_radius=40):
        super(Generator, self).__init__()
        self.positional_attention = PositionalAttention(1, img_size, focus_radius)
        model = [
            nn.Conv3d(input_channels, 64, kernel_size=7, padding=3),
            nn.InstanceNorm3d(64),
            nn.ReLU(inplace=True)
        ]
        
        # Downsampling
        in_features = 64
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

        model += [nn.Conv3d(64, output_channels, kernel_size=7, padding=3)]
        model += [nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        check_tensor_size(x, (x.size(0), 1, 128, 128, 128), "Generator input")  # Added dimension check
        x = self.positional_attention(x)
        output = self.model(x)
        check_tensor_size(output, (x.size(0), 1, 128, 128, 128), "Generator output")  # Added dimension check
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
            *discriminator_block(input_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.Conv3d(512, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        check_tensor_size(x, (x.size(0), 1, 128, 128, 128), "Discriminator input")
        output = self.model(x)
        check_tensor_size(output, (x.size(0), 1, 8, 8, 8), "Discriminator output")
        return output

def plot_model(model, input_tensor, filename):
    """
    Plot the model architecture and save it as a PNG file.
    
    Parameters:
    model (torch.nn.Module): The model to be visualized.
    input_tensor (torch.Tensor): A sample input tensor.
    filename (str): The filename to save the plot.
    """
    model.eval()
    output = model(input_tensor)
    dot = make_dot(output, params=dict(list(model.named_parameters()) + [('input', input_tensor)]))
    dot.format = 'png'
    dot.render(filename)