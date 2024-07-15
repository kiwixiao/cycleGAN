# Add at the top of the file
from utils import logger, check_tensor_size

class Generator(nn.Module):
    def forward(self, x):
        check_tensor_size(x, (x.size(0), 1, 128, 128, 128), "Generator input")
        output = self.model(x)
        check_tensor_size(output, (x.size(0), 1, 128, 128, 128), "Generator output")
        return output

class Discriminator(nn.Module):
    def forward(self, x):
        check_tensor_size(x, (x.size(0), 1, 128, 128, 128), "Discriminator input")
        output = self.model(x)
        check_tensor_size(output, (x.size(0), 1, 8, 8, 8), "Discriminator output")
        return output