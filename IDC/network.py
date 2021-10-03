import torch
import torch.nn as nn
from util import swap_axis

class Generator(nn.Module):
    def __init__(self, conf):
        super(Generator, self).__init__()
        struct = conf.G_structure
        # First layer - Converting RGB image to latent space
        self.first_layer = nn.Conv2d(in_channels=1, out_channels=conf.G_chan, kernel_size=struct[0], bias=False)

        feature_block = []  # Stacking intermediate layer
        for layer in range(1, len(struct) - 1):
            feature_block += [nn.Conv2d(in_channels=conf.G_chan, out_channels=conf.G_chan, kernel_size=struct[layer], bias=False)]
        self.feature_block = nn.Sequential(*feature_block)
        # Final layer - Down-sampling and converting back to image
        self.final_layer = nn.Conv2d(in_channels=conf.G_chan, out_channels=1, kernel_size=struct[-1],
                                     stride=int(1 / conf.scale_factor), bias=False)

        # Calculate number of pixels shaved in the forward pass
        self.output_size = self.forward(torch.FloatTensor(torch.ones([1, 1, conf.input_crop_size, conf.input_crop_size]))).shape[-1]
        self.forward_shave = int(conf.input_crop_size * conf.scale_factor) - self.output_size

    def forward(self, input_tensor):
        # Swap axis of RGB image for the network to get a "batch" of size = 3 rather the 3 channels
        input_tensor = swap_axis(input_tensor)
        downscaled = self.first_layer(input_tensor)
        features = self.feature_block(downscaled)
        output = self.final_layer(features)
        return swap_axis(output)