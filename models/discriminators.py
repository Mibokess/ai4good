from torch import nn
from models.layers import Downsample


class CycleGANDiscriminator(nn.Module):
    def __init__(self, in_channels, filter=64):
        super(CycleGANDiscriminator, self).__init__()

        self.block = nn.Sequential(
            Downsample(in_channels, filter, kernel_size=4, stride=2, apply_instancenorm=False),
            Downsample(filter, filter * 2, kernel_size=4, stride=2),
            Downsample(filter * 2, filter * 4, kernel_size=4, stride=2),
            Downsample(filter * 4, filter * 8, kernel_size=4, stride=1),
        )

        self.last = nn.Conv2d(filter * 8, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, x):
        x = self.block(x)
        x = self.last(x)

        return x