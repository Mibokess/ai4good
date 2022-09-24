from torch import nn
from models.layers import Downsample, Upsample


class CycleGANGenerator(nn.Module):
    def __init__(self, in_channels, out_channels, filter=64):
        super(CycleGANGenerator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsamples = nn.ModuleList(
            [
                Downsample(self.in_channels, filter, kernel_size=4, apply_instancenorm=True),
                Downsample(filter, filter * 2),
                Downsample(filter * 2, filter * 4),
                Downsample(filter * 4, filter * 8),
                Downsample(filter * 8, filter * 8),
                Downsample(filter * 8, filter * 8),
                Downsample(filter * 8, filter * 8),
            ]
        )

        self.upsamples = nn.ModuleList(
            [
                Upsample(filter * 8, filter * 8),
                Upsample(filter * 16, filter * 8),
                Upsample(filter * 16, filter * 8),
                Upsample(filter * 16, filter * 4, dropout=False),
                Upsample(filter * 8, filter * 2, dropout=False),
                Upsample(filter * 4, filter, dropout=False),
            ]
        )

        self.last = nn.Sequential(
            nn.ConvTranspose2d(filter * 2, self.out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        skips = []
        for l in self.downsamples:
            x = l(x)
            skips.append(x)

        skips = reversed(skips[:-1])
        for l, s in zip(self.upsamples, skips):
            x = l(x, s)

        out = self.last(x)

        return out