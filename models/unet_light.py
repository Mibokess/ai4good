import torch
import torch.nn as nn

from models.layers import DoubleConv

class UnetLight(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()

        self.skip1 = DoubleConv(in_channels, 16)
        self.skip2 = DoubleConv(16, 32)
        self.skip3 = DoubleConv(32, 64)
        self.skip4 = DoubleConv(64, 128)
        self.last_conv = DoubleConv(128, 256)

        self.pool = nn.MaxPool2d(2, stride=2)

        self.up4 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.cat_double_conv4 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.cat_double_conv3 = DoubleConv(128, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.cat_double_conv2 = DoubleConv(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.cat_double_conv1 = DoubleConv(32, 16)

        # add conditional for clasess
        self.out = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1),
            #nn.Softmax(dim=1) pytorch cross entropy already uses softmax
        )


    def forward(self, x):

        s1 = self.skip1(x)
        d1 = self.pool(s1)
        s2 = self.skip2(d1)
        d2 = self.pool(s2)
        s3 = self.skip3(d2)
        d3 = self.pool(s3)
        s4 = self.skip4(d3)
        d4 = self.pool(s4)

        d5 = self.last_conv(d4)
        
        u4 = self.up4(d5)
        u4 = torch.cat((s4, u4), 1)
        u4 = self.cat_double_conv4(u4)
        u3 = self.up3(u4)
        u3 = torch.cat((s3, u3), 1)
        u3 = self.cat_double_conv3(u3)
        u2 = self.up2(u3)
        u2 = torch.cat((s2, u2), 1)
        u2 = self.cat_double_conv2(u2)
        u1 = self.up1(u2)
        u1 = torch.cat((s1, u1), 1)
        u1 = self.cat_double_conv1(u1)

        return self.out(u1)