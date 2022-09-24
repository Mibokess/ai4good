import torch
import torch.nn as nn
import numpy as np
from kornia.augmentation import RandomGaussianNoise

img_sz = 256

class Upsample(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=4,
        stride=2,
        padding=1,
        dropout=True,
    ):
        super(Upsample, self).__init__()
        self.dropout = dropout
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=nn.InstanceNorm2d,
            ),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.dropout_layer = nn.Dropout2d(0.5)

    def forward(self, x, shortcut=None):
        x = self.block(x)
        if self.dropout:
            x = self.dropout_layer(x)

        if shortcut is not None:
            x = torch.cat([x, shortcut], dim=1)

        return x


class Downsample(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=4,
        stride=2,
        padding=1,
        apply_instancenorm=True,
    ):
        super(Downsample, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=nn.InstanceNorm2d,
        )
        self.norm = nn.InstanceNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.apply_norm = apply_instancenorm

    def forward(self, x):
        x = self.conv(x)
        if self.apply_norm:
            x = self.norm(x)
        x = self.relu(x)

        return x



class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class CycleGANGenerator(nn.Module):
    def __init__(self, filter=64):
        super(CycleGANGenerator, self).__init__()
        self.downsamples = nn.ModuleList(
            [
                Downsample(4, filter, kernel_size=4, apply_instancenorm=True),
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
            nn.ConvTranspose2d(filter * 2, 4, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

        self.gaussian_noise = RandomGaussianNoise(0., 0.001, p=1.)

    def forward(self, x, noise=True):
        skips = []
        for l in self.downsamples:
            x = l(x)
            skips.append(x)

        skips = reversed(skips[:-1])
        for l, s in zip(self.upsamples, skips):
            x = l(x, s)

        out = self.last(x)

        if noise:
            out = self.gaussian_noise(out)

        return out

class CycleGANDiscriminator(nn.Module):
    def __init__(self, filter=64):
        super(CycleGANDiscriminator, self).__init__()

        self.block = nn.Sequential(
            Downsample(4, filter, kernel_size=4, stride=2, apply_instancenorm=False),
            Downsample(filter, filter * 2, kernel_size=4, stride=2),
            Downsample(filter * 2, filter * 4, kernel_size=4, stride=2),
            Downsample(filter * 4, filter * 8, kernel_size=4, stride=1),
        )

        self.last = nn.Conv2d(filter * 8, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, x):
        x = self.block(x)
        x = self.last(x)

        return x


class ResBlock(nn.Module):

    def __init__(self, in_channels: int, apply_dp: bool = True):

        """
                            Defines a ResBlock
        X ------------------------identity------------------------
        |-- Convolution -- Norm -- ReLU -- Convolution -- Norm --|
        """

        """
        Parameters:
            in_channels:  Number of input channels
            apply_dp:     If apply_dp is set to True, then activations are 0'ed out with prob 0.5
        """

        super().__init__()

        conv = nn.Conv2d(in_channels = in_channels, out_channels = in_channels, kernel_size = 3, stride = 1)
        layers =  [nn.ReflectionPad2d(1), conv, nn.InstanceNorm2d(in_channels), nn.ReLU(True)]

        if apply_dp:
            layers += [nn.Dropout(0.5)]

        conv = nn.Conv2d(in_channels = in_channels, out_channels = in_channels, kernel_size = 3, stride = 1)
        layers += [nn.ReflectionPad2d(1), conv, nn.InstanceNorm2d(in_channels)]

        self.net = nn.Sequential(*layers)


    def forward(self, x): return x + self.net(x)



class Generator(nn.Module):

    def __init__(self, in_channels: int = 4, out_channels: int = 64, apply_dp: bool = True):

        """
                                Generator Architecture (Image Size: 256)
        c7s1-64, d128, d256, R256, R256, R256, R256, R256, R256, R256, R256, R256, u128, u64, c7s1-3,
        where c7s1-k denote a 7 × 7 Conv-InstanceNorm-ReLU layer with k filters and stride 1, dk denotes a 3 × 3
        Conv-InstanceNorm-ReLU layer with k filters and stride 2, Rk denotes a residual block that contains two
        3 × 3 Conv layers with the same number of filters on both layer. uk denotes a 3 × 3 DeConv-InstanceNorm-
        ReLU layer with k filters and stride 1.
        """

        """
        Parameters:
            in_channels:  Number of input channels
            out_channels: Number of output channels
            apply_dp:     If apply_dp is set to True, then activations are 0'ed out with prob 0.5
        """

        super().__init__()

        f = 1
        nb_downsampling = 2
        nb_resblks = 6 if img_sz == 128 else 9

        conv = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 7, stride = 1)
        self.layers = [nn.ReflectionPad2d(3), conv, nn.InstanceNorm2d(out_channels), nn.ReLU(True)]

        for i in range(nb_downsampling):
            conv = nn.Conv2d(out_channels * f, out_channels * 2 * f, kernel_size = 3, stride = 2, padding = 1)
            self.layers += [conv, nn.InstanceNorm2d(out_channels * 2 * f), nn.ReLU(True)]
            f *= 2

        for i in range(nb_resblks):
            res_blk = ResBlock(in_channels = out_channels * f, apply_dp = apply_dp)
            self.layers += [res_blk]

        for i in range(nb_downsampling):
            conv = nn.ConvTranspose2d(out_channels * f, out_channels * (f//2), 3, 2, padding = 1, output_padding = 1)
            self.layers += [conv, nn.InstanceNorm2d(out_channels * (f//2)), nn.ReLU(True)]
            f = f // 2

        conv = nn.Conv2d(in_channels = out_channels, out_channels = in_channels, kernel_size = 7, stride = 1)
        self.layers += [nn.ReflectionPad2d(3), conv, nn.Tanh()]

        self.net = nn.Sequential(*self.layers)
        
        self.gaussian_noise = RandomGaussianNoise(0., 0.001, p=1.)

    def forward(self, x, noise=True): 
        out = self.net(x)

        if noise:
            out = self.gaussian_noise(out)

        return out



class Discriminator(nn.Module):

    def __init__(self, in_channels: int = 4, out_channels: int = 64, nb_layers: int = 3):

        """
                                    Discriminator Architecture!
        C64 - C128 - C256 - C512, where Ck denote a Convolution-InstanceNorm-LeakyReLU layer with k filters
        """

        """
        Parameters:
            in_channels:    Number of input channels
            out_channels:   Number of output channels
            nb_layers:      Number of layers in the 70*70 Patch Discriminator
        """

        super().__init__()
        in_f  = 1
        out_f = 2

        conv = nn.Conv2d(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1)
        self.layers = [conv, nn.LeakyReLU(0.2, True)]

        for idx in range(1, nb_layers):
            conv = nn.Conv2d(out_channels * in_f, out_channels * out_f, kernel_size = 4, stride = 2, padding = 1)
            self.layers += [conv, nn.InstanceNorm2d(out_channels * out_f), nn.LeakyReLU(0.2, True)]
            in_f   = out_f
            out_f *= 2

        out_f = min(2 ** nb_layers, 8)
        conv = nn.Conv2d(out_channels * in_f,  out_channels * out_f, kernel_size = 4, stride = 1, padding = 1)
        self.layers += [conv, nn.InstanceNorm2d(out_channels * out_f), nn.LeakyReLU(0.2, True)]

        conv = nn.Conv2d(out_channels * out_f, out_channels = 1, kernel_size = 4, stride = 1, padding = 1)
        self.layers += [conv]

        self.net = nn.Sequential(*self.layers)


    def forward(self, x): return self.net(x)



class Initializer:

    def __init__(self, init_type: str = 'normal', init_gain: float = 0.02):

        """
        Initializes the weight of the network!
        Parameters:
            init_type: Initializer type - 'kaiming' or 'xavier' or 'normal'
            init_gain: Standard deviation of the normal distribution
        """

        self.init_type = init_type
        self.init_gain = init_gain


    def init_module(self, m):

        cls_name = m.__class__.__name__;
        if hasattr(m, 'weight') and (cls_name.find('Conv') != -1 or cls_name.find('Linear') != -1):

            if   self.init_type == 'kaiming': nn.init.kaiming_normal_(m.weight.data, a = 0, mode = 'fan_in')
            elif self.init_type == 'xavier' : nn.init.xavier_normal_ (m.weight.data,  gain = self.init_gain)
            elif self.init_type == 'normal' : nn.init.normal_(m.weight.data, mean = 0, std = self.init_gain)
            else: raise ValueError('Initialization not found!!')

            if m.bias is not None: nn.init.constant_(m.bias.data, val = 0);

        if hasattr(m, 'weight') and cls_name.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, mean = 1.0, std = self.init_gain)
            nn.init.constant_(m.bias.data, val = 0)


    def __call__(self, net):

        """
        Parameters:
            net: Network
        """

        net.apply(self.init_module)

        return net



class ImagePool:

    """
    This class implements an image buffer that stores previously generated images! This buffer enables to update
    discriminators using a history of generated image rather than the latest ones produced by generator.
    """

    def __init__(self, pool_sz: int = 50):

        """
        Parameters:
            pool_sz: Size of the image buffer
        """

        self.nb_images = 0
        self.image_pool = []
        self.pool_sz = pool_sz


    def push_and_pop(self, images):

        """
        Parameters:
            images: latest images generated by the generator
        Returns a batch of images from pool!
        """

        images_to_return = []
        for image in images:
            image = torch.unsqueeze(image, 0)

            if  self.nb_images < self.pool_sz:
                self.image_pool.append (image)
                images_to_return.append(image)
                self.nb_images += 1
            else:
                if np.random.uniform(0, 1) > 0.5:

                    rand_int = np.random.randint(0, self.pool_sz)
                    temp_img = self.image_pool[rand_int].clone()
                    self.image_pool[rand_int] = image
                    images_to_return.append(temp_img)
                else:
                    images_to_return.append(image)

        return torch.cat(images_to_return, 0)
