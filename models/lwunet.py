import torch
from torch import nn
import torch.nn.functional as F


class FactorizedConv(nn.Module):
    """
    Factorized convolution block with two convolutions.
    """

    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super(FactorizedConv, self).__init__()

        # first convolution: kx1
        self.conv1 = nn.Conv2d(
            in_ch, out_ch, kernel_size=(kernel_size, 1), padding=(padding, 0)
        )

        # second convolution: 1xk
        self.conv2 = nn.Conv2d(
            out_ch, out_ch, kernel_size=(1, kernel_size), padding=(0, padding)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        return x


class DilatedConv(nn.Module):
    """
    Dilated convolution block. The output dimension is the same as the input dimension.
    """

    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=2, padding=-1):
        super(DilatedConv, self).__init__()
        actual_padding = (kernel_size - 1) * dilation // 2
        self.dilated_conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=actual_padding,
        )

    def forward(self, x):
        return self.dilated_conv(x)


class UpConv(nn.Module):
    """
    Upsampling convolution block.
    """

    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.up(x)


class ConvBlock(nn.Module):
    """
    Configurable double-convolution block.
    Has activation and optional batch normalization.
    """

    def __init__(
        self,
        in_ch,
        out_ch,
        conv1: nn.Module,
        conv2: nn.Module,
        act=nn.LeakyReLU,
        bn=False,
    ):
        super(ConvBlock, self).__init__()

        self.conv1 = conv1(in_ch, out_ch, kernel_size=3, padding=1)  # best: dilated
        self.bn1 = nn.BatchNorm2d(out_ch) if bn else None
        self.act1 = act()  # best: leakyrelu

        self.conv2 = conv2(
            out_ch, out_ch, kernel_size=3, padding=1
        )  # best: kernel factorized
        self.bn2 = nn.BatchNorm2d(out_ch) if bn else None
        self.act2 = act()  # best: leakyrelu

    def forward(self, x):
        if self.bn1 is not None and self.bn2 is not None:
            x = self.act1(self.bn1(self.conv1(x)))
            x = self.act2(self.bn2(self.conv2(x)))
        else:
            x = self.act1(self.conv1(x))
            x = self.act2(self.conv2(x))

        return x


class LWUNetBase4(nn.Module):
    def __init__(
        self,
        conv1: nn.Module,
        conv2: nn.Module,
        act=nn.LeakyReLU,
        bn=True,
        k=4,
        in_ch=1,
        out_ch=1,
    ):
        super().__init__()

        print(f"LWUNet: {conv1=} {conv2=}")

        # encoder
        self.enc1 = ConvBlock(in_ch, k, conv1=conv1, conv2=conv2, act=act, bn=bn)
        self.enc2 = ConvBlock(k, k * 2, conv1=conv1, conv2=conv2, act=act, bn=bn)
        self.enc3 = ConvBlock(k * 2, k * 4, conv1=conv1, conv2=conv2, act=act, bn=bn)
        self.enc4 = ConvBlock(k * 4, k * 8, conv1=conv1, conv2=conv2, act=act, bn=bn)
        self.pool = nn.MaxPool2d(
            kernel_size=2, stride=2
        )  # identical for all encoding blocks

        self.bottleneck = ConvBlock(
            k * 8, k * 16, conv1=conv1, conv2=conv2, act=act, bn=bn
        )

        # decoder
        self.up1 = UpConv(k * 16, k * 8)
        self.dec1 = ConvBlock(k * 16, k * 8, conv1=conv1, conv2=conv2, act=act, bn=bn)
        self.up2 = UpConv(k * 8, k * 4)
        self.dec2 = ConvBlock(k * 8, k * 4, conv1=conv1, conv2=conv2, act=act, bn=bn)
        self.up3 = UpConv(k * 4, k * 2)
        self.dec3 = ConvBlock(k * 4, k * 2, conv1=conv1, conv2=conv2, act=act, bn=bn)
        self.up4 = UpConv(k * 2, k)
        self.dec4 = ConvBlock(k * 2, k, conv1=conv1, conv2=conv2, act=act, bn=bn)

        self.pointwise_conv = nn.Conv2d(k, out_ch, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        pool1 = self.pool(enc1)
        enc2 = self.enc2(pool1)
        pool2 = self.pool(enc2)
        enc3 = self.enc3(pool2)
        pool3 = self.pool(enc3)
        enc4 = self.enc4(pool3)
        pool4 = self.pool(enc4)

        bottleneck = self.bottleneck(pool4)

        up1 = self.up1(bottleneck)
        cat1 = torch.cat((up1, enc4), dim=1)
        dec1 = self.dec1(cat1)

        up2 = self.up2(dec1)
        cat2 = torch.cat((up2, enc3), dim=1)
        dec2 = self.dec2(cat2)

        up3 = self.up3(dec2)
        cat3 = torch.cat((up3, enc2), dim=1)
        dec3 = self.dec3(cat3)

        up4 = self.up4(dec3)
        cat4 = torch.cat((up4, enc1), dim=1)
        dec4 = self.dec4(cat4)

        out = self.pointwise_conv(dec4)
        out = F.sigmoid(out)

        return out


class LWUNet(LWUNetBase4):
    """
    Best LWUNet model. Experimentally determined with the Needle dataset.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__(
            conv1=DilatedConv,  # type: ignore
            conv2=FactorizedConv,  # type: ignore
            act=nn.LeakyReLU,
            bn=True,
            k=4,
            in_ch=in_ch,
            out_ch=out_ch,
        )
