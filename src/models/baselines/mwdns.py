import numpy as np
import torch
import torch.nn as nn

from ._blocks import DoubleConv, Down, OutConv, Up


def wiener(blur, psf, delta):
    blur_fft = torch.fft.rfft2(blur)
    psf_fft = torch.fft.rfft2(psf, s=blur.shape[-2:])
    weight = torch.conj(psf_fft) / (torch.abs(psf_fft) ** 2 + delta)
    return torch.fft.ifftshift(torch.fft.irfft2(blur_fft * weight), (-2, -1)).real


class MWDNet_CPSF(nn.Module):
    def __init__(self, in_channels, out_channels, psf, psf_channels=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, out_channels)
        self.delta = nn.Parameter(torch.tensor(np.ones(5) * 0.01, dtype=torch.float32))
        self.w = nn.Parameter(torch.tensor(np.ones((1, psf_channels, 1, 1)) * 0.01, dtype=torch.float32))
        self.inc0 = DoubleConv(psf_channels, 64)
        self.down11 = Down(64, 128)
        self.down22 = Down(128, 256)
        self.down33 = Down(256, 512)
        self.psf = psf

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        psf1 = self.inc0(self.w * self.psf)
        psf2 = self.down11(psf1)
        psf3 = self.down22(psf2)
        psf4 = self.down33(psf3)
        x4 = wiener(x4, psf4, self.delta[3])
        x3 = wiener(x3, psf3, self.delta[2])
        x2 = wiener(x2, psf2, self.delta[1])
        x1 = wiener(x1, psf1, self.delta[0])
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)
