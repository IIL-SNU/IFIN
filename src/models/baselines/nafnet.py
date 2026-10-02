# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Original source comments are preserved from the audited research file named in docs/baselines.md.

'''
Simple Baselines for Image Restoration

@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from ._blocks import LayerNorm2d
# from basicsr.models.archs.arch_util import LayerNorm2d

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class NAFNet(nn.Module):

    def __init__(self, img_channel=3, out_channel=3, width=32, middle_blk_num=12, enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2]):
    #img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()
    # enc_blks = [2, 2, 4, 8]
    # middle_blk_num = 12
    # dec_blks = [2, 2, 2, 2]

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=out_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

class BackPropPhysics(nn.Module):
    """
    Conv 없음. 순수 ASM 역전파만.
    hologram intensity → sqrt → zero-phase 복소장 → ASM back-prop → amplitude
    학습 파라미터: z (전파 거리)
    """
    def __init__(self, wavelength: float = 532e-9, pixel_size: float = 5.08e-6):
        super().__init__()
        self.z          = nn.Parameter(torch.tensor(0.03))
        self.k0         = 1.0 / wavelength
        self.pixel_size = pixel_size

    def _H_conj(self, H, W, device):
        fx  = torch.fft.fftfreq(W, d=self.pixel_size, device=device)
        fy  = torch.fft.fftfreq(H, d=self.pixel_size, device=device)
        FY, FX = torch.meshgrid(fy, fx, indexing='ij')
        kz   = torch.sqrt(torch.clamp(self.k0**2 - FX**2 - FY**2, min=0.0))
        mask = (FX**2 + FY**2 < self.k0**2).float()
        return torch.exp(-1j * 2 * torch.pi * kz * self.z) * mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, H, W)  hologram intensity
        amp    = x.clamp(min=0).sqrt()                    # intensity → amplitude
        U      = amp.to(torch.complex64)                  # zero phase 가정

        _, _, H, W = x.shape
        U_back = torch.fft.ifft2(
            torch.fft.fft2(U, dim=(-2,-1)) * self._H_conj(H, W, x.device),
            dim=(-2,-1)
        )
        return U_back.abs()                               # (B, 1, H, W)  amplitude


class BackPropNAFNet(nn.Module):
    def __init__(self,
                 img_channel:    int   = 1,
                 out_channel:    int   = 1,
                 width:          int   = 32,
                 middle_blk_num: int   = 12,
                 enc_blk_nums          = [2, 2, 4, 8],
                 dec_blk_nums          = [2, 2, 2, 2],
                 wavelength:     float = 532e-9,
                 pixel_size:     float = 5.08e-6):
        super().__init__()
        self.bp     = BackPropPhysics(wavelength, pixel_size)
        self.nafnet = NAFNet(
            img_channel    = img_channel,   # BackProp 전후 채널 수 동일
            out_channel    = out_channel,
            width          = width,
            middle_blk_num = middle_blk_num,
            enc_blk_nums   = enc_blk_nums,
            dec_blk_nums   = dec_blk_nums,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.nafnet(self.bp(x))
