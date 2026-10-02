# Original source comments are preserved from the audited research file named in docs/baselines.md.
import torch
from torch import nn
import torch.nn.functional as F

from ._blocks import ConvBnRelu2d, StackDecoder, StackEncoder
from utils.operators import gaus_t, generate_roi


# ---- 최소 보조 모듈 (FiLM) ----
class _FiLM(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.c = c
    def forward(self, x, gamma, beta):
        # x: (N,C,H,W), gamma/beta: (N,C)
        return x * gamma.unsqueeze(-1).unsqueeze(-1) + beta.unsqueeze(-1).unsqueeze(-1)

class _ScalarEncoder(nn.Module):
    """
    입력 스칼라 s -> 각 적용 지점별 (gamma,beta) 생성
    channels: 각 지점의 채널 수 리스트
    """
    def __init__(self, channels, hidden=128, in_dim=1):
        super().__init__()
        self.channels = list(channels)
        self.total = sum(self.channels)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2 * self.total),  # gamma || beta
        )

    def forward(self, s: torch.Tensor):
        # s: (N,) or (N,1)
        if s is None:
            return None, None
        if s.dim() == 1:
            s = s.unsqueeze(-1)
        h = self.mlp(s)  # (N, 2*total)
        N = h.size(0)
        total = h.size(1) // 2
        gamma_flat, beta_flat = h[:, :total], h[:, total:]
        gammas, betas = [], []
        idx = 0
        for c in self.channels:
            gammas.append(gamma_flat[:, idx:idx + c])
            betas.append(beta_flat[:, idx:idx + c])
            idx += c
        return gammas, betas

class PadToMultiple(nn.Module):
    def __init__(self, multiple=32, mode='reflect'):
        super().__init__()
        self.multiple = multiple
        self.mode = mode
    def forward(self, x):
        _, _, h, w = x.shape
        H = ((h + self.multiple - 1) // self.multiple) * self.multiple
        W = ((w + self.multiple - 1) // self.multiple) * self.multiple
        pad_h = H - h
        pad_w = W - w
        # left, right, top, bottom (F.pad는 (left,right,top,bottom))
        pad = (pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2)
        return F.pad(x, pad, mode=self.mode), pad  # pad 저장해두면 나중에 크롭에 사용

class UNet_LF(nn.Module):
    """U-Net model class"""

    def __init__(self,in_channels=3, out_channels=3):
        super().__init__()
        self.padder = PadToMultiple(32, mode='reflect')
        self.down1 = StackEncoder(in_channels, 32, kernel_size=3)
        self.down2 = StackEncoder(32, 64, kernel_size=3)
        self.down3 = StackEncoder(64, 128, kernel_size=3)
        self.down4 = StackEncoder(128, 256, kernel_size=3)
        self.down5 = StackEncoder(256, 512, kernel_size=3)

        self.up5 = StackDecoder(512, 512, 256, kernel_size=3)
        self.up4 = StackDecoder(256, 256, 128, kernel_size=3)
        self.up3 = StackDecoder(128, 128, 64, kernel_size=3)
        self.up2 = StackDecoder(64, 64, 32, kernel_size=3)
        self.up1 = StackDecoder(32, 32, 32, kernel_size=3)
        self.classify = nn.Conv2d(32, out_channels, kernel_size=1, bias=True)

        self.center = nn.Sequential(ConvBnRelu2d(512, 512, kernel_size=3, padding=1))
        film_channels = [32, 64, 128, 256, 512, 512, 256, 128, 64, 32, 32]
        self._scalar_encoder = _ScalarEncoder(film_channels, hidden=128, in_dim=1)
        self._films = nn.ModuleList([_FiLM(c) for c in film_channels])

    def forward(self, x, s=None):
        """
        x: (N,C,H,W)
        s: (N,) 또는 (N,1) 스칼라. None이면 조건 없이 기존과 동일.
        """
        # s -> (gamma,beta) 리스트 (또는 None)
        x, pad = self.padder(x)     # 1300→1312
        gammas, betas = self._scalar_encoder(s)
        fi = 0  # film index helper

        out = x
        down1, out = self.down1(out)  # down1: 24ch
        down1 = self._films[fi](down1, gammas[fi], betas[fi]); fi += 1

        down2, out = self.down2(out)  # 64
        down2 = self._films[fi](down2, gammas[fi], betas[fi]); fi += 1

        down3, out = self.down3(out)  # 128
        down3 = self._films[fi](down3, gammas[fi], betas[fi]); fi += 1

        down4, out = self.down4(out)  # 256
        down4 = self._films[fi](down4, gammas[fi], betas[fi]); fi += 1

        down5, out = self.down5(out)  # 512
        down5 = self._films[fi](down5, gammas[fi], betas[fi]); fi += 1

        out = self.center(out)        # 512
        out = self._films[fi](out, gammas[fi], betas[fi]); fi += 1

        out = self.up5(out, down5)    # 256
        out = self._films[fi](out, gammas[fi], betas[fi]); fi += 1

        out = self.up4(out, down4)    # 128
        out = self._films[fi](out, gammas[fi], betas[fi]); fi += 1

        out = self.up3(out, down3)    # 64
        out = self._films[fi](out, gammas[fi], betas[fi]); fi += 1

        out = self.up2(out, down2)    # 24
        out = self._films[fi](out, gammas[fi], betas[fi]); fi += 1

        out = self.up1(out, down1)    # 24
        out = self._films[fi](out, gammas[fi], betas[fi]); fi += 1

        out = self.classify(out)
        # return out
        top, bottom = pad[2], pad[3]
        left, right = pad[0], pad[1]
        return out[..., top:out.shape[-2]-bottom, left:out.shape[-1]-right]


class UNet_LF_0(nn.Module):
    """U-Net model class"""

    def __init__(self,in_channels=3, out_channels=3):
        super().__init__()
        self.padder = PadToMultiple(24, mode='reflect')
        self.down1 = StackEncoder(in_channels, 24, kernel_size=3)
        self.down2 = StackEncoder(24, 64, kernel_size=3)
        self.down3 = StackEncoder(64, 128, kernel_size=3)
        self.down4 = StackEncoder(128, 256, kernel_size=3)
        self.down5 = StackEncoder(256, 512, kernel_size=3)

        self.up5 = StackDecoder(512, 512, 256, kernel_size=3)
        self.up4 = StackDecoder(256, 256, 128, kernel_size=3)
        self.up3 = StackDecoder(128, 128, 64, kernel_size=3)
        self.up2 = StackDecoder(64, 64, 24, kernel_size=3)
        self.up1 = StackDecoder(24, 24, 24, kernel_size=3)
        self.classify = nn.Conv2d(24, out_channels, kernel_size=1, bias=True)

        self.center = nn.Sequential(ConvBnRelu2d(512, 512, kernel_size=3, padding=1))
        film_channels = [24, 64, 128, 256, 512, 512, 256, 128, 64, 24, 24]
        self._scalar_encoder = _ScalarEncoder(film_channels, hidden=128, in_dim=1)
        self._films = nn.ModuleList([_FiLM(c) for c in film_channels])

    def forward(self, x, s=None):
        """
        x: (N,C,H,W)
        s: (N,) 또는 (N,1) 스칼라. None이면 조건 없이 기존과 동일.
        """
        # s -> (gamma,beta) 리스트 (또는 None)
        gammas, betas = self._scalar_encoder(s)
        fi = 0  # film index helper

        out = x
        down1, out = self.down1(out)  # down1: 24ch
        down1 = self._films[fi](down1, gammas[fi], betas[fi]); fi += 1

        down2, out = self.down2(out)  # 64
        down2 = self._films[fi](down2, gammas[fi], betas[fi]); fi += 1

        down3, out = self.down3(out)  # 128
        down3 = self._films[fi](down3, gammas[fi], betas[fi]); fi += 1

        down4, out = self.down4(out)  # 256
        down4 = self._films[fi](down4, gammas[fi], betas[fi]); fi += 1

        down5, out = self.down5(out)  # 512
        down5 = self._films[fi](down5, gammas[fi], betas[fi]); fi += 1

        out = self.center(out)        # 512
        out = self._films[fi](out, gammas[fi], betas[fi]); fi += 1

        out = self.up5(out, down5)    # 256
        out = self._films[fi](out, gammas[fi], betas[fi]); fi += 1

        out = self.up4(out, down4)    # 128
        out = self._films[fi](out, gammas[fi], betas[fi]); fi += 1

        out = self.up3(out, down3)    # 64
        out = self._films[fi](out, gammas[fi], betas[fi]); fi += 1

        out = self.up2(out, down2)    # 24
        out = self._films[fi](out, gammas[fi], betas[fi]); fi += 1

        out = self.up1(out, down1)    # 24
        out = self._films[fi](out, gammas[fi], betas[fi]); fi += 1

        out = self.classify(out)
        return out


class UNet(nn.Module):
    """U-Net model class"""

    def __init__(self,in_channels=3, out_channels=3):
        super().__init__()
        self.down1 = StackEncoder(in_channels, 24, kernel_size=3)
        self.down2 = StackEncoder(24, 64, kernel_size=3)
        self.down3 = StackEncoder(64, 128, kernel_size=3)
        self.down4 = StackEncoder(128, 256, kernel_size=3)
        self.down5 = StackEncoder(256, 512, kernel_size=3)

        self.up5 = StackDecoder(512, 512, 256, kernel_size=3)
        self.up4 = StackDecoder(256, 256, 128, kernel_size=3)
        self.up3 = StackDecoder(128, 128, 64, kernel_size=3)
        self.up2 = StackDecoder(64, 64, 24, kernel_size=3)
        self.up1 = StackDecoder(24, 24, 24, kernel_size=3)
        self.classify = nn.Conv2d(24, out_channels, kernel_size=1, bias=True)

        self.center = nn.Sequential(ConvBnRelu2d(512, 512, kernel_size=3, padding=1))

    def forward(self, x):
        out = x
        down1, out = self.down1(out)
        down2, out = self.down2(out)
        down3, out = self.down3(out)
        down4, out = self.down4(out)
        down5, out = self.down5(out)

        out = self.center(out)
        out = self.up5(out, down5)
        out = self.up4(out, down4)
        out = self.up3(out, down3)
        out = self.up2(out, down2)
        out = self.up1(out, down1)

        out = self.classify(out)
        return out


class W(nn.Module):
    """
    'W' 클래스를 기반으로, WieNer_SV(멀티 커널, k 차원, kernel_weights 등) 기능을 결합하되
    추가적인 패딩 없이 FFT를 수행하는 예시.

    Args:
        channels (int): 입력/출력 채널 수
        height (int): 이미지 높이(H)
        width (int):  이미지 너비(W)
        k (int): 추가로 사용할 PSF(또는 ROI) 개수
    """
    def __init__(self, channels, height, width, height_p, width_p, k=1):
        super(W, self).__init__()
        self.height_freq = height + height_p
        self.width_freq = (width + width_p)// 2 + 1

        self.psf_weights = nn.Parameter(
            torch.ones(k, channels, self.height_freq, self.width_freq) * 0.01
        )
        self.group_norm = nn.GroupNorm(num_groups=1, num_channels=channels)

        self.alpha = nn.Parameter(torch.ones(k, 1, 1, 1) * 0.1)
        self.kernel_weights = nn.Parameter(generate_roi(k, height, width),
                                           requires_grad=True)
        self.relu = nn.ReLU()
        self.k = k

    def forward(self, raw: torch.Tensor, psf: torch.Tensor, epsilon=1e-6) -> torch.Tensor:
        """
        Args:
            raw (torch.Tensor): (B, C, H, W) 형태의 입력
            psf (torch.Tensor): (B, C, H_p, W_p) 형태의 PSF (혹은 동일 크기 H, W일 수도 있음)
            epsilon (float): Regularization 파라미터

        Returns:
            torch.Tensor: (B, C, H, W) 형태의 복원 결과
        """
        B, C, H, W = raw.shape
        _, _, H_p, W_p = psf.shape

        psf = psf.reshape(self.k,-1,psf.size(-2),psf.size(-1))
        psf_sum = psf.sum(dim=(-2, -1), keepdim=True)          # (B, C, 1, 1)
        psf_normalized = psf / (psf_sum.abs() * self.alpha + 1e-12)

        # Apply symmetric padding to raw input
        raw_padded = F.pad(
            raw,
            (W_p // 2, W_p - W_p // 2, H_p // 2, H_p - H_p // 2),
            mode='replicate'
        )
        raw_padded = gaus_t(raw_padded, fwhm=2)
        psf_padded = F.pad(
            psf_normalized,
            (W // 2, W - W // 2, H // 2, H - H // 2),
            mode='constant'
        )

        # Compute FFT with 'ortho' normalization to maintain energy
        raw_fft = torch.fft.rfft2(raw_padded, dim=(-2, -1))  # Shape: (B, C, H_freq, W_freq)
        psf_fft = torch.fft.rfft2(psf_padded, s=(raw_padded.size(-2), raw_padded.size(-1)), dim=(-2, -1))  # Shape: (B, C, H_freq, W_freq)

        pw = self.relu(self.psf_weights)
        wiener_filter = psf_fft.conj() / (psf_fft.abs()**2 + epsilon + pw)
        out_fft = raw_fft.unsqueeze(0) * wiener_filter.unsqueeze(1)  # (k, B, C, H, W//2+1)

        out_spatial = torch.fft.irfft2(out_fft, dim=(-2, -1))
        out_spatial = torch.fft.ifftshift(out_spatial, dim=(-2, -1))
        start_H = H_p // 2
        start_W = W_p // 2
        out_cropped = out_spatial[..., start_H:start_H + H, start_W:start_W + W]  # Shape: (N, B, C, H, W)
        kw = self.kernel_weights.unsqueeze(1).unsqueeze(1)  # (k, 1, 1, H, W)
        out_cropped = ((out_cropped) * kw).sum(dim=0)         # (B, C, H, W)

        return self.group_norm(out_cropped.real)

class WienerU(nn.Module):
    """U-Net model class"""

    def __init__(self,in_channels, out_channels, psf, height=336, width=336, k=1):
        super().__init__()
        self.psf = nn.Parameter(psf.repeat(1,k,1,1), requires_grad=True)
        _, _, height_p, width_p = psf.size()
        self.W = W(in_channels, height, width, height_p, width_p)
        self.down1 = StackEncoder(in_channels, 24, kernel_size=3)
        self.down2 = StackEncoder(24, 64, kernel_size=3)
        self.down3 = StackEncoder(64, 128, kernel_size=3)
        self.down4 = StackEncoder(128, 256, kernel_size=3)
        self.down5 = StackEncoder(256, 512, kernel_size=3)

        self.up5 = StackDecoder(512, 512, 256, kernel_size=3)
        self.up4 = StackDecoder(256, 256, 128, kernel_size=3)
        self.up3 = StackDecoder(128, 128, 64, kernel_size=3)
        self.up2 = StackDecoder(64, 64, 24, kernel_size=3)
        self.up1 = StackDecoder(24, 24, 24, kernel_size=3)
        self.classify = nn.Conv2d(24, out_channels, kernel_size=1, bias=True)

        self.center = nn.Sequential(ConvBnRelu2d(512, 512, kernel_size=3, padding=1))

    def forward(self, x):
        out = self.W(x, self.psf)
        down1, out = self.down1(out)
        down2, out = self.down2(out)
        down3, out = self.down3(out)
        down4, out = self.down4(out)
        down5, out = self.down5(out)

        out = self.center(out)
        out = self.up5(out, down5)
        out = self.up4(out, down4)
        out = self.up3(out, down3)
        out = self.up2(out, down2)
        out = self.up1(out, down1)

        out = self.classify(out)
        return out
