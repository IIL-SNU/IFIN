
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.operators import gaus_t, generate_roi

BN_EPS = 1e-4

def get_num_groups(channels):
    # Ensure num_groups divides channels
    for num_groups in [32, 16, 8, 4, 2, 1]:
        if channels % num_groups == 0:
            return num_groups
    return 1

def exists(val):
    return val is not None


class ConvG(nn.Module):
    """(Convolution => [GroupNorm] => GELU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, num_groups=None):
        super(ConvG, self).__init__()
        mid_channels = out_channels
        self.single_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.single_conv(x)

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class RecB(nn.Module):
    def __init__(self, dim, drop_prob=0.1):
        super().__init__()
        # 1st sub-block
        self.norm1 = nn.GroupNorm(1, dim)            # per-pixel LN over channels
        self.pw1   = nn.Conv2d(dim, dim, 1)       # pointwise
        self.dw    = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.pw2   = nn.Conv2d(dim, dim*2, 1)     # for gating: 채널수 2배
        self.sg1   = SimpleGate()
        self.se    = nn.Sequential(               # AdaptiveAvgPool + 1×1 Conv
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1),
        )
        self.conv3 = nn.Conv2d(dim, dim, 3, padding=1)
        self.drop1 = nn.Dropout(drop_prob)
        self.beta  = nn.Parameter(torch.zeros(1, dim, 1, 1))

        # 2nd sub-block
        self.norm2 = nn.GroupNorm(1, dim)
        self.pw3   = nn.Conv2d(dim, dim*2, 1)
        self.sg2   = SimpleGate()
        self.pw4   = nn.Conv2d(dim, dim, 1)
        self.drop2 = nn.Dropout(drop_prob)
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        # --- 1st sub-block ---
        y = self.norm1(x)

        y = self.pw1(y)
        y = self.dw(y)
        y = self.pw2(y)
        y = self.sg1(y)                   # simple gate

        # spatial fusion via SE
        se = self.se(x)
        y = y * se

        y = self.conv3(y)
        y = self.drop1(y)
        y = x + self.beta * y            # 잔차 연결

        # --- 2nd sub-block ---
        z = self.norm2(y)

        z = self.pw3(z)
        z = self.sg2(z)
        z = self.pw4(z)
        z = self.drop2(z)
        z = y + self.gamma * z

        return z

class RB(nn.Module):
    """
    3×3 Conv → LN → GELU → RecB → 3×3 Conv → LN → GELU
    """
    def __init__(self, in_ch, out_ch, num_groups=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(1, out_ch)
        self.act   = nn.GELU()

        self.rec   = RecB(out_ch)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(1, out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)

        x = self.rec(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)
        return x
class ISO(nn.Module):
    """
    'W' 클래스를 기반으로, WieNer_SV(멀티 커널, k 차원, kernel_weights 등) 기능을 결합하되
    추가적인 패딩 없이 FFT를 수행하는 예시.

    Args:
        channels (int): 입력/출력 채널 수
        height (int): 이미지 높이(H)
        width (int):  이미지 너비(W)
        k (int): 추가로 사용할 PSF(또는 ROI) 개수
    """
    def __init__(self, channels, height, width, height_p, width_p, k=16):
        super(ISO, self).__init__()
        self.height_freq = height + height_p
        self.width_freq = (width + width_p)// 2 + 1

        self.psf_weights = nn.Parameter(
            torch.ones(k, channels, self.height_freq, self.width_freq) * 0.01
        )
        num_groups = get_num_groups(channels)
        self.group_norm = nn.GroupNorm(num_groups=1, num_channels=channels)

        self.alpha = nn.Parameter(torch.ones(k, 1, 1, 1) * 1,
                                           requires_grad=True)
        self.kernel_weights = nn.Parameter(generate_roi(k, height, width),
                                           requires_grad=True)
        self.relu = nn.ReLU()#Maxout(channels, channels)#
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
        out_cropped = (out_cropped * kw).sum(dim=0)         # (B, C, H, W)

        return self.group_norm(out_cropped.real)

class FSO(nn.Module):
    """
    Performs 2D convolution using Fast Fourier Transform (FFT) operations with enhanced normalization and scaling control.

    Args:
        in_channels (int): Number of input channels.
        init_scale (float): Initial scaling factor for the output.
    """
    def __init__(self, in_channels, height, width, height_p, width_p, init_scale=1.0, k=16):
        super(FSO, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, 1, 1, 1) * 1)
        num_groups = get_num_groups(in_channels)
        self.group_norm = nn.GroupNorm(num_groups=1, num_channels=in_channels)
        # self.kernel_weights = nn.Parameter(generate_roi(k, height, width), requires_grad=False)
        self.k = k

    def forward(self, x, p):
        """
        Forward pass for FFT-based convolution with enhanced normalization and scaling.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            p (torch.Tensor): Point Spread Function tensor of shape (B, C, H_p, W_p)

        Returns:
            torch.Tensor: Convolved and normalized output tensor of shape (B, C, H, W)
        """
        # Compute padding sizes
        B, C, H, W = x.shape
        _, _, H_p, W_p = p.shape

        # Ensure p is normalized to prevent scale amplification
        p = p.reshape(self.k, -1, p.size(-2), p.size(-1)).mean(dim=0, keepdim=True)
        p_sum = p.sum(dim=(-2, -1), keepdim=True)
        p_normalized = p / (abs(p_sum * self.alpha) + 1e-12)  # Prevent division by zero

        # Apply symmetric padding to raw input
        x_padded = F.pad(
            x,
            (W_p // 2, W_p - W_p // 2, H_p // 2, H_p - H_p // 2),
            mode='constant'
        )
        p_normalized = F.pad(
            p_normalized,
            (W // 2, W - W // 2, H // 2, H - H // 2),
            mode='constant'
        )

        X = torch.fft.rfft2(x_padded, dim=(-2, -1))  # Shape: (B, C, H_freq, W_freq)
        P = torch.fft.rfft2(p_normalized, s=(x_padded.size(-2), x_padded.size(-1)), dim=(-2, -1))  # Shape: (B, C, H_freq, W_freq)

        # Element-wise multiplication in frequency domain
        Y = X * P
        # Inverse FFT to spatial domain
        y = torch.fft.irfft2(Y, s=(x_padded.size(-2), x_padded.size(-1)), dim=(-2, -1))  # Shape: (B, C, H, W)
        # Shift zero frequency component to the center
        y = torch.fft.ifftshift(y, dim=(-2, -1))

#         kw = self.kernel_weights.unsqueeze(1).unsqueeze(2)  # (k, 1, 1, H, W)
#         y = y * kw             # (k, B, C, H, W)
#         y = y.sum(dim=0, keepdim=False)

        # Crop to original input size
        start_H = H_p // 2
        start_W = W_p // 2
        y_cropped = y[:, :, start_H:start_H + H, start_W:start_W + W]

        # Normalize using GroupNorm
        y_normalized = self.group_norm(y_cropped.real)
        return y_normalized.real

class IFIB(nn.Module):

    def __init__(self, in_channels, out_channels, height, width, height_p, width_p, block_cls=RB, exchange=0.2, k=16):
        super(IFIB, self).__init__()
        self.iso = ISO(in_channels, height, width, height_p, width_p, k=k)
        self.fso = FSO(in_channels, height, width, height_p, width_p, k=k)

        self.residual = False
        self.block_cls = block_cls
        self.in_channels = in_channels

        self.conv1_w = self.create_block(block_cls, in_channels, out_channels)
        self.conv1_c = self.create_block(block_cls, in_channels, out_channels)
        # self.conv2_w = self.create_block(block_cls, in_channels, out_channels)
        # self.conv2_c = self.create_block(block_cls, in_channels, out_channels)
        self.res_conv_w = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.res_conv_c = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        self.alpha_c = nn.Parameter(torch.full((1, 1, 1, 1), 1-exchange, dtype=torch.float32),
                                    requires_grad=True)
        self.delta_c = nn.Parameter(torch.full((1, 1, 1, 1), exchange, dtype=torch.float32),
                                    requires_grad=True)
        self.alpha_w = nn.Parameter(torch.full((1, 1, 1, 1), 1-exchange, dtype=torch.float32),
                                    requires_grad=True)
        self.delta_w = nn.Parameter(torch.full((1, 1, 1, 1), exchange, dtype=torch.float32),
                                    requires_grad=True)

    def create_block(self, block_cls, in_channels, out_channels):
        num_groups = get_num_groups(in_channels)
        return block_cls(in_channels, out_channels, num_groups=1)

    def forward(self, w, c, p):
        w_skip = w
        c_skip = c

        # Apply W and C modules
        w_ = self.iso(c, p)
        # w_ = w
        c_ = self.fso(w, p)
        # c_ = c

        w = self.conv1_w(w * self.alpha_w + w_ * self.delta_w)
        c = self.conv1_c(c * self.alpha_c + c_ * self.delta_c)

        # Combine with skips and residuals
        if self.residual:
            w = w + self.res_conv_w(w_skip)
            c = c + self.res_conv_c(c_skip)
        return w, c

class UpsampleIFIB(nn.Module):
    """Upscaling then DoubleConvG with IFIB."""

    def __init__(self, in_channels, out_channels, mid_channels, height, width, height_p, width_p, block_cls=RB, exchange=0.2, k=16):
        super(UpsampleIFIB, self).__init__()
        self.upw = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
        self.upc = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
        self.ifib = IFIB(mid_channels, out_channels, height, width, height_p, width_p, block_cls=block_cls, exchange=exchange, k=k)
        self.convw1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.convc1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)

    def forward(self, w1, w2, c1, c2, p):
        w1 = self.upw(w1)
        c1 = self.upc(c1)
        diffY = w2.size()[2] - w1.size()[2]
        diffX = w2.size()[3] - w1.size()[3]
        w1 = F.pad(w1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        c1 = F.pad(c1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        w = torch.cat([w2, w1], dim=1)
        c = torch.cat([c2, c1], dim=1)
        w = self.convw1(w)
        c = self.convc1(c)
        w, c = self.ifib(w, c, p)
        return w, c

class DownsampleIFIB(nn.Module):
    """Downscaling with average pooling followed by IFIB."""

    def __init__(self, in_channels, out_channels, height, width, height_p, width_p, block_cls=RB, exchange=0.2, k=16):
        super(DownsampleIFIB, self).__init__()
        self.pool = nn.AvgPool2d(2)
        self.ifib = IFIB(in_channels, out_channels, height, width, height_p, width_p, block_cls=block_cls, exchange=exchange, k=k)
        self.conv_block = ConvG(in_channels * k, out_channels * k)

    def forward(self, w, c, p):
        w = self.pool(w)
        c = self.pool(c)
        p = self.pool(p)
        w, c = self.ifib(w, c, p)
        p = self.conv_block(p)
        return w, c, p



class IFINNet(nn.Module):
    def __init__(self, in_channels, out_channels, psf, height=270, width=480, dim=32, depth=3, block_cls=RB, exchange=0.2, k=16, repeat=True, random=False,
                 seed_blocks="rb", bottleneck=False, residual=False,
                 upsample="bicubic", regularizer_activation="relu"):
        super().__init__()
        if depth < 2:
            raise ValueError("IFIN requires depth >= 2")
        if seed_blocks not in {"rb", "conv"}:
            raise ValueError("seed_blocks must be rb or conv")
        if upsample not in {"bicubic", "bilinear"}:
            raise ValueError("upsample must be bicubic or bilinear")
        if regularizer_activation not in {"relu", "sigmoid"}:
            raise ValueError("regularizer_activation must be relu or sigmoid")
        self.height, self.width = height, width
        self.psf = psf
        _, _, height_p, width_p = psf.size()
        self.depth = depth
        if repeat: self.psf = nn.Parameter(psf.repeat(1,k,1,1), requires_grad=True)
        else: self.psf = nn.Parameter(psf, requires_grad=True)
        if random: nn.init.xavier_uniform_(self.psf)
        channels = [dim * (2 ** i) for i in range(depth)]
        h = [height // (2 ** i) for i in range(depth+1)]
        w = [width // (2 ** i) for i in range(depth+1)]
        h_p = [height_p // (2 ** i) for i in range(depth+1)]
        w_p = [width_p // (2 ** i) for i in range(depth+1)]
        self.initial_iso = ISO(in_channels, h[0], w[0], h_p[0], w_p[0], k=k)

        # self.start_w = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1, bias=True)
        # self.start_c = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1, bias=True)
        # self.start_p = nn.Conv2d(self.psf.size(1), channels[0] * k, kernel_size=3, padding=1, bias=True)

        if seed_blocks == "conv":
            self.start_w = nn.Conv2d(in_channels, channels[0], 3, padding=1, bias=True)
            self.start_c = nn.Conv2d(in_channels, channels[0], 3, padding=1, bias=True)
            self.start_p = nn.Conv2d(self.psf.size(1), channels[0] * k, 3, padding=1, bias=True)
        else:
            self.start_w = self.create_block(block_cls, in_channels, channels[0])
            self.start_c = self.create_block(block_cls, in_channels, channels[0])
            self.start_p = ConvG(self.psf.size(1), channels[0] * k)


        # Create downsampling layers
        self.down_layers = nn.ModuleList()
        for i in range(depth):
            if i == 0:
                self.down_layers.append(
                    DownsampleIFIB(channels[i], channels[i+1], h[i+1], w[i+1], h_p[i+1], w_p[i+1], block_cls=block_cls, exchange=exchange, k=k))
            elif i < depth - 1:
                self.down_layers.append(
                    DownsampleIFIB(channels[i], channels[i+1], h[i+1], w[i+1], h_p[i+1], w_p[i+1], block_cls=block_cls, exchange=exchange, k=k))
            else:
                # For the last downsampling layer, in_channels = out_channels
                self.down_layers.append(
                    DownsampleIFIB(channels[i], channels[i], h[i+1], w[i+1], h_p[i+1], w_p[i+1], block_cls=block_cls, exchange=exchange, k=k))

        # Create upsampling layers
        self.up_layers = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            if i == 0:
                self.up_layers.append(
                    UpsampleIFIB(channels[0]*2, channels[0], channels[0], h[0], w[0], h_p[0], w_p[0], block_cls=block_cls, exchange=exchange, k=k)
                )
            else:
                self.up_layers.append(
                    UpsampleIFIB(channels[i]*2, channels[i-1], channels[i], h[i], w[i], h_p[i], w_p[i],block_cls=block_cls, exchange=exchange, k=k)
                )

        if bottleneck:
            self.center_layers = IFIB(channels[-1], channels[-1], h[-1], w[-1],
                                     h_p[-1], w_p[-1], block_cls=block_cls, exchange=exchange, k=k)

        self.refine_w = self.create_block(block_cls, channels[0], channels[0])
        self.out_w = nn.Conv2d(channels[0], out_channels, kernel_size=3, padding=1, stride=1, bias=True)
        self.refine_c = self.create_block(block_cls, channels[0], channels[0])
        self.out_c = nn.Conv2d(channels[0], out_channels, kernel_size=3, padding=1, stride=1, bias=True)
        for module in self.modules():
            if isinstance(module, ISO):
                module.relu = nn.ReLU() if regularizer_activation == "relu" else nn.Sigmoid()
            elif isinstance(module, IFIB):
                module.residual = residual
            elif isinstance(module, nn.Upsample):
                module.mode = upsample


    def create_block(self, block_cls, in_channels, out_channels):
        num_groups = get_num_groups(in_channels)
        return block_cls(in_channels, out_channels, num_groups=1)

    def forward(self, x):
        if tuple(x.shape[-2:]) != (self.height, self.width):
            raise ValueError(f"Expected {self.height}x{self.width} input; got {tuple(x.shape[-2:])}")
        p = self.psf
        x_wiener = self.initial_iso(x, p)
        w = self.start_w(x_wiener)
        c = self.start_c(x)
        p = self.start_p(p)

        w_feats = [w]
        c_feats = [c]
        p_feats = [p]

        # Downsampling
        for i in range(self.depth):
            w, c, p = self.down_layers[i](w_feats[-1], c_feats[-1], p_feats[-1])
            p_feats.append(p)
            w_feats.append(w)
            c_feats.append(c)

        if hasattr(self, "center_layers"):
            w, c = self.center_layers(w, c, p)

        # Upsampling
        for i in range(self.depth):
            idx = -(i + 2)
            p_feat = p_feats[idx]
            w_prev = w_feats[idx]
            c_prev = c_feats[idx]
            up_layer = self.up_layers[i]
            w, c = up_layer(w, w_prev, c, c_prev, p_feat)

        out_w = self.out_w(self.refine_w(w))
        out_c = self.out_c(self.refine_c(c))
        return out_w, out_c, x_wiener


IFIN = IFINNet


def build_ifin_model(config, psf, checkpoint=None):
    options = dict(config["model"])
    options.pop("name", None)
    options.pop("options", None)
    repeat = options.pop("repeat_psf", True)
    random_init = options.pop("random_init", False)
    center_index = config["data"].get("psf_center_index", 4)
    if checkpoint is not None:
        from utils.checkpoint import extract_state_dict
        state = extract_state_dict(checkpoint)
        learned_psf = state.get("psf")
        if learned_psf is not None:
            psf = learned_psf.detach().clone()
            repeat = False
            random_init = False
    elif config["data"].get("dataset") == "multiwienernet":
        psf = psf[:, center_index:center_index + 1]
    return IFINNet(psf=psf, repeat=repeat, random=random_init, block_cls=RB, **options)
