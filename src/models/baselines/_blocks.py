import torch
import torch.nn as nn
import torch.nn.functional as F


BN_EPS = 1e-4


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        return weight.view(1, -1, 1, 1) * y + bias.view(1, -1, 1, 1)

    @staticmethod
    def backward(ctx, grad_output):
        y, var, weight = ctx.saved_tensors
        grad = grad_output * weight.view(1, -1, 1, 1)
        mean_grad = grad.mean(dim=1, keepdim=True)
        mean_grad_y = (grad * y).mean(dim=1, keepdim=True)
        grad_x = (grad - mean_grad - y * mean_grad_y) / torch.sqrt(var + ctx.eps)
        grad_weight = (grad_output * y).sum(dim=(0, 2, 3))
        grad_bias = grad_output.sum(dim=(0, 2, 3))
        return grad_x, grad_weight, grad_bias, None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class ConvBnRelu2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.batchnorm = nn.BatchNorm2d(out_channels, eps=BN_EPS)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))


class StackEncoder(nn.Module):
    def __init__(self, x_channels, y_channels, kernel_size=(3, 3)):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.encode = nn.Sequential(
            ConvBnRelu2d(x_channels, y_channels, kernel_size, padding),
            ConvBnRelu2d(y_channels, y_channels, kernel_size, padding),
        )

    def forward(self, x):
        x = self.encode(x)
        return x, F.max_pool2d(x, kernel_size=2, stride=2)


class StackDecoder(nn.Module):
    def __init__(self, x_big_channels, x_channels, y_channels, kernel_size=3):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.decode = nn.Sequential(
            ConvBnRelu2d(x_big_channels + x_channels, y_channels, kernel_size, padding),
            ConvBnRelu2d(y_channels, y_channels, kernel_size, padding),
            ConvBnRelu2d(y_channels, y_channels, kernel_size, padding),
        )

    def forward(self, x, down_tensor):
        x = F.interpolate(x, size=down_tensor.shape[-2:], mode="bilinear", align_corners=False)
        return self.decode(torch.cat([x, down_tensor], dim=1))


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        mid_channels = mid_channels or out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels, eps=BN_EPS),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=BN_EPS),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.avgpool_conv = nn.Sequential(nn.AvgPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.avgpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
