# Original source comments are preserved from the audited research file named in docs/baselines.md.
import time
import torch
import numpy as np
import torch.fft as fft
import torch.nn.functional as f
import torch.nn as nn
import torchvision
from ._utils import gaus_t_multiple, norm, to_tensor_or_numpy

class WieNerDeconv(nn.Module):
    def __init__(self, psf, iteration=0, device='cpu', g=3):
        super().__init__()
        self.device = device
        self.h = psf
        if not torch.is_tensor(self.h):
            self.h = to_tensor_or_numpy(self.h)
        self.h = self.h.to(self.device)
        self.h_size = psf.size(-2)
        self.w_size = psf.size(-1)
        self.padding_height = self.h_size//2
        self.padding_width = self.w_size//2
        self.g = g

        self.h = f.pad(self.h, (self.padding_width, self.padding_width, self.padding_height, self.padding_height), mode='constant')

        # self.h = norm(self.h, normalization='minmax')
        self.H = fft.rfft2(self.h)
        self.iteration = iteration

    def preprocess(self, img):
        if not torch.is_tensor(img):
            img = to_tensor_or_numpy(img)
            img = img.to(self.device)
        # img = norm(img, normalization='minmax')
        img = f.pad(img, (self.padding_width, self.padding_width, self.padding_height, self.padding_height), mode='replicate')
        img = gaus_t_multiple(img, self.g)
        return img

    def forward(self, img, delta=[1.0e-2]):
        delta = torch.tensor(delta, device=img.device).view(-1, 1, 1, 1,1)  # Delta 확장 (B, C, H, W와 호환되도록)

        img = self.preprocess(img)
        Y = fft.rfft2(img)
        # H_WieNer = torch.conj(self.H) / (torch.abs(self.H) ** 2 + delta)
        H_WieNer = torch.conj(self.H).unsqueeze(0) / ((torch.abs(self.H) ** 2).unsqueeze(0) + delta)  # Delta와 broadcasting
        x_hat = fft.ifftshift(fft.irfft2(H_WieNer.unsqueeze(0) * Y.unsqueeze(1)), (-2, -1))
        x_hat = x_hat.mean(dim=[0, 1], keepdim=False)
        return torchvision.transforms.CenterCrop((self.h_size, self.w_size))(x_hat.real)
