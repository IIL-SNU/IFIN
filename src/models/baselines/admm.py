# Original source comments are preserved from the audited research file named in docs/baselines.md.
import time
import torch
import numpy as np
import torch.fft as fft
import torch.nn.functional as f
import torch.nn as nn
from ._utils import norm, to_tensor_or_numpy

class ADMMs(nn.Module):
    def __init__(self, psf, iterations, stacks, device='cpu', train=True):
        """
        Initialize the ADMMs model.

        Parameters:
        - iterations (int): Number of ADMM iterations.
        - stacks (int): Number of stacks.
        - psf (numpy.ndarray or torch.Tensor): Point spread function.
        - device (str): Device to run the model on (default is 'cpu').
        """
        super(ADMMs, self).__init__()
        self.device = device
        if not torch.is_tensor(psf):
            psf = to_tensor_or_numpy(psf)
        self.psf_init = psf

        # Trainable parameters
        if train:
            self.mu1 = torch.nn.Parameter(torch.tensor(np.ones(stacks) * 1.0e-03, dtype=torch.float32))
            self.mu2 = torch.nn.Parameter(torch.tensor(np.ones(stacks) * 1.0e-03, dtype=torch.float32))
            self.mu3 = torch.nn.Parameter(torch.tensor(np.ones(stacks) * 1.0e-03, dtype=torch.float32))
            self.tau = torch.nn.Parameter(torch.tensor(np.ones(stacks) * 1.0e-03, dtype=torch.float32))
        else:
            self.mu1 = [1e4]
            self.mu2 = [1e4]
            self.mu3 = [1e4]
            self.tau = [1e4]

        self.iterations = iterations
        self.stacks = stacks

    def forward(self, raw):
        """
        Perform ADMM iterations on the input.

        Parameters:
        - x_in (numpy.ndarray or torch.Tensor): Input data.

        Returns:
        - torch.Tensor: Processed output.
        """
        if not torch.is_tensor(raw):
            raw = to_tensor_or_numpy(raw)
        # self.psf = self.psf.to(self.device)
        self.psf = norm(self.psf_init, normalization='linalg').to(self.device)
        # self.psf = self.psf_init.to(self.device)
        self.raw = raw.to(self.device)
        # self.raw = norm(self.raw, normalization='linalg')

        self.sensor_size = self.psf.shape
        self.full_size = tuple(x for x in self.psf.shape[:-2]) + tuple(2*x for x in self.psf.shape[-2:])
        X, U, self.V, W, xi, eta, rho = self.init_Matrices()
        total_time = 0

        for j in range(self.iterations):
            self.i = 0
            for i in range(self.stacks):
                self.i = i
                start_time = time.time()
                X, U, self.V, W, xi, eta, rho = self.ADMM_Step(X, U, self.V, W, xi, eta, rho)
        img = self.Crop(torch.clone(self.V), self.full_size, self.sensor_size)
        return img

    def init_Matrices(self):
        """
        Initialize matrices and variables for ADMM.

        Returns:
        - Tuple of torch.Tensor: Initialized matrices and variables.
        """
        self.H_fft = self.fft_shift_2d(self.CropT(self.psf, self.full_size, self.sensor_size))
        X_divmat_num = self.CropT(torch.ones(self.sensor_size).to(self.device), self.full_size, self.sensor_size)
        self.X_divmat = torch.ones_like(X_divmat_num).to(self.device) / (X_divmat_num + self.mu1[0])

        MTM_component = self.mu1[0] * (torch.abs(torch.conj(self.H_fft) * self.H_fft))

        PsiTPsi = torch.zeros(self.full_size).to(self.device)
        PsiTPsi[...,0,0] = 4
        PsiTPsi[...,0,1] = PsiTPsi[...,1,0] = PsiTPsi[...,0,-1] = PsiTPsi[...,-1,0] = -1
        PsiTPsi = fft.fftn(PsiTPsi, dim=(-2,-1))

        self.R_divmat = torch.ones_like(MTM_component).to(self.device) / (
                MTM_component + self.mu2[0] * torch.abs(PsiTPsi) + self.mu3[0])

        X = torch.zeros(self.full_size).to(self.device)
        U = torch.zeros(self.full_size + (2,)).to(self.device)
        self.V = torch.zeros(self.full_size).to(self.device)
        W = torch.zeros(self.full_size).to(self.device)
        xi = torch.zeros_like(self.M(self.V, self.H_fft)).to(self.device)
        eta = torch.zeros_like(self.Psi(self.V)).to(self.device)
        rho = torch.zeros_like(W).to(self.device)

        return X, U, self.V, W, xi, eta, rho

    def ADMM_Step(self, X, U, V, W, xi, eta, rho):
        U = self.U_update(eta, self.V)
        X = self.X_update(xi, V, self.raw)
        self.V = self.V_update(W, rho, U, eta, X, xi)
        W = self.W_update(rho, self.V)
        xi = self.xi_update(xi, self.V, X)
        eta = self.eta_update(eta, self.V, U)
        rho = self.rho_update(rho, self.V, W)
        return X, U, self.V, W, xi, eta, rho

    def U_update(self, eta, image_est):
        return self.SoftThresh(self.Psi(image_est) + eta / self.mu2[self.i], self.tau[self.i] / self.mu2[self.i])

    def X_update(self, xi, image_est, sensor_reading):
        return self.X_divmat * (xi + self.mu1[self.i] * self.M(image_est, self.H_fft) +
                                self.CropT(sensor_reading, self.full_size, self.sensor_size))

    def W_update(self, rho, image_est):
        return torch.maximum(rho / self.mu3[self.i] + image_est, torch.zeros_like(image_est).to(self.device))

    def V_update(self, w, rho, u, eta, x, xi):
        freq_space_result = self.R_divmat * self.fft_shift_2d(self.r_calc(w, rho, u, eta, x, xi))
        return torch.real(self.ifft_shift_2d(freq_space_result))

    def r_calc(self, w, rho, u, eta, x, xi):
        return (self.mu3[self.i] * w - rho) + self.PsiT(self.mu2[self.i] * u - eta) + self.MT(
            self.mu1[self.i] * x - xi, self.H_fft)

    def xi_update(self, xi, V, X):
        return xi + self.mu1[self.i] * (self.M(V, self.H_fft) - X)

    def eta_update(self, eta, V, U):
        return eta + self.mu2[self.i] * (self.Psi(V) - U)

    def rho_update(self, rho, V, W):
        return rho + self.mu3[self.i] * (V - W)

    def SoftThresh(self, x, tau_c):
        theta = tau_c * 0.1
        return torch.sign(x) * (
                (torch.abs(x) - tau_c - theta) + torch.sqrt(
            torch.pow((torch.abs(x) - tau_c - theta), 2) + 4 * theta * torch.abs(x))) / 2

    def Psi(self, v):
        return torch.stack((torch.roll(v, 1, dims = -2) - v, torch.roll(v, 1, dims = -1) - v), dim = -1)

    def PsiT(self, U):
        diff1 = torch.roll(U[..., 0], -1, dims=-2) - U[..., 0]
        diff2 = torch.roll(U[..., 1], -1, dims=-1) - U[..., 1]
        return diff1 + diff2

    def fft_shift_2d(self, *args, **kwargs):
        return fft.fftn(fft.ifftshift(*args, dim=(-2,-1)), dim=(-2,-1))

    def ifft_shift_2d(self, *args, **kwargs):
        return fft.fftshift(fft.ifftn(*args, dim=(-2,-1)), dim=(-2,-1))

    def M(self, vk, H_fft):
        return torch.real(self.ifft_shift_2d((self.fft_shift_2d(vk) * H_fft)))

    def MT(self, x, H_fft):
        return torch.real(self.ifft_shift_2d((self.fft_shift_2d(x) * torch.conj(H_fft))))

    def Crop(self, M, full_size, sensor_size):
        top = (full_size[-2] - sensor_size[-2]) // 2
        bottom = (full_size[-2] + sensor_size[-2]) // 2
        left = (full_size[-1] - sensor_size[-1]) // 2
        right = (full_size[-1] + sensor_size[-1]) // 2
        return M[...,top:bottom,left:right]

    def CropT(self, b, full_size, sensor_size):
        v_pad = (full_size[-2] - sensor_size[-2]) // 2
        h_pad = (full_size[-1] - sensor_size[-1]) // 2
        return f.pad(b, (h_pad, h_pad, v_pad, v_pad))
