import numpy as np
import torch


def to_tensor_or_numpy(data):
    if torch.is_tensor(data):
        data = data.cpu().detach() if data.is_cuda else data
        if data.ndim == 3:
            data = data.permute(1, 2, 0)
        elif data.ndim == 4:
            data = data.permute(0, 2, 3, 1)
        return data.numpy()
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
        if data.ndim == 3:
            data = data.permute(2, 0, 1)
        elif data.ndim == 4:
            data = data.permute(0, 3, 1, 2)
        return data
    raise ValueError("Input must be a PyTorch Tensor or a Numpy array")


def norm(img, normalization="max"):
    if normalization == "max":
        if torch.is_tensor(img):
            maximum = img.reshape(img.size(0), -1).max(1)[0].reshape(img.size(0), 1, 1, 1)
            return img / maximum
        return ((img / np.max(img)) * 255).astype(np.uint8)
    if normalization == "linalg":
        if torch.is_tensor(img):
            norms = torch.linalg.norm(img.reshape(img.size(0), -1), dim=1).reshape(img.size(0), 1, 1, 1)
            return (img / norms) / norms
        return (img / np.linalg.norm(img.reshape(-1)) * 255).astype(np.uint8)
    if torch.is_tensor(img):
        minimum = img.reshape(img.size(0), -1).min(1)[0].reshape(img.size(0), 1, 1, 1)
        maximum = img.reshape(img.size(0), -1).max(1)[0].reshape(img.size(0), 1, 1, 1)
        return (img - minimum) / (maximum - minimum)
    return ((img - np.min(img)) / (np.max(img) - np.min(img)) * 255).astype(np.uint8)


def gaus_t_multiple(x, fwhms):
    _, _, width, height = x.size()
    fwhms = torch.tensor(fwhms, dtype=x.dtype, device=x.device).view(-1, 1, 1)
    x_w = torch.arange(width, dtype=x.dtype, device=x.device) - (width - 1) / 2
    x_h = torch.arange(height, dtype=x.dtype, device=x.device) - (height - 1) / 2
    ga_w = torch.exp(-0.5 * (x_w.view(1, -1) / (width / fwhms)) ** 2)
    ga_h = torch.exp(-0.5 * (x_h.view(1, -1) / (height / fwhms)) ** 2)
    ga = (ga_w.unsqueeze(-1) * ga_h.unsqueeze(-2)).unsqueeze(1)
    return x.unsqueeze(0) * ga
