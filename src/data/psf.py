from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import scipy.io
import torch

from .benchmarks import _resolve_path


def _crop_from_center(psf: torch.Tensor, bbox: tuple[int, int, int, int]) -> torch.Tensor:
    center_y, center_x, height, width = bbox
    y1 = max(0, center_y - height // 2)
    x1 = max(0, center_x - width // 2)
    return psf[..., y1 : min(psf.shape[-2], y1 + height), x1 : min(psf.shape[-1], x1 + width)]


def _signal_bbox(psf: torch.Tensor) -> tuple[int, int, int, int]:
    indices = (psf > 0).nonzero(as_tuple=False)
    if indices.numel() == 0:
        raise ValueError("PSF has no positive pixels after background subtraction")
    y_min, y_max = indices[:, 2].min(), indices[:, 2].max()
    x_min, x_max = indices[:, 3].min(), indices[:, 3].max()
    return (
        int(torch.div(y_min + y_max, 2, rounding_mode="floor")),
        int(torch.div(x_min + x_max, 2, rounding_mode="floor")),
        int(y_max - y_min + 1),
        int(x_max - x_min + 1),
    )


def _load_image_psf(path: Path, data_config: Dict[str, Any], height: int, width: int) -> torch.Tensor:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot read PSF image: {path}")
    interpolation = cv2.INTER_AREA if data_config.get("psf_interpolation") == "area" else cv2.INTER_LINEAR
    image = cv2.resize(image, (width, height), interpolation=interpolation)
    psf = torch.from_numpy(np.asarray(image)).to(dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    if "psf_background" in data_config:
        background = float(data_config["psf_background"])
    else:
        patch = int(data_config.get("psf_bg_patch", 15))
        background = float(psf[:, :, :patch, :patch].mean())
    psf = torch.clamp(psf - background, min=0)
    normalization = data_config.get("psf_normalization", "none")
    if normalization == "max":
        psf = psf / psf.max()
    elif normalization == "l2":
        psf = psf / torch.linalg.norm(psf)
    elif normalization != "none":
        raise ValueError(f"Unknown PSF normalization: {normalization}")
    crop = data_config.get("psf_crop")
    if crop == "signal":
        psf = _crop_from_center(psf, _signal_bbox(psf))
    elif crop:
        psf = _crop_from_center(psf, tuple(int(value) for value in crop))
    return psf


def _load_multiwiener_psf(path: Path, data_config: Dict[str, Any]) -> torch.Tensor:
    key = data_config.get("psf_key", "multiWienerPSFStack_40z")
    psfs = scipy.io.loadmat(path)[key]
    psfs = psfs[18:466, 4:644]
    psfs = psfs[..., int(data_config.get("psf_depth_index", 0))]
    psfs = psfs.transpose(2, 0, 1)
    psfs = cv2.resize(psfs.transpose(1, 2, 0), (0, 0), fx=0.5, fy=0.5).transpose(2, 0, 1)
    tensor = torch.tensor(psfs, dtype=torch.float32).unsqueeze(0)
    normalization = data_config.get("psf_normalization", "max")
    if normalization == "max":
        tensor = tensor / tensor.max()
    elif normalization == "l2":
        tensor = tensor / torch.linalg.norm(tensor)
    elif normalization == "byte":
        tensor = tensor / 255.0
    return torch.clamp(tensor, min=0)


def build_psf(config: Dict[str, Any]) -> torch.Tensor:
    data_config = config["data"]
    if data_config["dataset"].lower() == "synthetic":
        size = int(data_config["psf_size"])
        generator = torch.Generator(device="cpu").manual_seed(int(config["seed"]))
        return torch.rand((1, 1, size, size), generator=generator, dtype=torch.float32)
    root = _resolve_path(data_config["root"])
    path = _resolve_path(data_config["psf_path"], root)
    if data_config["dataset"].lower() == "multiwienernet":
        psf = _load_multiwiener_psf(path, data_config)
    else:
        model_config = config["model"]
        psf = _load_image_psf(
            path,
            data_config,
            height=int(model_config["height"]),
            width=int(model_config["width"]),
        )
    if psf.ndim != 4 or psf.shape[0] != 1:
        raise ValueError(f"Expected PSF shape (1,N,H,W), got {tuple(psf.shape)}")
    return psf.cpu().float()
