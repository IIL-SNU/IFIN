from __future__ import annotations

import torch
from typing import Any, Dict

from torch import nn


class ReconstructionLoss(nn.Module):
    def __init__(
        self,
        config: Dict[str, Any] | None = None,
        use_lpips: bool | None = None,
        lpips_weight: float | None = None,
    ) -> None:
        super().__init__()
        self.mse = nn.MSELoss()
        weights = (config or {}).get("loss", {})
        self.weights = {
            "image": float(weights.get("image", 1.0)),
            "consistency_image": float(weights.get("consistency_image", 0.0)),
            "consistency_fourier": float(weights.get("consistency_fourier", 0.0)),
            "wiener": float(weights.get("wiener", 0.0)),
            "psf": float(weights.get("psf", 0.0)),
            "lpips": float(
                weights.get("lpips", 0.0) if lpips_weight is None else lpips_weight
            ),
        }
        self.fourier_applied = bool(weights.get("consistency_fourier_applied", True))
        self.lpips_normalize = bool(weights.get("lpips_normalize", True))
        enabled = self.weights["lpips"] > 0 if use_lpips is None else use_lpips
        self.lpips: nn.Module | None = None
        if enabled:
            try:
                import lpips  # type: ignore
            except ImportError as error:
                raise ImportError(
                    "loss.lpips requires the optional 'lpips' package"
                ) from error
            self.lpips = lpips.LPIPS(net="vgg").requires_grad_(False).eval()

    def forward(
        self,
        img_recon: torch.Tensor,
        img_target: torch.Tensor,
        meas_recon: torch.Tensor | None = None,
        meas_input: torch.Tensor | None = None,
        iso_recon: torch.Tensor | None = None,
        psf: torch.Tensor | None = None,
        train: bool = True,
        normalize: bool = True,
    ) -> torch.Tensor:
        del train
        image_loss = self.mse(img_recon, img_target)
        if self.lpips is not None:
            pred, target = img_recon, img_target
            if pred.shape[1] == 1:
                pred, target = pred.repeat(1, 3, 1, 1), target.repeat(1, 3, 1, 1)
            image_loss = (
                image_loss
                + self.weights["lpips"]
                * self.lpips(
                    pred, target, normalize=normalize and self.lpips_normalize
                ).mean()
            )
        loss = self.weights["image"] * image_loss
        if meas_recon is not None and meas_input is not None:
            loss = loss + self.weights["consistency_image"] * self.mse(
                meas_recon, meas_input
            )
            if self.fourier_applied:
                loss = loss + self.weights["consistency_fourier"] * self.mse(
                    torch.fft.rfft2(meas_recon).abs(), torch.fft.rfft2(meas_input).abs()
                )
        if iso_recon is not None:
            loss = loss + self.weights["wiener"] * self.mse(iso_recon, img_target)
        if psf is not None:
            loss = loss + self.weights["psf"] * torch.relu(-psf).mean()
        return loss
