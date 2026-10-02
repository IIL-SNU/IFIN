from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .admm import ADMMs
from ._external import load_external
from .leadmm_unet import UNet270480
from .lensnet import LensNet
from .multiwiener import MyEnsemble, WienerDeconvolution3D
from .mwdns import MWDNet_CPSF
from .nafnet import NAFNet
from .unet import UNet
from .updn import ImageOptimizer, ImageOptimizerMixColors
from .wiener import WieNerDeconv


class LeADMMU(nn.Module):
    def __init__(self, unet: nn.Module, admms: nn.Module):
        super().__init__()
        self.unet = unet
        self.admms = admms

    def forward(self, x):
        self.admms_out = self.admms(x)
        self.admms_out = self.admms_out / self.admms_out.max()
        self.unet_out = self.unet(self.admms_out)
        return self.unet_out


def _options(config: Mapping[str, Any]) -> tuple[str, dict[str, Any], Mapping[str, Any]]:
    model_config = config.get("model", config)
    if not isinstance(model_config, Mapping):
        raise TypeError("config.model must be a mapping")
    name = str(model_config.get("name", "")).lower()
    options = dict(model_config.get("options", {}))
    if not name:
        raise ValueError("config.model.name is required")
    return name, options, config


def _batch_size(config: Mapping[str, Any], options: Mapping[str, Any]) -> int:
    phase = str(options.get("phase", config.get("phase", "eval")))
    phase_config = config.get(phase, {})
    return int(options.get("batch_size", phase_config.get("batch_size", 1)))


def _prepare_psf(psf: torch.Tensor, options: Mapping[str, Any], full_stack: bool = False) -> torch.Tensor:
    if psf.ndim != 4 or psf.shape[0] != 1:
        raise ValueError(f"psf must have shape (1,N,H,W), got {tuple(psf.shape)}")
    prepared = psf.detach().clone().float()
    if not full_stack:
        center = int(options.get("center_index", 4 if prepared.shape[1] >= 9 else prepared.shape[1] // 2))
        prepared = prepared[:, center : center + 1]
    mode = str(options.get("psf_norm", "none")).lower()
    if mode == "l2":
        prepared = prepared / torch.linalg.norm(prepared)
    elif mode == "max":
        prepared = prepared / prepared.max()
    elif mode != "none":
        raise ValueError(f"unknown psf_norm: {mode}")
    return prepared * float(options.get("psf_scale", 1.0))


def _channels(options: Mapping[str, Any]) -> tuple[int, int]:
    return int(options.get("in_channels", 3)), int(options.get("out_channels", 3))


def _build_wiener(options, psf, device, config):
    return WieNerDeconv(_prepare_psf(psf, options), device=device, g=int(options.get("gaussian_windows", 3)))


def _build_admm(options, psf, device, config):
    cls = load_external("modl", options).ADMMs if options.get("implementation") == "modl" else ADMMs
    model = cls(
        _prepare_psf(psf, options),
        int(options.get("iterations", 30)),
        int(options.get("stacks", 1)),
        device=device,
        train=bool(options.get("trainable", False)),
    )
    if "fixed_parameters" in options:
        model.mu1, model.mu2, model.mu3, model.tau = options["fixed_parameters"]
    return model


def _build_unet(options, psf, device, config):
    in_channels, out_channels = _channels(options)
    cls = load_external("modl", options).UNet if options.get("implementation") == "modl" else UNet
    return cls(in_channels, out_channels)


def _build_lensnet(options, psf, device, config):
    in_channels, out_channels = _channels(options)
    if bool(options.get("use_input_psf", False)):
        initial_psf = _prepare_psf(psf, options)
    else:
        model_config = config.get("model", {})
        height = int(model_config.get("height", psf.shape[-2]))
        width = int(model_config.get("width", psf.shape[-1]))
        initial_psf = torch.rand(1, in_channels, height, width, device=device)
    return LensNet(
        in_channels, out_channels, ngf=int(options.get("ngf", 64)), psf=initial_psf,
        w_init=float(options.get("w_init", 0.001)),
        clamp_output=bool(options.get("clamp_output", True)),
    )


def _build_nafnet(options, psf, device, config):
    in_channels, out_channels = _channels(options)
    return NAFNet(img_channel=in_channels, out_channel=out_channels)


def _build_updn(options, psf, device, config):
    prepared = _prepare_psf(psf, options)
    depth = int(options.get("depth", 10))
    if bool(options.get("mix_colors", True)):
        return ImageOptimizerMixColors(prepared, depth=depth, c=int(options.get("in_channels", 3)))
    return ImageOptimizer(prepared.squeeze(0), depth=depth)


def _build_mwdns(options, psf, device, config):
    in_channels, out_channels = _channels(options)
    return MWDNet_CPSF(in_channels, out_channels, _prepare_psf(psf, options))


def _build_multiwiener(options, psf, device, config):
    prepared = _prepare_psf(psf, options, full_stack=True)
    count = int(options.get("k", 9))
    if prepared.shape[1] == 1 and count > 1:
        prepared = prepared[0].repeat(count, 1, 1)
    elif bool(options.get("psf_batch_dim", False)):
        prepared = prepared[:, :count]
    else:
        prepared = prepared[0, :count]
    ks = np.ones((count, 1, 1), dtype=np.float32)
    wiener_model = WienerDeconvolution3D(prepared, ks)
    out_channels = int(options.get("out_channels", 3))
    unet_cls = load_external("modl", options).UNet if options.get("implementation") == "modl" else UNet
    return MyEnsemble(wiener_model, unet_cls(count, out_channels))


def _build_modl(options, psf, device, config):
    prepared = _prepare_psf(psf, options)
    module = load_external("modl", options)
    if bool(options.get("three_dimensional", False)):
        return module.MoDLNet3D(
            prepared,
            ch=int(options.get("out_channels", 1)),
            psf_in_channels=int(options.get("psf_in_channels", 1)),
            iteration=int(options.get("iterations", 5)),
            device=device,
        )
    return module.MoDLNet(
        prepared,
        psf_in_channels=int(options.get("psf_in_channels", 1)),
        iteration=int(options.get("iterations", 5)),
        device=device,
    )


def _build_leadmmu(options, psf, device, config):
    channels = int(options.get("in_channels", 3))
    height = int(options.get("height", psf.shape[-2]))
    width = int(options.get("width", psf.shape[-1]))
    unet = UNet270480((channels, height, width))
    cls = load_external("modl", options).ADMMs if options.get("implementation") == "modl" else ADMMs
    admms = cls(_prepare_psf(psf, options).squeeze(0), int(options.get("iterations", 5)), 1, device=device)
    return LeADMMU(unet, admms)


def _build_deeplir(options, psf, device, config):
    prepared = _prepare_psf(psf, options).squeeze()
    module = load_external("deeplir", options)
    admm = module.build_model(
        prepared,
        batch_size=_batch_size(config, options),
        iterations=int(options.get("iterations", 5)),
        device=device,
    )
    denoiser = module.Unet(
        dim=int(options.get("dim", 32)),
        channels=int(options.get("in_channels", 3)),
        dim_mults=tuple(options.get("dim_mults", (1, 2, 4, 8))),
    )
    return module.EnsembleModel(admm, denoiser)


REGISTRY: dict[str, Callable[..., nn.Module]] = {
    "wiener": _build_wiener,
    "admm": _build_admm,
    "unet": _build_unet,
    "lensnet": _build_lensnet,
    "nafnet": _build_nafnet,
    "updn": _build_updn,
    "mwdns": _build_mwdns,
    "multiwienernet": _build_multiwiener,
    "modl": _build_modl,
    "leadmmu": _build_leadmmu,
    "deeplir": _build_deeplir,
}


def build_baseline(config: Mapping[str, Any], psf: torch.Tensor, device: str | torch.device) -> nn.Module:
    name, options, root_config = _options(config)
    target_device = torch.device(device)
    psf = psf.to(target_device)
    try:
        builder = REGISTRY[name]
    except KeyError as error:
        raise ValueError(f"unknown baseline {name!r}; choose from {sorted(REGISTRY)}") from error
    return builder(options, psf, target_device, root_config).to(target_device)


__all__ = ["REGISTRY", "build_baseline"]
