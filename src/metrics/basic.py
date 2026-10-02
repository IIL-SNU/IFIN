from __future__ import annotations

import torch


def normalize_images(
    images: torch.Tensor, mode: str | bool | None = "max"
) -> torch.Tensor:
    if mode in (True, "max"):
        maximum = images.amax(dim=(-3, -2, -1), keepdim=True).clamp_min(1e-12)
        images = images / maximum
    return images.clamp(0, 1)


def psnr(
    img_pred: torch.Tensor, img_target: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    mse = (img_pred - img_target).square().mean(dim=(-3, -2, -1))
    return 10.0 * torch.log10(1.0 / mse.clamp_min(eps))


def build_metrics(
    names: list[str], device: torch.device
) -> dict[str, torch.nn.Module | None]:
    metrics: dict[str, torch.nn.Module | None] = (
        {"psnr": None} if "psnr" in names else {}
    )
    external = [name for name in names if name != "psnr"]
    if external:
        try:
            import pyiqa  # type: ignore
        except ImportError as error:
            raise ImportError(
                "SSIM/LPIPS evaluation requires the optional 'pyiqa' package"
            ) from error
        for name in external:
            metric_name = "lpips-vgg" if name == "lpips" else name
            metrics[name] = pyiqa.create_metric(
                metric_name, device=device, as_loss=False
            )
    return metrics


def metric_values(
    metrics: dict[str, torch.nn.Module | None],
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    values: dict[str, torch.Tensor] = {}
    for name, metric in metrics.items():
        if name == "psnr":
            values[name] = psnr(prediction, target)
        else:
            pred, ref = prediction, target
            if name == "lpips" and pred.shape[1] == 1:
                pred, ref = pred.repeat(1, 3, 1, 1), ref.repeat(1, 3, 1, 1)
            values[name] = torch.cat(
                [
                    metric(one_pred.unsqueeze(0), one_ref.unsqueeze(0)).reshape(-1)
                    for one_pred, one_ref in zip(pred, ref)
                ]  # type: ignore[misc]
            )
    return values
