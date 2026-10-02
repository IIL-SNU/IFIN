from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

from data import build_dataset
from data.benchmarks import evaluation_view
from data.psf import build_psf
from losses.basic import ReconstructionLoss
from metrics.basic import build_metrics, metric_values, normalize_images
from models import build_model
from utils.checkpoint import load_checkpoint, load_model_state_compat, save_checkpoint
from utils.seed import seed_everything


def _select_device(value: str) -> torch.device:
    return torch.device(
        "cuda"
        if value == "auto" and torch.cuda.is_available()
        else "cpu"
        if value == "auto"
        else value
    )


def _build_loader(config: Dict[str, Any], split: str) -> DataLoader:
    options = config[split]
    return DataLoader(
        build_dataset(config, split),
        batch_size=options["batch_size"],
        shuffle=split == "train",
        num_workers=options.get("num_workers", 0),
        pin_memory=options.get("pin_memory", False),
    )


def _outputs(
    config: Dict[str, Any], output: Any
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    name = config["model"].get("name", "ifin").lower()
    if name == "ifin":
        image, measurement, initial = output
        return image, measurement, initial
    index = 1 if name == "deeplir" else 0
    return (output[index] if isinstance(output, (tuple, list)) else output), None, None


def _rng_state() -> Dict[str, Any]:
    numpy_state = np.random.get_state()
    state: Dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
        "random": random.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: Dict[str, Any]) -> None:
    torch.set_rng_state(state["torch"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:])
    )
    random.setstate(state["random"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _run_loader(
    config: Dict[str, Any],
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: ReconstructionLoss,
    optimizer: optim.Optimizer | None = None,
    psf_optimizer: optim.Optimizer | None = None,
    metric_names: list[str] | None = None,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    metrics = build_metrics(metric_names or ["psnr"], device)
    totals = {"loss": 0.0, **{name: 0.0 for name in metrics}}
    view = config.get("eval", {}).get("metric_view") if metric_names is not None else None
    if view:
        totals.update({f"{name}_{view}": 0.0 for name in metrics})
    samples = 0
    for measurement, target in loader:
        measurement, target = measurement.to(device), target.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
            if psf_optimizer:
                psf_optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            image, measurement_recon, initial = _outputs(config, model(measurement))
            model_psf = (
                getattr(model, "psf", None)
                if config["model"].get("name", "ifin").lower() == "ifin"
                else None
            )
            loss = criterion(
                image,
                target,
                measurement_recon,
                measurement,
                initial,
                model_psf,
                train=training,
            )
            if training:
                loss.backward()
                optimizer.step()
                if psf_optimizer:
                    psf_optimizer.step()
        batch_size = target.shape[0]
        totals["loss"] += float(loss.detach()) * batch_size
        mode = config.get("eval", {}).get("normalize", "max")
        prediction, reference = (
            normalize_images(image.detach(), mode),
            target.detach().clamp(0, 1),
        )
        for name, values in metric_values(metrics, prediction, reference).items():
            totals[name] += float(values.sum())
        if view:
            for name, values in metric_values(
                metrics, evaluation_view(prediction, config), evaluation_view(reference, config)
            ).items():
                totals[f"{name}_{view}"] += float(values.sum())
        samples += batch_size
    if not samples:
        raise ValueError("Dataset is empty")
    return {name: value / samples for name, value in totals.items()}


def _setup(config: Dict[str, Any], checkpoint_path: str | None = None):
    seed_everything(config["seed"], deterministic=config["deterministic"])
    device = _select_device(config["device"])
    checkpoint = load_checkpoint(checkpoint_path, "cpu") if checkpoint_path else None
    model = build_model(
        config, build_psf(config).to(device), device=device, checkpoint=checkpoint
    )
    if checkpoint:
        load_model_state_compat(model, checkpoint, strict=True)
    return device, model, checkpoint


def train(config: Dict[str, Any], checkpoint_path: str | None = None) -> Dict[str, Any]:
    config = copy.deepcopy(config)
    config["phase"] = "train"
    device, model, checkpoint = _setup(config, checkpoint_path)
    psf = (
        getattr(model, "psf", None)
        if config["model"].get("name", "ifin").lower() == "ifin"
        else None
    )
    separate_psf = (
        psf is not None
        and psf.requires_grad
        and config.get("optimizer", {}).get("psf_lr") is not None
    )
    checkpoint_psf_state = None
    if checkpoint:
        checkpoint_psf_state = checkpoint.get(
            "psf_optimizer_state_dict", checkpoint.get("optimizer_psf_state_dict")
        )
        if bool(checkpoint_psf_state) != separate_psf:
            raise ValueError(
                "Checkpoint PSF optimizer grouping does not match optimizer.psf_lr; "
                "use the optimizer layout recorded by the checkpoint"
            )
    network_parameters = [
        parameter
        for parameter in model.parameters()
        if not separate_psf or parameter is not psf
    ]
    if not network_parameters:
        raise ValueError(
            f"Model {config['model'].get('name')!r} has no trainable network parameters; use eval"
        )
    optimizer = optim.AdamW(
        network_parameters,
        lr=float(config.get("optimizer", {}).get("lr", 1e-4)),
    )
    psf_optimizer = None
    if separate_psf:
        psf_optimizer = optim.AdamW([psf], lr=float(config["optimizer"]["psf_lr"]))
    scheduler_options = {
        "factor": 0.5,
        "patience": 25,
        "threshold": 1e-4,
        **config.get("scheduler", {}),
    }
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", **scheduler_options
    )
    psf_scheduler = (
        optim.lr_scheduler.ReduceLROnPlateau(
            psf_optimizer, mode="min", **scheduler_options
        )
        if psf_optimizer
        else None
    )
    start_epoch = 0
    best_loss = float("inf")
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if psf_optimizer and checkpoint_psf_state:
            psf_optimizer.load_state_dict(checkpoint_psf_state)
        if checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if psf_scheduler and checkpoint.get("psf_scheduler_state_dict"):
            psf_scheduler.load_state_dict(checkpoint["psf_scheduler_state_dict"])
        elif psf_scheduler and checkpoint.get("scheduler_psf_state_dict"):
            psf_scheduler.load_state_dict(checkpoint["scheduler_psf_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0))
        if "epoch" in checkpoint and "config" not in checkpoint:
            start_epoch += 1
        best_loss = float(checkpoint.get("best_loss", best_loss))
        if checkpoint.get("rng_state"):
            _restore_rng(checkpoint["rng_state"])

    criterion = ReconstructionLoss(config).to(device)
    train_loader = _build_loader(config, "train")
    eval_loader = _build_loader(config, "eval")
    history = []
    epochs = int(config.get("train", {}).get("epochs", 100))
    for epoch in range(start_epoch, epochs):
        training = _run_loader(
            config,
            model,
            train_loader,
            device,
            criterion,
            optimizer,
            psf_optimizer,
        )
        validation = _run_loader(
            config, model, eval_loader, device, criterion
        )
        scheduler.step(validation["loss"])
        if psf_scheduler:
            psf_scheduler.step(validation["loss"])
        record = {
            "epoch": epoch + 1,
            **{f"train_{name}": value for name, value in training.items()},
            **{f"val_{name}": value for name, value in validation.items()},
        }
        history.append(record)
        if config.get("checkpoint", {}).get("save", True):
            improved = validation["loss"] < best_loss
            best_loss = min(best_loss, validation["loss"])
            payload = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": config,
                "metrics": record,
                "best_loss": best_loss,
                "rng_state": _rng_state(),
            }
            if psf_optimizer:
                payload["psf_optimizer_state_dict"] = psf_optimizer.state_dict()
                payload["psf_scheduler_state_dict"] = psf_scheduler.state_dict()
            name = config["model"].get("name", "ifin")
            output_dir = Path(config["checkpoint"]["output_dir"])
            save_checkpoint(output_dir / f"{name}_last.pth", payload)
            if improved:
                best_name = "ifin_best.pth" if name == "ifin" else "model_best.pth"
                save_checkpoint(output_dir / best_name, payload)
    final = history[-1] if history else {"epoch": start_epoch}
    return {"epochs": epochs, "history": history, "device": str(device), **final}


def train_one_epoch(config: Dict[str, Any]) -> Dict[str, float]:
    one_epoch = copy.deepcopy(config)
    one_epoch.setdefault("train", {})["epochs"] = 1
    result = train(one_epoch)
    record = result["history"][-1]
    return {
        "loss": record["train_loss"],
        "psnr": record["train_psnr"],
        "device": result["device"],
    }


def evaluate(
    config: Dict[str, Any], checkpoint_path: str | None = None
) -> Dict[str, float]:
    config = copy.deepcopy(config)
    config["phase"] = "eval"
    device, model, _ = _setup(config, checkpoint_path)
    names = [
        name.lower()
        for name in config.get("eval", {}).get("metrics", ["psnr", "ssim", "lpips"])
    ]
    result = _run_loader(
        config,
        model,
        _build_loader(config, "eval"),
        device,
        ReconstructionLoss(config).to(device),
        metric_names=names,
    )
    return {**result, "device": str(device), "weights_loaded": checkpoint_path is not None}


def _load_input(path: str, config: Dict[str, Any]) -> torch.Tensor:
    if Path(path).suffix.lower() in {".npy", ".npz"}:
        loaded = np.load(path)
        array = (
            loaded[loaded.files[0]]
            if isinstance(loaded, np.lib.npyio.NpzFile)
            else loaded
        )
    else:
        from PIL import Image

        array = np.asarray(Image.open(path))
    if array.ndim == 2:
        array = array[..., None]
    if np.issubdtype(array.dtype, np.integer):
        dtype = array.dtype
        array = array.astype(np.float32) / np.iinfo(dtype).max
    tensor = torch.as_tensor(array, dtype=torch.float32)
    expected = (
        int(config["model"]["in_channels"]),
        int(config["model"]["height"]),
        int(config["model"]["width"]),
    )
    if (
        tensor.ndim == 3
        and tuple(tensor.shape) != expected
        and tensor.shape[-1] in (1, 3, 4)
    ):
        tensor = tensor.permute(2, 0, 1)
    if tensor.shape[0] == 4:
        tensor = tensor[:3]
    if tuple(tensor.shape) != expected:
        raise ValueError(f"Expected input shape {expected}, got {tuple(tensor.shape)}")
    if Path(path).suffix.lower() in {".npy", ".npz"} and tensor.shape[0] == 3:
        default_order = "bgr" if config["data"]["dataset"] in {"diffusercam", "waller"} else "rgb"
        if config["data"].get("npy_color_order", default_order).lower() == "bgr":
            tensor = tensor[[2, 1, 0]]
    return tensor.unsqueeze(0)


def _save_image(image: torch.Tensor, path: Path, mode: str | bool | None = "max") -> None:
    from PIL import Image

    array = (
        (normalize_images(image, mode)[0].permute(1, 2, 0).cpu().numpy() * 255)
        .round()
        .astype(np.uint8)
    )
    if array.shape[-1] == 1:
        array = array[..., 0]
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def infer(
    config: Dict[str, Any],
    checkpoint_path: str | None = None,
    input_path: str | None = None,
    output_dir: str | None = None,
) -> Dict[str, Any]:
    device, model, _ = _setup(config, checkpoint_path)
    measurement = (
        _load_input(input_path, config)
        if input_path
        else build_dataset(config, "eval")[0][0].unsqueeze(0)
    )
    measurement = measurement.to(device)
    model.eval()
    with torch.no_grad():
        image, measurement_recon, initial = _outputs(config, model(measurement))
    result: Dict[str, Any] = {
        "input_shape": tuple(measurement.shape),
        "img_recon_shape": tuple(image.shape),
        "device": str(device),
        "weights_loaded": checkpoint_path is not None,
    }
    if measurement_recon is not None:
        result["meas_recon_shape"] = tuple(measurement_recon.shape)
        result["iso_recon_shape"] = tuple(initial.shape)
    if input_path or output_dir:
        artifact = Path(output_dir or "outputs/inference") / "reconstruction.png"
        _save_image(image, artifact, config.get("eval", {}).get("normalize", "max"))
        result["output"] = str(artifact)
    return result
