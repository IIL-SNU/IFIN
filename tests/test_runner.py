from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import runner  # noqa: E402
from config import load_config  # noqa: E402
from losses.basic import ReconstructionLoss  # noqa: E402
from metrics.basic import normalize_images  # noqa: E402
from utils.checkpoint import load_checkpoint  # noqa: E402


def test_yaml_measurement_and_iso_loss_weights_are_applied():
    criterion = ReconstructionLoss({"loss": {"measurement": 2.0, "iso": 3.0}})
    zero = torch.zeros(1, 1, 2, 2)
    one = torch.ones_like(zero)
    loss = criterion(zero, zero, one, zero, one)
    assert loss.item() == pytest.approx(5.0)


def test_psf_penalty_matches_squared_training_only_recipe():
    criterion = ReconstructionLoss({"loss": {"psf": 2.0}})
    zero = torch.zeros(1, 1, 2, 2)
    psf = torch.tensor([-2.0, 3.0])
    assert criterion(zero, zero, psf=psf).item() == pytest.approx(4.0)
    assert criterion(zero, zero, psf=psf, train=False).item() == pytest.approx(0.0)


def test_notebook_clip_before_max_normalization():
    image = torch.tensor([[[[2.0, 0.5]]]])
    assert torch.equal(normalize_images(image, "clip_max"), torch.tensor([[[[1.0, 0.5]]]]))
    assert torch.equal(normalize_images(image, "max"), torch.tensor([[[[1.0, 0.25]]]]))
    dim_image = torch.tensor([[[[-0.2, 0.25, 0.5]]]])
    assert torch.equal(normalize_images(dim_image, "clip_max"), torch.tensor([[[[0.0, 0.5, 1.0]]]]))
    assert torch.equal(normalize_images(torch.zeros_like(image), "clip_max"), torch.zeros_like(image))


def test_evaluation_does_not_rescale_ground_truth():
    config = {"model": {"name": "ifin"}, "eval": {"normalize": "max"}}
    dataset = TensorDataset(torch.ones(1, 1, 2, 2), torch.full((1, 1, 2, 2), 0.5))
    result = runner._run_loader(
        config, TinyIFIN(), torch.utils.data.DataLoader(dataset),
        torch.device("cpu"), ReconstructionLoss(config), metric_names=["psnr"],
    )
    assert result["psnr"] == pytest.approx(10 * np.log10(4))


def test_diffusercam_numpy_input_matches_dataset_rgb_order(tmp_path):
    measurement = np.zeros((2, 2, 3), dtype=np.float32)
    measurement[..., 0] = 0.2
    measurement[..., 2] = 0.8
    path = tmp_path / "raw.npy"
    np.save(path, measurement)
    config = {
        "model": {"in_channels": 3, "height": 2, "width": 2},
        "data": {"dataset": "diffusercam"},
    }
    tensor = runner._load_input(str(path), config)
    assert tensor[0, 0, 0, 0] == pytest.approx(0.8)
    assert tensor[0, 2, 0, 0] == pytest.approx(0.2)


class TinyIFIN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.5))
        self.psf = nn.Parameter(torch.ones(1, 1, 2, 2))

    def forward(self, measurement: torch.Tensor):
        reconstruction = measurement * self.weight
        return reconstruction, reconstruction, reconstruction


def _config(tmp_path: Path) -> dict:
    return {
        "seed": 7,
        "deterministic": True,
        "device": "cpu",
        "model": {"name": "ifin", "in_channels": 1, "height": 2, "width": 2},
        "data": {"dataset": "synthetic"},
        "train": {"epochs": 1, "num_samples": 3, "batch_size": 2, "num_workers": 0},
        "eval": {
            "num_samples": 3,
            "batch_size": 2,
            "num_workers": 0,
            "normalize": False,
        },
        "optimizer": {"lr": 1e-4, "psf_lr": 1e-3},
        "loss": {"image": 1.0, "lpips": 0.0},
        "checkpoint": {"save": True, "output_dir": str(tmp_path)},
    }


def test_config_inheritance_env_and_baseline_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_DATA", str(tmp_path / "dataset"))
    (tmp_path / "base.yaml").write_text(
        "data: {dataset: widercam, root: '${TEST_DATA:-data/fallback}', psf_crop: signal}\n"
        "model: {name: ifin, in_channels: 3, out_channels: 3}\n"
        "eval: {batch_size: 4}\n",
        encoding="utf-8",
    )
    (tmp_path / "child.yaml").write_text(
        "inherits: base.yaml\neval: {batch_size: 2}\n", encoding="utf-8"
    )
    (tmp_path / "profiles.yaml").write_text(
        "baselines:\n  wiener:\n    name: wiener\n    options: {in_channels: 1, out_channels: 1}\n"
        "    eval: {normalize: none}\n",
        encoding="utf-8",
    )
    config = load_config(
        str(tmp_path / "child.yaml"),
        {"model": {"name": "wiener"}},
        str(tmp_path / "profiles.yaml"),
    )
    assert config["data"]["root"] == str(tmp_path / "dataset")
    assert config["data"]["psf_crop"] is None
    assert config["data"]["psf_normalization"] == "none"
    assert config["eval"]["batch_size"] == 2
    assert config["eval"]["normalize"] == "none"
    assert config["model"]["name"] == "wiener"
    assert config["model"]["in_channels"] == 1


def test_reconstruction_loss_uses_only_available_terms() -> None:
    config = {
        "loss": {
            "image": 2,
            "consistency_image": 3,
            "consistency_fourier": 99,
            "consistency_fourier_applied": False,
            "wiener": 4,
            "psf": 5,
            "lpips": 0,
        }
    }
    criterion = ReconstructionLoss(config)
    zeros, ones = torch.zeros(1, 1, 2, 2), torch.ones(1, 1, 2, 2)
    loss = criterion(zeros, ones, zeros, ones, zeros, -ones)
    assert loss.item() == pytest.approx(2 + 3 + 4 + 5)
    assert criterion(zeros, ones).item() == pytest.approx(2)


def test_train_resume_and_real_input_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    dataset = TensorDataset(torch.ones(3, 1, 2, 2), torch.zeros(3, 1, 2, 2))
    monkeypatch.setattr(runner, "build_dataset", lambda _config, _split: dataset)
    monkeypatch.setattr(runner, "build_psf", lambda _config: torch.ones(1, 1, 2, 2))
    monkeypatch.setattr(runner, "build_model", lambda *_args, **_kwargs: TinyIFIN())

    first = runner.train(config)
    checkpoint_path = tmp_path / "ifin_last.pth"
    checkpoint = load_checkpoint(checkpoint_path)
    assert first["epoch"] == 1
    assert (tmp_path / "ifin_best.pth").is_file()
    assert checkpoint["best_loss"] == first["val_loss"]
    assert {
        "psf_optimizer_state_dict",
        "scheduler_state_dict",
        "psf_scheduler_state_dict",
        "rng_state",
    } <= checkpoint.keys()

    config["train"]["epochs"] = 2
    resumed = runner.train(config, str(checkpoint_path))
    assert resumed["epoch"] == 2

    input_path = tmp_path / "measurement.npy"
    np.save(input_path, np.ones((1, 2, 2), dtype=np.float32))
    result = runner.infer(
        config, input_path=str(input_path), output_dir=str(tmp_path / "reconstruction")
    )
    assert Path(result["output"]).is_file()
