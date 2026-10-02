from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from models.baselines import REGISTRY, build_baseline
from config import load_config


torch.set_num_threads(1)


OPTIONS = {
    "wiener": {},
    "admm": {"iterations": 1},
    "unet": {},
    "lensnet": {},
    "nafnet": {},
    "updn": {"depth": 1},
    "mwdns": {},
    "multiwienernet": {"k": 9},
    "modl": {"iterations": 1},
    "leadmmu": {"height": 32, "width": 32, "iterations": 1},
    "deeplir": {"iterations": 1, "dim": 8, "dim_mults": [1, 2]},
}


EXTERNAL_DIR = Path(__file__).resolve().parents[1] / "external_baselines"


def require_external(name: str) -> None:
    filename = {"deeplir": "deeplir.py", "modl": "modl.py"}.get(name)
    if filename and not (EXTERNAL_DIR / filename).is_file():
        pytest.skip(f"local external baseline source is not prepared: {filename}")


@pytest.mark.parametrize("name", sorted(OPTIONS))
def test_all_registered_baselines_construct(name: str) -> None:
    require_external(name)
    psf = torch.rand(1, 9, 32, 32)
    config = {"model": {"name": name, "options": OPTIONS[name]}, "eval": {"batch_size": 1}}
    model = build_baseline(config, psf, "cpu")
    assert isinstance(model, torch.nn.Module)
    assert name in REGISTRY


@pytest.mark.parametrize("name", ["unet", "nafnet", "multiwienernet"])
def test_small_forward_subset(name: str) -> None:
    psf = torch.rand(1, 9, 32, 32)
    config = {"model": {"name": name, "options": OPTIONS[name]}, "eval": {"batch_size": 1}}
    model = build_baseline(config, psf, "cpu").eval()
    channels = 3
    with torch.no_grad():
        output = model(torch.rand(1, channels, 32, 32))
    if isinstance(output, tuple):
        output = output[0]
    assert output.shape == (1, 3, 32, 32)
    assert torch.isfinite(output).all()


def test_rgb_multiwiener_repeats_single_psf_nine_times() -> None:
    config = {
        "model": {"name": "multiwienernet", "options": {"k": 9}},
        "eval": {"batch_size": 1},
    }
    model = build_baseline(config, torch.rand(1, 1, 32, 32), "cpu")
    assert model.wiener_model.psfs.shape == (9, 32, 32)


def test_multiwiener_profile_preserves_psf_batch_dimension() -> None:
    config = {
        "model": {"name": "multiwienernet", "options": {"k": 9, "psf_batch_dim": True}},
        "eval": {"batch_size": 1},
    }
    model = build_baseline(config, torch.rand(1, 9, 32, 32), "cpu")
    assert model.wiener_model.psfs.shape == (1, 9, 32, 32)


def test_multiwiener_grayscale_measurement_uses_nine_channel_unet() -> None:
    config = {
        "model": {
            "name": "multiwienernet",
            "options": {
                "k": 9,
                "psf_batch_dim": True,
                "in_channels": 1,
                "out_channels": 1,
            },
        },
        "eval": {"batch_size": 1},
    }
    model = build_baseline(config, torch.rand(1, 9, 32, 32), "cpu").eval()
    assert model.unet_model.down1.encode[0].conv.in_channels == 9
    with torch.no_grad():
        output, _ = model(torch.rand(1, 1, 32, 32))
    assert output.shape == (1, 1, 32, 32)


def test_factory_moves_plain_psf_attributes_to_target_device() -> None:
    config = {
        "model": {
            "name": "mwdns",
            "options": {"in_channels": 1, "out_channels": 1},
        }
    }
    model = build_baseline(config, torch.rand(1, 1, 32, 32), "meta")
    assert model.psf.device.type == "meta"


def test_deeplir_uses_requested_phase_batch_size() -> None:
    require_external("deeplir")
    config = {
        "phase": "train",
        "model": {
            "name": "deeplir",
            "options": {"iterations": 1, "dim": 8, "dim_mults": [1, 2]},
        },
        "train": {"batch_size": 3},
        "eval": {"batch_size": 1},
    }
    model = build_baseline(config, torch.rand(1, 1, 32, 32), "cpu")
    assert model.admm_model.batch_size == 3


def test_lensnet_can_use_checkpoint_psf_shape() -> None:
    config = {
        "model": {
            "name": "lensnet",
            "options": {"in_channels": 1, "out_channels": 1, "use_input_psf": True, "center_index": 4},
        }
    }
    model = build_baseline(config, torch.rand(1, 9, 224, 320), "cpu")
    assert model.psf.shape == (1, 1, 224, 320)


def test_official_lensnet_dataset_profiles_preserve_research_options() -> None:
    for name, channels, height, width in [("widercam", 3, 270, 480), ("diffusercam", 3, 270, 480), ("multiwienernet", 1, 224, 320)]:
        config = load_config(str(PROJECT_ROOT / "configs" / f"{name}.yaml"), {"model": {"name": "lensnet"}})
        count = 9 if name == "multiwienernet" else channels
        model = build_baseline(config, torch.rand(1, count, height, width), "cpu")
        assert model.psf.shape == (1, channels, height, width)
        assert model.w.item() == pytest.approx(0.01)
        assert model.clamp_output is False


def test_safe_baselines_work_without_external_sources(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("IFIN_EXTERNAL_BASELINES", str(tmp_path))
    psf = torch.rand(1, 1, 32, 32)
    model = build_baseline({"model": {"name": "unet", "options": {}}}, psf, "cpu")
    assert isinstance(model, torch.nn.Module)
    with pytest.raises(RuntimeError, match="prepare_external_baselines.py"):
        build_baseline({"model": {"name": "modl", "options": {}}}, psf, "cpu")
