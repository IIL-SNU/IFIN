from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import scipy.io
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import load_config
from data import build_dataset, build_psf
from data.benchmarks import WiderCamTargetTransform


def test_diffusercam_evaluation_crop_uses_flipped_coordinates():
    from data.benchmarks import evaluation_view

    image = torch.arange(270 * 480, dtype=torch.float32).reshape(1, 1, 270, 480)
    cropped = evaluation_view(image, {"eval": {"metric_view": "crop"}})
    assert cropped.shape == (1, 1, 210, 380)
    assert torch.equal(cropped, image.flip(-2)[..., 60:, 62:-38])


def test_widercam_evaluation_dewarping_undoes_forward_translation():
    from data.benchmarks import evaluation_view

    image = np.zeros((3, 4, 1), dtype=np.uint8)
    image[1, 1, 0] = 255
    matrix = [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
    forward = WiderCamTargetTransform((3, 4), matrix)(image).unsqueeze(0)
    config = {"eval": {"metric_view": "dewarped"}, "data": {"affine": {"matrix": matrix}}}
    restored = evaluation_view(forward.expand(2, -1, -1, -1), config)
    expected = torch.from_numpy(image).permute(2, 0, 1).float().div(255).unsqueeze(0)
    torch.testing.assert_close(restored, expected.expand_as(restored), atol=1e-6, rtol=0)


def test_benchmark_config_runtime_schema() -> None:
    model_keys = {
        "name",
        "in_channels",
        "out_channels",
        "height",
        "width",
        "dim",
        "depth",
        "k",
        "exchange",
        "repeat_psf",
        "random_init",
        "seed_blocks",
        "regularizer_activation",
        "residual",
        "bottleneck",
        "upsample",
    }
    loss_keys = {"image", "measurement", "iso", "psf", "lpips"}
    for name in ("widercam", "diffusercam", "multiwienernet"):
        config = load_config(str(PROJECT_ROOT / "configs" / f"{name}.yaml"))
        assert set(config["model"]) == model_keys
        expected_loss_keys = loss_keys | ({"lpips_normalize"} if name == "widercam" else set())
        assert set(config["loss"]) == expected_loss_keys
        assert config["eval"]["normalize"] == ("clip_max" if name == "widercam" else "none")
        baseline = load_config(str(PROJECT_ROOT / "configs" / f"{name}.yaml"), {"model": {"name": "wiener"}})
        assert baseline["eval"]["normalize"] == ("max" if name == "diffusercam" else "clip_max")
    for name in ("widercam", "diffusercam", "multiwienernet"):
        assert load_config(str(PROJECT_ROOT / "configs" / f"{name}.yaml"))["model"]["k"] == 1
    assert load_config(str(PROJECT_ROOT / "configs" / "smoke.yaml"))["model"]["k"] == 1
    default_config = load_config(str(PROJECT_ROOT / "configs" / "default.yaml"))
    assert "random_init_psf" not in default_config["model"]
    assert default_config["model"]["random_init"] is False


def test_synthetic_psf_is_seeded_and_path_independent() -> None:
    config = load_config(str(PROJECT_ROOT / "configs" / "smoke.yaml"))
    first = build_psf(config)
    second = build_psf(config)
    changed_seed = dict(config)
    changed_seed["seed"] = config["seed"] + 1

    assert first.shape == (1, 1, 16, 16)
    assert first.device.type == "cpu"
    assert first.dtype == torch.float32
    assert torch.equal(first, second)
    assert not torch.equal(first, build_psf(changed_seed))


def test_widercam_affine_and_data_root(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "widercam"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    pd.DataFrame(["sample.jpg.tiff"]).to_csv(root / "dataset_train.csv", index=False)

    raw = np.zeros((3, 4, 3), dtype=np.uint8)
    raw[..., 0] = 10
    raw[..., 1] = 20
    raw[..., 2] = 30
    target = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
    cv2.imwrite(str(root / "images" / "sample_rgb8.png"), raw)
    cv2.imwrite(str(root / "labels" / "sample.png"), target)

    monkeypatch.setenv("DATA_ROOT", str(root))
    config = load_config(str(PROJECT_ROOT / "configs" / "widercam.yaml"))
    assert config["data"]["root"] == str(root)
    assert config["data"]["psf_crop"] == [133, 236, 130, 133]
    assert config["provenance"]["checkpoint_psf_shape_candidates"] == [[1, 9, 130, 133]]
    assert config["model"]["seed_blocks"] == "rb"
    assert config["model"]["regularizer_activation"] == "relu"
    config["data"]["affine"] = {
        "target_size": [3, 4],
        "pad": 0,
        "align_corners": False,
        "bbox": None,
        "matrix": [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
    }
    config["data"]["target_suffix"] = ".png"
    measurement, warped_target = build_dataset(config, "train")[0]

    assert measurement.shape == (3, 3, 4)
    assert torch.equal(measurement[:, 0, 0], torch.tensor([30, 20, 10]) / 255.0)
    assert warped_target.shape == (3, 3, 4)
    assert torch.allclose(warped_target[:, :, 0], torch.zeros(3, 3), atol=1e-7)
    assert torch.allclose(
        warped_target[:, 1, 2],
        torch.tensor([17.0, 16.0, 15.0]) / 255.0,
        atol=1e-6,
    )


def test_widercam_notebook_affine_pixel_regression() -> None:
    affine = load_config(str(PROJECT_ROOT / "configs" / "widercam.yaml"))["data"]["affine"]
    assert affine["matrix"] == [
        [0.741431858, -0.0103940972, 62.9320307],
        [0.0103940972, 0.741431858, 32.5871991],
    ]
    y, x = np.indices((270, 480))
    image = np.stack(
        ((3 * y + 5 * x) % 256, (7 * y + 11 * x) % 256, (13 * y + 17 * x) % 256),
        axis=-1,
    ).astype(np.uint8)
    transformed = WiderCamTargetTransform(
        tuple(affine["target_size"]),
        affine["matrix"],
        int(affine["pad"]),
        None,
        bool(affine["align_corners"]),
    )(image)

    assert torch.allclose(
        transformed[:, 100, 200],
        torch.tensor([0.6736698151, 0.4159618318, 0.8534965515]),
        atol=1e-6,
        rtol=0,
    )
    assert torch.allclose(
        transformed[:, 134, 239],
        torch.tensor([0.2400431931, 0.9918004274, 0.6751270294]),
        atol=1e-6,
        rtol=0,
    )


def test_diffusercam_adapter_and_signal_cropped_psf(tmp_path: Path) -> None:
    root = tmp_path / "diffusercam"
    (root / "diffuser_images").mkdir(parents=True)
    (root / "ground_truth_lensed").mkdir()
    pd.DataFrame(["pair.jpg.tiff"]).to_csv(root / "dataset_test.csv", index=False)
    raw = np.array([[[1, 2, 3]]], dtype=np.uint8)
    target = np.array([[[4, 5, 6]]], dtype=np.uint8)
    np.save(root / "diffuser_images" / "pair.npy", raw)
    np.save(root / "ground_truth_lensed" / "pair.npy", target)
    psf_image = np.zeros((6, 8), dtype=np.uint8)
    psf_image[2:5, 3:7] = 255
    cv2.imwrite(str(root / "psf.tiff"), psf_image)

    config = load_config(str(PROJECT_ROOT / "configs" / "diffusercam.yaml"))
    config["data"]["root"] = str(root)
    config["model"]["height"] = 6
    config["model"]["width"] = 8
    config["data"]["psf_bg_patch"] = 1
    config["data"]["psf_normalization"] = "l2"
    measurement, label = build_dataset(config, "eval")[0]
    psf = build_psf(config)

    assert torch.equal(measurement[:, 0, 0], torch.tensor([3, 2, 1]) / 255.0)
    assert torch.equal(label[:, 0, 0], torch.tensor([6, 5, 4]) / 255.0)
    assert psf.shape == (1, 1, 3, 4)
    expected_psf = torch.ones_like(psf)
    expected_psf[..., 0] = 0
    expected_psf = expected_psf / (12.0 ** 0.5)
    assert torch.allclose(psf, expected_psf)


def test_multiwienernet_crop_split_and_psf(tmp_path: Path) -> None:
    root = tmp_path / "multiwienernet"
    target_dir = root / "2D" / "Ground_truth_downsampled"
    measurement_dir = root / "2D" / "Simulated_Miniscope_2D_Training_data"
    psf_dir = root / "PSF"
    target_dir.mkdir(parents=True)
    measurement_dir.mkdir(parents=True)
    psf_dir.mkdir()

    image = np.arange(486 * 648, dtype=np.uint32).reshape(486, 648) % 251 + 1
    image = image.astype(np.uint8)
    for index in range(5):
        cv2.imwrite(str(target_dir / f"{index}.png"), image)
        cv2.imwrite(str(measurement_dir / f"{index}.png"), image)

    stack = np.zeros((486, 648, 9, 1), dtype=np.float32)
    for index in range(9):
        stack[..., index, 0] = (index + 1) * image.astype(np.float32)
    scipy.io.savemat(psf_dir / "multiWienerPSFStack_40z_aligned.mat", {"multiWienerPSFStack_40z": stack})

    config = load_config(str(PROJECT_ROOT / "configs" / "multiwienernet.yaml"))
    assert config["model"]["k"] == 1
    assert config["provenance"]["paper_main_k"] == 9
    assert config["model"]["seed_blocks"] == "conv"
    assert config["model"]["regularizer_activation"] == "sigmoid"
    assert config["model"]["residual"] is True
    assert config["model"]["bottleneck"] is True
    assert config["model"]["upsample"] == "bilinear"
    assert config["data"]["psf_center_index"] == 4
    config["data"]["root"] = str(root)
    train_dataset = build_dataset(config, "train")
    eval_dataset = build_dataset(config, "eval")
    measurement, target = train_dataset[0]
    psf = build_psf(config)

    assert len(train_dataset) == 4
    assert len(eval_dataset) == 1
    assert measurement.shape == target.shape == (1, 224, 320)
    assert torch.allclose(measurement, target)
    assert float(measurement.max()) == 1.0
    assert psf.shape == (1, 9, 224, 320)
    assert float(psf.min()) >= 0.0
    assert float(psf.max()) == 1.0
    assert torch.all(psf[:, 1:] >= psf[:, :-1])
