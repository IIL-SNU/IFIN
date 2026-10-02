from __future__ import annotations

import glob
import os
import random
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, Subset
from torchvision.transforms import InterpolationMode

from .synthetic import SyntheticLenslessDataset, SyntheticLenslessSpec
from .waller import WallerDataset


def _resolve_path(value: str, root: Path | None = None) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(value))
    path = Path(expanded)
    data_root = os.environ.get("DATA_ROOT")
    if data_root and path.parts[:1] == ("data",):
        path = Path(data_root, *path.parts[1:])
    elif root is not None and not path.is_absolute():
        path = root / path
    return path


def _opencv_matrix_to_theta(
    matrix: torch.Tensor,
    height: int,
    width: int,
    *,
    align_corners: bool = False,
    inverse: bool = True,
) -> torch.Tensor:
    identity = torch.eye(3, device=matrix.device, dtype=matrix.dtype)
    if align_corners:
        scale = identity.clone()
        scale[0, 0] = 2.0 / max(width - 1, 1)
        scale[1, 1] = 2.0 / max(height - 1, 1)
        scale[0, 2] = -1.0
        scale[1, 2] = -1.0
        scale_inverse = identity.clone()
        scale_inverse[0, 0] = (width - 1) / 2.0
        scale_inverse[1, 1] = (height - 1) / 2.0
        scale_inverse[0, 2] = (width - 1) / 2.0
        scale_inverse[1, 2] = (height - 1) / 2.0
    else:
        scale = identity.clone()
        scale[0, 0] = 2.0 / width
        scale[1, 1] = 2.0 / height
        scale[0, 2] = (1.0 / width) - 1.0
        scale[1, 2] = (1.0 / height) - 1.0
        scale_inverse = identity.clone()
        scale_inverse[0, 0] = width / 2.0
        scale_inverse[1, 1] = height / 2.0
        scale_inverse[0, 2] = (width / 2.0) - 0.5
        scale_inverse[1, 2] = (height / 2.0) - 0.5

    homogeneous = identity.clone()
    homogeneous[:2] = matrix
    mapping = torch.linalg.inv(homogeneous) if inverse else homogeneous
    return (scale @ mapping @ scale_inverse)[:2].unsqueeze(0)


def evaluation_view(images: torch.Tensor, config: Dict[str, Any]) -> torch.Tensor:
    view = config["eval"]["metric_view"]
    if view == "crop":
        return TF.resize(images, [270, 480]).flip(-2)[..., 60:, 62:-38]
    if view == "dewarped":
        affine = config["data"]["affine"]
        align_corners = bool(affine.get("align_corners", False))
        matrix = images.new_tensor(affine["matrix"])
        theta = _opencv_matrix_to_theta(
            matrix, *images.shape[-2:], align_corners=align_corners, inverse=False
        ).expand(images.shape[0], -1, -1)
        grid = F.affine_grid(theta, images.shape, align_corners=align_corners)
        return F.grid_sample(
            images, grid, mode="bicubic", padding_mode="zeros", align_corners=align_corners
        ).clamp(0, 1)
    raise ValueError(f"Unknown evaluation metric view: {view}")


class WiderCamTargetTransform:
    def __init__(
        self,
        target_size: tuple[int, int],
        matrix: list[list[float]],
        pad: int = 0,
        bbox: tuple[int, int, int, int] | None = None,
        align_corners: bool = False,
    ) -> None:
        self.target_size = target_size
        self.matrix = torch.tensor(matrix, dtype=torch.float32)
        self.pad = pad
        self.bbox = bbox
        self.align_corners = align_corners

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        source = TF.to_tensor(image)
        source = TF.resize(source, self.target_size, interpolation=InterpolationMode.BICUBIC)
        source = TF.pad(source, [self.pad] * 4)
        channels, height, width = source.shape
        theta = _opencv_matrix_to_theta(
            self.matrix.to(device=source.device, dtype=source.dtype),
            height,
            width,
            align_corners=self.align_corners,
        )
        grid = F.affine_grid(theta, (1, channels, height, width), align_corners=self.align_corners)
        warped = F.grid_sample(
            source.unsqueeze(0),
            grid,
            mode="bicubic",
            padding_mode="zeros",
            align_corners=self.align_corners,
        ).squeeze(0)

        y_min, x_min, y_max, x_max = self.bbox or (0, 0, height, width)
        masked = torch.zeros_like(source)
        masked[:, y_min : y_max + 1, x_min : x_max + 1] = warped[
            :, y_min : y_max + 1, x_min : x_max + 1
        ]
        return masked


class WiderCamDataset(Dataset):
    def __init__(self, root: Path, split: str, data_config: Dict[str, Any]) -> None:
        self.root = root
        self.raw_dir = data_config.get("raw_dir", "images")
        self.target_dir = data_config.get("target_dir", "labels")
        csv_name = data_config.get(f"{split}_csv", f"dataset_{'train' if split == 'train' else 'test'}.csv")
        frame = pd.read_csv(root / csv_name)
        raw_suffix = data_config.get("raw_suffix", "_rgb8.png")
        self.rows = [
            row
            for row in frame.iloc[:, 0].astype(str)
            if (root / self.raw_dir / row.replace(".jpg.tiff", raw_suffix)).exists()
        ]
        self.raw_suffix = raw_suffix
        self.target_suffix = data_config.get("target_suffix", ".jpg")
        affine = data_config["affine"]
        self.target_transform = WiderCamTargetTransform(
            target_size=tuple(affine["target_size"]),
            matrix=affine["matrix"],
            pad=int(affine.get("pad", 0)),
            bbox=tuple(affine["bbox"]) if affine.get("bbox") is not None else None,
            align_corners=bool(affine.get("align_corners", False)),
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        filename = self.rows[index]
        raw_path = self.root / self.raw_dir / filename.replace(".jpg.tiff", self.raw_suffix)
        target_path = self.root / self.target_dir / filename.replace(".jpg.tiff", self.target_suffix)
        raw = cv2.imread(str(raw_path), cv2.IMREAD_UNCHANGED)
        target = cv2.imread(str(target_path), cv2.IMREAD_UNCHANGED)
        if raw is None or target is None:
            raise FileNotFoundError(f"Cannot read WiderCam pair: {raw_path}, {target_path}")
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        return TF.to_tensor(raw), self.target_transform(target)


class MultiWienerNetDataset(Dataset):
    def __init__(self, target_files: list[str], measurement_dir: Path) -> None:
        self.target_files = target_files
        self.measurement_dir = measurement_dir

    def __len__(self) -> int:
        return len(self.target_files)

    @staticmethod
    def _preprocess(image: np.ndarray) -> torch.Tensor:
        image = image.astype(np.float32) / 255.0
        image = image[18:466, 4:644]
        image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
        image = image / np.max(image)
        return torch.from_numpy(image).unsqueeze(0)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        target_path = Path(self.target_files[index])
        target = cv2.imread(str(target_path), cv2.IMREAD_UNCHANGED)
        measurement = cv2.imread(str(self.measurement_dir / target_path.name), cv2.IMREAD_UNCHANGED)
        if target is None or measurement is None:
            raise FileNotFoundError(f"Cannot read MultiWienerNet pair for {target_path.name}")
        return self._preprocess(measurement), self._preprocess(target)


def _limit_dataset(dataset: Dataset, config: Dict[str, Any], split: str) -> Dataset:
    count = config.get(split, {}).get("num_samples")
    if count is None:
        return dataset
    return Subset(dataset, range(min(int(count), len(dataset))))


def build_dataset(config: Dict[str, Any], split: str) -> Dataset:
    if split not in {"train", "eval"}:
        raise ValueError("split must be 'train' or 'eval'")
    data_config = config["data"]
    dataset_name = data_config["dataset"].lower()

    if dataset_name == "synthetic":
        split_config = config[split]
        model_config = config["model"]
        dataset: Dataset = SyntheticLenslessDataset(
            SyntheticLenslessSpec(
                num_samples=int(split_config["num_samples"]),
                channels=int(model_config["in_channels"]),
                height=int(model_config["height"]),
                width=int(model_config["width"]),
                seed=int(config["seed"]) + (0 if split == "train" else 1),
            )
        )
        return dataset

    root = _resolve_path(data_config["root"])
    if dataset_name == "widercam":
        dataset = WiderCamDataset(root, split, data_config)
    elif dataset_name in {"diffusercam", "waller"}:
        dataset = WallerDataset(
            str(root),
            train=split == "train",
            transform_raw=TF.to_tensor,
            transform_lab=TF.to_tensor,
        )
    elif dataset_name == "multiwienernet":
        target_dir = _resolve_path(data_config["target_dir"], root)
        measurement_dir = _resolve_path(data_config["measurement_dir"], root)
        target_files = glob.glob(str(target_dir / "*"))
        random.Random(int(data_config.get("split_seed", 8))).shuffle(target_files)
        split_index = int(len(target_files) * (1.0 - float(data_config.get("test_fraction", 0.2))))
        selected = target_files[:split_index] if split == "train" else target_files[split_index:]
        dataset = MultiWienerNetDataset(selected, measurement_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return _limit_dataset(dataset, config, split)
