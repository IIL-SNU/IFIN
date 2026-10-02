from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class WallerDataset(Dataset):
    def __init__(self, path: str, train: bool = False, transform_raw=None, transform_lab=None):
        self.path = Path(path)
        self.transform_raw = transform_raw
        self.transform_lab = transform_lab
        csv_name = "dataset_train.csv" if train else "dataset_test.csv"
        self.df = pd.read_csv(self.path / csv_name)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        filename = self.df.iloc[idx, 0]
        raw_path = str((self.path / "diffuser_images" / filename)).replace(".jpg.tiff", ".npy")
        lab_path = str((self.path / "ground_truth_lensed" / filename)).replace(".jpg.tiff", ".npy")

        raw = np.load(raw_path)
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

        lab = np.load(lab_path)
        lab = cv2.cvtColor(lab, cv2.COLOR_BGR2RGB)

        if self.transform_raw is not None:
            raw = self.transform_raw(raw)
        if self.transform_lab is not None:
            lab = self.transform_lab(lab)

        return raw, lab
