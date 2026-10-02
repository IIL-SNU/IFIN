from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from utils.checkpoint import load_checkpoint, save_checkpoint


def test_save_over_memory_mapped_checkpoint(tmp_path):
    path = tmp_path / "checkpoint.pth"
    save_checkpoint(path, {"optimizer_tensor": torch.arange(64)})
    loaded = load_checkpoint(path)
    save_checkpoint(path, loaded)
    torch.testing.assert_close(loaded["optimizer_tensor"], torch.arange(64))
    torch.testing.assert_close(
        load_checkpoint(path)["optimizer_tensor"], torch.arange(64)
    )
    assert not list(tmp_path.glob("*.tmp"))
