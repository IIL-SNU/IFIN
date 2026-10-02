from __future__ import annotations

from pathlib import Path
from typing import Any
import os
import tempfile

import torch


def extract_state_dict(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    return {key.removeprefix("module."): value for key, value in state.items()}


def _remap_legacy_ifin_keys(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    renamed = {}
    for key, value in state_dict.items():
        parts = key.split(".")
        parts = [
            {"WieNerH": "initial_iso", "caw_block": "ifib", "W": "iso", "C": "fso"}.get(
                part, part
            )
            for part in parts
        ]
        new_key = ".".join(parts)
        if new_key in renamed:
            raise ValueError(f"Checkpoint key collision: {new_key}")
        renamed[new_key] = value
    return renamed


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def load_checkpoint(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> dict[str, Any]:
    return torch.load(
        Path(path), map_location=map_location, weights_only=True, mmap=True
    )


def load_model_state_compat(
    model: torch.nn.Module, checkpoint: dict[str, Any], strict: bool = True
) -> None:
    state = extract_state_dict(checkpoint)
    if hasattr(model, "initial_iso"):
        state = _remap_legacy_ifin_keys(state)
    model.load_state_dict(state, strict=strict)
