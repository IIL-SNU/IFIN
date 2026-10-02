from __future__ import annotations

import importlib.util
import os
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def external_source_dir(options) -> Path:
    configured = options.get("external_source_dir") or os.environ.get("IFIN_EXTERNAL_BASELINES")
    return Path(configured).expanduser().resolve() if configured else PROJECT_ROOT / "external_baselines"


@lru_cache(maxsize=None)
def _load_path(path: str) -> ModuleType:
    source = Path(path)
    module_name = f"_ifin_external_{source.stem}_{abs(hash(source))}"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load external baseline module: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_external(module_name: str, options) -> ModuleType:
    source = external_source_dir(options) / f"{module_name}.py"
    if not source.is_file():
        raise RuntimeError(
            f"baseline requires local external source {source}; "
            "run scripts/prepare_external_baselines.py --source-dir PATH or set "
            "IFIN_EXTERNAL_BASELINES/config.model.options.external_source_dir"
        )
    return _load_path(str(source))
