from __future__ import annotations

import os
import re
import json
from pathlib import Path
from typing import Any, Dict

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        value = _ENV.sub(
            lambda match: os.environ.get(match.group(1), match.group(2) or ""), value
        )
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    return value


def _load(path: Path, seen: set[Path]) -> Dict[str, Any]:
    path = path.expanduser().resolve()
    if path in seen:
        raise ValueError(f"Circular config inheritance: {path}")
    seen.add(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    parent = config.pop("inherits", config.pop("base", None))
    if parent:
        parent_path = Path(_expand(parent))
        if not parent_path.is_absolute():
            parent_path = path.parent / parent_path
        config = _deep_update(_load(parent_path, seen), config)
    seen.remove(path)
    return config


def load_config(
    config_path: str | None = None,
    overrides: Dict[str, Any] | None = None,
    model_config_path: str | None = None,
) -> Dict[str, Any]:
    selected_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    config = _load(selected_path, set())
    requested_model = str(
        (overrides or {})
        .get("model", {})
        .get("name", config.get("model", {}).get("name", "ifin"))
    )
    if requested_model.lower() != "ifin" and not model_config_path:
        dataset = str(config["data"]["dataset"]).lower()
        filename = (
            "multiwiener.yaml" if dataset == "multiwienernet" else f"{dataset}.yaml"
        )
        candidate = selected_path.resolve().parent / "baselines" / filename
        if candidate.exists():
            model_config_path = str(candidate)
    if model_config_path:
        catalog = _load(Path(model_config_path), set())
        profiles = catalog.pop("baselines", {})
        catalog.pop("dataset", None)
        _deep_update(config, catalog)
        if requested_model.lower() != "ifin":
            try:
                profile = profiles[requested_model.lower()]
            except KeyError as error:
                raise ValueError(
                    f"No {requested_model!r} profile in {model_config_path}"
                ) from error
            config["model"] = _deep_update(
                config.get("model", {}),
                {"name": profile["name"], "options": profile.get("options", {})},
            )
            _deep_update(config["data"], profile.get("data", {}))
            config["data"]["psf_crop"] = profile.get("data", {}).get("psf_crop")
            for key in ("in_channels", "out_channels"):
                if key in profile.get("options", {}):
                    config["model"][key] = profile["options"][key]
            if config["data"].get("psf_normalization") is None:
                config["data"]["psf_normalization"] = "none"
    if overrides:
        _deep_update(config, overrides)
    return _expand(config)


def cli_overrides(args: Any, splits: tuple[str, ...]) -> Dict[str, Any]:
    override: Dict[str, Any] = {}
    if args.model is not None:
        override.setdefault("model", {})["name"] = args.model
    if args.data_root is not None:
        override.setdefault("data", {})["root"] = args.data_root
    if args.psf_path is not None:
        override.setdefault("data", {})["psf_path"] = args.psf_path
    if args.device is not None:
        override["device"] = args.device
    for split in splits:
        values = {
            key: getattr(args, key)
            for key in ("num_samples", "batch_size", "num_workers")
            if getattr(args, key) is not None
        }
        if values:
            override.setdefault(split, {}).update(values)
    return override


def add_config_arguments(parser: Any) -> None:
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--model")
    parser.add_argument("--model-config")
    parser.add_argument("--data-root")
    parser.add_argument("--psf-path")
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--device")


def output_result(result: Dict[str, Any], output: str | None = None) -> None:
    text = json.dumps(result, indent=2)
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)
