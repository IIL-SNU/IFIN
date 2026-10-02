from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import add_config_arguments, cli_overrides, load_config, output_result  # noqa: E402
from runner import train  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IFIN training entry point")
    add_config_arguments(parser)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(
        args.config, cli_overrides(args, ("train", "eval")), args.model_config
    )
    output_result(train(config, checkpoint_path=args.checkpoint), args.output)


if __name__ == "__main__":
    main()
