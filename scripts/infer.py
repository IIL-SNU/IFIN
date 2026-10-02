from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import add_config_arguments, cli_overrides, load_config, output_result  # noqa: E402
from runner import infer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IFIN inference entry point")
    add_config_arguments(parser)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--input")
    parser.add_argument("--output-dir")
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, cli_overrides(args, ("eval",)), args.model_config)
    result = infer(
        config,
        checkpoint_path=args.checkpoint,
        input_path=args.input,
        output_dir=args.output_dir,
    )
    output_result(result, args.output)


if __name__ == "__main__":
    main()
