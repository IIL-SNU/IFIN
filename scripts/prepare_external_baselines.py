from __future__ import annotations

import argparse
import ast
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FILES = {"DeepLIR.py": "deeplir.py", "MoDL_SV.py": "modl.py"}
DROP_MODULES = ("matplotlib", "torchmetrics", "tqdm")


def removable(node: ast.AST) -> bool:
    if isinstance(node, ast.Import):
        return any(alias.name == "sys" or alias.name.startswith(DROP_MODULES) for alias in node.names)
    if isinstance(node, ast.ImportFrom):
        return bool(node.module and node.module.startswith(DROP_MODULES))
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        return False
    function = node.value.func
    return (
        isinstance(function, ast.Attribute)
        and function.attr == "append"
        and isinstance(function.value, ast.Attribute)
        and function.value.attr == "path"
        and isinstance(function.value.value, ast.Name)
        and function.value.value.id == "sys"
    )


def sanitize(source: str) -> str:
    tree = ast.parse(source)
    removed = set()
    for node in tree.body:
        if removable(node):
            removed.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    lines = [line for number, line in enumerate(source.splitlines(), 1) if number not in removed]
    marker = "# Original source comments are preserved from the user-supplied research file."
    return marker + "\n" + "\n".join(lines).lstrip("\n") + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("IFIN_EXTERNAL_BASELINES", PROJECT_ROOT / "external_baselines")),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for source_name, output_name in FILES.items():
        source_path = args.source_dir / source_name
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        output_path = args.output_dir / output_name
        output_path.write_text(sanitize(source_path.read_text(encoding="utf-8")), encoding="utf-8")
        print(output_path)


if __name__ == "__main__":
    main()
