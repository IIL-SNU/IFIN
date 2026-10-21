from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import re
import tempfile


BLOCK_SIZE = 8 * 1024 * 1024


def _filename(value):
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or PureWindowsPath(value).drive
    ):
        raise ValueError(f"Unsafe filename in manifest: {value!r}")
    return value


def _file_spec(spec):
    _filename(spec["filename"])
    if type(spec["size_bytes"]) is not int or spec["size_bytes"] <= 0:
        raise ValueError("File size must be a positive integer")
    if not isinstance(spec["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", spec["sha256"]):
        raise ValueError("Invalid SHA-256 in manifest")


def load_manifest(path):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint manifest version")
    entries = manifest.get("checkpoints")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Manifest must contain checkpoints")
    checkpoints, filenames = {}, set()
    for entry in entries:
        _file_spec(entry)
        identifier = entry["id"]
        if not isinstance(identifier, str) or not identifier or identifier in checkpoints:
            raise ValueError("Checkpoint IDs must be nonempty and unique")
        if entry["filename"] in filenames:
            raise ValueError("Checkpoint output filenames must be unique")
        parts = entry["parts"]
        if not isinstance(parts, list) or not parts:
            raise ValueError("Checkpoint must contain ordered parts")
        for part in parts:
            _file_spec(part)
        if len({part["filename"] for part in parts}) != len(parts):
            raise ValueError("Duplicate part filename")
        if sum(part["size_bytes"] for part in parts) != entry["size_bytes"]:
            raise ValueError("Part sizes do not match checkpoint size")
        checkpoints[identifier] = entry
        filenames.add(entry["filename"])
    return checkpoints


def _matches(path, spec):
    if not path.is_file() or path.stat().st_size != spec["size_bytes"]:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest() == spec["sha256"]


def restore_checkpoint(entry, parts_dir, output_dir):
    parts_dir, output_dir = Path(parts_dir), Path(output_dir)
    output = output_dir / entry["filename"]
    if output.exists() or output.is_symlink():
        if _matches(output, entry):
            print(f"Verified existing checkpoint: {output}", flush=True)
            return output
        raise FileExistsError(f"Refusing to overwrite an unverified file: {output}")
    for part in entry["parts"]:
        source = parts_dir / part["filename"]
        if not source.is_file():
            raise FileNotFoundError(f"Missing checkpoint part: {source}")
        if source.stat().st_size != part["size_bytes"]:
            raise ValueError(f"Incorrect size for checkpoint part: {source}")
    output_dir.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{entry['filename']}.", suffix=".partial", dir=output_dir
    )
    temporary = Path(temporary_name)
    digest, total = hashlib.sha256(), 0
    try:
        with os.fdopen(descriptor, "wb") as target:
            for part in entry["parts"]:
                part_digest, size = hashlib.sha256(), 0
                source = parts_dir / part["filename"]
                with source.open("rb") as stream:
                    for block in iter(lambda: stream.read(BLOCK_SIZE), b""):
                        target.write(block)
                        digest.update(block)
                        part_digest.update(block)
                        size += len(block)
                if size != part["size_bytes"] or part_digest.hexdigest() != part["sha256"]:
                    raise ValueError(f"Checksum mismatch for checkpoint part: {source}")
                total += size
                print(f"Verified part: {part['filename']}", flush=True)
            target.flush()
            os.fsync(target.fileno())
        if total != entry["size_bytes"] or digest.hexdigest() != entry["sha256"]:
            raise ValueError(f"Restored checkpoint checksum mismatch: {entry['filename']}")
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Restored and SHA-256 verified: {output}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description="Restore and verify IFIN checkpoint downloads")
    parser.add_argument("--manifest", type=Path, default=Path("IFIN_checkpoints.json"))
    parser.add_argument("--parts-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--checkpoint", action="append", help="Checkpoint ID; default: all")
    args = parser.parse_args()
    try:
        checkpoints = load_manifest(args.manifest)
        selected = args.checkpoint or list(checkpoints)
        unknown = set(selected) - checkpoints.keys()
        if unknown:
            raise ValueError(f"Unknown checkpoint IDs: {', '.join(sorted(unknown))}")
        parts_dir = args.parts_dir or args.manifest.parent
        output_dir = args.output_dir or parts_dir
        for identifier in dict.fromkeys(selected):
            restore_checkpoint(checkpoints[identifier], parts_dir, output_dir)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
