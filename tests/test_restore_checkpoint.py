import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "restore_checkpoint.py"
spec = importlib.util.spec_from_file_location("restore_checkpoint", SCRIPT)
restore = importlib.util.module_from_spec(spec)
spec.loader.exec_module(restore)


def file_spec(filename, data):
    return {"filename": filename, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def checkpoint(tmp_path, identifier="k16", chunks=(b"first", b"second")):
    filename = f"IFIN_{identifier}.pth"
    parts = []
    for index, chunk in enumerate(chunks, 1):
        part = file_spec(f"{filename}.part{index:03d}", chunk)
        (tmp_path / part["filename"]).write_bytes(chunk)
        parts.append(part)
    return {"id": identifier, **file_spec(filename, b"".join(chunks)), "parts": parts}


def manifest_file(tmp_path, entries):
    path = tmp_path / "IFIN_checkpoints.json"
    path.write_text(json.dumps({"format_version": 1, "checkpoints": entries}))
    return path


def test_restores_exact_bytes_and_preserves_parts(tmp_path):
    entry = checkpoint(tmp_path)
    manifest = manifest_file(tmp_path, [entry])
    parsed = restore.load_manifest(manifest)["k16"]
    result = restore.restore_checkpoint(parsed, tmp_path, tmp_path)
    assert result.read_bytes() == b"firstsecond"
    assert all((tmp_path / part["filename"]).is_file() for part in entry["parts"])
    assert not list(tmp_path.glob("*.partial"))
    before = result.stat().st_mtime_ns
    assert restore.restore_checkpoint(parsed, tmp_path, tmp_path) == result
    assert result.stat().st_mtime_ns == before


@pytest.mark.parametrize("failure", ["missing", "wrong_size", "corrupt", "wrong_order"])
def test_bad_parts_do_not_leave_a_checkpoint_or_temporary_file(tmp_path, failure):
    entry = checkpoint(tmp_path)
    source = tmp_path / entry["parts"][0]["filename"]
    if failure == "missing":
        source.unlink()
    elif failure == "wrong_size":
        source.write_bytes(b"shorter than expected")
    elif failure == "corrupt":
        source.write_bytes(b"wrong")
    else:
        entry["parts"].reverse()
    with pytest.raises((ValueError, FileNotFoundError)):
        restore.restore_checkpoint(entry, tmp_path, tmp_path)
    assert not (tmp_path / entry["filename"]).exists()
    assert not list(tmp_path.glob("*.partial"))


def test_existing_unverified_file_is_never_overwritten(tmp_path):
    entry = checkpoint(tmp_path)
    output = tmp_path / entry["filename"]
    output.write_bytes(b"user file")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        restore.restore_checkpoint(entry, tmp_path, tmp_path)
    assert output.read_bytes() == b"user file"


def test_checkpoint_directory_restores_inside_matching_folder(tmp_path):
    folder = tmp_path / "WiderCam_k9"
    folder.mkdir()
    entry = checkpoint(folder, identifier="widercam_k9")
    entry["directory"] = folder.name
    parsed = restore.load_manifest(manifest_file(tmp_path, [entry]))[entry["id"]]
    result = restore.restore_checkpoint(parsed, tmp_path, tmp_path)
    assert result == folder / entry["filename"]
    assert result.read_bytes() == b"firstsecond"
    assert restore.restore_checkpoint(parsed, tmp_path, tmp_path) == result
    assert all((folder / part["filename"]).is_file() for part in entry["parts"])


def test_directory_manifest_accepts_legacy_flat_downloads(tmp_path):
    entry = checkpoint(tmp_path)
    entry["directory"] = "DiffuserCam_k16"
    result = restore.restore_checkpoint(entry, tmp_path, tmp_path)
    assert result == tmp_path / entry["directory"] / entry["filename"]
    assert result.read_bytes() == b"firstsecond"
    assert all((tmp_path / part["filename"]).is_file() for part in entry["parts"])


def test_cli_selected_directory_works_without_other_checkpoint_parts(tmp_path):
    folder = tmp_path / "MultiWienerNet_k9"
    folder.mkdir()
    selected = checkpoint(folder, identifier="multiwienernet_k9")
    selected["directory"] = folder.name
    absent = {"id": "not_downloaded", **file_spec("absent.pth", b"absent"),
              "parts": [file_spec("absent.part001", b"absent")]}
    manifest = manifest_file(tmp_path, [absent, selected])
    output = tmp_path / "restored"
    command = [sys.executable, str(SCRIPT), "--manifest", str(manifest),
               "--checkpoint", selected["id"], "--output-dir", str(output)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (output / folder.name / selected["filename"]).read_bytes() == b"firstsecond"
    assert not (output / "absent.pth").exists()


@pytest.mark.parametrize("directory", ["../outside", "/outside", "..\\outside", "C:outside", "", ".", ".."])
def test_manifest_rejects_unsafe_checkpoint_directories(tmp_path, directory):
    entry = checkpoint(tmp_path)
    entry["directory"] = directory
    with pytest.raises(ValueError, match="Unsafe filename"):
        restore.load_manifest(manifest_file(tmp_path, [entry]))


@pytest.mark.parametrize("filename", ["../outside.pth", "/outside.pth", "..\\outside.pth", "C:outside.pth"])
def test_manifest_rejects_unsafe_filenames(tmp_path, filename):
    entry = checkpoint(tmp_path)
    entry["parts"][0]["filename"] = filename
    with pytest.raises(ValueError, match="Unsafe filename"):
        restore.load_manifest(manifest_file(tmp_path, [entry]))


@pytest.mark.parametrize("failure", ["duplicate_id", "duplicate_part", "size", "sha"])
def test_manifest_rejects_inconsistent_specs(tmp_path, failure):
    entry = checkpoint(tmp_path)
    entries = [entry]
    if failure == "duplicate_id":
        entries.append(entry.copy())
    elif failure == "duplicate_part":
        entry["parts"].append(entry["parts"][0])
    elif failure == "size":
        entry["size_bytes"] += 1
    else:
        entry["sha256"] = "not-a-sha256"
    with pytest.raises(ValueError):
        restore.load_manifest(manifest_file(tmp_path, entries))


def test_cli_restores_split_checkpoint_and_verifies_unsplit_checkpoint(tmp_path):
    split = checkpoint(tmp_path)
    data = b"complete k1 checkpoint"
    whole = {"id": "k1", **file_spec("IFIN_k1.pth", data), "parts": [file_spec("IFIN_k1.pth", data)]}
    (tmp_path / whole["filename"]).write_bytes(data)
    manifest = manifest_file(tmp_path, [split, whole])
    command = [sys.executable, str(SCRIPT), "--manifest", str(manifest)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / split["filename"]).read_bytes() == b"firstsecond"
    assert (tmp_path / whole["filename"]).read_bytes() == data
    assert "Verified existing checkpoint" in result.stdout
    result = subprocess.run(command + ["--checkpoint", "missing"], capture_output=True, text=True)
    assert result.returncode == 1
    assert "Unknown checkpoint IDs" in result.stderr
