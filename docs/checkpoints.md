# Pretrained Checkpoints

## Download

Download from the [checkpoint folder on Google Drive](https://drive.google.com/drive/folders/1UZiyWqdqK75IQwusnfYNoKRT-fOa0neX).
Weight files are not committed to Git.

| k | Restored filename | Size | Configuration |
| --- | --- | ---: | --- |
| 16 | `IFIN_DiffuserCam_k16_20251101_best.pth` | 3,381,734,300 bytes | `configs/diffusercam_k16.yaml` |
| 1 | `IFIN_DiffuserCam_k1_20260507_best.pth` | 218,858,833 bytes | `configs/diffusercam.yaml` |

For k=16, download all 34 files ending in `.pth.part001` through
`.pth.part034`. For k=1, download all three files ending in `.pth.part001`
through `.pth.part003`. Parts are at most 96 MiB each. Keep their original names.
Also download `IFIN_checkpoints.json`. The folder includes a standalone
`restore_checkpoint.py` and `README_RESTORE.md` for use without cloning GitHub.

These are separately trained November 2025 (k=16) and May 2026 (k=1)
checkpoints, not a controlled k-only ablation pair or the historical FIX_SV
file referenced by the research notebook.

## Restore And Verify

Python 3.10 or newer is sufficient; restoring does not require PyTorch.
With the repository cloned:

```bash
python scripts/restore_checkpoint.py --manifest /path/to/downloads/IFIN_checkpoints.json
```

Or, in the downloaded folder, using the included standalone script:

```bash
python restore_checkpoint.py --manifest IFIN_checkpoints.json
```

This restores both checkpoints into their original `.pth` files.
To restore only k=16, append `--checkpoint diffusercam_k16`.
To restore only k=1, append `--checkpoint diffusercam_k1`.
Use `--output-dir /path/to/checkpoints` to write to a different directory.
On Windows, `py` can be used instead of `python`.

Every part and the restored file are SHA-256 checked. Parts are preserved;
an already valid output is verified and reused, while an unverified existing
output is never overwritten. A failed restoration does not leave a partial
`.pth` file. Allow about 8 GB of free disk space for both downloads and restored files.

Expected restored SHA-256 checksums:

```text
IFIN_DiffuserCam_k16_20251101_best.pth
b818f42028ffa230cf66d1c1d6ff4448cb628e2a1499ba7e47861ce5f708592c

IFIN_DiffuserCam_k1_20260507_best.pth
3491e2257646d4b76e8a8df81026d3ce1c094d9f892293c202fc69e44efe2346
```

## Evaluate

Both checkpoints use depth 3, width 32, and 270 x 480 RGB measurements.
Their learned PSFs have support 262 x 278 and are loaded without resizing.
Legacy state keys are mapped to IFIN names and all weights are loaded strictly.
A separate PSF calibration file is unnecessary when loading these checkpoints.

```bash
python eval.py --config configs/diffusercam_k16.yaml --data-root /path/to/DiffuserCam --checkpoint /path/to/IFIN_DiffuserCam_k16_20251101_best.pth --output outputs/diffusercam_metrics.json
python eval.py --config configs/diffusercam.yaml --data-root /path/to/DiffuserCam --checkpoint /path/to/IFIN_DiffuserCam_k1_20260507_best.pth --output outputs/diffusercam_k1_metrics.json
python infer.py --config configs/diffusercam_k16.yaml --checkpoint /path/to/IFIN_DiffuserCam_k16_20251101_best.pth --input /path/to/measurement.npy --output-dir outputs/diffusercam
python infer.py --config configs/diffusercam.yaml --checkpoint /path/to/IFIN_DiffuserCam_k1_20260507_best.pth --input /path/to/measurement.npy --output-dir outputs/diffusercam_k1
```

The data root for evaluation must contain `dataset_test.csv`,
`diffuser_images/`, and `ground_truth_lensed/`. NumPy pairs use the source
BGR-to-RGB and ToTensor preprocessing. Predictions and targets are clipped
to [0,1] without per-image maximum normalization. Unsuffixed metrics use the
full frame; `_crop` metrics are separate visualization-crop diagnostics.

Evaluation follows the supplied CSV without redefining the split. The
validated available split contains 998 test pairs; it is not presented as a
new 1,000-image evaluation. An explicit `--input` needs only the measurement,
not the paired dataset or calibration file.

The general dataset presets retain k=1. Select another field size with `--k`
only when it matches the checkpoint; a mismatched k is rejected.
