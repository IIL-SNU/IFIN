# Pretrained Checkpoints

## DiffuserCam

File: `IFIN_DiffuserCam_k16_20251101_best.pth` (3,381,734,300 bytes).

SHA-256:

```text
b818f42028ffa230cf66d1c1d6ff4448cb628e2a1499ba7e47861ce5f708592c
```

This is a separately trained November 2025 DiffuserCam checkpoint, not the
historical FIX_SV file referenced by the research notebook. It is distributed
separately from this repository; do not add weight files to Git.

Use `configs/diffusercam_k16.yaml`: k=16, depth 3, width 32, and 270 x 480 RGB
measurements. The 16 learned PSFs have support 262 x 278 and are loaded without
resizing. Legacy state keys are mapped to IFIN names and all weights are
loaded strictly. A separate PSF calibration file is unnecessary when loading
this checkpoint.

```bash
sha256sum /path/to/IFIN_DiffuserCam_k16_20251101_best.pth
python eval.py --config configs/diffusercam_k16.yaml --data-root /path/to/DiffuserCam --checkpoint /path/to/IFIN_DiffuserCam_k16_20251101_best.pth --output outputs/diffusercam_metrics.json
python infer.py --config configs/diffusercam_k16.yaml --checkpoint /path/to/IFIN_DiffuserCam_k16_20251101_best.pth --input /path/to/measurement.npy --output-dir outputs/diffusercam
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
