# Pretrained Checkpoints

## Download

Download from the [checkpoint folder on Google Drive](https://drive.google.com/drive/folders/1UZiyWqdqK75IQwusnfYNoKRT-fOa0neX).
Weight files are not committed to Git.

| Checkpoint ID | Restored filename | Parts |
| --- | --- | ---: |
| `diffusercam_k16` | `IFIN_DiffuserCam_k16_20251101_best.pth` | 34 |
| `diffusercam_k1` | `IFIN_DiffuserCam_k1_20260507_best.pth` | 3 |
| `widercam_k9` | `IFIN_WiderCam_k9_20250923_best.pth` | 12 |
| `widercam_k4` | `IFIN_WiderCam_k4_20251008_best.pth` | 6 |
| `widercam_k1` | `IFIN_WiderCam_k1_20251004_best.pth` | 2 |
| `multiwienernet_k9` | `IFIN_MultiWienerNet_k9_20250910_best.pth` | 15 |
| `multiwienernet_k4` | `IFIN_MultiWienerNet_k4_20251030_best.pth` | 7 |
| `multiwienernet_k1` | `IFIN_MultiWienerNet_k1_20251030_best.pth` | 2 |

Choose the checkpoint for your dataset and download its folder, containing
every file ending in `.pth.part001` through the part count above. Parts are at
most 96 MiB each. Keep folder and file names unchanged.
Also download `IFIN_checkpoints.json`. The Drive root includes a standalone
`restore_checkpoint.py` and `README_RESTORE.md` for use without cloning GitHub.

The manifest records exact file sizes and SHA-256 checksums. The learned PSFs
are included in each checkpoint; a separate calibration file is unnecessary.

Keep the common files at the download root and checkpoint parts in their folders:

```text
downloads/
  IFIN_checkpoints.json
  restore_checkpoint.py
  README_RESTORE.md
  DiffuserCam_k16/
  DiffuserCam_k1/
  WiderCam_k9/
  WiderCam_k4/
  WiderCam_k1/
  MultiWienerNet_k9/
  MultiWienerNet_k4/
  MultiWienerNet_k1/
```

Only the selected checkpoint folders are required. The original `WiderCam.zip`
dataset archive remains separate at the Drive root.

## Restore And Verify

Python 3.10 or newer is sufficient; restoring does not require PyTorch.
With the repository cloned:

```bash
python scripts/restore_checkpoint.py --manifest /path/to/downloads/IFIN_checkpoints.json --checkpoint widercam_k9
```

Or, in the downloaded folder, using the included standalone script:

```bash
python restore_checkpoint.py --manifest IFIN_checkpoints.json --checkpoint widercam_k9
```

Replace `widercam_k9` with the ID of the checkpoint you downloaded.
For example, use `--checkpoint multiwienernet_k9` for MultiWienerNet k=9.
Repeat `--checkpoint` to select several downloaded checkpoints. Omit it only
when all eight sets of parts are present, because the default restores every
entry in the manifest.
Use `--output-dir /path/to/checkpoints` to write to a different directory.
Checkpoint subfolders are created inside that output root. For example,
restoring `widercam_k9` creates
`WiderCam_k9/IFIN_WiderCam_k9_20250923_best.pth`. Previous flat part downloads
are also accepted; old manifests without folder metadata retain flat output.
On Windows, `py` can be used instead of `python`.

Every part and the restored file are SHA-256 checked. Parts are preserved;
an already valid output is verified and reused, while an unverified existing
output is never overwritten. A failed restoration does not leave a partial
`.pth` file. Allow at least twice the combined size of the selected checkpoints
for downloaded parts and restored files (about 16 GB for the whole collection).

Expected restored SHA-256 checksums:

```text
IFIN_DiffuserCam_k16_20251101_best.pth
b818f42028ffa230cf66d1c1d6ff4448cb628e2a1499ba7e47861ce5f708592c

IFIN_DiffuserCam_k1_20260507_best.pth
3491e2257646d4b76e8a8df81026d3ce1c094d9f892293c202fc69e44efe2346

IFIN_WiderCam_k9_20250923_best.pth
6efd8ded5cf7e811cec363bafeb4d3798b03cb90c583c474eda743515b9e339a

IFIN_WiderCam_k4_20251008_best.pth
c036f367ddb0b94e1768cef0c6b1885c68705825dec16608276b115a72af624b

IFIN_WiderCam_k1_20251004_best.pth
fac964927347fdd7d1996969c305c06dae2545665b3068388cd65774ebd06efc

IFIN_MultiWienerNet_k9_20250910_best.pth
2caf50fd2a406af3734efb350a687f641982844496e75432f129884401e527d9

IFIN_MultiWienerNet_k4_20251030_best.pth
7f55787dfddd8435611a1618e3b03d4af53a2002fccc4ac0b4dd795a6b78f04c

IFIN_MultiWienerNet_k1_20251030_best.pth
07d851846eeb8eaea8220b058745db7160e128d3f1a80fb0574e529f9c60c381
```

## Evaluate

Use the matching configuration, not another dataset's preset:

| Checkpoint ID | Configuration | Measurement | Learned PSF support |
| --- | --- | --- | --- |
| `diffusercam_k16` | `configs/diffusercam_k16.yaml` | RGB 270 x 480 | 262 x 278 |
| `diffusercam_k1` | `configs/diffusercam.yaml` | RGB 270 x 480 | 262 x 278 |
| `widercam_k9` | `configs/widercam_k9.yaml` | RGB 270 x 480 | 130 x 133 |
| `widercam_k4` | `configs/widercam_k4.yaml` | RGB 270 x 480 | 135 x 135 |
| `widercam_k1` | `configs/widercam_k1.yaml` | RGB 270 x 480 | 135 x 135 |
| `multiwienernet_k9` | `configs/multiwienernet_k9.yaml` | Grayscale 224 x 320 | 224 x 320 |
| `multiwienernet_k4` | `configs/multiwienernet_k4.yaml` | Grayscale 224 x 320 | 224 x 320 |
| `multiwienernet_k1` | `configs/multiwienernet.yaml` | Grayscale 224 x 320 | 224 x 320 |

All use depth 3 and width 32. Learned PSFs are loaded without resizing.
Legacy state keys are mapped to IFIN names and all weights are loaded strictly.
A separate PSF calibration file is unnecessary when loading these checkpoints.
WiderCam k=1/4 use the historical `refinement_block: rb_convg` option in the
same IFIN implementation. Use their matching presets; overriding only `--k`
on the k=9 preset does not select the correct architecture.

```bash
python eval.py --config configs/diffusercam_k16.yaml --data-root /path/to/DiffuserCam --checkpoint /path/to/downloads/DiffuserCam_k16/IFIN_DiffuserCam_k16_20251101_best.pth --output outputs/diffusercam_metrics.json
python eval.py --config configs/diffusercam.yaml --data-root /path/to/DiffuserCam --checkpoint /path/to/downloads/DiffuserCam_k1/IFIN_DiffuserCam_k1_20260507_best.pth --output outputs/diffusercam_k1_metrics.json
python eval.py --config configs/widercam_k9.yaml --data-root /path/to/WiderCam --checkpoint /path/to/downloads/WiderCam_k9/IFIN_WiderCam_k9_20250923_best.pth --output outputs/widercam_metrics.json
python eval.py --config configs/widercam_k4.yaml --data-root /path/to/WiderCam --checkpoint /path/to/downloads/WiderCam_k4/IFIN_WiderCam_k4_20251008_best.pth --output outputs/widercam_k4_metrics.json
python eval.py --config configs/widercam_k1.yaml --data-root /path/to/WiderCam --checkpoint /path/to/downloads/WiderCam_k1/IFIN_WiderCam_k1_20251004_best.pth --output outputs/widercam_k1_metrics.json
python eval.py --config configs/multiwienernet_k9.yaml --data-root /path/to/MultiWienerNet --checkpoint /path/to/downloads/MultiWienerNet_k9/IFIN_MultiWienerNet_k9_20250910_best.pth --output outputs/multiwienernet_k9_metrics.json
python eval.py --config configs/multiwienernet_k4.yaml --data-root /path/to/MultiWienerNet --checkpoint /path/to/downloads/MultiWienerNet_k4/IFIN_MultiWienerNet_k4_20251030_best.pth --output outputs/multiwienernet_k4_metrics.json
python eval.py --config configs/multiwienernet.yaml --data-root /path/to/MultiWienerNet --checkpoint /path/to/downloads/MultiWienerNet_k1/IFIN_MultiWienerNet_k1_20251030_best.pth --output outputs/multiwienernet_k1_metrics.json
python infer.py --config configs/diffusercam_k16.yaml --checkpoint /path/to/downloads/DiffuserCam_k16/IFIN_DiffuserCam_k16_20251101_best.pth --input /path/to/measurement.npy --output-dir outputs/diffusercam
python infer.py --config configs/diffusercam.yaml --checkpoint /path/to/downloads/DiffuserCam_k1/IFIN_DiffuserCam_k1_20260507_best.pth --input /path/to/measurement.npy --output-dir outputs/diffusercam_k1
```

### Dataset Conventions

The DiffuserCam data root must contain `dataset_test.csv`,
`diffuser_images/`, and `ground_truth_lensed/`. NumPy pairs use the source
BGR-to-RGB and ToTensor preprocessing. Predictions and targets are clipped
to [0,1] without per-image maximum normalization. Unsuffixed metrics use the
full frame; `_crop` metrics are separate visualization-crop diagnostics.

WiderCam requires `dataset_test.csv`, `images/*_rgb8.png`, and `labels/*.jpg`.
The loader retains the research notebook's RGB conversion and affine target
alignment. Predictions are clipped and then max-normalized. `_dewarped`
metrics are secondary diagnostics, not the full aligned-frame metrics.
The dataset is available from the [WiderCam dataset page](https://iilab.io/WiderCam).

MultiWienerNet requires `2D/Ground_truth_downsampled/` and
`2D/Simulated_Miniscope_2D_Training_data/` containing paired PNG files.
The original crop, downsampling, and per-image input/target normalization are
preserved; predictions are only clipped. The source uses an unsorted filesystem
glob followed by a seed-8 shuffle and an 80/20 split. The inspected dataset has
22,126 pairs and 4,426 test samples. Copying it to another filesystem can change
the split order; see [Research Code Provenance](code_provenance.md).

CSV-based evaluation follows the supplied CSV without redefining the split.
The available DiffuserCam and WiderCam CSV loaders each yield 998 test pairs;
these runs are not presented as new 1,000-image evaluations.
An explicit `--input` needs only the measurement,
not the paired dataset or calibration file.

The general dataset presets retain k=1. Select another field size with `--k`
only when it matches the checkpoint; a mismatched k is rejected.

## Provenance

### Release Validation

The following complete GPU evaluations used `python eval.py` and the matching
presets above, with strict checkpoint loading and no sample limit:

| Checkpoint ID | Test pairs | PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: | ---: |
| `widercam_k9` | 998 | 25.4437 | 0.8235 | 0.2013 |
| `widercam_k4` | 998 | 25.2553 | 0.8214 | 0.2038 |
| `widercam_k1` | 998 | 25.0899 | 0.8138 | 0.2109 |
| `multiwienernet_k9` | 4,426 | 30.9906 | 0.8662 | 0.1747 |
| `multiwienernet_k4` | 4,426 | 30.2083 | 0.8551 | 0.1847 |
| `multiwienernet_k1` | 4,426 | 29.7333 | 0.8477 | 0.1969 |

These are checks of the released files under the dataset conventions above,
not replacements for the paper's reported results. WiderCam values use the
aligned full frame, not the secondary `_dewarped` metrics.
The October WiderCam k=1/k=4 releases do not exactly reproduce the
supplementary table's 24.963/25.148 dB values; the measurements above are
reported for the actual distributed checkpoint files.

### Checkpoint Sources

The WiderCam release is the notebook's September 2025 `FIX` branch checkpoint
`IFIN_9_20250923-105638_best.pth`, renamed for clarity without changing its bytes.
The October k=1/k=4 files are `IFIN_1_20251004-193707_best.pth` (epoch 97)
and `IFIN_4_20251008-084033_best.pth` (epoch 98). They contain 528 state entries
and use the historical ConvG-based RB rather than the k=9 refinement block.
The original `IFIN2.py` and October training logs were not recovered; this
configuration is reconstructed from checkpoint tensors and surviving model
components, with strict loading and numerical fidelity tests. WiderCam k=4
attribution was initially inferred from this matching topology and PSF support.
The paper reports WiderCam k=1/4/9 in its supplementary PSF-field ablation;
its k=16 result is on DiffuserCam, not WiderCam.

MultiWienerNet k=9 was recovered as `IFIN_20250910-031003_best.pth` (stored
epoch 93, nine learned PSFs). Its original training summary was unavailable;
release compatibility is established with strict state loading and the public
evaluation code. The October 2025 k=1/k=4 weights are the notebook's
`IFIN_k_1_20251030-140124_best.pth` and `IFIN_k_4_20251030-131015_best.pth`.
They use the historical CAW6 options encoded in the dataset YAML, not the RGB
FIXNet options. Interrupted and fixed-PSF May 2026 ablations are not distributed.

DiffuserCam k=16 and k=1 were separately trained in November 2025 and May 2026,
respectively; neither is identified as the missing historical FIX_SV file.
Different training runs are not a controlled k-only ablation pair.
