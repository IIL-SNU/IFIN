# Research Code Provenance

The release packages the implementations used by the research evaluation
notebooks. Dataset-specific IFIN settings live in YAML; there is only one
`IFINNet` implementation. No checkpoints or dataset samples are committed.

## Evaluation Sources

| Dataset | Research notebook | Released notebook | Configuration |
| --- | --- | --- | --- |
| WiderCam | `eval/lensless/run_model_IFIN.ipynb` | `notebooks/eval_widercam.ipynb` | `configs/widercam.yaml` |
| DiffuserCam | `eval/lensless/run_model-Copy1.ipynb` | `notebooks/eval_diffusercam.ipynb` | `configs/diffusercam.yaml` |
| MultiWienerNet | `eval/lensless/run_model_M3D.ipynb` | `notebooks/eval_multiwienernet.ipynb` | `configs/multiwienernet.yaml` |

Released notebooks call the same dataset, model, and evaluation functions as
the CLI. Research notebooks contain exploratory cells and stale paths; they
are not treated as sequential, executable benchmark specifications.

## IFIN Implementation

The `FIX` branch in the RGB evaluation notebooks resolves to `models/FIXNet.py`.
The available MultiWienerNet k=1/k=4 checkpoints match the historical
`models/CAW6_buffer.py` topology. These implementations differ in seed blocks,
ISO regularizer activation, IFIB residual connections, a bottleneck IFIB, and
upsampling interpolation. The differences are explicit model options, not
separate dataset model classes. Numerical forward and input-gradient
equivalence is tested against both originals.

| Research name | Released name |
| --- | --- |
| `CAWNet` | `IFINNet` (also exported as `IFIN`) |
| `CAWBlock` | `IFIB` |
| `W` | `ISO` |
| `C` | `FSO` |
| `DoubleConvLN` | `RB` |
| `DownCAW`, `UpCAW` | `DownsampleIFIB`, `UpsampleIFIB` |

FFT padding/cropping, normalization, the full learned 2D ISO regularizer,
learnable ROI weights, PSF encoding, and refinement computations are retained.
The FSO in these sources uses the mean PSF; it is not replaced by a newly
invented region-wise forward operator. Legacy state keys are renamed during
strict loading. Learned tensors are never resized and unmatched keys are not
silently discarded. Earlier simplified prerelease checkpoints are not
equivalent to these research checkpoints.

## Verified Checkpoint Metadata

| Filename | PSF shape | State entries | Architecture |
| --- | --- | ---: | --- |
| `IFIN_9_20250923-105638_best.pth` | `1 x 9 x 130 x 133` | 560 | FIXNet |
| `IFIN_k_4_20251030-131015_best.pth` | `1 x 4 x 224 x 320` | 577 | historical CAW6 |
| `IFIN_k_1_20251030-140124_best.pth` | `1 x 1 x 224 x 320` | 577 | historical CAW6 |

The DiffuserCam notebook references `FIX_SV_20250918-061739_best.pth`, which
was not present in the inspected weights directory. Do not replace it with an
unrelated checkpoint and report the result as the published benchmark.
The WiderCam notebook's experimental `IFIN2` branch references a missing
module. The verified `FIX` branch is packaged instead.

Paper support sizes and checkpoint support sizes are not always identical.
Evaluation initializes IFIN
from the learned PSF stored in the checkpoint to preserve its exact support.
The available MultiWienerNet k=1/k=4 experiments are not the paper's k=9 result.

## Dataset Reproduction Limits

The MultiWienerNet source builds its 22,126-image list with an unsorted
filesystem glob and then applies `random.Random(8).shuffle`. The inspected
dataset contains PNG pairs without unrelated suffixes, but the exact split
still depends on the original filesystem enumeration order. Copying the same
files to a different filesystem can therefore change the split. The release
preserves this behavior instead of silently sorting it; the limitation is also
recorded in `configs/multiwienernet.yaml`.

WiderCam target alignment uses the notebook matrix and `grid_sample` inverse
mapping exactly. Its checkpoint-confirmed center PSF crop is
`(center_y=133, center_x=236, height=130, width=133)`; the commented 135 x 135
alternative is not used by the released configuration.

All released IFIN YAML files default to `k=1`. Use `model.k` or `--k` to select
1, 4, 9, or 16 kernels; the paper uses k=9 for WiderCam and MultiWienerNet.
The MultiWienerNet YAML preserves the historical CAW6 architecture, independently
of the selected k. Checkpoint evaluation requires the original matching k.

The DiffuserCam research notebook selects the FIX configuration with `k=16`.
The referenced best checkpoint is unavailable, so real-data inference without a
checkpoint is a pipeline smoke check only and is not a benchmark result.

## Evaluation Convention

Ground truth is clipped to `[0,1]` but is not independently max-normalized
by the evaluator. The MultiWienerNet dataset loader already normalizes each
measurement and target as in its source. Prediction normalization follows
the selected notebook's test-dataset loop, not its separate simulation cells:

| Dataset | IFIN prediction | Baseline prediction | Additional metric frame |
| --- | --- | --- | --- |
| WiderCam | Clip, then divide by the image maximum | Same | Undo target alignment; suffix `_dewarped` |
| DiffuserCam | Clip only | Divide by the image maximum, then clip; Le-ADMM-U clips only | Resize to 270 x 480, flip vertically, crop `[60:,62:-38]`; suffix `_crop` |
| MultiWienerNet | Clip only | Clip, then divide by the image maximum; Le-ADMM-U reverses that order | None; the source's duplicate full-frame metrics are omitted |

Metrics are calculated per image and averaged by sample count. Reported
unsuffixed metrics always use the full aligned frame; suffixed metrics use
the explicitly listed transformed frame. Transformations are shared by the
CLI and released notebooks. Set `eval.metric_view: null` to omit secondary
metrics. These evaluation transformations do not change the training loss.

## Training Sources

- WiderCam: `train/lensless/CAW/SVLensless/` scripts and their saved run files.
- DiffuserCam: `train/lensless/CAW/` scripts, including
  `skeleton_XCD_initial_iso_wallerlab.py`. Its current imports/defaults have
  changed; it is a recipe source, not proof of the missing checkpoint's origin.
- MultiWienerNet: archived `CAW/3D/skeleton_XCD_3D.py` in runs
  `run-20251030_131016-q6vaobe6` and `run-20251030_140125-qp7xzfac`.

The shared runner exposes image/perceptual loss, measurement consistency,
initial ISO supervision, and PSF non-negativity weights in YAML. It saves
optimizer/scheduler state and configuration for new runs. See the baseline
documentation for the comparison-model recipes and source mapping.

## Source SHA-256

```text
run_model_IFIN.ipynb  7df75c2f10c9829673f84d0b32373837f0abcd779487b4bf52db6de437f131fd
run_model-Copy1.ipynb 26f3ebf13aa1f14bb62258f1359b978eb11787790685be10da0bf1caf064b0a1
run_model_M3D.ipynb   f0c8631e6363c521e957df58c833484af5bcfb393cc7485105aba5e612a2bf48
FIXNet.py            815d1a1c7012a28c05b260eb3023852dd86197205c4ff109cc5e7df379d74c3d
CAW6_buffer.py       d77af23cca28ba647e9ddf42604256f9e824454203dd55fca06705371f149260
```
