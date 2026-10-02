# Codebase Update

## Scope

Use the models selected in `run_model_IFIN.ipynb` (WiderCam),
`run_model-Copy1.ipynb` (DiffuserCam), and `run_model_M3D.ipynb`
(MultiWienerNet), rather than the simplified prerelease implementation.
Keep the research model operations intact. Publish one IFIN implementation
with dataset and historical architecture differences configured in YAML.
Do not publish data, weights, notebook outputs, or machine-specific paths.

## Checklist

- [x] Trace notebook imports, constructors, datasets, and training sources.
- [x] Restore research IFIN operations and verify legacy checkpoint compatibility.
- [x] Package the evaluated baselines and dataset-specific training/evaluation recipes.
- [x] Run tests, real-data inference, and CLI checks.
- [x] Commit and push the code update.

Implementation commit: `5a68b38ab473c1ece55aaf31b54bcf7c35c92a59` on
`IIL-SNU/IFIN` main. The unrelated local Fourier-PSF notebook was not included.

## Verification

Compare the renamed model directly against both research implementations
using the same state dictionary and inputs. Checkpoint loading must be strict;
no dropping parameters, reshaping learned tensors, or silently substituting
one historical architecture for another. Test dataset transformations on
small fixtures, then run the public CLI on actual available data and weights.

## Known Source Discrepancies

- The WiderCam notebook also contains an experimental `IFIN2` import whose
  source is not available. The working `FIX` branch resolves to `FIXNet`.
- MultiWienerNet k=1/k=4 checkpoints have a different historical architecture
  from the current `FIXNet` branch. Preserve it through explicit YAML settings.
- Notebook checkpoint filenames, paper settings, and current training-script
  defaults are not interchangeable. Record verified provenance separately.

## Verified Results

- 36 tests passed, including direct forward/input-gradient equivalence against
  both research IFIN sources and atomic checkpoint replacement after mmap loading.
- 30 comparison-model/dataset real-size forward cases passed, with strict loading
  for the available learned weights. Some cases intentionally used random weights
  when the notebook did not select a checkpoint for that dataset/model pair.
- Actual WiderCam and MultiWienerNet IFIN weights passed CLI evaluation with
  PSNR, SSIM, and LPIPS and saved reconstruction images.
- DiffuserCam passed real-data inference with untrained IFIN; the named notebook
  IFIN checkpoint is missing, so no pretrained DiffuserCam metric is claimed.
- WiderCam completed one real-data training step and validation.
- DeepLIR/MoDL external preparation and local forwards passed. Their source copies
  remain gitignored; the public code contains adapters and preparation tooling.
- Core Ruff diagnostics and compileall passed. Python/YAML LSPs were unavailable;
  configuration parsing and runtime tests were used instead.
- A staged, clean public-source copy passed train/eval/infer smoke commands and
  31 tests. Five optional checks skipped because research/external sources were
  intentionally absent from that copy.
