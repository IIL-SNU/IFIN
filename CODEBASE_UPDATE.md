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

## Evaluation Fidelity Follow-Up

- [x] Verify auxiliary loss weights and archived PSF penalty against training sources.
- [x] Match notebook prediction normalization without rescaling ground truth.
- [x] Expose the notebook crop/dewarped metrics and verify their coordinate transforms.
- [x] Re-run tests and real-data CLI evaluation.

## Requested Cleanup

- [x] Remove stale initial-codebase notebooks and unused path validation; retain
  runtime dependencies and working regression tests regardless of file age.
- [x] Audit the official LensNet model directory and use its LensNet source;
  retain research baseline variants where mathematical operations differ.
- [x] Default all IFIN configurations and constructors to k=1; expose --k and
  verify the paper's k=1/4/9/16 settings without changing model operations.
- [x] Recheck MoDL provenance and run tests and CLI checks.

## Verification Criteria

The stale initial notebooks removed are `notebooks/inference_waller.ipynb`
and `tests/waller_usage_template.ipynb`; they referenced obsolete configuration
keys or removed runner helpers. The unused `validate_waller_paths` API was also
removed. Unchanged CLI wrappers, synthetic data, FFT/ROI helpers, seeding, package
exports, and working tests remain because the current runtime still uses them.

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

- 49 tests passed, including direct forward/input-gradient equivalence against
  both research IFIN sources and atomic checkpoint replacement after mmap loading.
- 30 comparison-model/dataset real-size forward cases passed, with strict loading
  for the available learned weights. Some cases intentionally used random weights
  when the notebook did not select a checkpoint for that dataset/model pair.
- Actual WiderCam and MultiWienerNet IFIN weights passed CLI evaluation with
  PSNR, SSIM, and LPIPS and saved reconstruction images.
  Evaluation now uses notebook-specific prediction normalization without
  rescaling ground truth. Crop/dewarped transforms are bit-identical to the
  original source helpers; they are reported separately from full-frame metrics.
- DiffuserCam passed real-data inference with untrained IFIN; the named notebook
  IFIN checkpoint is missing, so no pretrained DiffuserCam metric is claimed.
- WiderCam and MultiWienerNet each completed one real-data training step and validation.
- NAFNet completed one real-data training step and validation through the same
  shared runner. YAML measurement/ISO loss weights have a direct regression check.
  The archived squared PSF non-negativity penalty is applied during training only.
- DeepLIR/MoDL external preparation and local forwards passed. Their source copies
  remain gitignored; the public code contains adapters and preparation tooling.
- Official LensNet and the configured research variant have bit-identical outputs
  against their respective reference sources with the same 343 state keys.
  Strict-loaded RGB/grayscale research checkpoints passed real-data CLI inference.
- The k=1/4/9/16 settings passed model forwards; --k 4 passed CLI train/eval/infer.
- Core Ruff diagnostics and compileall passed. Python/YAML LSPs were unavailable;
  configuration parsing and runtime tests were used instead.
- A staged, clean public-source copy passed train/eval/infer smoke commands and
  44 tests. Five optional checks skipped because research/external sources were
  intentionally absent from that copy.
