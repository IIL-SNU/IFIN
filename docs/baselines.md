# Baseline implementations

`models.baselines.build_baseline(config, psf, device)` builds the evaluated comparison models without modifying `sys.path`. `psf` must already be dataset-preprocessed, full/uncropped, and have shape `(1, N, Hp, Wp)`. The factory selects PSF index 4 when `N >= 9` for single-PSF methods. `multiwienernet` passes the first nine PSFs when a stack is available and repeats a single RGB PSF nine times exactly as the DiffuserCam/WiderCam notebooks do.

DeepLIR and MoDL are not distributed in the tracked package because the audited local copies do not establish redistribution permission. To use them locally, place sanitized `deeplir.py` and `modl.py` files in the gitignored `external_baselines/` directory:

```bash
python scripts/prepare_external_baselines.py --source-dir /path/to/user-supplied/research/models
```

The source directory must contain `DeepLIR.py` and `MoDL_SV.py`. The script performs only mechanical import sanitization and does not download either source. Override the local directory with `IFIN_EXTERNAL_BASELINES` or `model.options.external_source_dir`. All other registry entries import and construct without these files.

Each factory config has this form:

```yaml
model:
  name: unet
  options:
    in_channels: 3
    out_channels: 3
    psf_norm: l2
    psf_scale: 1.0
eval:
  batch_size: 1
```

The notebook-derived option sets are in `configs/baselines/{diffusercam,widercam,multiwiener}.yaml`. Each profile disables shared IFIN PSF cropping/normalization with null data overrides; baseline-specific normalization then occurs once in the factory. Each `baselines.<name>` entry can be supplied as `config.model`. Set root `phase` to `train` or `eval`; DeepLIR reads the corresponding root batch size while constructing its physics stage.

The profiles record checkpoint-compatible constructor choices explicitly. WiderCam/MultiWiener `unet` and `multiwienernet` select the shared `UNet` because their selected checkpoints fail strict loading against the notebook-overwritten `MoDL_SV.UNet` schema. MultiWienerNet keeps its checkpoint's `(1,9,224,320)` PSF batch dimension, while RGB checkpoints use `(9,H,W)`. MultiWiener LensNet sets `use_input_psf: true` because its checkpoint contains a `(1,1,224,320)` `psf` parameter; no keys are removed, remapped, or loaded non-strictly.

## Source map

LensNet now uses the pinned official source listed below. Its constructor keeps
the upstream defaults (320 x 320 RGB PSF, `w_init=0.001`, clamped output); the
factory adapts PSF dimensions/channels to the selected dataset. Dataset profiles
explicitly set `w_init: 0.01` and `clamp_output: false` to preserve the evaluated
research variant. MultiWienerNet initializes it from the selected center PSF.
Set `model.options.w_init: 0.001` and `model.options.clamp_output: true` for
official initialization/output behavior at the dataset's dimensions. The
upstream architecture uses `ngf=64`; unsupported widths are rejected.

| Registry name | Evaluated source and class | Training source |
| --- | --- | --- |
| `wiener` | `models/WieNerDeconv.py::WieNerDeconv` | Analytic; no checkpoint |
| `admm` | DiffuserCam/WiderCam: `models/ADMM.py::ADMMs`; MultiWiener: `models/MoDL_SV.py::ADMMs` after wildcard overwrite | Analytic notebook parameters; no checkpoint |
| `unet` | DiffuserCam: `models/UNet.py::UNet`; WiderCam/MultiWiener: `models/MoDL_SV.py::UNet` after wildcard overwrite | `CAW/SVLensless/skeleton_U.py`, `CAW/3D/skeleton_U.py`; DiffuserCam producer is not uniquely identified by current filenames |
| `lensnet` | [Official LensNet](https://github.com/baijiesong/Lensnet/blob/a6977ad9f1a84971b9960acb97e3f370f96d302e/models/LensNet.py), with dataset options | `CAW/SVLensless/skeleton_LensNet.py`, `CAW/3D/skeleton_LensNet.py` |
| `nafnet` | `models/NAF.py::NAFNet` | `train/lensless/skeleton.py` (`Name='NAF'`) |
| `updn` | RGB: `models/UPDN.py::ImageOptimizerMixColors`; MultiWiener: `ImageOptimizer` | `train/lensless/skeleton_UPDN.py`, `CAW/SVLensless/skeleton_UPDN.py`, `CAW/3D/skeleton_UPDN.py` |
| `mwdns` | Effective notebook symbol: `models/MWDN.py::MWDNet_CPSF` | `train/lensless/skeleton_MWDN.py`, `CAW/SVLensless/skeleton_MWDN.py`, `CAW/3D/skeleton_MWDN.py` |
| `multiwienernet` | `CAW/3D/models/wiener_model.py::{WienerDeconvolution3D,MyEnsemble}` plus effective notebook `UNet` | `train/lensless/skeleton_MW3D.py`, `CAW/SVLensless/skeleton_MW.py`, `CAW/3D/skeleton_MW_3D.py` |
| `modl` | RGB: `models/MoDL_SV.py::MoDLNet`; MultiWiener: `MoDLNet3D` | `train/lensless/skeleton_MoDL.py`, `CAW/SVLensless/skeleton_MoDL.py`, `CAW/3D/skeleton_MoDL.py` |
| `leadmmu` | `models/unet.py::UNet270480` plus ADMM and notebook-local ensemble | No matching live training constructor was found; `UNet270480` is commented out in `train/lensless/skeleton.py` |
| `deeplir` | `models/DeepLIR.py::{build_model,Unet,EnsembleModel}` | `train/lensless/skeleton_DeepLIR.py`, `CAW/SVLensless/skeleton_DeepLIR.py`, `CAW/3D/skeleton_DeepLIR.py` |

## Effective notebook selections

Execution counts indicate prior execution, not current reproducibility. Commented constructors and cells with `execution_count: null` are excluded below.

### DiffuserCam

Source: `run_model-Copy1.ipynb`; dataset `WallerDataset` over the DiffuserCam data root, PSF `psf.tiff`, resized to `270x480`.

| Baseline | Effective constructor settings | Checkpoint basename |
| --- | --- | --- |
| NAFNet | `NAFNet()`; PSF unused | `NAF_20250923-222630_best.pth` |
| MultiWienerNet | 9 repeated PSFs, L2 norm; `WienerDeconvolution3D + UNet(9,3)` | `MW3D_20250919-174000_best.pth` |
| UNet | `UNet(3,3)` | `U_20250919-214327_best.pth` |
| DeepLIR | `build_model(psf.squeeze(), batch=1, iterations=5) + Unet(dim=32, channels=3, dim_mults=(1,2,4,8))` | `DeepLIR_20241202-172048_best.pth` |
| MWDNs | `MWDN.py::MWDNet_CPSF(3,3,psf)` | `MWDN_20250920-115627_best.pth` |
| Wiener | max-normalized PSF times `0.003`; no checkpoint load | n/a |
| ADMM | `ADMMs(psf/255,30,1)` | n/a |
| UPDN | `ImageOptimizerMixColors(psf,depth=10)`; uppercase `L2` skips every normalization branch | `UPDN_20241204-105841_best.pth` |

`LeADMMU` and FIX cells are live source variants but have no recorded execution count.

### WiderCam (`IFIN_SV` in the notebook)

Source: `run_model_IFIN.ipynb`; dataset `IFINDataset` over the WiderCam data root (historically named `IFIN_SV`), fixed affine alignment, PSF `point_4_4_rgb8.png`, resize `270x480`, subtract `15/255`, clamp negative values.

| Baseline | Effective constructor settings | Checkpoint basename |
| --- | --- | --- |
| Le-ADMM-U | `UNet270480((3,270,480)) + ADMMs(psf,5,1)` | `Le-ADMM-U-SV_20250923-232545_best.pth` |
| MultiWienerNet | notebook resolves `MoDL_SV.UNet(9,3)`; checkpoint-compatible profile selects shared `UNet(9,3)` | `MW_20250924-152852_best.pth` |
| UNet | notebook resolves `MoDL_SV.UNet(3,3)`; checkpoint-compatible profile selects shared `UNet(3,3)` | `U_20250919-214327_best.pth` |
| DeepLIR | `build_model(...,batch=1,iterations=5) + Unet(dim=32,channels=3)` | `DeepLIR_20250923-213824_best.pth` |
| MWDNs | `MWDN.py::MWDNet_CPSF(3,3,psf)` | `MWDN_20250921-185800_best.pth` |
| MoDL | `MoDLNet(psf,device)` | `MoDL_20251121-022442_best.pth` |
| LensNet | `LensNet()` | `LensNet_20251122-221830_best.pth` |
| Wiener | max-normalized PSF times `0.05`; no checkpoint load | n/a |
| UPDN | `ImageOptimizerMixColors(psf,depth=10)`; uppercase `L2` skips every normalization branch | `UPDN_20250923-073258_best.pth` |

Recorded own-model branches are FIX/CAWNet with `k=9` and IFIN with `k=1`; they are intentionally outside this baseline package. The ADMM cell is unexecuted.

### MultiWienerNet

Source: `run_model_M3D.ipynb`; effective dataset is `MiniscopeDataset_2D` over `MultiWieNer/2D/{Ground_truth_downsampled,Simulated_Miniscope_2D_Training_data}`. PSFs are `multiWienerPSFStack_40z_aligned.mat`, preprocessed to nine planes and downsampled by `0.5`.

| Baseline | Effective constructor settings | Checkpoint basename |
| --- | --- | --- |
| Le-ADMM-U | center PSF index 4; `UNet270480((1,224,320)) + ADMMs(psf,5,1)` | `Le_ADMM-U-MINI_20250923-174527_best.pth` |
| UNet | notebook resolves `MoDL_SV.UNet(1,1)`; checkpoint-compatible profile selects shared `UNet(1,1)` | `U_20250919-085928_best.pth` |
| DeepLIR | center PSF, L2 norm; `build_model(...,batch=1,iterations=5) + Unet(dim=32,channels=1)` | `DeepLIR_20250922-063054_best.pth` |
| MWDNs | center PSF; `MWDN.py::MWDNet_CPSF(1,1,psf)` | `MWDN_20250910-133851_best.pth` |
| MoDL | intended `MoDLNet3D(center_psf,ch=1,psf_in_channels=1)` | `MoDL_20251122-172556_best.pth` |
| LensNet | `LensNet(1,1)` | `LensNet_20251124-094905_best.pth` |
| MultiWienerNet | full 9-PSF stack; notebook resolves `MoDL_SV.UNet(9,1)`, checkpoint-compatible profile selects shared `UNet(9,1)` | `MW_20250920-023906_best.pth` |
| ADMM | center PSF; `ADMMs(psf,100,1)` | n/a |

FIX `k=4` and `k=1` are recorded own-model branches. Wiener and UPDN source cells are unexecuted.

## Checkpoint contract and audit findings

Learned branches load `torch.load(path, map_location='cpu')['model_state_dict']` with `strict=True`. Training scripts save dictionaries containing `epoch`, `model_state_dict`, and `optimizer_state_dict`; DiffuserCam/WiderCam MWDNs/IFIN scripts may also include `optimizer_psf_state_dict`. Analytic Wiener and ADMM branches ignore their configured `PTH_PATH` values.

All recorded learned checkpoint paths existed during the audit. All 21 selected learned dataset/model combinations were also loaded into their profile constructors with `strict=True`. The following source/checkpoint inconsistencies remain:

- Every audited notebook's first code cell is syntactically invalid because each `sys.path.append(` line lacks `)`. Execution counts therefore describe an older kernel state, not a clean rerun.
- WiderCam imports `IFIN2` from an unavailable external source directory. The baseline package does not depend on it.
- WiderCam Le-ADMM-U resolves `EnsembleModel` from `DeepLIR.py` and passes `(unet, admm)` to a constructor declared `(admm_model, denoise_model)`, unlike the inline DiffuserCam/MultiWiener ensemble. The factory uses the checkpoint-facing inline `unet`/`admms` attributes and preserves its forward order.
- DiffuserCam/WiderCam evaluation resolves `MWDNet_CPSF` from `MWDN.py`, while their current training scripts import `MWDNs.py`; the selected checkpoints strictly load into the evaluated `MWDN.py` class, but provenance still points at inconsistent source filenames.
- MultiWiener MoDL reads `TPARAMS['PSF']` before assigning it in that branch. The factory passes the selected center PSF directly without changing `MoDLNet3D` math.
- MultiWiener LensNet requests one channel, but the audited research `LensNet` source hardcodes a trainable `(1,3,270,480)` PSF. Its checkpoint stores `(1,1,224,320)`, so the profile explicitly initializes the same parameter from the selected full-size input PSF before strict loading.
- Several configured checkpoint paths are stale but harmless because Wiener/ADMM skip loading; DiffuserCam UNet also points to the same checkpoint basename as the WiderCam notebook.

## Official LensNet Repository Comparison

Audited [model directory](https://github.com/baijiesong/Lensnet/tree/a6977ad9f1a84971b9960acb97e3f370f96d302e/models)
at commit `a6977ad9f1a84971b9960acb97e3f370f96d302e`. A shared model name does
not mean that two implementations reproduce the same experiment.

| Released model | Official repository difference | Release decision |
| --- | --- | --- |
| LensNet | Same core at width 64; official PSF 320 x 320, Wiener scale 0.001, and final clipping differ from research defaults | Use official core with explicit dataset options |
| MWDNs | [Official MWDNs](https://github.com/baijiesong/Lensnet/blob/a6977ad9f1a84971b9960acb97e3f370f96d302e/models/MWDNs.py) has a trainable 3 x 256 x 256 PSF, fixed RGB PSF encoder, scalar scale 0.001, and clipping; research accepts dataset PSFs and grayscale, with different regularizer shape/initialization | Keep research implementation |
| UPDN | [Official UPDN](https://github.com/baijiesong/Lensnet/blob/a6977ad9f1a84971b9960acb97e3f370f96d302e/models/UPDN.py) loads a fixed RGB PSF asset internally; research has injected PSFs and separate grayscale/RGB color-mixing variants | Keep research implementation |
| DeepLIR | [Official DeepLIR](https://github.com/baijiesong/Lensnet/blob/a6977ad9f1a84971b9960acb97e3f370f96d302e/models/DeepLIR.py) uses ADMM plus SwinIR; evaluated source uses ADMM plus a diffusion-style U-Net | Keep evaluated external implementation |
| Le-ADMM-U | [Official Le-ADMM-U](https://github.com/baijiesong/Lensnet/blob/a6977ad9f1a84971b9960acb97e3f370f96d302e/models/Le_ADMM_U.py) uses embedded ADMM and a diffusion/ConvNeXt U-Net; research uses ADMMs plus UNet270480 | Keep research implementation |
| Wiener | [Official Wiener file](https://github.com/baijiesong/Lensnet/blob/a6977ad9f1a84971b9960acb97e3f370f96d302e/models/Wiener.py) is an offline scikit-image unsupervised-Wiener script, not the evaluated Torch FFT module | Keep research implementation |
| ADMM | Only embedded variants, with different parameters, PSF loading, and crop/state conventions | Keep research implementation |
| UNet | LenslessGAN has a similar denoiser topology, but different interpolation, output handling, and state names | Keep research implementation |
| NAFNet | Not provided; WoNAF is a LensNet ablation without NAF blocks, not NAFNet | Keep research implementation |
| MultiWienerNet, MoDL | Not provided | Keep existing integrations |

Additional upstream models are FlatNet, MMCN, UDN, ULAMPNet, MDGAN, TikNet,
ThreeDown, LenslessGAN, and the WoNAF/WoWNFB ablations. PSF.py is also exported.
These are not automatically added to the released benchmark set because the
selected IFIN evaluation notebooks did not use them. No complete upstream
baseline was a drop-in numerical replacement for its research counterpart.

## Runtime Dependencies

The tracked package uses existing project dependencies: Python, PyTorch, torchvision, NumPy, and einops. No NAS paths, datasets, checkpoints, W&B, pyiqa, matplotlib, tqdm, torchmetrics, or scipy are imported eagerly by the baseline package. DeepLIR and MoDL source files and all checkpoint/dataset files remain local external inputs.
