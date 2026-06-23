<div align="center">

# Integrated Forward-Inverse Network for Lensless Image Reconstruction

**Accepted to ECCV 2026**

[Donggeon Bae](mailto:donggeonbae@snu.ac.kr)<sup>1</sup>, Jaewoo Jung<sup>2,3</sup>, Yong Guk Kang<sup>2</sup>, Kyung Chul Lee<sup>4</sup>, Taeyoung Kim<sup>3</sup>, Jongho Kim<sup>1</sup>, Sangjun Byun<sup>1</sup>, Joonsik Park<sup>3</sup>, Seung Ah Lee<sup>1,2</sup>

<sup>1</sup>Seoul National University, Department of Mechanical Engineering<br>
<sup>2</sup>Seoul National University, School of Mechanical and Aerospace Engineering/SNU-IAMD<br>
<sup>3</sup>Yonsei University, Department of Electrical and Electronic Engineering<br>
<sup>4</sup>University of Michigan, Department of Biomedical Engineering

<p>
  <a href="https://iilab.io/IFIN/"><img src="https://img.shields.io/badge/Project-Page-315c5f" alt="Project page"></a>
  <a href="https://github.com/IIL-SNU/IFIN/releases/download/v0.1/ifin_eccv2026_paper.pdf"><img src="https://img.shields.io/badge/Paper-PDF-bb4b4b" alt="Paper PDF"></a>
  <a href="https://github.com/IIL-SNU/IFIN/releases/download/v0.1/ifin_eccv2026_supplementary.pdf"><img src="https://img.shields.io/badge/Supplement-PDF-6b5f2a" alt="Supplement PDF"></a>
  <a href="https://iilab.io/SVLensless-dataset"><img src="https://img.shields.io/badge/WiderCam-Dataset-4d6f3b" alt="WiderCam dataset"></a>
</p>

</div>

## Abstract

Lensless imaging enables compact and versatile computational cameras by replacing bulky optics with thin coded elements. However, reconstruction from the resulting measurements is challenging: large-footprint PSFs produce highly multiplexed observations, making inversion severely ill-conditioned and sensitive to calibration errors and model mismatch. We propose the **Integrated Forward-Inverse Network (IFIN)**, a physics-guided architecture that interleaves differentiable forward projections with learnable inverse updates at every stage. This bidirectional coupling supports progressive, physics-consistent refinement and permits system-constrained PSF kernel refinement under model uncertainty. On challenging lensless benchmarks, including our newly introduced **WiderCam** dataset, IFIN achieves state-of-the-art reconstruction quality. IFIN also remains effective on Gaussian deblurring and simulated inline holography, suggesting that forward-inverse interleaving can extend beyond a single lensless camera.

## Method

IFIN uses an encoder-decoder backbone with **Integrated Forward-Inverse Blocks (IFIBs)** at every scale. Each IFIB exchanges information between an image-domain stream and a measurement-domain stream using:

- **Forward System Operator (FSO):** projects the current image-domain representation through the optical forward model.
- **Inverse System Operator (ISO):** applies a learnable Wiener-like inverse update from the measurement domain.
- **Learnable PSF field:** refines system kernels end-to-end for calibration mismatch and shift-variant degradation.

Rather than applying a single inversion and then relying only on learned refinement, IFIN repeatedly exchanges forward projections and inverse updates throughout the feature hierarchy. This keeps measurement-domain evidence available during reconstruction.

<img src="docs/assets/images/architecture.webp" alt="IFIN architecture" width="100%">

<p align="center"><sub>Overall architecture of IFIN. IFIBs are inserted at each encoder-decoder scale to jointly apply FSO and ISO with a shared learnable PSF field.</sub></p>

<img src="docs/assets/images/ifib.webp" alt="Integrated Forward-Inverse Block" width="100%">

<p align="center"><sub>Integrated Forward-Inverse Block. IFIN couples forward projection and inverse restoration in either a single-PSF or PSF-field setting.</sub></p>

## Results

IFIN improves reconstruction quality across three lensless benchmarks: DiffuserCam, WiderCam, and the MultiWienerNet dataset.

| Method | DiffuserCam PSNR | DiffuserCam LPIPS | DiffuserCam SSIM | WiderCam PSNR | WiderCam LPIPS | WiderCam SSIM | MultiWienerNet PSNR | MultiWienerNet LPIPS | MultiWienerNet SSIM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ADMM | 12.252 | 0.607 | 0.346 | 11.843 | 0.643 | 0.323 | 19.189 | 0.557 | 0.420 |
| Wiener Deconv. | 12.552 | 0.591 | 0.384 | 12.405 | 0.607 | 0.369 | 18.658 | 0.640 | 0.302 |
| **ISO (Ours)** | 16.528 | 0.544 | 0.404 | 17.240 | 0.462 | 0.444 | 20.202 | 0.623 | 0.380 |
| UNet | 21.230 | 0.394 | 0.656 | 21.890 | 0.474 | 0.646 | 23.859 | 0.389 | 0.589 |
| NAFNet | 24.830 | 0.239 | 0.810 | 23.857 | 0.245 | 0.769 | 24.657 | 0.282 | 0.712 |
| Le-ADMM-U | 23.261 | 0.312 | 0.765 | 21.956 | 0.278 | 0.748 | 23.732 | 0.335 | 0.702 |
| DeepLIR | 25.958 | 0.260 | 0.829 | 20.523 | 0.339 | 0.642 | 22.556 | 0.379 | 0.642 |
| MWNet | 24.832 | 0.247 | 0.810 | 23.001 | 0.255 | 0.766 | 25.660 | 0.260 | 0.728 |
| UPDN | <u>28.228</u> | 0.194 | 0.877 | 23.920 | 0.229 | 0.801 | 24.364 | 0.287 | 0.707 |
| MWDNs | 27.298 | 0.217 | 0.845 | 24.525 | 0.224 | 0.801 | 27.436 | 0.236 | 0.780 |
| LensNet | 27.650 | 0.201 | 0.868 | 24.615 | 0.219 | 0.806 | 27.546 | 0.221 | 0.809 |
| MoDL | 27.958 | <u>0.183</u> | <u>0.878</u> | <u>24.791</u> | <u>0.202</u> | <u>0.810</u> | <u>28.504</u> | <u>0.202</u> | <u>0.831</u> |
| **IFIN (Ours)** | **29.862** | **0.174** | **0.893** | **25.444** | **0.201** | **0.824** | **31.083** | **0.175** | **0.866** |

<img src="docs/assets/images/diffusercam.webp" alt="DiffuserCam reconstruction comparison" width="100%">

<p align="center"><sub>Visual comparison on DiffuserCam display-capture data. IFIN preserves color fidelity and high-frequency textures while suppressing artifacts.</sub></p>

<img src="docs/assets/images/widercam.webp" alt="WiderCam reconstruction comparison" width="100%">

<p align="center"><sub>Comparison on WiderCam. IFIN mitigates field-dependent peripheral blur and geometric distortion while preserving fine textures and edges.</sub></p>

<img src="docs/assets/images/mwnet.webp" alt="MultiWienerNet reconstruction comparison" width="100%">

<p align="center"><sub>Comparison on MWNet. Results include simulated spatially variant measurements and experimental miniscope captures.</sub></p>

More qualitative results and system details are available on the [project page](https://iilab.io/IFIN/).

## Inline Holography Reconstruction

IFIN also transfers to simulated inline holography reconstruction by replacing the system operators with angular-spectrum forward propagation and corresponding back-propagation.

| Method | PSNR | LPIPS | SSIM |
| --- | ---: | ---: | ---: |
| RCAN | 23.764 | 0.354 | 0.702 |
| NAFNet | 23.224 | 0.389 | 0.634 |
| NAFNet+ | <u>25.751</u> | <u>0.242</u> | <u>0.817</u> |
| **IFIN (Ours)** | **28.302** | **0.166** | **0.890** |

<img src="docs/assets/images/inline.webp" alt="Inline holography reconstruction" width="100%">

<p align="center"><sub>Inline holography reconstruction. IFIN adapts to the holographic forward model and recovers cleaner object structures from a single intensity hologram.</sub></p>

## WiderCam Dataset

We introduce **WiderCam**, a wide-field lensless reconstruction benchmark captured with a compact phase-mask camera. The dataset contains 25,000 paired measurements, split into 24,000 training and 1,000 test images, with strong field-dependent degradation over a wide field of view. Raw measurements are captured at 4608 x 2592 using a Sony IMX708 sensor and resized to 480 x 270; supervision is affine-aligned offline from the display-capture pair.

WiderCam is available here:

**Dataset:** [WiderCam dataset page](https://iilab.io/SVLensless-dataset)

## Installation

```bash
git clone https://github.com/IIL-SNU/IFIN.git
cd IFIN
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

The default configuration follows the DiffuserCam/Waller-style paired dataset layout. For a dependency and code-path smoke test without external data, switch `data.dataset` in `configs/default.yaml` to `synthetic`.

Train:

```bash
python train.py --config configs/default.yaml
```

Evaluate:

```bash
python eval.py --config configs/default.yaml --checkpoint outputs/checkpoints/ifin_last.pth
```

Run inference:

```bash
python infer.py --config configs/default.yaml --checkpoint outputs/checkpoints/ifin_last.pth
```

Run tests:

```bash
python -m pytest -q tests
```

## Dataset Layout

For DiffuserCam/Waller-style data, set `data.waller_path` and `data.psf_path` in `configs/default.yaml`.

Expected layout:

```text
dataset_root/
  dataset_train.csv
  dataset_test.csv
  diffuser_images/
  ground_truth_lensed/
  psf.tiff
```

## Citation

```bibtex
@inproceedings{bae2026ifin,
  title     = {Integrated Forward-Inverse Network for Lensless Image Reconstruction},
  author    = {Bae, Donggeon and Jung, Jaewoo and Kang, Yong Guk and Lee, Kyung Chul and Kim, Taeyoung and Kim, Jongho and Byun, Sangjun and Park, Joonsik and Lee, Seung Ah},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

## Contact

For access to additional datasets, checkpoints, or code, please contact [donggeonbae@snu.ac.kr](mailto:donggeonbae@snu.ac.kr).
