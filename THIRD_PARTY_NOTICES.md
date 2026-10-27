# Third-party notices

The [project MIT license](LICENSE), Copyright (c) 2026 Donggeon Bae, covers
original IFIN project code and its accompanying code documentation. It does
not relicense third-party baseline implementations under
`src/models/baselines/`, dependencies, or externally supplied source. Existing
upstream copyrights, license conditions, and notices continue to apply.
Datasets, weights, papers, media, and out-of-scope local experiments are not
covered by the project software license.

The tracked files under `src/models/baselines/` include author-written implementations and third-party adaptations. Research model copies were integrated with imports localized and unused experiment dependencies removed; that mechanical integration does not by itself establish third-party authorship. Mathematical operations and checkpoint-facing module attributes were retained. DeepLIR and MoDL copies are kept only in the gitignored local `external_baselines/` directory and are not part of the tracked source package.

## NAFNet

`nafnet.py` retains the source header: Copyright (c) 2022 megvii-model. All Rights Reserved. The cited upstream project is [megvii-research/NAFNet](https://github.com/megvii-research/NAFNet), associated with "Simple Baselines for Image Restoration" by Chen et al. Its license file at audited commit `2b4af71ebe098a92a75910c233a3965a3e93ede4` contains combined MIT and Apache License 2.0 terms: [immutable LICENSE](https://raw.githubusercontent.com/megvii-research/NAFNet/2b4af71ebe098a92a75910c233a3965a3e93ede4/LICENSE). The official text is included at `licenses/NAFNet.txt`; this notice does not assert a new license for the code.

## MultiWienerNet

`multiwiener.py` is derived from the local research copy associated with [Waller-Lab/MultiWienerNet](https://github.com/Waller-Lab/MultiWienerNet). The upstream license at audited commit `f49a38e74a73bcf58f91a46d3ff0d2360d213283` is BSD 3-Clause: [immutable LICENSE](https://raw.githubusercontent.com/Waller-Lab/MultiWienerNet/f49a38e74a73bcf58f91a46d3ff0d2360d213283/LICENSE). The official text is included at `licenses/MultiWienerNet.txt`.

The local Wiener class bodies and Wiener-normalize-U-Net ensembles correspond
to [the pinned PyTorch source](https://github.com/Waller-Lab/MultiWienerNet/blob/f49a38e74a73bcf58f91a46d3ff0d2360d213283/pytorch/models/wiener_model.py).

## Lensless Learning

`leadmm_unet.py` and the stack-convolution blocks in `_blocks.py` correspond
to [Waller-Lab/LenslessLearning's U-Net source](https://github.com/Waller-Lab/LenslessLearning/blob/eaab0fb694a4f51fdda382a53f92832af98fd692/models/unet.py)
at commit `eaab0fb694a4f51fdda382a53f92832af98fd692`. The research copy
generalizes channels and retains batch dimensions. The BSD-3-Clause copyright
notice, conditions, and disclaimer are included in `licenses/LenslessLearning.txt`.

## LensNet

`lensnet.py` uses [the official LensNet source](https://github.com/baijiesong/Lensnet/blob/a6977ad9f1a84971b9960acb97e3f370f96d302e/models/LensNet.py), with explicit dataset PSF, Wiener initialization, and clipping options. The upstream license at audited commit `a6977ad9f1a84971b9960acb97e3f370f96d302e` is the MIT License with its Nick Chen copyright notice: [immutable LICENSE](https://raw.githubusercontent.com/baijiesong/Lensnet/a6977ad9f1a84971b9960acb97e3f370f96d302e/LICENSE). The official text is included at `licenses/LensNet.txt`, and the source file carries its attribution.

The research `mwdns.py`, `updn.py`, and their DoubleConv/Down/Up/OutConv helpers
in `_blocks.py` also correspond to that repository's
[MWDNs.py](https://github.com/baijiesong/Lensnet/blob/a6977ad9f1a84971b9960acb97e3f370f96d302e/models/MWDNs.py)
and [UPDN.py](https://github.com/baijiesong/Lensnet/blob/a6977ad9f1a84971b9960acb97e3f370f96d302e/models/UPDN.py).
Dataset PSF injection, grayscale support, and output handling are research
adaptations; those differences do not make the original portions independently
licensed code. LayerNorm helpers in `_blocks.py` correspond to the NAFNet source
and retain its separately included notice.

These upstream license findings identify the referenced projects and commits for third-party portions. They do not replace the project MIT license for independently author-written portions or relicense third-party code under the project copyright.

## Author-written baseline implementations

Donggeon Bae explicitly confirmed authorship of `src/models/baselines/admm.py`,
`src/models/baselines/wiener.py`, `src/models/baselines/_utils.py`, and the
previously unverified local portions of `src/models/baselines/unet.py`.
Those original implementations and additions are covered by the project
[MIT license](LICENSE), Copyright (c) 2026 Donggeon Bae. The previous
source-origin questions for these portions are resolved by that confirmation;
they are not pending redistribution-permission blockers.

The confirmation does not claim authorship of the known third-party code
identified elsewhere in this notice. In particular, the LenslessLearning-derived
stack helpers used by `unet.py` and the separately identified NAFNet, LensNet,
and MultiWienerNet portions retain their upstream terms and notices.

## External baseline inputs

The local gitignored `external_baselines/deeplir.py` and `modl.py` are not part
of the tracked source package or this authorship confirmation. The existing
audit found no license file in the audited DeepLIR upstream and identified
LenslessPiCam-derived GPLv3 code in MoDL, with complete local-wrapper provenance
unresolved. Keep these external inputs untracked unless redistribution
permission is established. This does not restrict the separately licensed
original IFIN implementations above.

## MoDL Source Audit

The local `UNetRes` body matches the distinctive compensation and background
paths of [LenslessPiCam's U-Net at commit d0261b4](https://github.com/LCAV/LenslessPiCam/blob/d0261b4bc79ef05228b135e6898deb4f7793d1aa/lensless/recon/drunet/network_unet.py).
The [license at that revision](https://github.com/LCAV/LenslessPiCam/blob/d0261b4bc79ef05228b135e6898deb4f7793d1aa/LICENSE)
is GPLv3. Upstream [changed to AGPLv3 later](https://github.com/LCAV/LenslessPiCam/commit/672ba558af20df85c4e69127024fb7ea2006d446);
the current license alone does not establish the local copy's revision or terms.
The inlined helper blocks correspond to the separately
[MIT-licensed DPIR source](https://github.com/cszn/DPIR/blob/15bca3fcc1f3cc51a1f99ccf027691e278c19354/models/basicblock.py).
No exact public match was established for the complete local `MoDLNet`,
`MoDLNet3D`, and Torch ADMM wrappers. The official LensNet repository contains
no MoDL implementation. These findings do not justify publishing the local
module under LensNet's MIT license; it remains a user-supplied external input.

Relevant method references already cited by the project are:

- ADMM: Boyd et al., "Distributed Optimization and Statistical Learning via ADMM"
- Wiener deconvolution: Wiener, "Extrapolation, Interpolation, and Smoothing of Stationary Time Series"
- UNet: Ronneberger et al., "U-Net"
- Le-ADMM-U: Waller Lab, "Lensless Learning"
- DeepLIR: `arpanpoudel/lenslessimaging`
- MultiWienerNet: Waller Lab, "MultiWienerNet"
- UPDN: Optics Express, DOI `10.1364/OE.475521`
- MWDNs: Optics Express, DOI `10.1364/OE.501970`
- LensNet: arXiv `2505.01755`
- MoDL: IEEE Transactions on Computational Imaging, DOI `10.1109/TCI.2025.3539448`

## Baseline source comparison (2026-10-27)

The scoped MIT grant for original IFIN code is based on the project owner's
confirmation of ownership and the specific authorship statement above.
Scientific methods such as ADMM, Wiener filtering,
and U-Net architectures may be reimplemented; a method name or paper citation
alone does not establish that a particular source file was independently
written or that copied source is licensed for redistribution.

A comparison of the tracked baseline files with the pinned upstream sources
above found matching source structure, including complete classes or methods:

- `nafnet.py`: `NAFBlock` matches the pinned NAFNet source. Existing NAFNet and
  BasicSR notices and license texts remain applicable.
- `lensnet.py`: multiple blocks match the pinned LensNet source, including
  `NAFBlock`, LayerNorm helpers, and decoder blocks. The NAFBlock and LayerNorm
  bodies also match NAFNet/BasicSR; preserve both sets of existing notices.
- `leadmm_unet.py`: complete `ConvBnRelu2d`, `StackEncoder`, `StackDecoder`, and
  `UNet_small` classes match the pinned LenslessLearning U-Net source.
- `multiwiener.py`: Wiener constructors/forward methods and ensemble
  constructors match the pinned MultiWienerNet source.
- `updn.py`: camera/operator blocks, its U-Net, and primal-dual blocks match
  the pinned LensNet `UPDN.py` source. `mwdns.py` retains that repository's
  multiscale PSF-encoder/decoder structure with local PSF injection, channel
  options, and Wiener helper changes. This is evidence of adaptation, not
  merely a citation to a method.
- `_blocks.py`: shared portions map to the pinned LensNet, LenslessLearning,
  and NAFNet sources as recorded above.
- `unet.py`: the encoder/decoder layout shares the LenslessLearning U-Net
  architecture and uses the separately attributed stack helpers in `_blocks.py`.
  Architecture correspondence alone does not establish copied source. The
  author confirmed the previously unverified local implementation/additions,
  including FiLM, padding, Wiener, and ensemble portions; those original
  portions are MIT, while the known third-party helpers retain their terms.

The bounded comparison found no concrete copied-source match contradicting the
confirmed authorship of `admm.py`, `wiener.py`, `_utils.py`, or the previously
unverified additions in `unet.py`. Historical research filenames such as
`models/ADMM.py` and `models/WieNerDeconv.py` are not independent evidence of
third-party authorship. No exclusion of these author-written implementations
from the IFIN release is required by this review. No source file or experiment
was removed or moved.

## Evaluation dependencies and downloaded assets

`requirements.txt` includes unpinned `pyiqa`; `src/metrics/basic.py` uses it for
configured metrics other than PSNR, including SSIM and LPIPS. As checked on
2026-10-27, the [upstream license](https://github.com/chaofengc/IQA-PyTorch/blob/main/LICENSE)
is PolyForm Noncommercial License 1.0.0. This finding does not determine which
version or license was installed for earlier experiments. Check the exact
version's terms before use or redistribution, especially for commercial use.
This dependency condition does not change the MIT license on original IFIN code.

`src/losses/basic.py` optionally uses `lpips` with a VGG backbone. The
[LPIPS source license](https://github.com/richzhang/PerceptualSimilarity/blob/master/LICENSE)
is BSD-2-Clause; that finding does not establish rights to every downloaded
backbone or metric weight. All dependencies and downloaded model assets retain
their respective terms. The requirements file is not a lockfile or a complete
license inventory of installed and transitive packages.
