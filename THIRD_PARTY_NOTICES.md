# Third-party notices

The tracked files under `src/models/baselines/` were mechanically derived from research copies in `/mnt/nas/Research/DGBae/models` and `/mnt/nas/Research/DGBae/train/lensless/CAW/3D/models`, with imports localized and unused experiment dependencies removed. Mathematical operations and checkpoint-facing module attributes were retained. DeepLIR and MoDL copies are kept only in the gitignored local `external_baselines/` directory and are not part of the tracked source package.

## NAFNet

`nafnet.py` retains the source header: Copyright (c) 2022 megvii-model. All Rights Reserved. The cited upstream project is [megvii-research/NAFNet](https://github.com/megvii-research/NAFNet), associated with "Simple Baselines for Image Restoration" by Chen et al. Its license file at audited commit `2b4af71ebe098a92a75910c233a3965a3e93ede4` contains combined MIT and Apache License 2.0 terms: [immutable LICENSE](https://raw.githubusercontent.com/megvii-research/NAFNet/2b4af71ebe098a92a75910c233a3965a3e93ede4/LICENSE). The official text is included at `licenses/NAFNet.txt`; this notice does not assert a new license for the code.

## MultiWienerNet

`multiwiener.py` is derived from the local research copy associated with [Waller-Lab/MultiWienerNet](https://github.com/Waller-Lab/MultiWienerNet). The upstream license at audited commit `f49a38e74a73bcf58f91a46d3ff0d2360d213283` is BSD 3-Clause: [immutable LICENSE](https://raw.githubusercontent.com/Waller-Lab/MultiWienerNet/f49a38e74a73bcf58f91a46d3ff0d2360d213283/LICENSE). The official text is included at `licenses/MultiWienerNet.txt`.

## LensNet

`lensnet.py` is derived from the local research copy associated with [baijiesong/Lensnet](https://github.com/baijiesong/Lensnet). The upstream license at audited commit `a6977ad9f1a84971b9960acb97e3f370f96d302e` is the MIT License with its Nick Chen copyright notice: [immutable LICENSE](https://raw.githubusercontent.com/baijiesong/Lensnet/a6977ad9f1a84971b9960acb97e3f370f96d302e/LICENSE). The official text is included at `licenses/LensNet.txt`.

These upstream license findings identify the referenced projects and commits. They do not by themselves prove that every local research copy is identical to, derived solely from, or redistributable under the corresponding upstream license.

## Research baseline copies

The following derived modules had no license header in the audited source files and no applicable license file was found in the shared `models`, `functools`, or `train/lensless/CAW/3D` roots:

- `admm.py`
- local `external_baselines/deeplir.py` (the audited upstream repository has no license file)
- `leadmm_unet.py`
- `lensnet.py` (upstream license identified above; local-copy correspondence still requires confirmation)
- local `external_baselines/modl.py` (relationship to AGPL-licensed work is unclear; no license conclusion is made for this local implementation)
- `multiwiener.py` (upstream license identified above; local-copy correspondence still requires confirmation)
- `mwdns.py`
- `unet.py`
- `updn.py`
- `wiener.py`
- `_blocks.py` and `_utils.py`, which contain only direct helper subsets required by those modules

Their inclusion here is provenance documentation, not a declaration that they are relicensed under any project-wide license. A local file without a license header does not establish its license status. DeepLIR and MoDL must remain untracked unless redistribution authorization is established; local-copy correspondence for all other derived modules must also be confirmed before public release. No AGPL conclusion is inferred for the local MoDL implementation merely from a possible relationship to other work.

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
