# Where to personalize: split-point study for federated 3D segmentation

Which parts of a 3D U-Net should hospitals share, and which should each keep local?
This repository tests whether the answer depends on the kind of heterogeneity
(site shift, missing MRI modalities) and on how much data a site holds.

## Status

| Step | State |
| --- | --- |
| Network, block naming, split specifications, partial federated averaging | Done, unit-tested on CPU |
| Data pipeline (FeTS 2022 / MSD Task01), preprocessing, patch sampling | Not started |
| Training loop, baselines (local, FedAvg, FedAvg + fine-tuning) | Not started |
| Heterogeneity generators (missing modalities, site-size subsampling) | Not started |
| Pilot grid, diagnostics, statistics | Not started |

## Layout

- `pflseg/model.py` — MONAI DynUNet (22.6M parameters, as in Manthe et al. 2024) and the
  mapping from parameter names to ten blocks: stem, enc1-3, bottleneck, dec1-4, head.
- `pflseg/splits.py` — every split in the grid (`all_splits()`): input-side and
  output-side prefixes, normalization-only, and non-prefix shapes.
- `pflseg/fed.py` — weighted averaging over the shared parameters only.
- `tests/` — `python -m pytest -q tests`
- `get_data.sh` — downloads the public dataset and documents the gated one.
- `partitions/` — client definitions (see below).

## Data

- **FeTS 2022** (main dataset, gated): register and submit the data request at
  https://www.synapse.org/Synapse:syn54079892/wiki/626845
- **MSD Task01 Brain Tumour** (public): fetched by `get_data.sh`.

## Partition files and their sources

- `partitions/fets2022/` — institution partitions from https://github.com/FeTS-AI/Challenge
- `partitions/manthe_folds/` — five-fold federated cross-validation folds from
  https://github.com/MatthisManthe/Benchmark_FeTS2022
- `partitions/fedmepd_brats/` — BraTS 2018/2020 client splits from https://github.com/ccarliu/FedMEPD

These files belong to their authors; check each source's licence before redistributing.
