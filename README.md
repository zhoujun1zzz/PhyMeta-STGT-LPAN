# PhyMeta-STGT

## Unified Spatio-Temporal Graph Modeling for Low-Pilot RIS Channel Completion

[![CI](https://github.com/zhoujun1zzz/PhyMeta-STGT-LPAN/actions/workflows/ci.yml/badge.svg)](https://github.com/zhoujun1zzz/PhyMeta-STGT-LPAN/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Research code accompanying the manuscript:

> **A Unified Spatio-Temporal Graph Model with Parameter-Efficient Adaptation for Low-Pilot RIS Cascaded Channel Completion**

PhyMeta-STGT provides a unified implementation for quasi-static spatial completion and time-varying spatio-temporal completion on the two LPAN RIS cascaded-channel datasets. The repository includes non-learning baselines, neural baselines, independent and balanced joint training, and parameter-efficient transfer from quasi-static to time-varying channels.

> [!IMPORTANT]
> The manuscript is currently a draft. Dataset files and pretrained checkpoints are not distributed in this repository. Download the datasets from IEEE DataPort and follow the licensing and citation requirements on the corresponding dataset pages.

## Overview

The two LPAN tasks are represented through one sparse-observation-to-dense-query interface:

| Domain | Sparse input | Dense target | Task |
|---|---:|---:|---|
| Quasi-static indoor | `[B, 1, 32, 64, 2]` | `[B, 1, 256, 64, 2]` | Spatial channel completion |
| Time-varying outdoor | `[B, 2, 32, 64, 2]` | `[B, 6, 256, 64, 2]` | Joint spatial-temporal completion |

The final dimension stores the real and imaginary components. The current official-data loader supplies only the 32 measured RIS tokens and emits an all-true `observation_mask`; it does not create a dense zero-filled input or a separate query mask. The mask is therefore an attention-padding interface for variable or padded inputs, not a learned missing-node representation in the current data path.

### PhyMeta-STGT architecture

The current `PhyMetaSTGT` implementation combines:

- sparse observed-token encoding and learned queries for all 256 RIS elements;
- observed-to-all multi-head cross-attention for global access to pilot observations;
- local edge-aware refinement on the four-neighbor RIS grid;
- learned scalar-time embeddings and multi-head temporal cross-attention for both `1 -> 1` and `2 -> 6` tasks;
- a domain-conditioned FiLM adapter;
- a shared node-wise complex-channel decoder; and
- sample-level complex NMSE, Charbonnier, observation-consistency, and temporal-difference objectives.

With the default settings (`hidden=64`, four heads, and two graph layers), the current implementation contains 188,360 parameters. It has no uncertainty head. The adaptation-policy parameter counts are reported by each run and should be taken from its saved configuration rather than copied from an earlier manuscript draft.

## Result provenance

Smoke-test outputs are implementation checks and are not reportable results. Numerical manuscript claims, multi-seed statistics, confidence intervals, latency, memory, and MAC values must be regenerated from complete server-side run directories before release. No draft result is treated as reproduced merely because it appears in a figure or table.

## Installation

Python 3.10 or later is required. CUDA is optional for smoke tests and recommended for full training.

```bash
git clone https://github.com/zhoujun1zzz/PhyMeta-STGT-LPAN.git
cd PhyMeta-STGT-LPAN
python -m pip install -e .
```

For development and tests:

```bash
python -m pip install -e ".[dev]"
pytest
```

## Datasets

- [LPAN quasi-static dataset](https://doi.org/10.21227/3c2t-dz81)
- [LPAN time-varying dataset](https://doi.org/10.21227/pz7h-q132)
- [Official LPAN repository](https://github.com/WiCi-Lab/LPAN)

Recommended directory structure:

```text
data/
|-- quasi/
|   |-- indoorH_LS_Data6users_1B32pilot/
|   |   `-- indoorH_LS_Data6users_1B32pilot.mat
|   |-- indoorH_LSval_Data6users_1B32pilot/
|   |   `-- indoorH_LSval_Data6users_1B32pilot.mat
|   `-- indoorH_LStest_Data6users_1B32pilot/
|       `-- indoorH_LStest_Data6users_1B32pilot.mat
`-- mobility/
    |-- OutdoorH_LS_Data6users_60B32pilot/
    |   `-- OutdoorH_LS_Data6users_60B32pilot.mat
    |-- OutdoorH_LSval_Data6users_60B32pilot/
    |   `-- OutdoorH_LSval_Data6users_60B32pilot.mat
    `-- OutdoorH_LStest_Data6users_60B32pilot/
        `-- OutdoorH_LStest_Data6users_60B32pilot.mat
```

Dataset paths are resolved in the following order:

1. command-specific `--train-path`, `--val-path`, or `--data-path`;
2. `--data-root /path/to/data`;
3. the `LPAN_DATA_ROOT` environment variable; and
4. the repository-local `data/` directory.

Run the read-only dataset audit before training:

```bash
python main.py audit --data-root /path/to/data
```

The audit records file locations, HDF5 keys, raw shapes, dtypes, and canonical interface shapes. It reads metadata and one sample from each split. The loader does not fit or apply an additional normalization transform; it consumes the upstream preprocessed tensors stored in the LPAN files.

### Data semantics

For the official LPAN files, the implementation uses:

```text
observed RIS indices: 0, 8, 16, ..., 248
complex layout: grouped
grid index: 16 * row + column
time-varying pilot blocks: 0, 1
```

The `16 x 16` RIS grid therefore contains observations at columns 0 and 8 of every row under zero-based indexing. The time-varying channel layout is:

```text
Yd = [Re(t1), Re(t2), Im(t1), Im(t2)]
Hd = [Re(t1), ..., Re(t6), Im(t1), ..., Im(t6)]
```

The corresponding command-line options remain configurable for other datasets, but should not be changed for the official LPAN files without an independently verified reason:

```bash
--obs-ris-indices 0,8,16,...,248
--complex-layout grouped
--obs-times 0,1
```

## Reproducing the experiments

### Non-learning baselines

```bash
python main.py interpolate --domain quasi --split validation
python main.py interpolate --domain mobility --split validation \
  --spatial linear --temporal linear

python main.py ridge --domain quasi --max-train 512 --max-val 128
python main.py ridge --domain mobility --max-train 512 --max-val 128
```

Remove the `--max-*` limits for full experiments. Ridge regression reads the test split only when `--test` is supplied.

> [!CAUTION]
> The current linear spatial interpolation baseline operates along the flattened RIS index `0..255`. It is not a two-dimensional grid-aware interpolator and must not be described as physical 2D interpolation in the manuscript.

### Neural-model smoke tests

```bash
python main.py train --domain quasi --model edsr_lite --mode smoke
python main.py train --domain quasi --model spatial_gcn --mode smoke
python main.py train --domain mobility --model cnn_gru --mode smoke
python main.py train --domain mobility --model gcn_gru --mode smoke
python main.py train --domain mobility --model phymeta_stgt --mode smoke
```

Smoke mode uses 64 training samples, 16 validation samples, and one epoch. Its outputs are implementation checks, not reportable performance results.

### Independent training and held-out evaluation

```bash
python main.py train --domain quasi --model phymeta_stgt --mode full \
  --epochs 100 --batch-size 8 --run-name quasi_stgt_seed123

python main.py train --domain mobility --model phymeta_stgt --mode full \
  --epochs 100 --batch-size 2 --run-name mobility_stgt_seed123

python main.py evaluate \
  --checkpoint runs/mobility_stgt_seed123/checkpoints/best_checkpoint.pth \
  --domain mobility --split test --per-snr \
  --output runs/mobility_stgt_seed123/results/independent_test.json
```

Training uses only the official training and validation splits. Test evaluation is a separate command. By default, evaluation inherits the domain, pilot times, observed RIS indices, and complex layout from the checkpoint; conflicting explicit settings are rejected unless `--allow-semantic-override` is intentionally supplied.

### Balanced joint training

```bash
python main.py joint --mode smoke
python main.py joint --mode full --epochs 100 --run-name joint_stgt_seed123
```

Joint training alternates task-homogeneous mini-batches at approximately equal task frequency. It does not replicate quasi-static labels across six artificial time blocks.

### Parameter-efficient transfer

First train a quasi-static PhyMeta-STGT checkpoint, then adapt it using a fixed fraction of the time-varying training split:

```bash
python main.py train --domain mobility --model phymeta_stgt --mode full \
  --pretrained runs/quasi_stgt_seed123/checkpoints/best_checkpoint.pth \
  --fraction 0.05 --adaptation selective --seed 123 \
  --run-name transfer_5pct_selective_seed123
```

Supported adaptation protocols are:

| Protocol | Option |
|---|---|
| Target-only training | omit `--pretrained` and use `--adaptation full` |
| Full fine-tuning | `--adaptation full` |
| Frozen spatial encoder | `--adaptation frozen_spatial` |
| Domain adapter only | `--adaptation adapter_only` |
| Selective temporal adaptation | `--adaptation selective` |

In the current implementation, `selective` and `frozen_spatial` expose the same trainable parameter set: the time encoder, temporal attention, temporal normalization, domain embedding, and decoder. They should not be reported as distinct adaptation methods unless the code is changed to differentiate them. `adapter_only` updates the temporal normalization, domain embedding, and decoder.

Recommended target fractions are `0.01`, `0.05`, `0.10`, `0.20`, and `1.0`. For a fixed seed, subsets are nested prefixes of one shuffled index manifest so that every smaller support set is contained in the larger sets.

The `--pretrained` option accepts only a structurally compatible PhyMeta-STGT checkpoint and performs strict model-configuration and state-dictionary validation.

### Resume a run

```bash
python main.py train ... \
  --resume runs/<run>/checkpoints/last_checkpoint.pth \
  --run-name <run>
```

Checkpoints preserve Python, NumPy, PyTorch CPU/CUDA, and DataLoader random states. Resume configuration is validated before training continues.

## Models and protocols included

- LS coarse-channel interpolation;
- empirical Ridge regression with validation-selected regularization;
- EDSR-lite;
- Spatial GCN;
- CNN-GRU;
- GCN-GRU; and
- PhyMeta-STGT.

Official LPAN and LPAN-L architectures are not registered model options in this repository. They must be integrated or run through a separately validated official pipeline before the manuscript describes them as reproduced baselines under the same protocol.

CNN-GRU and GCN-GRU encode the two pilot blocks and autoregressively decode positions `0..5`. They perform sequence-to-sequence reconstruction of the complete six-block frame, including the two pilot positions; they are not strict future-only predictors beginning at position 2.

## Evaluation and reproducibility

- Model selection uses the training and validation splits only.
- Test data are read only through the independent evaluation entry point.
- NMSE is computed per sample in the linear domain before aggregation and a single conversion to decibels.
- Completion-only metrics report observed versus unobserved RIS elements and pilot versus non-pilot time blocks separately. The current evaluator does not report their intersection (unobserved RIS elements at non-pilot blocks) as a dedicated metric.
- Every run stores the complete command, best and last checkpoints, training history, and JSON results.
- Data, checkpoints, and experiment outputs are excluded by `.gitignore`.
- The time-varying task is frame-internal `2 -> 6` reconstruction; samples are not concatenated into trajectories.

For per-SNR evaluation, declare the test-file grouping explicitly:

```bash
python main.py evaluate \
  --checkpoint runs/<run>/checkpoints/best_checkpoint.pth \
  --domain mobility --split test --per-snr \
  --snr-values=-10,-5,0,5,10,15,20,25,30 \
  --samples-per-snr 1000
```

The command refuses to assign SNR labels when the declared grouping does not match the sample count or when the test data have been sampled or reordered.

## Citation

The manuscript citation will be added after a public preprint or final publication becomes available. In the meantime, please cite the LPAN datasets and their associated paper:

```bibtex
@article{xiao2024multiscale,
  title   = {Multi-Scale Attention Based Channel Estimation for RIS-Aided Massive MIMO Systems},
  author  = {Xiao, Jian and Wang, Ji and Wang, Zhaolin and Xie, Wenwu and Liu, Yuanwei},
  journal = {IEEE Transactions on Wireless Communications},
  volume  = {23},
  number  = {6},
  pages   = {5969--5984},
  year    = {2024},
  doi     = {10.1109/TWC.2023.3329387}
}
```

## License

The source code in this repository is released under the [MIT License](LICENSE). The LPAN datasets and upstream LPAN code are not covered by this license; their original terms apply.

## Acknowledgments

This project builds on the LPAN datasets and the public implementation released by the WiCi Lab. The repository is an independent research implementation and is not affiliated with or endorsed by the original LPAN authors.
