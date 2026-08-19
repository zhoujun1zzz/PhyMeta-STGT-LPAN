# PhyMeta-STGT

## Unified Spatio-Temporal Graph Modeling for Low-Pilot RIS Channel Completion

[![CI](https://github.com/zhoujun1zzz/PhyMeta-STGT-LPAN/actions/workflows/ci.yml/badge.svg)](https://github.com/zhoujun1zzz/PhyMeta-STGT-LPAN/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Research code accompanying the manuscript:

> **A Unified Spatio-Temporal Graph Model with Parameter-Efficient Adaptation for Low-Pilot RIS Cascaded Channel Completion**

PhyMeta-STGT provides a unified implementation for quasi-static spatial completion and time-varying spatio-temporal completion on the two LPAN RIS cascaded-channel datasets. The repository includes non-learning baselines, neural baselines, independent and balanced joint training, and parameter-efficient transfer from quasi-static to time-varying channels.

The historical identifier `PhyMeta-STGT` does **not** denote meta-learning in this repository. The implemented model uses shared spatio-temporal parameters, physical RIS structure, domain conditioning, and ordinary supervised pretraining/fine-tuning; it has no episodic meta-training or inner/outer optimization loop.

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

The two recent protocol-hardening rounds are summarized in Chinese in [`docs/two_round_modification_summary_zh.md`](docs/two_round_modification_summary_zh.md).

Verified baseline evidence that predates this unified repository is recorded in [`docs/experiment_evidence.md`](docs/experiment_evidence.md). In particular, the official progressive Mobility LPAN-L run has 1,112,904 parameters, validation NMSE `-21.6269 dB`, and frozen 9000-frame independent-test NMSE `-19.7268 dB`. These values belong to the progressive official architecture and must not be assigned to `lpan_l_direct`.

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

The audit records file locations, byte sizes, HDF5 keys, raw shapes, dtypes, full sample counts, and canonical interface shapes. It reads metadata and one sample from each split. Under the default `official_lpan` profile, the Mobility counts must be exactly 20,000 training, 1,800 validation, and 9,000 test samples. The loader does not fit or apply an additional normalization transform; it consumes the upstream preprocessed tensors stored in the LPAN files.

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

These labels are enforced by the default `--semantic-profile official_lpan`; changing only a label cannot silently reinterpret the same raw columns. Use `--semantic-profile custom` only for a separately rearranged or regenerated dataset whose column meanings have been independently verified:

```bash
--obs-ris-indices 0,8,16,...,248
--complex-layout grouped
--obs-times 0,1
--semantic-profile custom
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

Spatial interpolation is grid-aware and row-wise on the physical `16 x 16` RIS. A query uses observations from its own row only; linear mode interpolates between observed columns and applies nearest-value extension outside their range. Nearest mode also remains within the same row. It never interpolates across a flattened row boundary.

### Neural-model smoke tests

```bash
python main.py train --domain quasi --model edsr_lite --mode smoke
python main.py train --domain quasi --model lpan_progressive --mode smoke
python main.py train --domain quasi --model lpan_l_progressive --mode smoke
python main.py train --domain quasi --model lpan_l_direct --mode smoke
python main.py train --domain quasi --model spatial_gcn --mode smoke
python main.py train --domain mobility --model cnn_gru --mode smoke
python main.py train --domain mobility --model gcn_gru --mode smoke
python main.py train --domain mobility --model phymeta_stgt --mode smoke
```

Smoke mode uses 64 training samples, 16 validation samples, and one epoch. Its outputs are implementation checks, not reportable performance results.

Spatial GCN now initializes every dense RIS node from the verified row-wise
physical-grid interpolation, concatenates the sparse observation mask and
normalized coordinates, and uses the GCN only as a residual graph refinement.
On Mobility it is a spatial-only control: query 0 uses observed block 0 and all
queries from 1 onward hold the spatial output of observed block 1. CNN-GRU and
GCN-GRU directly decode the GRU states aligned with observed queries 0 and 1;
their recurrent decoder is invoked only four times for future queries 2--5.

### Progressive LPAN and LPAN-L baselines

The main LPAN baselines are `lpan_progressive` (displayed as LPAN) and
`lpan_l_progressive` (displayed as LPAN-L). Both execute the complete
`32 -> 64 -> 128 -> 256` reconstruction path. Training preserves the public
multi-scale targets at RIS indices `1::4`, `1::2`, and the full grid, using
equal-weight FP32 Charbonnier losses. Validation, test, and complexity profiling
continue to use only the final HR8 unified tensor. Mobility LPAN-L follows
`Mobility_LPAN_L1.py`; Mobility LPAN is explicitly recorded as a channel-adapted
LPAN architecture rather than an official time-varying model.

```bash
python main.py train --domain quasi --model lpan_progressive --mode full \
  --epochs 100 --training-profile lpan_public_code
python main.py train --domain mobility --model lpan_l_progressive --mode full \
  --epochs 100 --training-profile lpan_public_code
```

The public LPAN-L same-input loops are mathematically redundant: each iteration
overwrites the output with `block(x)`. The registered model evaluates the block
once, records this equivalence in checkpoint metadata, and does not reinterpret
the loop as recursive residual processing.

### Supplementary LPAN-L-Direct baseline

`LPAN-L-Direct` (CLI key: `lpan_l_direct`) is retained only as a supplementary task-adapted comparison. It is not used as the main LPAN-L baseline.

```bash
python main.py train --domain quasi --model lpan_l_direct --mode full \
  --epochs 100 --run-name quasi_lpan_l_direct_seed123

python main.py train --domain mobility --model lpan_l_direct --mode full \
  --epochs 100 --run-name mobility_lpan_l_direct_seed123
```

### Independent training and held-out evaluation

```bash
python main.py train --domain quasi --model phymeta_stgt --mode full \
  --epochs 100 --batch-size 32 --eval-batch-size 64 --workers 8 \
  --min-epochs 40 --patience 15 --run-name quasi_stgt_seed123

python main.py train --domain mobility --model phymeta_stgt --mode full \
  --epochs 100 --batch-size 32 --eval-batch-size 64 --workers 8 \
  --min-epochs 40 --patience 15 --run-name mobility_stgt_seed123

python main.py evaluate \
  --checkpoint runs/mobility_stgt_seed123/checkpoints/best_checkpoint.pth \
  --domain mobility --split test --per-snr \
  --output runs/mobility_stgt_seed123/results/independent_test.json
```

Training uses only the official training and validation splits. Test evaluation is a separate command. By default, evaluation inherits the domain, pilot times, observed RIS indices, and complex layout from the checkpoint; conflicting explicit settings are rejected unless `--allow-semantic-override` is intentionally supplied.

### Optional balanced joint training

```bash
python main.py joint --mode smoke
python main.py joint --mode full --epochs 100 --run-name joint_stgt_seed123
```

Joint training alternates task-homogeneous mini-batches at approximately equal task frequency. It does not replicate quasi-static labels across six artificial time blocks.

Joint training is not part of the frozen fast formal protocol. It remains optional until its resume/recovery path is brought to the same standard as single-domain training and the manuscript explicitly requires the result.

### Parameter-efficient transfer

The frozen Table 2 protocol uses Quasi-static pretraining followed by Mobility
adaptation at `1%, 5%, 10%, 20%, 100%`, with seed 123 only. Use the dedicated
runner rather than assembling individual commands:

```bash
python scripts/run_v1_lowdata_transfer.py --stage 1 \
  --source-checkpoint /path/to/quasi_seed123/best_checkpoint.pth \
  --run-root runs/v1_lowdata_transfer_seed123_<commit>_<timestamp> \
  --data-root /path/to/data --device cuda
```

The five formal policies are `scratch`, `full_finetune`, `frozen_spatial`,
`domain_adapter_only`, and `adapter_head`. Scratch rejects a source checkpoint;
all other policies require one. `adapter_head` trains the domain-conditioned
FiLM embedding and prediction decoder, while `frozen_spatial` additionally
trains the temporal stack. The runner shares one deterministic nested subset
per fraction across all methods, never opens the test split, validates exact
historical reuse, and records measured counts and timing in manifests.

Complete commands, parameter boundaries, resume rules, and the single-seed
summary format are documented in
[`docs/v1_lowdata_transfer.md`](docs/v1_lowdata_transfer.md).

### Hyperparameter search

The historical `two_round_validation_promotion` protocol remains the default for
reproducibility. The new formal protocol is `targeted_boundary`: it first trains
hidden sizes 96/128/160 for a fixed 20 epochs at LR `5e-4`, ranks the epoch
16--20 median in linear NMSE, then trains LR `5e-4`/`8e-4`/`1e-3` from scratch
for 40 epochs at the selected hidden size and ranks the epoch 31--40 linear-NMSE
median. Mean, standard deviation, and best validation NMSE are diagnostics. The
winner resumes exactly from its epoch-40 `last_checkpoint.pth` to at most epoch
100. The test split is never read.

```bash
python main.py tune --domain mobility --mode full \
  --tuning-protocol targeted_boundary --epochs 100 \
  --batch-size 32 --eval-batch-size 64 --workers 8 \
  --min-epochs 40 --patience 15 \
  --study-name mobility_targeted_boundary_seed123
```

The selected configuration and checkpoint are written to `runs/<study_name>/best_result.json`; `search_plan.json`, `trials.csv`, and `trials.json` retain the round-1 ranking, promotion decisions, resume paths, and final ranking. A smoke search verifies only the round-1 plumbing and is not a reportable optimum.

### Frozen batch and early-stopping protocol

Formal Mobility training uses batch size 32, evaluation batch size 64, eight DataLoader workers, FP32, `OMP_NUM_THREADS=1`, and `MKL_NUM_THREADS=1`. Quasi-static training selects one batch size from `16,32,64,128` using a non-reportable 1,024-sample throughput benchmark; the fastest configuration that does not OOM is frozen for all subsequent Quasi runs.

Every independently trainable neural model uses the same validation-only stopping rule: at most 100 epochs, at least 40 epochs, and patience 15 after the minimum epoch. Progressive LPAN models use their public-code-derived AdamW/cosine/multi-scale-loss profile while keeping this unified selection rule. The test split never participates.

### Parameters, GMACs, and GFLOPs

Use the built-in profiler to compare registered models under exactly the same domain-specific input, batch size, precision, and forward-pass scope:

```bash
python main.py profile --domain mobility --batch-size 1 --device cpu \
  --models lpan_progressive,lpan_l_progressive,edsr_lite,spatial_gcn,cnn_gru,gcn_gru,phymeta_stgt \
  --output runs/mobility_complexity.json

python main.py profile --domain quasi --batch-size 1 --device cpu \
  --models lpan_progressive,lpan_l_progressive,edsr_lite,spatial_gcn,phymeta_stgt \
  --output runs/quasi_complexity.json
```

The command writes both JSON and CSV and prints `parameters`, `GMACs`, and `GFLOPs`. The fixed convention is one FP32 forward pass with batch size one and `1 MAC = 2 FLOPs`; backward, bias, normalization, activation, softmax, indexing, and all spatial/temporal interpolation paths are excluded. This includes sparse-to-dense expansion einsums and framework interpolation kernels for every model. Every training run also records the same canonical complexity profile in `results/final_result.json`.

The checked-in same-condition tables and machine-readable outputs are in [`reports/complexity_summary.md`](reports/complexity_summary.md). For the mobility task, the current PhyMeta-STGT measures `188,360` parameters, `0.114 GMACs`, and `0.228 GFLOPs`; the separately profiled official progressive LPAN-L measures `1,112,904` parameters, `6.338 GMACs`, and `12.676 GFLOPs` under the identical convention.

Do not place THOP MACs, paper-reported FLOPs, and profiler FLOPs in one column without reconciling their definitions. The official progressive LPAN-L and the task-adapted single-stage `lpan_l_direct` must be reported as separate architectures.

### Compact controlled ablation study

The repaired V1 protocol runs eight strict one-factor Mobility ablations at seed
123 only. The full-model `none` reference is reused from the frozen Stage-B/C
seed-123 result and is not trained again. `nmse_only` is retained only for
historical CLI reproduction and is excluded from the compact formal protocol
because it removes several losses simultaneously. Every ablation inherits the
same Stage-B hyperparameters and uses validation-only selection.

```bash
python main.py ablate --domain mobility --mode full --epochs 100 \
  --best-result runs/mobility_hparam_seed123/best_result.json \
  --reuse-full-reference \
  --variants no_spatial_cross_attention,no_graph,no_temporal_attention,\
no_domain_adapter,no_coordinate_encoding,no_charbonnier_loss,\
no_observation_loss,no_temporal_delta_loss \
  --seed 123 --study-name mobility_compact_ablation_seed123
```

Each result is selected independently on validation NMSE. `summary.json` records the inherited Stage-B artifact and hyperparameters and reports the dB change relative to the full model; final test evaluation should be run only after the ablation protocol and checkpoint-selection rule are frozen.

### V1 repaired-baseline protocol

Use [`scripts/run_v1_repair_protocol.py`](scripts/run_v1_repair_protocol.py) to
validate and reference the trusted artifacts under
`runs/formal_d51be59_20260817_005210` without copying or modifying them. It
reruns only fixed Quasi Spatial GCN and fixed Mobility CNN-GRU/GCN-GRU, gates
seed 456/789 on a finite sub-0-dB seed-123 validation result, and runs the eight
compact ablations. It has no Stage F and never opens the test split.

```bash
python scripts/run_v1_repair_protocol.py \
  --reuse-formal-root /path/to/runs/formal_d51be59_20260817_005210 \
  --output-root runs/v1_repair_<new_commit> \
  --data-root /path/to/data --device cuda --phase smoke

python scripts/run_v1_repair_protocol.py \
  --reuse-formal-root /path/to/runs/formal_d51be59_20260817_005210 \
  --output-root runs/v1_repair_<new_commit> \
  --data-root /path/to/data --device cuda --phase seed123
```

Continue with `--phase remaining` only after seed 123 passes the gate, then use
`--phase ablation`. The complete provenance and server commands are documented
in [`docs/v1_repair_protocol.md`](docs/v1_repair_protocol.md).

### Historical automated fast formal pipeline

The pre-repair FP32 protocol is orchestrated by [`scripts/run_fast_formal_protocol.py`](scripts/run_fast_formal_protocol.py). It remains available for historical reproduction, but its old Spatial GCN/CNN-GRU/GCN-GRU and 22-run Stage E must not be used in the repaired final comparison.

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p runs/formal_fast/logs
nohup python -u scripts/run_fast_formal_protocol.py \
  --data-root /path/to/data \
  > runs/formal_fast/logs/formal_pipeline.log 2>&1 &
```

Do not reuse one pipeline output directory across different commits. Audit and semantic verification are intentionally not repeated by this script because they must already have passed before formal execution. The frozen details and server acceptance checks are documented in [`docs/fast_formal_protocol.md`](docs/fast_formal_protocol.md).

### Resume a run

```bash
python main.py train ... \
  --resume runs/<run>/checkpoints/last_checkpoint.pth \
  --run-name <run>
```

Checkpoints preserve Python, NumPy, PyTorch CPU/CUDA, and DataLoader random states. They are written to a temporary file and atomically replace the final checkpoint only after serialization succeeds. During resume, RNG tensors are explicitly normalized to contiguous CPU byte tensors before being passed to the PyTorch RNG APIs; this remains correct even when the full checkpoint is loaded with `map_location=cuda`. Resume configuration is validated before training continues. If `training_history.csv` is ahead of the selected checkpoint (for example, after an interrupted save), it is truncated to the matching checkpoint epoch and the repair is recorded in `recovery.log`; missing, duplicated, unordered, or irreconcilable history still fails loudly.

## Models and protocols included

- LS coarse-channel interpolation;
- empirical Ridge regression with validation-selected regularization;
- progressive LPAN and LPAN-L (`lpan_progressive`, `lpan_l_progressive`);
- LPAN-L-Direct as a supplementary task-adapted baseline (`lpan_l_direct`);
- EDSR-lite;
- Spatial GCN;
- CNN-GRU;
- GCN-GRU; and
- PhyMeta-STGT.

`LPAN-L-Direct` must not be labeled as LPAN-L. The registered main LPAN-L option is the progressive model; historical external evidence and new in-repository runs must retain separate provenance.

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
