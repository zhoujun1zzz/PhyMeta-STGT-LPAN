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
python main.py train --domain quasi --model lpan_l_direct --mode smoke
python main.py train --domain quasi --model spatial_gcn --mode smoke
python main.py train --domain mobility --model cnn_gru --mode smoke
python main.py train --domain mobility --model gcn_gru --mode smoke
python main.py train --domain mobility --model phymeta_stgt --mode smoke
```

Smoke mode uses 64 training samples, 16 validation samples, and one epoch. Its outputs are implementation checks, not reportable performance results.

### LPAN-L-Direct baseline

`LPAN-L-Direct` (CLI key: `lpan_l_direct`) is an LPAN-L-derived comparison model adapted to this repository's task contract. It retains dilated residual processing and channel attention, but replaces the original progressive `32 -> 64 -> 128 -> 256` reconstruction with one direct `32 -> 256` resize-and-reconstruction stage. It accepts only the verified official input ordering `(0, 8, ..., 248)` and returns the final dense channel tensor, with shape `[B, 1, 256, 64, 2]` for quasi-static data and `[B, 6, 256, 64, 2]` for mobility data. The ambiguous `lpan_l` alias is intentionally unsupported.

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
| Frozen spatial encoder | `--adaptation frozen_spatial`: time path, domain adapter, and decoder |
| Selective temporal adaptation | `--adaptation selective`: time path and domain adapter |
| Domain adapter only | `--adaptation adapter_only`: domain embedding only |

The trainable sets are strictly nested: `adapter_only < selective < frozen_spatial < full`. Every run records the exact trainable parameter names, top-level modules, counts, and fraction in its metadata; use that artifact when describing an adaptation result.

Recommended target fractions are `0.01`, `0.05`, `0.10`, `0.20`, and `1.0`. For a fixed seed, subsets are nested prefixes of one shuffled index manifest so that every smaller support set is contained in the larger sets.

The `--pretrained` option accepts only a structurally compatible PhyMeta-STGT checkpoint and performs strict model-configuration and state-dictionary validation.

### Hyperparameter search

The `tune` command performs deterministic two-round multi-fidelity search using only the training and validation splits. In the formal protocol, all 12 seeded-random candidates train for 25 epochs, are ranked by their minimum sample-level linear validation NMSE, and the top three resume from their exact epoch-25 checkpoints to a maximum of 100 epochs. Checkpoint and DataLoader RNG states are preserved during promotion. This reduces the nominal per-domain budget from 1,200 to 525 epochs before early stopping. The test split is never read by this command.

```bash
python main.py tune --domain mobility --mode full \
  --strategy random --max-trials 12 --search-seed 2026 \
  --round1-epochs 25 --promote-top-k 3 --epochs 100 \
  --batch-size 32 --eval-batch-size 64 --workers 8 \
  --min-epochs 40 --patience 15 \
  --hidden-values 48,64,96 \
  --graph-layer-values 1,2,3 \
  --head-values 4,8 \
  --dropout-values 0,0.1 \
  --learning-rate-values 1e-4,2e-4,5e-4 \
  --weight-decay-values 0,1e-5 \
  --study-name mobility_hparam_seed123
```

The selected configuration and checkpoint are written to `runs/<study_name>/best_result.json`; `search_plan.json`, `trials.csv`, and `trials.json` retain the round-1 ranking, promotion decisions, resume paths, and final ranking. A smoke search verifies only the round-1 plumbing and is not a reportable optimum.

### Frozen batch and early-stopping protocol

Formal Mobility training uses batch size 32, evaluation batch size 64, eight DataLoader workers, FP32, `OMP_NUM_THREADS=1`, and `MKL_NUM_THREADS=1`. Quasi-static training selects one batch size from `16,32,64,128` using a non-reportable 1,024-sample throughput benchmark; the fastest configuration that does not OOM is frozen for all subsequent Quasi runs.

Every independently trainable neural model—including PhyMeta-STGT, LPAN-L-Direct, and all trainable baselines—uses the same validation-only stopping rule: at most 100 epochs, at least 40 epochs, and patience 15 after the minimum epoch. The test split never participates. Runs retain the best checkpoint even when training stops early. Disable the rule only for an intentional diagnostic with `--no-early-stopping`.

### Parameters, GMACs, and GFLOPs

Use the built-in profiler to compare registered models under exactly the same domain-specific input, batch size, precision, and forward-pass scope:

```bash
python main.py profile --domain mobility --batch-size 1 --device cpu \
  --models lpan_l_direct,edsr_lite,spatial_gcn,cnn_gru,gcn_gru,phymeta_stgt \
  --output runs/mobility_complexity.json

python main.py profile --domain quasi --batch-size 1 --device cpu \
  --models lpan_l_direct,edsr_lite,spatial_gcn,phymeta_stgt \
  --output runs/quasi_complexity.json
```

The command writes both JSON and CSV and prints `parameters`, `GMACs`, and `GFLOPs`. The fixed convention is one FP32 forward pass with batch size one and `1 MAC = 2 FLOPs`; backward, bias, normalization, activation, softmax, indexing, and all spatial/temporal interpolation paths are excluded. This includes sparse-to-dense expansion einsums and framework interpolation kernels for every model. Every training run also records the same canonical complexity profile in `results/final_result.json`.

The checked-in same-condition tables and machine-readable outputs are in [`reports/complexity_summary.md`](reports/complexity_summary.md). For the mobility task, the current PhyMeta-STGT measures `188,360` parameters, `0.114 GMACs`, and `0.228 GFLOPs`; the separately profiled official progressive LPAN-L measures `1,112,904` parameters, `6.338 GMACs`, and `12.676 GFLOPs` under the identical convention.

Do not place THOP MACs, paper-reported FLOPs, and profiler FLOPs in one column without reconciling their definitions. The official progressive LPAN-L and the task-adapted single-stage `lpan_l_direct` must be reported as separate architectures.

### Controlled ablation study

The `ablate` command runs the full model and one-factor variants under the same seed, data fraction, optimizer settings, epoch budget, and Stage-B best hyperparameters. In full mode, `--best-result` is required and every variant automatically inherits `hidden`, graph layers, heads, dropout, learning rate, and weight decay from that validation-only search artifact. The spatial-attention ablation replaces that module with deterministic row-wise grid-aware interpolation; the temporal-attention ablation replaces it with deterministic linear temporal interpolation and nearest-value extension. Other variants remove graph refinement, the domain adapter, coordinate encoding, or individual auxiliary losses. Every result row stores both a stable variant ID and a publication-facing display name/replacement mechanism. Quasi-static studies automatically omit the inapplicable temporal-difference-loss ablation.

```bash
python main.py ablate --domain mobility --mode full --epochs 100 \
  --best-result runs/mobility_hparam_seed123/best_result.json \
  --study-name mobility_ablation_seed123

python main.py ablate --domain mobility --mode full --epochs 100 \
  --best-result runs/mobility_hparam_seed123/best_result.json \
  --variants none,no_spatial_cross_attention,no_graph,no_temporal_attention,\
no_domain_adapter,no_coordinate_encoding,nmse_only,no_charbonnier_loss,\
no_observation_loss,no_temporal_delta_loss \
  --study-name mobility_ablation_selected_seed123
```

Each result is selected independently on validation NMSE. `summary.json` records the inherited Stage-B artifact and hyperparameters and reports the dB change relative to the full model; final test evaluation should be run only after the ablation protocol and checkpoint-selection rule are frozen.

### Automated fast formal pipeline

The complete FP32 protocol is orchestrated by [`scripts/run_fast_formal_protocol.py`](scripts/run_fast_formal_protocol.py). It benchmarks the Quasi batch size, runs validation-only Stage A, executes two-round Stage B for both domains, runs three-seed PhyMeta and baseline studies, executes the Mobility ablations, freezes a checkpoint manifest, and only then opens the independent test split in Stage F. Each step has a separate log and an entry in `pipeline_state.json`; a failed command stops the pipeline without deleting completed artifacts.

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
- LPAN-L-Direct, the task-adapted single-stage baseline (`lpan_l_direct`);
- EDSR-lite;
- Spatial GCN;
- CNN-GRU;
- GCN-GRU; and
- PhyMeta-STGT.

`LPAN-L-Direct` is intentionally a task-adapted, single-stage LPAN-L-derived baseline and must not be labeled as an unchanged reproduction of the official progressive LPAN-L architecture.
The official progressive LPAN-L is not a registered training option in this repository. Its source implementation is handled only by the separate validated profiling script and external baseline pipeline.

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
