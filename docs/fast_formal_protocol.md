# Fast formal experiment protocol

This document freezes the accelerated paper experiment protocol after server validation of commit `627657b`. It does not change the two LPAN tasks, model inputs or targets, sample-level NMSE definition, principal comparison models, or test-isolation rule.

## Invariants

- FP32 only; BF16/AMP is not enabled in this protocol.
- Mobility remains within-sample `2 -> 6` reconstruction, not future-frame or trajectory prediction.
- Hyperparameter search, early stopping, promotion, and checkpoint selection use training and validation data only.
- The independent test split is opened only after Stage F writes `frozen_model_manifest.json`. That manifest also freezes the Stage-A Ridge regularization values and interpolation settings before any test command runs.
- Joint training is excluded from the main pipeline.
- `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and eight DataLoader workers are used throughout server training.

## Throughput settings

Mobility is frozen at training batch 32 and evaluation batch 64, based on the server benchmark of the heaviest candidate. Quasi runs a one-time, non-reportable benchmark over 1,024 real training samples with candidate batches `16,32,64,128`. The highest-throughput candidate that does not OOM is then used by every formal Quasi neural run.

The benchmark does not read validation or test data and does not produce a paper metric.

## Stage B: two-round search

For each domain, the deterministic candidate generator keeps random search seed 2026, shared training seed 123, and the existing 12-candidate search space.

```text
Round 1: 12 candidates x 25 epochs
             |
             +-- rank by best validation linear NMSE
             |
Round 2: top 3 resume from epoch-25 last_checkpoint.pth
             |
             +-- train to at most epoch 100
```

Promotion uses the same run directory and restores model, optimizer, Python, NumPy, PyTorch CPU/CUDA, and DataLoader generator states. `search_plan.json` records the protocol; `trials.json` and `trials.csv` record both rankings and promotion status; `best_result.json` is the only Stage-B artifact consumed by later stages.

The nominal budget is `12 x 25 + 3 x 75 = 525` epochs per domain, 43.75% of the previous `12 x 100` budget.

## Unified early stopping

All independently trained neural models and ablations use:

```text
maximum epochs: 100
minimum epochs: 40
patience:       15
metric:         validation sample-level linear NMSE
```

Patience begins after epoch 40, so the earliest stopping point is epoch 55. The best checkpoint is always retained. History stores the improvement flag and stale-epoch counter; resume reconstructs the counter from the auditable history and rejects a mismatch with checkpoint `best_nmse`.

## Pipeline stages

```text
Batch benchmark
  -> Stage A: complexity + interpolation/ridge validation
  -> Stage B: two-round Quasi and Mobility search
  -> Stage C: best PhyMeta-STGT at seeds 123/456/789
  -> Stage D: all domain-applicable trainable baselines at 123/456/789
  -> Stage E: all Mobility ablations at 123; structural ablations at 456/789
  -> freeze model/checkpoint, Ridge, and interpolation manifest
  -> Stage F: independent test, per-SNR, Ridge test, mean/std, complexity links
```

Quasi baselines are LPAN-L-Direct, EDSR-lite, and Spatial GCN. Mobility additionally includes CNN-GRU and GCN-GRU. Stage E covers spatial cross-attention, graph, temporal attention, domain adapter, coordinate encoding, and the registered loss ablations. Structural variants are repeated for seeds 456 and 789.

## Operation and recovery

Run from the repository checkout whose commit will be reported:

```bash
mkdir -p runs/formal_fast/logs
nohup python -u scripts/run_fast_formal_protocol.py \
  --data-root /path/to/data \
  > runs/formal_fast/logs/formal_pipeline.log 2>&1 &
```

Every command receives a dedicated log under `runs/formal_fast/logs/`. `pipeline_state.json` is updated atomically before and after every step. A nonzero exit or missing expected artifact stops the pipeline and preserves prior outputs. Re-running skips completed steps whose outputs still exist. The script refuses to continue a state directory created by another commit. Stage F reads each domain's selected Ridge regularization from Stage A and passes only that frozen value to the Ridge test command.

If a step fails partway through a multi-run study, inspect its dedicated log and study artifacts before retrying; do not delete completed evidence blindly.

## Validation before launch

After merging the optimization commit, run:

1. the full pytest suite;
2. a Quasi and a Mobility PhyMeta-STGT CUDA smoke;
3. the 1,024-sample Quasi batch benchmark;
4. one short resume check confirming the new early-stopping metadata remains compatible across promotion.

The previous data audit, semantic verification, and already successful model smoke tests need not be repeated solely because of this throughput patch. Formal Stage A/B must not begin until the new CUDA smoke and benchmark succeed.
