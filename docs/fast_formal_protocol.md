# Fast formal experiment protocol

> Historical protocol notice: this document describes the pre-repair pipeline.
> Its Spatial GCN, CNN-GRU, GCN-GRU and 22-run Stage E are invalidated for the
> final V1 comparison. Use `docs/v1_repair_protocol.md` and
> `scripts/run_v1_repair_protocol.py` for the repaired, validation-only flow.

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

## Stage B: targeted boundary search

Each domain runs independently with shared training seed 123 and validation-only selection.

```text
Capacity: hidden 96/128/160 x 20 epochs at LR 5e-4
          -> rank epochs 16-20 median linear NMSE
Learning rate: 5e-4/8e-4/1e-3 x 40 epochs from scratch
          -> rank epochs 31-40 median linear NMSE
Final: winner resumes from epoch-40 last_checkpoint.pth
          -> train to at most epoch 100
```

Final continuation restores model, optimizer, Python, NumPy, PyTorch CPU/CUDA, DataLoader generator states, and history. Phase A and B use fixed budgets without early truncation. `best_result.json` remains compatible with later stages and records boundary hits without automatically extending the search.

The nominal budget is `3 x 20 + 3 x 40 + 60 = 240` epochs per domain. The historical two-round protocol remains available through the CLI for reproduction.

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
  -> Stage B: targeted-boundary Quasi and Mobility search
  -> Stage C: best PhyMeta-STGT at seeds 123/456/789
  -> Stage D: all domain-applicable trainable baselines at 123/456/789
  -> Stage E: all Mobility ablations at 123; structural ablations at 456/789
  -> freeze model/checkpoint, Ridge, and interpolation manifest
  -> Stage F: independent test, per-SNR, Ridge test, mean/std, complexity links
```

Quasi baselines are progressive LPAN, progressive LPAN-L, EDSR-lite, and Spatial GCN. Mobility additionally includes CNN-GRU and GCN-GRU; its LPAN is explicitly channel-adapted while LPAN-L follows the public Mobility structure. `--exclude-mobility-adapted-lpan` can omit only the adapted LPAN. LPAN-L-Direct is supplementary and is not in Stage D by default.

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
