# V1 repaired baselines and compact ablation protocol

This document describes the earlier repair protocol at source commit
`d51be59183456af81591993ae2458f46153718ca`. Its Mobility reuse assumptions are
superseded by `v1_mobility_semantic_correction.md`. Existing formal and
historical run directories remain read-only.

## Frozen task and metric contracts

- Quasi input/target: `[B,1,32,64,2] -> [B,1,256,64,2]`.
- Mobility input/target: `[B,2,32,64,2] -> [B,6,256,64,2]` inside one sample.
- Observed RIS: `0,8,...,248` on `index = 16 * row + column`.
- Corrected Mobility raw layout: interleaved; observed time `[1,4]`, query time
  `[0..5]`. Earlier grouped/`[0,1]` artifacts are legacy-only.
- Main metric: mean of sample-level linear NMSE, converted to dB only after the
  linear mean.
- Training, early stopping and selection use train/validation only. The repair
  runner has no Stage F and contains no command that evaluates test.

## Model repairs

Spatial GCN first calls the verified row-wise physical-grid interpolation for
every observed block. Each of its 256 node tokens contains the interpolated 128
real channel features, the true sparse observation mask and normalized physical
coordinates. The shallow GCN is a residual refinement, not the mechanism that
must diffuse observations across the whole grid. On Mobility, this model is
only a spatial control using anchor interpolation between q1/q4 and nearest
extension outside the anchors.

CNN-GRU and GCN-GRU retain the GRU output sequence. q1/q4 directly use their
observed states. q0/q2/q3/q5 use piecewise-linear anchor states with nearest
extension, followed by one time-conditioned GRUCell update. GCN-GRU uses the
same interpolated dense prior before graph refinement.

## Result classifications

Only Quasi results remain reusable after the semantic correction. Every
Mobility result from this protocol is retained as legacy evidence and must be
rerun under the corrected contract.

The runner verifies source commit, artifact existence, final training result,
checkpoint existence, model name, domain, seed, semantic profile, complex
layout, observed times and RIS indices before accepting a reused neural result.
It writes JSON and JSONL manifests containing absolute source references; it
does not copy checkpoints.

## Compact ablation

Mobility seed 123 runs exactly:

```text
no_spatial_cross_attention
no_graph
no_temporal_attention
no_domain_adapter
no_coordinate_encoding
no_charbonnier_loss
no_observation_loss
no_temporal_delta_loss
```

`none` is reused from the Stage-B/C seed-123 frozen result with
`reference_retrained=false`. `nmse_only` is excluded because it changes several
loss terms at once. No seed 456/789 ablation is part of this protocol.

## Server execution

Use a new output root. Never point `--output-root` at, or below, the reused
formal root.

```bash
python scripts/run_v1_repair_protocol.py \
  --reuse-formal-root /path/to/runs/formal_d51be59_20260817_005210 \
  --output-root runs/v1_repair_<repair_commit> \
  --data-root /path/to/data --phase manifest

python scripts/run_v1_repair_protocol.py \
  --reuse-formal-root /path/to/runs/formal_d51be59_20260817_005210 \
  --output-root runs/v1_repair_<repair_commit> \
  --data-root /path/to/data --device cuda --phase pytest

python scripts/run_v1_repair_protocol.py \
  --reuse-formal-root /path/to/runs/formal_d51be59_20260817_005210 \
  --output-root runs/v1_repair_<repair_commit> \
  --data-root /path/to/data --device cuda --phase smoke

python scripts/run_v1_repair_protocol.py \
  --reuse-formal-root /path/to/runs/formal_d51be59_20260817_005210 \
  --output-root runs/v1_repair_<repair_commit> \
  --data-root /path/to/data --device cuda --phase seed123
```

Seed 123 must have finite validation metrics and pass the default sub-0-dB gate.
If it remains anomalously poor, stop and audit. Otherwise continue:

```bash
python scripts/run_v1_repair_protocol.py \
  --reuse-formal-root /path/to/runs/formal_d51be59_20260817_005210 \
  --output-root runs/v1_repair_<repair_commit> \
  --data-root /path/to/data --device cuda --phase remaining

python scripts/run_v1_repair_protocol.py \
  --reuse-formal-root /path/to/runs/formal_d51be59_20260817_005210 \
  --output-root runs/v1_repair_<repair_commit> \
  --data-root /path/to/data --device cuda --phase ablation
```

Add `--include-mobility-spatial-control` consistently to all relevant phases
only if the supplementary spatial-only control is required. The final files are
`result_manifest.json`, `result_manifest.jsonl`,
`result_classification.json`, per-step logs and `pipeline_state.json`.

Independent test remains locked until V1/V3 architectures, checkpoints and
protocols are frozen under a separately authorized test procedure.
