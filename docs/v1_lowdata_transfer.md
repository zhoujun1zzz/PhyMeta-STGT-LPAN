# V1 Low-Data Target Transfer Protocol

## Scope

This protocol fills Table 2 only. The source is Quasi-static, the target is
Mobility, and the target task remains frame-internal `2 observed blocks -> 6
target blocks`. Formal runs use seed 123. The runner never evaluates the test
split; validation selects the best checkpoint.

The theoretical matrix contains 25 cells: five methods by five Mobility-train
fractions (`1%, 5%, 10%, 20%, 100%`). Results are reported as single NMSE-dB
values. Do not report a standard deviation unless additional formal seeds are
run under a separately frozen protocol.

## Implementation audit

Before this protocol, V1 provided `full`, `frozen_spatial`, `adapter_only`, and
`selective` policies. The optimizer already filtered parameters by
`requires_grad`; the train fraction affected Mobility train only; validation
was complete and independent; and training did not call the test evaluator.

The gaps were:

- `full` ambiguously represented both random target-only training and
  checkpoint-based full fine-tuning;
- `selective` trained the temporal path and adapter but not the prediction
  decoder, so it was not the manuscript's Adapter + head method;
- source checkpoints were architecture-checked but their Quasi domain and
  SHA256 provenance were not enforced/recorded;
- frozen modules were not explicitly returned to evaluation mode after
  `model.train()`;
- no Table 2 runner, exact-match reuse validator, safe completed-run check, or
  single-seed summary existed.

Legacy options remain accepted for old command reproduction. New Table 2 runs
must use the five names below.

## Exact parameter policies

Counts below are computed for the default V1 configuration (`hidden=64`, four
heads, two graph layers), total 188,360 parameters. Runtime manifests are the
source of truth if the model configuration changes.

| Method | Trainable modules | Frozen modules | Trainable | Percent |
|---|---|---|---:|---:|
| `scratch` | all modules | none | 188,360 | 100.000000% |
| `full_finetune` | all modules | none | 188,360 | 100.000000% |
| `frozen_spatial` | time encoder, temporal attention/norm, domain embedding, decoder | channel/coordinate encoders, node query, spatial cross-attention, graph layers | 46,272 | 24.565725% |
| `domain_adapter_only` | domain embedding (FiLM) | all other modules | 256 | 0.135910% |
| `adapter_head` | domain embedding (FiLM), prediction decoder | all shared spatial and temporal modules | 25,216 | 13.387131% |

`scratch` is random initialization, loads no source checkpoint, and trains all
parameters. The other four methods require the exact Quasi seed-123 source
checkpoint. Optimizers contain only parameters with `requires_grad=True`.

## Deterministic subsets and provenance

One NumPy permutation of the complete Mobility train set is generated with
seed 123. Every fraction is a prefix of that permutation, so
`1% subset 5% subset 10% subset 20% subset 100%`; methods at the same fraction
therefore share the same index SHA256. Validation is never fractioned in formal
runs.

Each manifest records the git state, source checkpoint path/SHA256 and source
metadata, dataset identities, subset size/hash, model and optimization
configuration, trainable counts, best validation metric/epoch, measured
adaptation time, runtime versions, and `test_split_used=false`.

After the Mobility semantic correction, every manifest also records the full
canonical semantic contract (`semantic_profile`, `complex_layout`, q1/q4
`obs_time_index`, q0..q5 `query_time`, and observed RIS indices) plus its SHA256
fingerprint. Legacy grouped/q0-q1 manifests fail exact reuse validation. The
completed legacy 25-cell run remains immutable but every cell must be rerun;
its earlier frozen-strategy interpretation is withdrawn pending correction.

Historical reuse requires an exact match for implementation commit, source
checkpoint, datasets, fraction and subset hash, seed, optimizer/scheduler,
training budget, validation protocol, metric definition, completed checkpoint,
and result provenance. A missing or different field makes the run non-reusable.
If an older source checkpoint does not embed its originating git commit, pass
the commit recorded by its formal pipeline as `--source-git-commit <sha>`; the
runner refuses an untraceable source.

The local development audit found no completed transfer manifest satisfying
this contract. Thus the current plan is 25 theoretical cells, 0 reusable cells,
and 25 new formal runs. Server-side historical roots can be supplied with
repeated `--history-root`; the same strict validator decides reuse there.

## Dry-run and smoke

Show all 25 planned cells without training:

```bash
python scripts/run_v1_lowdata_transfer.py --dry-run \
  --source-checkpoint /path/to/quasi_seed123/best_checkpoint.pth \
  --data-root /path/to/data
```

Run a one-epoch capped smoke for all five policies:

```bash
python scripts/run_v1_lowdata_transfer.py --smoke \
  --source-checkpoint /path/to/quasi_seed123/best_checkpoint.pth \
  --run-root runs/v1_lowdata_transfer_smoke \
  --data-root /path/to/data --device cuda --max-train 8 --max-val 4
```

Smoke artifacts are diagnostic only and must not be reused as formal cells.

## Formal Stage 1 (five 5% runs)

```bash
python scripts/run_v1_lowdata_transfer.py --stage 1 \
  --source-checkpoint /path/to/quasi_seed123/best_checkpoint.pth \
  --run-root runs/v1_lowdata_transfer_seed123_<commit>_<timestamp> \
  --data-root /path/to/data --device cuda
```

Inspect the five manifests, validation curves, trainable groups, checkpoint
files, and 5% timing before continuing.

## Formal Stage 2 (remaining 20 runs)

Use the same run root after Stage 1 passes:

```bash
python scripts/run_v1_lowdata_transfer.py --stage 2 \
  --source-checkpoint /path/to/quasi_seed123/best_checkpoint.pth \
  --run-root runs/v1_lowdata_transfer_seed123_<commit>_<timestamp> \
  --data-root /path/to/data --device cuda
```

Use `--resume` only for reviewed incomplete directories. A completed cell is
skipped only when its manifest, best checkpoint, result, method/fraction/seed,
subset hash, source hash, and full protocol metadata validate. Existing but
invalid directories fail rather than being overwritten.

## Summary

```bash
python scripts/summarize_lowdata_transfer.py \
  --run-root runs/v1_lowdata_transfer_seed123_<commit>_<timestamp>
```

This writes `results/raw_runs.csv`, `table2_seed123.csv`,
`table2_seed123.md`, and `table2_seed123.json`. Missing cells remain blank and
the Markdown note explicitly states that all results use seed 123.

No test split was used during implementation/development.
