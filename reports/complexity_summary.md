# Unified model-complexity results

All entries below were measured with the repository profiler under one fixed
protocol: batch size 1, FP32, one forward pass, model operations only, and
`1 MAC = 2 FLOPs`. Bias, normalization, activation, softmax, indexing and
interpolation are excluded. Parameter counts include all model parameters.

## Mobility task

Canonical input: `[1, 2, 32, 64, 2]`; output: `[1, 6, 256, 64, 2]`.

| Model | Parameters | GMACs | GFLOPs |
|---|---:|---:|---:|
| Official progressive Mobility LPAN-L | 1,112,904 | 6.338 | 12.676 |
| LPAN-L-derived direct 32-to-256 baseline | 669,396 | 2.634 | 5.268 |
| EDSR-lite | 304,716 | 4.985 | 9.970 |
| Spatial GCN | 25,344 | 0.021 | 0.043 |
| CNN-GRU | 32,962 | 1.437 | 2.873 |
| GCN-GRU | 79,552 | 0.072 | 0.143 |
| PhyMeta-STGT | 188,360 | 0.114 | 0.228 |

The official progressive LPAN-L row was imported from
`D:\数据1\LPAN_mobility_baseline\Mobility_LPAN_L1.py` and profiled through an
input/output adapter; it is not the same architecture as `lpan_l_direct`.

## Quasi-static task

Canonical input: `[1, 1, 32, 64, 2]`; output: `[1, 1, 256, 64, 2]`.

| Model | Parameters | GMACs | GFLOPs |
|---|---:|---:|---:|
| LPAN-L-derived direct 32-to-256 baseline | 658,602 | 2.483 | 4.965 |
| EDSR-lite | 297,794 | 4.871 | 9.741 |
| Spatial GCN | 25,344 | 0.006 | 0.013 |
| PhyMeta-STGT | 188,360 | 0.040 | 0.081 |

## Interpretation for the manuscript

- Do not report the draft values `3.60 M / 22.40 GMACs` as the official
  progressive LPAN-L complexity. They do not match either its source model or
  the unified measurement above.
- The LPAN paper's reported `1.09 M / 6.06 GFLOPs` is close to this profiler's
  `1.113 M / 6.338 GMACs`. This is consistent with a MAC-versus-FLOP convention
  difference, but the original profiling code or its exact definition would be
  required to prove that explanation. Preserve the paper number only in a
  clearly labeled “reported by source” column.
- For a same-condition comparison of the implementations used in this project,
  use the measured GMAC and GFLOP columns above. Under this convention the
  current PhyMeta-STGT is `0.114 GMACs / 0.228 GFLOPs` on mobility, not `16.5 G`.

Machine-readable results are stored in `mobility_complexity.json`,
`mobility_complexity.csv`, `quasi_complexity.json`,
`quasi_complexity.csv`, and `official_mobility_lpan_l_complexity.json`.
