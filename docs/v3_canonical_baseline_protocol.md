# V1 canonical Mobility baseline protocol

This protocol changes the V1 repository only. PriST-RIS/V3 code and artifacts
are outside its modification boundary.

## Frozen semantic contract

The formal profile is `v3_mobility_q0_q3`, version
`mobility_q0_q3_v1`:

- raw `Yd`: `[Re(q0), Re(q3), Im(q0), Im(q3)]`;
- raw `Hd`: `[Re(q0)..Re(q5), Im(q0)..Im(q5)]`;
- observed times: q0 and q3; queries: q0 through q5;
- observed RIS indices: `0,8,...,248`;
- metric: per-sample linear NMSE, sample mean, then one dB conversion.

The historical `official_lpan`/`legacy_v1` profiles preserve the previous
interleaved q1/q4 interpretation. They are provenance paths, not formal
defaults.

## LPAN reuse gate

`audit-v3-baseline-semantics` uses the same deterministic 64 TRAIN and 64
VALIDATION sample IDs for both decoders. For LPAN and LPAN-L it requires:

1. bit-exact effective input equality;
2. bit-exact effective target equality;
3. progressive Charbonnier equality within `2e-6` FP32 reduction tolerance;
4. per-sample NMSE invariance within `2e-6` FP32 reduction tolerance;
5. independence from observation/query time labels;
6. strict three-seed checkpoint loading with no missing or unexpected keys.

Only all-pass evidence yields `REUSE_VERIFIED`. Missing data, missing
checkpoints, or any failed check yields `RERUN_REQUIRED`. The adapter only
reorders tensors and adds no state-dict parameters. Formal progressive training
uses the three-scale Charbonnier loss only; observation consistency, temporal
delta, and time labels are not consumed.

Reused checkpoints must still be evaluated with `evaluate-v3-baselines` on the
complete canonical VALIDATION split. Results record checkpoint SHA256, Git
HEAD, contract/fingerprint, dataset provenance, grouped metrics (overall,
q0..q5, q0/q3 anchors, non-pilots, interpolation, extrapolation), and
`test_split_used: false`.

## Formal matrix and execution

The matrix contains only LPAN, LPAN-L, EDSR-lite, and CNN-GRU, each at seeds
123, 456, and 789. GCN-GRU code and historical results remain preserved but are
excluded. EDSR-lite and CNN-GRU always retrain under the canonical profile.

Planning writes a JSON manifest and `launch_3gpu.sh`. GPU 0/1/2 own seeds
123/456/789 respectively, and each worker executes its models serially. Every
training-history row records epoch, training loss, validation linear/dB NMSE,
learning rate, epoch seconds, and cumulative wall-clock seconds.

No command in this protocol reads TEST. Independent TEST evaluation remains a
separate, explicitly authorized post-freeze action.
