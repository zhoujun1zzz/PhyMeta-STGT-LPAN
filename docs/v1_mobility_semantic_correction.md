# V1 Mobility semantic correction

## Correct contract

The official Mobility MAT files store complex channels interleaved by time:

```text
Yd: Re(q1), Im(q1), Re(q4), Im(q4)
Hd: Re(q0), Im(q0), ..., Re(q5), Im(q5)
```

The internal representation is `[B,T,RIS,BS,2]`, with the last dimension
holding real and imaginary components. The two pilots are q1 and q4, all six
queries q0..q5 are reconstructed, and observed RIS indices are
`0,8,...,248` under `index = 16*row + column`.

## Root cause and independent validation

The earlier verifier decoded each candidate layout but assumed pilot0 mapped to
q0 and pilot1 to q1 before scoring the layouts. In addition, the validation raw
file satisfies `H[0] == H[2]` and `H[6] == H[8]`. Under grouped decoding these
equalities manufacture an exact duplicate complex target block, so the circular
q0-first check can prefer a physically incorrect grouped interpretation.

The corrected verifier streams the file in bounded chunks and jointly searches
both Yd layouts, both Hd layouts, and all 15 increasing two-of-six pilot
position pairs. Complex correlation and sample-level normalized error are the
primary mapping evidence. A fixed small guardrail rejects exact duplicate
complex-block degeneracy; it does not encode q1/q4.

On all 1800 validation samples:

```text
SHA256   86b1c8320513bd12acd69c5fa85b58710868784767a1b8d6f7ec3a2b996648b9
Yd       (4, 32, 64, 1800)
Hd       (12, 256, 64, 1800)
best     interleaved, pilot positions (1,4)
runner   grouped, pilot positions (0,3)
margin   0.00920203980559375
status   verified
```

The grouped candidate contains exact duplicate decoded blocks `(0,2)`; the
interleaved candidate contains none. Test data were not opened during this
correction.

## Code impact

- `official_lpan` is domain-specific: Quasi keeps its single-pair-equivalent
  behavior, while Mobility requires interleaved layout and q1/q4 pilots.
- PhyMeta-STGT already uses explicit time embeddings and requires no structural
  redesign. Observation consistency now naturally selects q1/q4 from batch
  metadata.
- CNN-GRU and GCN-GRU use exact observed states at q1/q4. q0/q2/q3/q5 start
  from piecewise-linear anchor states with nearest extension, then receive one
  shared time-conditioned GRUCell update.
- Progressive LPAN wrappers restore the official raw interleaved channel order
  from unified complex-last tensors for both inputs and outputs.
- Transfer manifests include the complete semantic contract and a canonical
  SHA256 fingerprint, so legacy grouped/q0-q1 results cannot be reused.

## Historical-result boundary

Quasi data contain one complex pair and are not affected; existing Quasi
results and the Quasi seed-123 source checkpoint remain usable. All Mobility
model, baseline, ablation, interpolation, Ridge, and transfer results produced
under grouped/q0-q1 semantics are legacy-only and must be rerun for corrected
tables. Historical directories and checkpoints remain immutable.

Legacy LPAN/LPAN-L overall NMSE may be permutation-insensitive in aggregate,
but it was not produced under the corrected wrapper and per-time semantics.
It therefore remains historical evidence rather than a reusable corrected
number. All 25 low-data transfer cells must be rerun, and frozen-spatial,
domain-adapter-only, and adapter-head conclusions must be reevaluated.
