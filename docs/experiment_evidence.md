# Experiment evidence and reporting boundaries

This file records verified work that predates the unified repository. It is
evidence context, not a substitute for copying the original run directories
into the repository or rerunning a model under the unified protocol.

## Quasi-static LPAN baseline

- Project: `D:\数据1\LPAN_baseline`
- Training seed: `123`
- Batch size: `8`
- Training budget: `100` epochs
- Best epoch: `100`
- Best internal-validation NMSE: `-24.6236766 dB`
- Linear NMSE: `0.0034485167`

This is an internal validation result from the training file. It is not an
independent-test, OOD, user-level or few-shot result. The separate 9000-sample
quasi-static test file exists, but an independent-test result must not be
claimed until a frozen-checkpoint evaluation artifact is located or generated.

## Official progressive Mobility LPAN-L baseline

- Project: `D:\数据1\LPAN_mobility_baseline`
- Architecture: official progressive Mobility LPAN-L (`32 -> 64 -> 128 -> 256`)
- Input/output channels: `4 -> 12`
- Parameters: `1,112,904`
- Training seed: `123`
- Batch size: `8`
- Training budget: `100` epochs
- Best epoch: `95`
- Best validation NMSE: `-21.6269 dB`
- Epoch-100 validation NMSE: `-21.6186 dB`
- Frozen independent-test sample count: `9000`
- Independent-test linear NMSE: `0.010649267638723056`
- Independent-test NMSE: `-19.726802580841884 dB`
- Best checkpoint:
  `D:\数据1\LPAN_mobility_baseline\runs\mobility_lpan_l\full_seed123_bs8\checkpoints\best_checkpoint.pth`
- Test artifact:
  `D:\数据1\LPAN_mobility_baseline\runs\mobility_lpan_l\independent_test\independent_test_result.json`

The official progressive model and this repository's `lpan_l_direct` model are
different architectures. Their parameter counts, operation counts and results
must occupy separately labeled rows. The direct model must not inherit the
official model's `-19.7268 dB` test result.

## Mobility data contract

- Train: `20000` samples
- Validation: `1800` samples
- Independent test: `9000` samples
- Raw input: `[N, 4, 64, 32]`
- Raw target: `[N, 12, 64, 256]`
- Unified input: `[N, 2, 32, 64, 2]`
- Unified target: `[N, 6, 256, 64, 2]`
- Complex layout: grouped real time blocks followed by grouped imaginary time
  blocks
- Frame semantics: two pilot blocks reconstruct six target blocks inside one
  sample

No public `user_id`, `trajectory_id`, `sequence_id`, `timestamp`, position or
velocity metadata has been found. Consecutive MAT samples must not be described
as a physical trajectory, and the current task must not be labeled next-frame,
cross-trajectory or new-user prediction.

## Fair comparison protocol

Models in one comparison table must use the same train/validation/test files,
input and target tensors, complex layout, sample-level linear NMSE aggregation,
checkpoint-selection rule, seed policy and training budget. Hyperparameter
selection uses validation only. The independent test split is evaluated only
after the protocol and checkpoint have been frozen.

## Complexity reporting protocol

Use `python main.py profile` for models registered in the unified repository.
Report both GMACs and GFLOPs under these shared conditions:

- batch size `1`;
- FP32;
- one forward pass;
- no backward pass;
- domain-specific canonical sparse input;
- model operations only;
- `1 MAC = 2 FLOPs`.

The profiler counts convolution, linear, recurrent, attention-product and graph
aggregation MACs. Bias, normalization, activation, softmax, indexing and
interpolation are disclosed as excluded. Numbers produced by THOP, fvcore,
ptflops or a paper cannot be mixed into the same column unless their counting
scope and MAC/FLOP conversion have been reconciled.

The generated same-condition results are summarized in
`reports/complexity_summary.md`. In particular, the official progressive
Mobility LPAN-L has `1,112,904` parameters, `6.337769472 GMACs` and
`12.675538944 GFLOPs`, while the current PhyMeta-STGT has `188,360` parameters,
`0.113828352 GMACs` and `0.227656704 GFLOPs` on the mobility task. The source
paper's `6.06 GFLOPs` is close to the unified profiler's MAC count, which is
consistent with—but does not by itself prove—a MAC/FLOP naming difference.
