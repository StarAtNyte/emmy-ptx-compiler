# Measured optimization experiments

These patches are research artifacts, not active submission code. Their baseline
is commit `7b603f3` (compiler SHA-256
`749d9301189d77c9297c24b01b183a92e2ed84dcfa9efa94462d3039b3c1230d`).

## Explicit vector memory instructions

`vector-memory.patch` adds conservative modulo-residue alignment proofs and
emits grouped global vector loads/stores only when alignment is proven. Shared
accesses remain scalar. Eight typed-IR cases cover aligned strides, odd strides
and one-element offsets, and sanitizer checks passed.

The change was rejected from the active compiler because paired hosted timings
were neutral on five public kernels and two million-element pointwise problems.
This does not prove vector instructions can never help; it provides no measured
reason to add this logic to the current translator. The alignment test generator
remains useful independently.

## Unrolling global scalar loops

`global-loop-unroll.patch` expands straight-line scalar inner loops four ways.
Large matmul improved by about 1.33x, but public a01 attention regressed from
90.784 to 178.784 microseconds. The broad pass was rejected.

The active compiler applies the pass only when all loop loads read shared
memory, under bounded body/extent limits. This retains the matmul improvement
and leaves global-memory attention on its previously validated path. The guard
uses typed IR properties, not problem names or entry names. Scalar epilogues
preserve all remainder lengths; dedicated shared-loop cases exercise them.

All measurements use alternating baseline/candidate trials with checked outputs.
Runner clocks are not locked. Raw results and compiler hashes are archived in
the iteration's validation record.

## Frozen-compiler holdout provenance

`holdout-initial.py` preserves the original specification hash recorded before
tracing. Two of twelve attempts failed in Emmy's frontend (`amin`, `prod`). Nine
of the ten generated cases passed initially. The LayerNorm fixture incorrectly
precomputed centering outside the traced callable, so its IR took y as input and
performed RMS normalization while the reference centered again.

`holdout-layernorm-correction.py` keeps the whole computation inside forward.
The compiler was unchanged. All ten valid cases then passed locally and hosted.
The runnable `tools/expressions_holdout.py` reproduces those exact IRs, launch
metadata and expected arrays. Original and corrected results are retained in
`validation/2026-10-24-holdout.json`; no initial failure was silently discarded.
