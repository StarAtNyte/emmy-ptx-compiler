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

## Backend nounroll control — rejected 2026-10-24

`backend-nounroll.patch` inserts NVIDIA's documented loop-header
`.pragma "nounroll"` into the bounded shared-expansion compiler. It is an
experiment only, never active lowering. The [PTX specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#descriptions-of-pragma-strings)
requires statement-level placement before instructions in the loop header.

Despite the target/local register discrepancy motivating this experiment,
turning off backend unrolling made all four measured kernels slower. Seven
alternating paired RTX 5090 trials, full-output first-trial checks and checked
prefixes thereafter, unlocked clocks:

| Kernel | Bounded baseline → nounroll, µs |
| --- | --- |
| a01 attention | 90.752 → 563.616 |
| a02 attention | 53.920 → 193.056 |
| m01 matmul | 66.208 → 68.224 |
| r02 big sum | 21.152 → 72.320 |

Raw samples and both compiler hashes are preserved in the production validation
record. This rejects blanket backend-unroll suppression; it does not establish
that every possible per-loop policy would be slower.

## Paired shared reads — measured candidate, not accepted

`paired-shared-loads.patch` applies to the nested-scope compiler. It pairs
separate aligned adjacent shared-memory scalar reads across pure arithmetic in
straight-line read-only leaf bodies. The earlier neutral vector-memory patch
handled grouped global accesses; this is a different memory-space/scheduling
hypothesis. It requires proven 8-byte alignment for `ld.shared.v2.f32`, consistent
with [NVIDIA PTX access-size alignment rules](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#addresses-as-operands).

Seven alternating paired RTX 5090 trials checked full outputs first and prefixes
thereafter: m01 66.208 → 64.128 µs (1.032x); a01 90.784 → 90.784. Hosted registers
rise from 40 to 48 for m01. Its PTX shrinks from 187,430 to 151,531 bytes and
contains 64 paired shared reads. Clocks were unlocked; no general speedup claim.

**Not active compiler code.** Before adoption, verify dependency/alias guards,
intervening value uses/rebindings, weak alignment, shared writes, independent
matmul shapes and all semantic/sanitizer gates. The measured improvement alone
does not establish correct general load motion. Raw samples and hashes are in
the third production iteration record; keep its accepted compiler as fallback.
