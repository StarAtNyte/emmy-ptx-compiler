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

## Paired shared reads — accepted after semantic gates

`paired-shared-loads.patch` applies to the nested-scope compiler at 5791ed6.
The active compiler now includes this change; the patch preserves its provenance. It pairs
separate aligned adjacent shared-memory scalar reads across pure arithmetic in
straight-line read-only leaf bodies. The earlier neutral vector-memory patch
handled grouped global accesses; this is a different memory-space/scheduling
hypothesis. It requires proven 8-byte alignment for `ld.shared.v2.f32`, consistent
with [NVIDIA PTX access-size alignment rules](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#addresses-as-operands).

Seven alternating paired RTX 5090 trials checked full outputs first and prefixes
thereafter: m01 66.208 → 64.128 µs (1.032x); a01 90.784 → 90.784. Hosted registers
rise from 40 to 48 for m01. Its PTX shrinks from 187,430 to 151,531 bytes and
contains 64 paired shared reads. Clocks were unlocked; no general speedup claim.

Accepted after dependency/mutation/alignment probes, independent matmul checks,
232 prior local regressions, transformer, fresh fixtures and selected sanitizer
gates. The fourth production iteration record preserves these gates, standalone
runtime checks and the final seven-pair timing: m01 66.176 → 64.160 µs (1.031x),
independent vectorizing matmul 35.488 → 31.392 µs (1.130x). Attention and scalar
controls remain within noise. This bounded evidence does not establish general
load-motion safety for every possible IR or production readiness.

## Signed index floor correction — accepted bounded variant

`signed-index-floor.patch` reconstructs the accepted range-aware fix from compiler
at aa668e2. Installed Emmy 0.3.11's expression evaluator floors signed division,
unlike PTX div/rem. Unconditional correction fixed numerical probes but slowed
public m01 from 64.160 to 80.544 µs. Conservative signed-int32 bounds omit correction
when signs prove truncation equivalent; final seven-pair m01/m02 medians remain
64.160/5.056 µs. The fifth production iteration records exact-candidate semantic,
range-shadowing/overflow, sanitizer, transformer and standalone-runtime gates.
This correctness change is not a new speedup claim.

## Integer/address CSE — rejected after neutral target timing

`address-cse.patch` applies to accepted compiler at 1ad0dbf. It reuses pure integer
instructions by actual register identity and clears reuse at raw mutation/control
boundaries. Static PTX shrinks (m01 4547 → 4252 instructions), but seven paired
RTX 5090 trials on five workloads show no win. Main remains unchanged. Partial
80-case prior and 20-case fresh checks passed; the rest of the acceptance sweep
was intentionally stopped after timing rejection. No full safety claim.

`validation/2026-10-24-address-cse.json` preserves timings/resources/partial checks;
`validation/2026-10-24-native-instructions.json` compares local-driver-linked
sm86/sm120 cubins. Three workloads have identical disassembly; matmul native code
changes without a target win. These are not hosted SASS or counter measurements.

## Cross-iteration shared reads — validated, adoption pending

cross-iteration-pair.patch applies to compiler 7fbe5ee4 (current main source) and
reconstructs candidate 40a1de84. It pairs aligned affine unit-stride shared reads
across two unrolled iterations with immutable index dependencies. Final target
matmul gains are 1.387–1.482x; full safety and timing scope is recorded in
validation/2026-10-24-cross-iteration-discovery.json. Main has not yet adopted it;
CPU fixture retention, actual updated CI and refreshed packaging remain.
