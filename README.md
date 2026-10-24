# Emmy Kernel IR to PTX

A Python compiler for the [Emmy Hackathon](https://github.com/cloudrift-ai/emmy-hackathon-2026).
It loads Emmy's typed Kernel IR and emits PTX directly, without invoking an
existing IR-to-code compiler. All 15 public problems pass across all five categories.

## Run

Python 3.12 or newer is required. Compilation needs neither Torch nor a GPU.

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python compile.py path/to/kernel.ir.json > out.ptx
```

The neighboring `meta.json` supplies parameter order. The emitted entry name comes
from the IR; the launch remains the one prescribed by the problem. PTX uses
version 8.0 and target sm_89, tested on the organizers' RTX 5090.

## Implementation

[compile.py](compile.py) is the standalone translator. It handles tile coordinate
decoding and guards, recursive index expressions, scalar/grouped accesses, loops,
conditional folds, warp combines, shared memory, asynchronous copies, barriers,
atomics and dynamic shared allocation above 48 KiB. Virtual registers are left
for ptxas to allocate.

Online softmax state merges preserve both the running maximum and denominator,
including empty-state identities. Tree reductions preserve binary merge order and
synchronization. A conservative use analysis removes dead scratch writes; when
scratch is observed, an atomic ticket selects one writer per logical slot without
extra allocation. This also handles repeated lanes and named-barrier groups.

Targeted FMA contraction applies to adjacent multiply/add reduction steps.
Bounded scalar loops reading shared memory expand four ways, with scalar tails.
Both optimizations use typed IR properties, without problem-name recognition.
[DESIGN.md](DESIGN.md) explains the lowering and its tradeoffs.

## Validation

The latest [validation record](validation/2026-10-23-iteration5.json) identifies
the exact compiler hash and measured scope:

| Check | Result |
| --- | --- |
| Public problems, hosted RTX 5090 | 15/15 |
| Additional hosted cases | 73/73 |
| Local regression cases, RTX 3050 | 157/157 |
| Complete staged transformer seed/depth checks | 9/9, 147 stage outputs |
| Selected semantic audit, four sanitizer tools | 11 cases, no errors/hazards |

Corpora cover shapes, operation mixes, schedules, finite-value stress inputs,
alignment layouts, loop tails, actual PyTorch MLP/transformer comparisons and
semantic regressions. They have been used during development and are not an
estimate of hidden-test success. The pinned Emmy materializer fails on the fully
fused transformer; separately traced stages execute the same complete model math
with host transfers. That test measures correctness rather than model speed.

To run the public checker:

```sh
git clone https://github.com/cloudrift-ai/emmy-hackathon-2026 challenge
.venv/bin/python tools/validate.py
```

[TESTING.md](TESTING.md) contains complete reproduction commands, local execution,
paired timing and sanitizer details. A committed-source archive was also tested
from another directory in a fresh environment without Torch installed.

## Measured performance

Alternating paired RTX 5090 trials checked both compilers' outputs. Clocks were
not locked; these are observations on named kernels, not universal speedups.

| Optimization | Kernel | Median before → after | Speedup |
| --- | --- | --- | --- |
| Reduction FMA | Public a01 attention | 119.424 → 90.784 µs | 1.315× |
| Shared-memory loop expansion | Public m01 matmul | 98.944 → 74.400 µs | 1.330× |

The semantic audit retained those performance levels. Raw samples are preserved
in the validation records. Explicit vector accesses were timing-neutral, and
unrestricted scalar-loop expansion regressed attention; those versions are
archived under [experiments](experiments/README.md) rather than active.

## Scope and submission

The current storage/arithmetic path is f32, with static shapes, 32-bit index math
and 64-bit global addresses. Additional dtypes, auxiliary thread bands, shared
swizzles and some operations remain unsupported. Grouped accesses are scalarized.
No hidden-set result or prize outcome is claimed.

[SUBMISSION.md](SUBMISSION.md) holds the entry draft and required entrant fields.
[WALKTHROUGH.md](WALKTHROUGH.md) is an own-voice recording outline with understanding
questions. Repository publication and the actual entry are still pending.
