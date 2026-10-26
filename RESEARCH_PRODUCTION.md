# Production research and experiment gates

Original snapshot: 2026-10-24. Updated review: 2026-10-26. Submission is deferred. This document separates external
claims from our measurements; it does not certify production readiness.

## Primary material inspected

| Source | Pinned revision / date | Transfer to Emmy |
|---|---|---|
| [Joseph Suarez / PufferAI engineering account](https://puffer.ai/blog/engineering-4.0/) | 4.0 engineering retrospective; date not asserted | Profile end-to-end, reduce redundant memory traffic, verify numerical behavior. Fusion can matter more than tuning one small kernel. |
| [PufferLib](https://github.com/PufferAI/PufferLib/tree/ea7f989371270d2948f96728bda8aec5e7634a4c) | default branch **5.0**, `ea7f989371270d2948f96728bda8aec5e7634a4c`, 2026-10-14 | `tests/profile_kernels.cu` provides operation/shape sweeps, GPU warmup and event timing. It also benchmarks cuBLAS-based paths; these cannot replace our lowering. The older 4.0 `src/kernels.cu` has selectable float/BF16 precision. |
| [GPU MODE reference kernels](https://github.com/gpu-mode/reference-kernels/tree/f3295bb6bd559bae4659d2d8fc45edb1394251b1) | `f3295bb6bd559bae4659d2d8fc45edb1394251b1`, 2026-09-18 | `problems/nvidia/eval.py` separates correctness from repeated timing, clears L2, reports sample uncertainty, and supports changed-seed rechecks. These competition rules differ from Emmy's. |
| [GPU MODE kernelbot](https://github.com/gpu-mode/kernelbot/tree/ec4669669d368c395675565831f98b509db2b0b3) | `ec4669669d368c395675565831f98b509db2b0b3`, 2026-09-18 | Platform / evaluation tooling, not a drop-in compiler backend. Read README and example evaluator. |
| [SOL-ExecBench paper](https://arxiv.org/abs/2603.19173), [implementation](https://github.com/NVIDIA/SOL-ExecBench/tree/a9fa0804c793d438e70850c33fe34426e66d53dd) | paper 2026-03-19; code `a9fa0804c793d438e70850c33fe34426e66d53dd`, 2026-07-15 | Compare against hardware lower bounds, not merely a weak previous version. Paper describes clock locking, L2 clearing and fresh inputs. Pinned `timing.py` allocates **twice queried L2 size**, despite an older 256 MB docstring: inspect implementation, not just prose. Blackwell reduced-precision results do not establish f32 RTX 5090 performance. |
| [Atrex-Bench paper](https://arxiv.org/abs/2607.14541) | submitted 2026-07-16; abstract inspected, code not yet audited | Production-trace shapes and time-weighted importance suggest workload prioritization. Its reported results remain author claims, not reproduced evidence for our compiler. |
| [FastKernels paper](https://arxiv.org/abs/2605.23215), [repository](https://github.com/Snowflake-AI-Research/fastkernels/tree/9acebaa1349c78af8bb0ecb7e0719b5372d26fd8) | code `9acebaa1349c78af8bb0ecb7e0719b5372d26fd8`, 2026-09-20 | Integration contracts and composite model checks matter. Numerical acceptance depends on tolerance; retain normalized error margins, not only pass flags. Read paper sections and repository README; did not reproduce its benchmark. |

Local discovery manifests and selected unmodified source files are cached under
ignored `artifacts/research/`. No external compiler code was incorporated.

## Experiment matrix

| Hypothesis | Experiment / metric | Correctness gate | Decision boundary |
|---|---|---|---|
| Backend allocation differs by GPU | Query JIT function registers, local memory, shared memory, occupancy ceiling; compare hosted register counts | Compile every public case; preserve PTX/IR/meta/compiler hashes | Local occupancy is diagnostic only; target selects performance changes |
| Shared leaf loops remain instruction/loop limited | Sweep bounded expansion 4/8/16/32/64; target paired timing, registers and PTX size | Public matmul plus independent shared-loop tails, sanitizer and model regressions | Keep only robust improvements; bound code growth; preserve scalar tails |
| Apparent bandwidth is cache-dependent | Add local hot/eviction-conditioned timing with eviction outside captured events | Validate full outputs after repetitions, reset atomic outputs outside timings | Report both conditions; a memset is eviction conditioning, not proof all L2 lines are cold |
| Attention latency is dominated by small grid and backend expansion | Inspect loop structure, target register counts and code size; later SASS/counter analysis | Independent attention shapes, numerical stress, complete staged transformer | No launch changes or lower precision without contract proof |
| Address-expression work is redundant | SSA-aware integer CSE / loop-invariant addressing candidate | Reassigned Let/loop variables, branches, overflow semantics and shared aliases | Must prove lifetime validity; do not cache textual expressions across mutation |
| Existing tests miss semantic combinations | Deterministic typed-IR metamorphic/differential generation and failure shrinking | Compare conservative compiler and independent mathematical references | Keep generator failures distinct; retain seed and failure artifact |
| Kernel-level green hides integration defects | Complete model stages with fresh inputs and multiple seeds/depths; maximum normalized error | CPU reference plus compiled outputs, original tolerances | No hosted roundtrip timings as model throughput |

## Acceptance before a new release

1. No correctness regression in public, independent semantic/schedule/numerical
   suites; every new feature has a counterexample-oriented regression.
2. Memory/synchronization checks on changed shared-memory/async behavior, and
   fresh cases created after the candidate is frozen.
3. Reproducible runtime contract, explicit unsupported features, clear failure
   diagnostics, dependency pinning and fresh-environment smoke execution.
4. Performance evidence with paired order, raw samples, compiler/IR/metadata
   hashes, GPU identity, cache condition, and full output verification.
5. Explain remaining gaps honestly: static f32 coverage is not general ML
   production support; resource ceilings are not measured utilization. No SASS
   or hardware-counter evidence is available from the hosted API today.

Hardware lower bounds require the actual workload's arithmetic and minimum data
movement, the correct f32 execution path, and target clocks/bandwidth. Do not
use BF16/FP4 tensor-core peak figures to score scalar-f32 kernels. A first target calibration and conditional nominal-peak models now appear in
[TARGET_PERFORMANCE.md](TARGET_PERFORMANCE.md). The 224 MiB strided read measures
1654.52 effective GB/s; a cache-sized read measures 5366.47 GB/s, so a DRAM-only
roofline is inappropriate for repeated cache-resident workloads. Actual clocks
and counters remain unavailable; rigorous hardware lower bounds and measured
utilization remain unproven.

## 26 October refresh

[The focused review](RESEARCH_2026-10-26.md) covers GPU MODE, Modal, Unsloth,
tinygrad, PufferAI, DeepGEMM, FlashMLA, TileKernels and selected September/October
papers. Nine repository heads and twenty inspected files are pinned. It records
which ideas fit the current contract, which require separate experiments, and
which change precision, hardware or execution scope. Additional cancellation,
magnitude and pipelined-sanitizer checks were applied to the wider-load candidate.
