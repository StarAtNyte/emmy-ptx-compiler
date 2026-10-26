# Handoff — current state, 26 October 2026

## Goal and authorization

Build a production-level, competitive Emmy Kernel IR → PTX compiler and try to
win https://github.com/cloudrift-ai/emmy-hackathon-2026.
Read the user's attachment before continuing:
/home/nitiz/.codex-account-2/attachments/06fa33fe-9788-4edf-8b70-1012b95d62d6/pasted-text-1.txt.
Read PROGRESS.md too; it retains chronological decisions and failures.
Update both files after each meaningful completed step.

- User reaffirmed: make compiler win-ready first; public repo and own-voice video
  can happen later today. **Keep repository private and do not submit yet.**
- Routine pushes are authorized. Origin StarAtNyte/emmy-ptx-compiler, master.
- Do not change billing/spending or contact organizers/authors.
- Entrant fields, personal confirmations and video remain missing; do not invent.
- Official requirements were rechecked at ec8cde1943f88efea3827a95bda6085553d2d407:
  hidden-category correctness first, own code review next, kernel speed last.
  Eventual entry needs public repo and a personal3–5min walkthrough. Deadline
  Oct27 23:59 AoE / Oct28 17:44 Nepal. See HACKATHON_READINESS.md.
- Broader readiness and hidden-set success remain unproven. Do not mark goal
  complete merely because a milestone passes or submission is deferred.

## Runtime and current compiler

Workspace /home/nitiz/Projects/Self/Challenges/emmy. Use .venv/bin/python3.12,
Emmy0.3.11 and NumPy2.5.4. Dev Torch is CPU2.14.1; compiler needs no Torch/GPU.
Pinned CPU constraints: requirements-ci.lock. Ignored challenge/upstream-emmy/
artifacts/.venv are local assets, absent from a fresh checkout.

Local RTX3050 Laptop6GiB sm86, driver595.99.02. Official RTX5090 sm120:
http://runner.riftstack.ai:48701. Compiler emits PTX8.0 sm89, valid on target;
local test harness retargets to sm86. Local timing cannot select target changes.

**Accepted compile.py SHA256:**
6237caabcebbaf5503223ab90d56d19d57aa4725095ab8846f8b3d2349ae41a3.
Exact frozen source: artifacts/compiler-scalar-masks.py. Previous eeb79a3a is
reproducible from44bd6d3:compile.py and frozen as compiler-before-masks.py.
Previous c2f7e2b3 is reproducible from aacaa3c:compile.py. Earlier accepted sources/evidence are
recorded in PROGRESS.md and dated validation records; old patches target those
specific baselines, not automatically the current compiler.

## Latest acceptance — internal bool SSA and masks

Evidence: validation/2026-10-26-production-iteration17.json. All six scalar
comparisons, logical/legacy f32 masks, where and bool Assign/Let supported.
Write.value_dtype bool permits numeric conversion into still-f32 storage.
Global/shared bool buffers, bool Load/Init/Accum and general integer Assign
remain unsupported; storage ABI still f32. Comparisons normalize declared-f32
outputs; NaN truth/inequality and signed-zero selection retain scalar semantics.

All128 exact original/fresh/renamed typed probes and13 frontend workloads pass
both GPUs, including full masked attention. Nine transformer checks pass;
all four sanitizers clean on75 selected cases. 957 prior kernels emit identical
PTX. No new kernel speedup claim. Five positive CPU fixtures added, old33 unchanged.
Full157-check no-Torch matrix newly passes all3 Python versions (42+28+42+31+4+10).
Cold CLI public15×3processes, medians0.923–1.068sec while CPU matrix ran; no
before/after speed claim or judge time-limit guarantee. Hosted CPU CI unverified.

Initial candidate4/5 exposed source-bool→f32 Write conversion, refined5/5.
Initial attention oracle assumed3 inputs but an out-of-trace s=q@k made inputs
s,v. Two ReLU setup expressions had the same trace-boundary mistake. Final
specs inline intended QK/ReLU computations; all corrected6 workloads pass both
GPUs. Failed intermediate test versions preserved/excluded; compiler6237 unchanged.

## Previous acceptance — scalar logarithms and exp2

Evidence: validation/2026-10-26-production-iteration16.json.
Natural/base-two/base-ten logs lower via lg2 plus rounded f32 base conversion;
exp2 uses ex2. No .ftz. Unknown scalar ops reject before stdout with no traceback.
These are approximate functions, not bitwise libm replacements.

- Baseline c2f7e2b3 fails36 initial typed new-operation programs. Four unsupported
  but valid scalar op probes produce tracebacks before; candidate diagnostics pass.
- All104 final typed and four frontend compound cases pass both GPUs. Covers
  subnormal/normal ranges, neighbors of1, round trips, folds, log-softmax and
  normalization, negative/zero operands, infinities and NaNs created in-kernel.
  All transported values stay finite; result-class labels are exact.
- Four sanitizer tools clean on56 original/frontend cases. Nine staged-transformer
  comparisons pass. 845 retained kernels emit byte-identical PTX, preserving
  their previous execution/performance evidence. No new kernel speedup claimed.
- Initial power probe overrequired bitwise equality. Observed1ULP at2**-127 on
  both GPUs is within documented approximate behavior. Final test classifies
  independently computed2ULP bounds; separate probes check subnormals stay nonzero.
  Initial failed reports/generator preserved under artifacts; excluded from finals.
- Initial frontend8 attempts:4 successes,4 upstream CUDA-renderer failures for
  log2/log10/exp2. Final supported four specs regenerate identical IR/meta/oracle
  bytes. Typed IR probes cover the other ops; no false frontend-success claim.
- All15 public kernels compile deterministically in three isolated no-Torch
  processes each, local median0.629–0.703sec; no judge time-limit guarantee.
- Four GPU-verified CPU fixtures added, previous29 snapshots unchanged.
  All142 local checks newly pass each isolated no-Torch3.12.13/3.13.14/3.14.4:
  37 snapshot/feature +28 launch +42 ABI +31 shape +4 unsupported-op diagnostics.
  CPU workflow includes new diagnostic suite. **Hosted CPU CI remains unverified.**

## Contract and semantic invariants

`python compile.py path/to/kernel.ir.json > out.ptx` reads Graph.from_dict plus
adjacent meta.json. Own typed lowering; never invoke Emmy's CUDA renderer, nvcc,
emmy compile or another compiler backend in submission. Dev tracing/reference
math, driver JIT and disassembly do not change that boundary.

- Exactly one KernelOp and unique matching launch; positional f32 pointer args
  agree with arg_order and descriptor order. Consistent permutations are valid.
  Logical tensor shapes may differ from physical allocation shapes/scalar views.
- Address ranks: empty→0, one→flat, full rank→row-major, otherwise additive sum.
  Preserve permissive coordinates; do not impose logical/allocation rank equality.
- Launch uses1D grid/block; y/z1. Cooperative Tile.block_threads must match block.x.
  No aux thread bands. Empty Tile domain is a no-op; empty axis LIST is one cell.
- Typed tensor/Tile/loop extents are static nonnegative integers; shared extents
  positive (scalar() is1f32). Shared align0/natural or power-of-two. Symbolic
  dimensions and zero-size shared arrays unsupported. No complete bounds/grid/
  cardinality/type validator claim.
- Predicates materialize numeric0/1 in integer contexts; conditions retain predicates.
  Signed /,// floor and % follows divisor sign. Positive power-of-two shifts/masks
  preserve that behavior. Unknown/overflow intervals do not prove fast truncation.
- F32 scalar expressions support+,-,*, comparisons, truthiness, ternaries and
  explicit numeric casts. Floating indices reject; floating Expr /,//,% unsupported
  (typed Assign divide works). Nonfinite/out-of-range f32→i32 casts outside scope.
- Lexical maps isolate locals; Let snapshots capture values. Mutable reduction
  state crosses scopes explicitly. Nested fold discovery and seeded/persistent
  loops retain ownership. Strided end is captured after initializing axis;
  step depending on axis/end evaluates after body; axis_end is body-visible.
- Shared storage belongs to declaration identity, including shadowed names;
  dynamic pool above48KiB checks supplied size. Swizzles unsupported.
- Ordered online-softmax state handles neutral lanes. TreeHalve scratch that is
  observed uses writer tickets per logical slot; unobserved scratch can reduce
  registers. Preserve merge order, participant barriers and scratch visibility.

Accepted structural optimizations: fold FMA, shared-read-only bounded unroll
(max64, extent and1024-statement budget), aligned within-body v2 reads and
cross-iteration v2/v4 reads. Four-wide requires every v2 eligible load to also
prove width4; otherwise whole loop retains v2. Mutable indices, writes, global
reads and control flow guard motion; tails clear caches. No workload-name dispatch.

## Other current evidence and rejected experiments

- production-iteration15.json: four-wide acceptance,84 structural+8 floating cases,
  59 changed prior kernels pass both GPUs,641/700 byte-identical; four sanitizers
  clean on49 cases including three actual pipelined matmuls.
- Final seven alternating paired RTX5090 trials19workloads: m01 17.024→15.008us
  (1.134x), independent256×512×384 13.152→11.104us (1.184x). Third-shape early gain
  did not reproduce. Other16 max median regression0.032us. Unlocked clocks,
  reused buffers, full outputs first trial/prefixes later; no counter proof.
- wrap-addresses.json:44 in-bounds signed32-wrap address cases pass both GPUs;
  all four sanitizers clean on22 originals. New algebraic wrap tests do not prove
  bounds for all programs. Four retained CPU fixtures, old25 goldens unchanged.
- bounded-tuning.json: cg16 copy hints show only small selected differences and
  were not adopted; max32/max16 expansion regress m01 ~27%. Patches retained.
- Earlier unsigned tile rewrite regressed attention; blanket/global unroll,
  global vector pass and address CSE were rejected. Do not repeat unchanged ideas.
- Target calibration/RESEARCH_PRODUCTION.md distinguish cache-sized bandwidth,
  DRAM nominal specs, local-driver SASS and target timings. No hosted counters,
  locked-clock roofline or rigorous utilization proof.
- Requested GPU MODE, Modal, Unsloth, tinygrad, PufferAI and DeepSeek primary-source
  review: RESEARCH_2026-10-26.md and research-refresh.json,9 repos/20 cached files.
  Selected sections reviewed; external claims not reproduced. No external source
  incorporated, paid compute, Modal deployment or private-source upload performed.

Local runner initialization correction is retained: zero_outputs may name inputs;
zero them before initial correctness launch as well as timing. Output=true buffers
initially zero even with explicit data in tested hosted cases. Keep compiler role
checks separate. Runner SHA90a494a357d0368265a06fffdcdfd3895eaa87d3a19f8d51b6dd492b10531872.

## CI and packaging

GitHub run40295512213 (sourcebea1389) inspected: all3 CPU jobs rejected before
steps; annotation confirms account payments/spending limit. No compiler tests
ran. No account change, retry loop or visibility workaround authorized. Current
hosted CPU157-check suite remains unverified despite local157-check matrices.

Current verified archive source bea1389 / compiler6237caab:
ZIP041536e115f31a8b4561b9fbe1f33b36fe852cde2eb401ae575ab0122cb05c9e.
Record validation/2026-10-26-current-package.json. Previous scalar-log checkpoint
preserved as2026-10-26-scalar-logs-package.json. Masks/current/submission ZIP
aliases match. Actual compiler/harness/generator/manifest/fixture bytes checked.
Extracted hosted scalar-add/matmul/attention plus all5 retained mask/attention
fixtures pass in existing isolated no-Torch3.12 runtime. No fresh install claimed.
No rebuild needed for documentation/evidence-only updates.

## Next work

All semantic/CPU/transformer/sanitizer gates above have completed; do not restart.
Mask source/evidence published and archive verified. Next remaining frontend scalar ops.
Large dtype/Tensor Core expansions are not required by published evaluation scope.

HACKATHON_READINESS.md is the requirements audit. Personal video/publication later,
submission deferred, hidden-set and broad production-readiness claims unproven.

Two numeric bool-cast audits with zero/subnormal/max-finite inputs also pass
both GPUs and regenerate byte-identically; included in13 frontend count.
Mask acceptance is complete, source bea1389 published privately and archive verified.
All jobs terminal, including the next math discovery. Actual six frontend
programs generate: five fail current6237 compiler on square/erf/gelu, while
zero-pad-plus-add passes. Evidence: validation/2026-10-26-math-discovery.json,
including reproducible spec source and scalar op counts. No math candidate yet.
Next implement own accurate PTX square/erf/GELU lowering, verify real frontend
and adversarial numerical cases on both GPUs. Do not substitute a library
backend, reduced precision or narrow the full win-ready objective to this subset.
