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

Latest acceptance (steps76–77): all >256-thread blocks now get a safe register
ceiling based on complete physical warps. The state/math budget preserves measured
matmul and attention performance. Baseline384/512/768-thread copy/barrier probes
failed launch on both GPUs; fixed probes pass. Exact .reqntid and a state-only
budget were rejected after12% matmul/26% attention regressions. See
validation/2026-10-26-launch-resources.json and tools/check_launch_resources.py.

Final gates: hosted public15/15, frontend16/16, fresh pressure5/5; local22/22;
transformer9 runs/147 stage checks; four sanitizers clean on final attention and
shared matmul. Each isolated no-Torch3.12/3.13/3.14 passes176 contracts and3072
register-ceiling checks. Seven-pair target public matmul/attention medians unchanged;
independent shared matmul11.072→11.168us (<1%). Packaging needs rebuilding after
this source commit. No hidden score or new speedup claimed.

Workspace /home/nitiz/Projects/Self/Challenges/emmy. Use .venv/bin/python3.12,
Emmy0.3.11 and NumPy2.5.4. Dev Torch is CPU2.14.1; compiler needs no Torch/GPU.
Pinned CPU constraints: requirements-ci.lock. Ignored challenge/upstream-emmy/
artifacts/.venv are local assets, absent from a fresh checkout.

Local RTX3050 Laptop6GiB sm86, driver595.99.02. Official RTX5090 sm120:
http://runner.riftstack.ai:48701. Compiler emits PTX8.0 sm89, valid on target;
local test harness retargets to sm86. Local timing cannot select target changes.

**Accepted compile.py SHA256:**
c9184bfd67d79d2a267192936439e83b57000974a0f36859a516c0cc71fe19c8
(launch-resource fix; all1953 compared kernels retain identical PTX bodies,
1918 are completely byte-identical). Frozen: artifacts/compiler-safe-budget-one-state.py.
Previous51239969 is artifacts/compiler-review-baseline.py and compiler-loop-split.py.
1a2ad94e is frozen as artifacts/compiler-math-bounds.py. Previous 3586814e is frozen as
artifacts/compiler-erf-refined.py. Earlier accepted sources/evidence are
recorded in PROGRESS.md and dated validation records; old patches target those
specific baselines, not automatically the current compiler.

## Previous acceptance — accurate math, launch bounds, reduce identities

Evidence: validation/2026-10-26-production-iteration19.json (with a hash legend
for every intermediate candidate) and validation/iteration19-*.json raw reports.

- Own f32 math replaces bare hardware approximations: exp = ex2(rn(x*log2e))
  corrected by the product's exact rounding error (2 ULP, chain +1 FMA), tanh
  (odd minimax under 0.55, exp form above), log (fdlibm reduction, 1 ULP),
  sin/cos (3-part Cody-Waite, Payne-Hanek above 105615, 2 ULP), pow (f64 log
  and exp, C99 special cases, 0 ULP sampled). Emmy's reference kernels use
  accurate libm, so hidden tolerances may be tight. At rtol=atol=1e-6 on the
  RTX 5090 the old compiler failed tanh, gelu-tanh, sin, cos and pow.
- Launch bounds: .maxnreg 65536/block for 1024-thread blocks or estimated
  accumulator pressure (a 1024-thread fused matmul-softmax failed to launch
  with OUT_OF_RESOURCES before); .maxntid for blocks <= 256 (a02 1.295x).
  Always-on .maxntid/.maxnreg/.minnctapersm at 512 threads regressed m01 to
  0.88-0.90x and was rejected.
- Seeded max/min folds now take the IR op identity (Emmy -1e30/+1e30), not
  +-inf. Attention whose Tile has lanes with empty strided loops returned all
  NaN on both old and new compilers (8x8, 24x40, 100x16); all pass now.
- statement() split into lower_<kind> methods with a dispatch table plus a
  module overview; byte-identical PTX across 1718 kernels versus pre-refactor.
- New corpora: tools/expressions_precision.py (+ tools/measure_precision.py),
  expressions_multikernel.py (19 unpinned shapes; all fuse to one kernel) and
  expressions_hidden_proxy.py (43 fresh shape/op-mix cases across all five
  categories; old compiler 35/43, final 43/43 on the RTX 5090 after correcting
  two of my own trace-boundary spec mistakes).
- Local RTX 3050: no regression across 1718 corpus folders versus the old
  compiler; its 72 failures are shared, documented stale/obsolete cases.
  Sanitizers clean (4 tools). CPU contracts: 60+28+42+31+4+10 checks pass on
  isolated no-Torch 3.12/3.13/3.14. Public paired timing: a02 1.295x, f02
  0.959x (+0.13 us for accurate exp), others 0.99-1.002x. Cold compile <0.3 s.

## Previous acceptance — square/erf and GELU/RMS

Evidence: validation/2026-10-26-production-iteration18.json. Square uses native
rounded multiplication; own erf uses degree23 odd series over|x|<=1 and
A&S7.1.26 outside, with FMA Horner evaluation. No stock backend/device library.
All48 primitive/composition cases,6 frontend programs and3 seeded complete
GELU MLP runs pass both GPUs. Six parameter arrays restored from fixed CPU
PyTorch state, independent input seeds0/17/613. All3 regenerate exact bytes.
Nine transformer checks pass. All1104 prior supported kernels emit exact PTX.
Four sanitizers clean on31 selected primitive/frontend/full-model cases.

Initial algorithm2cd85520 passed tolerance but measured up to23ULP near.125.
Refined3586814e reduces selected fixed-input maximum to3ULP and1.7882e-7
absolute on both GPUs. Tiny1ULP, bounds/zero signs checked. Not a whole-f32
accuracy proof or bitwise libm claim. All162 local no-Torch checks pass each
Python3.12/3.13/3.14 (47+28+42+31+4+10); previous38 snapshots unchanged,5 added.
Cold CLI15public×3 medians0.854–0.971sec while CPU checks ran; no judge guarantee.

Initial model memcheck was started before fixture creation and did not exercise
kernel. Corrected check passes; final31-case4-tool run includes completed model.
Initial accuracy request iters0 rejected422 before execution; corrected1 works.
Initial candidate/reports preserved outside final acceptance; refined source frozen.

## Previous acceptance — internal bool SSA and masks

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

GitHub run40302035461 (sourced193ab1) inspected: all3 CPU jobs rejected before
steps for account payments/spending limit. No current hosted CPU results.
Do not alter billing, retry unchanged jobs or change visibility as a workaround.

Current verified archive source d193ab1 / compiler3586814e:
ZIP7dfc70d706ef2f1e9f4738ea73e36fa94bbb81f9c230584bc0022f950cf9a2a6.
Record validation/2026-10-26-current-package.json. Prior mask checkpoint preserved
as2026-10-26-masks-package.json. Erf/current/submission ZIP aliases match.
Actual compiler/harness/generators/manifest/fixture member bytes checked.
Extracted standard3 plus all5 archived math fixtures and a prepared full GELU
MLP case pass on official GPU in existing isolated no-Torch3.12 runtime.
The MLP descriptor is a prepared external artifact; compiler and5 fixtures were
extracted. No fresh installation claimed. No rebuild for docs/evidence-only edits.

## Next work

Judging order (official README; no numeric weights): hidden-set correctness
by category, then code review, then kernel speed. The video is an eligibility
gate. Deadline Oct 27 23:59 AoE; resubmission allowed until then.

Package for f1e9d7a / 51239969 is built and verified: extracted compiler and all
36 fixtures dated 2026-10-26 pass on the official GPU
(validation/2026-10-26-current-package.json). Rebuild after any code change.

1. Readability: README "How it works", DESIGN math/launch sections and the
   lower_loop split are done. Further refactors must stay byte-identical
   (compile both versions on every corpus folder and compare PTX strings).
2. Further hidden-set probing: more fresh shapes per category on the hosted
   runner, especially attention/softmax tiles with idle lanes and reductions
   with odd sizes. Fix any failure generally, never by workload name.
3. Feature freeze, then the user's own video, public repository and an early
   submission (resubmit if improved). Do not invent entrant details.

Win-track assessment (Step 69) stands: correctness and accuracy first, then
readability, then speed.
