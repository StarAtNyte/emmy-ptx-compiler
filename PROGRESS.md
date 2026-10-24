# Progress

## Working agreement

Update this file and `HANDOFF.md` after each meaningful completed step. Record
what changed, evidence, decisions, rejected ideas and the next action. Never
replace the full objective with a smaller milestone.

## Objective and current direction — 2026-10-24

Build a production-level, competitive Emmy Kernel IR → PTX compiler and try to
win the Emmy Hackathon. The user explicitly deferred submission until the work
is production-level and win-worthy. Research Joseph Suarez/PufferAI, GPU MODE,
recent papers and repositories; turn findings into measured improvements.

The competition entry has **not** been submitted. Repository publication was
approved and completed: https://github.com/StarAtNyte/emmy-ptx-compiler.

## Verified baseline

- Compiler: `compile.py`, SHA-256
  `6fb720f76e33e9a77f25767cc43de4f923e498028a9f2581b720ac018399558f`.
- Latest published checkpoint before these continuity files: `f58120b`.
- All 15 public problems pass, across pointwise, reductions, matmul, fused and
  attention categories. Main validation: 88 hosted checks and 157 local cases.
- Complete staged transformer: 9 seed/depth checks, 147 stage outputs. Full-block
  tracing fails upstream, so separately traced stages execute the same model.
- Four sanitizer tools clean on 11 semantic audit cases; these are selected
  cases, not proof of safety for every possible IR.
- A frozen-compiler follow-up passed 10 valid fresh cases on both GPUs. Two
  original expressions failed upstream tracing, and one fixture required its
  centering operation inside the traced callable. No compiler changes were made.
- Release archive from commit `49265af` passed three hosted smoke cases from an
  extracted directory in fresh Python 3.12 without Torch. Later changes so far
  are documentation/test data; the compiler hash remains the same.

Evidence: `validation/2026-10-23-iteration5.json`,
`validation/2026-10-24-holdout.json`, and local
`artifacts/emmy-ptx-submission.json`.

## Implemented and measured

- Direct typed visitor; metadata ABI/launch contract; index expressions, guarded
  tiles, loops, persistent/seeded folds, shared memory, async copies, barriers,
  atomics and dynamic shared allocation.
- Coupled online-softmax identity handling; ordered tree merges and barriers;
  observable scratch uses per-slot atomic writer tickets, including named groups.
- Fixed repeated Tile labels and integer Let literal typing.
- Adjacent multiply/add fold → FMA: public a01 attention about 119.4 → 90.8 µs.
- Shared-memory scalar loop expansion: public m01 matmul about 98.9 → 74.4 µs.
- Explicit vector-memory pass was timing-neutral and archived. Broad scalar
  unrolling doubled a01 runtime and was restricted to shared-memory loops.
- Timings are paired observations on the RTX 5090 without locked clocks, not
  universal speedups or a hardware-efficiency claim.

## Gaps defining the next phase

- Current f32/static-shape support and finite operation/index vocabulary are not
  production-complete. No auxiliary thread bands or shared swizzles.
- No hardware roofline/bottleneck study, SASS/occupancy/spill audit, cold-cache
  matrix, systematic compiler differential fuzzing, or long-running CI yet.
- Most test cases have become development regressions. Do not present their pass
  rate as hidden-test generalization or real-model throughput.
- Stronger target-appropriate optimization and packaging/error diagnostics need
  explicit acceptance criteria before calling this production-level.

## Step 1 — direction reset and research discovery

Completed 2026-10-24:

- Revalidated clean working tree, public checkpoint and local RTX 3050 6 GiB,
  driver 595.99.02. The contest target is hosted RTX 5090; local timings cannot
  select contest optimizations by themselves.
- Added continuity instructions in `CLAUDE.md` and established the update-after-
  each-step rule in both status files.
- Located primary sources for the next research pass:
  - Joseph Suarez/PufferAI engineering account:
    https://puffer.ai/blog/engineering-4.0/
  - PufferLib implementation:
    https://github.com/PufferAI/PufferLib
  - GPU MODE official reference kernels and tooling:
    https://github.com/gpu-mode/reference-kernels
    https://github.com/gpu-mode/kernelbot
    https://github.com/gpu-mode/popcorn-cli
  - SOL-ExecBench: https://arxiv.org/abs/2603.19173
  - Atrex production-trace benchmark/agent: https://arxiv.org/abs/2607.14541
  - FastKernels: https://arxiv.org/abs/2605.23215
  - Agentic kernel optimization: https://arxiv.org/abs/2608.14560

These are discovered sources, not yet validated performance guidance. Check
paper dates, hardware and repository revisions before transferring conclusions.
Do not copy a CUDA/Triton/LLVM compiler lowering path into the submission.

Next: read the primary material and selected source at pinned revisions, create
a target-specific evidence-to-experiment matrix, and establish an RTX 5090
profiling/benchmark baseline before modifying the frozen compiler.

## Step 2 — primary-source review and reproducible resource baseline

Completed 2026-10-24:

- Read PufferAI's engineering retrospective, current PufferLib profiling source,
  GPU MODE's evaluator, SOL-ExecBench timing implementation and paper, selected
  FastKernels paper sections and README, and Atrex's primary abstract.
- Pinned five repository revisions in `RESEARCH_PRODUCTION.md`. PufferLib's
  current default is 5.0, not the retrospective's 4.0. SOL implementation uses
  twice device L2 size despite an older 256 MB comment. Do not transfer stale
  documentation or reduced-precision/datacenter results blindly.
- Added `tools/profile_resources.py`: compile-only static PTX inventory, local
  CUDA-driver resource/occupancy queries and hosted compilation register counts.
  Preserves compiler, IR, metadata and PTX hashes. No compiler changes.
- Local all-15 inspection succeeded: zero reported local bytes per thread on all
  public kernels. This rules out local-memory spills for this local JIT result,
  not hosted spills. Local m01 has one resource-resident block/SM (73,728-byte
  dynamic shared launch); a01 grid has only four blocks.
- Hosted all-15 compilation succeeded. a01 uses 128 registers/thread versus 48
  locally; m01 uses 38 versus 40 locally. Hardware/backend differences matter.
- Raw reports: `artifacts/resources-baseline-local.json` and
  `artifacts/resources-baseline-hosted.json`; compiler remains `6fb720f...`.
- Created bounded shared-loop expansion candidates 8/16/32/64 under ignored
  artifacts, leaving `compile.py` frozen. Timing/acceptance is pending.

Next: target paired timing of the expansion sweep, correctness on independent
shared-loop tails, and cache-conditioned local measurement support. Do not call
resource inspection a SASS/counter profile or claim hardware efficiency yet.

## Step 3 — target expansion sweep and cache-conditioned diagnostics

Completed 2026-10-24:

- Seven alternating paired RTX 5090 trials each: factors 8/16/32/64 gave m01
  speedups 1.056x / 1.057x / 1.088x / 1.122x over the existing factor four.
  Factor 64 measured 74.272 → 66.208 µs. Full outputs checked first trial,
  checked prefixes thereafter; clocks unlocked. This is candidate evidence,
  not an accepted compiler change. PTX grew to 187,430 bytes for this case.
- Added optional local `--evict-l2`: query device L2, zero twice that size and
  synchronize outside captured events. Label it eviction-conditioned, not a
  guaranteed cold-cache measurement. Default timing behavior remains available.
- Verified both modes on r02, atomic m02, large add and large scale (31 timed
  iterations each, full final output checks). Local r02: warm 34.816 vs
  eviction-conditioned 66.560 µs. Separate runs, unlocked clocks: diagnostic,
  not a paired speedup claim. Large bandwidth cases were effectively unchanged.
- Published resource evidence under `validation/2026-10-24-resources-baseline-*.json`;
  combined sweep/cache samples in `validation/2026-10-24-production-sweep.json`.
- Documented commands and measurement limits in `TESTING.md`.

Next: constrain factor 64 by extent/body-size budget, validate the exact candidate
on independent cases and both GPUs, check sanitizers, then decide whether to
merge. `compile.py` remains the baseline until these checks establish acceptance.

## Step 4 — exact bounded candidate and rejected backend-unroll control

Checks completed so far, 2026-10-24:

- Candidate `a180f7ec3408d0a74d86293159eac8b80eff3253e5bbfbb28c892a50f8c305a3`
  chooses a power-of-two factor capped by 64, extent and 1,024 expanded body
  statements. Existing shared-memory/straight-line eligibility remains intact.
- 59/59 local cases passed (15 public, 20 shared boundaries, two cross-attention,
  ten schedules, one MLP, eleven semantic audit). All 20 shared cases passed the
  hosted checker. Complete transformer: nine seed/depth runs, 147 stage outputs,
  maximum absolute stage error 4.292e-6.
- Exact-candidate seven paired target trials: m01 74.304 → 66.176 µs (1.123x),
  a01 90.752 → 90.784, a02 53.920 → 53.888, m02 5.088 → 5.056. The last three
  gaps are timing noise, not claimed wins.
- Following NVIDIA's documented loop-header `.pragma "nounroll"`, tested a
  separate backend-control candidate. Rejected: a01 90.752 → 563.616 µs,
  a02 53.920 → 193.056, m01 66.208 → 68.224, r02 21.152 → 72.320.
  Do not infer that lowering register pressure necessarily helps latency.
- Expanded the shared-loop generator to 20 default extents covering 31/32/33,
  63/64/65, 127/128/129, and the 1024 eligibility boundary. Added --sizes/--out.
- Froze candidate before seven new extent cases; local/hosted checks and remaining
  sanitizer tools are in progress. Do not call these untested outcomes passed.

Next: finish fresh-case/sanitizer gates, exercise the body-size budget, archive
rejected pragma evidence and synchronize the active design/release documentation.

## Step 5 — bounded expansion accepted after verification

Completed 2026-10-24:

- All gates passed for compiler `a180f7ec3408d0a74d86293159eac8b80eff3253e5bbfbb28c892a50f8c305a3`:
  50 hosted cases (15 public + 20 shared boundaries + 7 fresh extents + 8 body
  budget cases); 74 local cases (59 regression + 7 fresh + 8 budget); nine
  complete transformer checks / 147 stage outputs. Counts overlap prior suites.
- memcheck, racecheck, synccheck and initcheck all clean on seven selected cases:
  m01 and shared extents 63/64/65/1023/1024/1025. No blanket sanitizer claim.
- Body repeats 8/17/32/33 test expanded-statement budget reductions and the
  original-body eligibility boundary at 64 statements. Eight cases pass both
  GPUs. Fresh seven-extent suite was generated after compiler freeze and also
  passes both GPUs with no compiler changes.
- Accepted bounded shared expansion; current measured m01 latency 66.176 µs
  vs 74.304 µs previous compiler. The two attention workloads remain unchanged
  within noise. Local JIT m01 retains 40 registers/thread and zero local bytes;
  do not assume hosted spill behavior from this.
- Preserved rejected backend pragma patch and measurements. Updated README,
  design, testing and entry/walkthrough drafts to match actual active behavior.
- Combined evidence: `validation/2026-10-24-production-iteration1.json`.
  The old submission ZIP still contains the previous compiler and is **stale**;
  no replacement entry has been submitted or final release claimed.

Next highest-value step: deterministic typed-IR differential/metamorphic fuzzing
with mathematical references and minimized retained failures. Focus on control
flow, mutable axis/value bindings, cooperative indexing and reduction scratch
observability before SSA-aware CSE/loop-invariant addressing. Continue target
profiling where accessible; SASS/counters and hardware roofline remain open.
Keep submission deferred and update both continuity files after each step.

Step 5 runtime verification: the existing isolated Python 3.12 runtime has no
Torch installed and emits byte-identical PTX to the development environment for
p00, m01 and a01 when invoked from `/tmp`. This checks current standalone
compilation only; it does not rebuild the stale archive or claim a new hosted
execution from a packaged release.

## Step 6 — lexical-scope differential probes found real failures

Completed 2026-10-24:

- Added `tools/fuzz_semantics.py`: four typed lexical-binding families, seeded
  extents/inputs, independent exact NumPy formulas, name-renamed counterparts,
  and a reduced two-element/no-noise mode. No compiler or Emmy renderer is used
  for the numerical oracle. This is a bounded grammar, not general IR fuzzing.
- Installed Emmy 0.3.11's Let/Cond/Loop definitions explicitly scope ordinary
  bindings to their lexical block and allow nested shadowing. Existing flat
  compiler value maps leak branch/loop-local names into later code.
- All eight reduced cases fail with the accepted `a180f7ec...` compiler in
  separate processes: branch integer bindings can cause illegal memory access;
  branch/loop float and nested-axis bindings give incorrect values.
- An initial shared-context run suffered CUDA context poisoning after the first
  illegal access. Its subsequent failures are NOT independent observations.
  Added `tools/validate_isolated.py` and reran each case in its own process;
  `artifacts/fuzz-scopes-before-isolated.json` is the authoritative baseline.
- Saved baseline source as ignored `artifacts/compiler-before-scope-fix.py`.

Next: implement lexical environment restoration while preserving explicitly
mutable carried states, then rerun reduced/full probes and prior semantic,
model, sanitizer and target-performance gates. Production readiness is disproven
by these concrete cases until the issue is fixed.

## Step 7 — scope restoration works; snapshot aliasing is a second bug

Completed 2026-10-24:

- Intermediate scope-only compiler `094c269a...` passes the eight original reduced
  probes, all 48 original four-family seeded/renamed cases, and 57 existing local
  regressions. These are intermediate results, not final acceptance evidence.
- Cond and Loop now restore private SSA/index bindings. External carried states
  rebound by reduction primitives are copied to enclosing registers; loop folds
  remain exported, while shadowed outer axis bindings are restored.
- Added a fifth family: Let binds a snapshot of an externally mutable accumulator.
  The scope-only compiler fails both reduced cases because Let(Var(...)) aliases
  the mutable register instead of copying its value. Source saved separately as
  `artifacts/compiler-scope-only.py` before changing that behavior.
- Reran the original accepted compiler on all ten reduced cases, each isolated:
  0/10 pass (`artifacts/fuzz-scopes-v2-before.json`). Added per-case IR hashes to
  isolated reports to disambiguate generator revisions.
- Expanded the grammar with conditional full-warp and named-barrier shared-tree
  state updates, including observed/dead scratch variants; checks pending.

Next: fix Let snapshot capture, then verify the exact final compiler against the
expanded grammar, baseline counterexamples, previous suites, sanitizers and target
paired timing. Do not present intermediate results as testing the final hash.

## Step 8 — final scope/snapshot candidate passes prior gates

Completed 2026-10-24:

- Candidate compiler `7a1a3dfe68ada47c57a2ec49da97a40cf752ec88fe3f35480956ea0638d2f94e`
  restores lexical environments and emits a value copy for Let(Var(...)).
- 74/74 prior local regressions pass, plus nine complete transformer runs / 147
  stages, maximum absolute stage error 4.292e-6.
- Seven paired RTX 5090 trials each against accepted bounded baseline: m01
  66.208 → 66.208 µs; a01 90.752 → 90.752; softmax 2.976 → 2.976;
  r02 21.152 → 21.152. No measured regression on these four workloads.
- Seven-family grammar now includes conditional warp reductions and named-barrier
  trees with observable/dead scratch. Small-integer arithmetic gives exact f32
  oracles; renamed counterparts share inputs and expected results.
- Corrected generator metadata: scalar Tile.block_threads must be None, not 0.
  Earlier expanded runs with marker 0 are excluded from acceptance, even though
  the compiler treated it as false. Corrected corpora are `artifacts/fuzz-proof*`;
  their isolated baseline/final, hosted and sanitizer checks are in progress.
- The --minimal mode is family-specific parameter reduction, NOT an arbitrary
  statement delta-debugger or a claim of globally smallest failures.

Next: complete corrected-corpus gates, freeze compiler for fresh seeds, preserve
all provenance and update active design/validation documentation before accepting.

## Step 9 — scope and snapshot fixes accepted

Completed 2026-10-24:

- Accepted compiler `7a1a3dfe68ada47c57a2ec49da97a40cf752ec88fe3f35480956ea0638d2f94e`.
  The previous accepted compiler fails all 14 corrected reduced counterexamples,
  each in an isolated process; the fixed compiler passes all 14.
- Final local checks: 74 prior regressions + 14 reduced + 56 seeded/renamed +
  28 frozen fresh-seed cases = 172 cases. Final hosted checks: 15 public + 56
  seeded/renamed + 28 fresh = 99. All pass. The counts are correlated suites,
  not independent random samples or hidden-test success estimates.
- Complete transformer: nine seed/depth runs, 147 stage checks, maximum absolute
  error 4.292e-6. All four sanitizer tools clean on 15 cases (14 corrected
  reduced cases plus m01), including conditional full-warp and named-barrier
  shared-tree state publication. Three no-Torch standalone compilation checks
  produce byte-identical PTX from `/tmp`.
- All four paired target medians preserve the previous accepted performance:
  m01 66.208 µs, a01 90.752, row softmax 2.976, large sum 21.152. Clocks unlocked.
- Combined evidence with per-case IR/meta/oracle hashes, generator manifests,
  baseline failures, compiler freeze, runtime and timing records:
  `validation/2026-10-24-production-iteration2.json`.
- Generator marker correction and context-poisoned preliminary runs remain
  disclosed and excluded. Reduced mode is bounded parameter reduction; no claim
  of general fuzzing or automatic arbitrary-program minimization.
- Updated active design, testing and reviewer overview. Rewrote HANDOFF as a
  concise current snapshot; this file retains the chronological evidence trail.

Next: broaden the grammar to nested shared/buffer scopes and loop-carried-state
interactions, add systematic shrinking of genuinely new failures, then evaluate
SSA-aware integer/address CSE against these gates and target timing. Production
readiness is still unproven. Submission stays deferred; existing ZIP is stale.

## Step 10 — nested scopes expose allocation and seed-analysis bugs

Completed 2026-10-24:

- Added tools/fuzz_nested.py with four typed-IR families and independent exact
  NumPy oracles: nested shared declarations in branches/loops, shadowed fold
  seeds/axes, and a control that carries an outer state through an inner loop.
- On accepted compiler 7a1a3dfe..., 6/8 reduced renamed probes fail in isolated
  processes. Shared shadowing produces invalid PTX (outer declarations lost);
  nested maximum incorrectly seeds the outer additive fold with -infinity.
  Both outer-through-inner control cases pass.
- Authoritative baseline: artifacts/nested-before-reduced.json. Saved source at
  artifacts/compiler-before-nested-fix.py, reproducible from commit cca5bf2.
- Generator also supports explicit 96 KiB dynamic shared-memory cases (two live
  48 KiB declarations); these are not yet verified.

Next: allocate shared storage per declaration, restore pointer/shape/address-space
bindings on scope exit, and make accumulator discovery respect shadowed axes and
locally seeded nested folds. Then run both new and previous semantic gates.

## Step 11 — declaration identity and nested fold discovery fixed

Completed 2026-10-24:

- All eight reduced nested cases now pass locally, including the two controls.
- Shared storage is allocated by declaration identity, with deterministic symbols
  and offsets. Header declarations are persistent, independent of lexical names.
  Scope exit restores buffer pointers, shapes and address-space membership.
- Accumulator discovery stops at a shadowing axis and excludes locally seeded
  nested folds, while retaining true updates to an externally carried state.
- Dynamic shared-pool alignment follows its declarations. A clear compile error
  rejects required dynamic allocation larger than the prescribed launch provides.
- All 15 public kernels emit byte-identical PTX versus the accepted compiler;
  artifacts/nested-public-equivalence.json records both hashes per entry. This
  establishes unchanged public code generation; no fresh speedup is claimed.
- Further completed checks: 32 seeded nested cases and four 96 KiB dynamic
  cases pass locally; nine transformer seed/depth runs / 147 stages pass.
  Full regression and hosted checks remain in flight.

Next: finish 96 KiB dynamic, semantic, sanitizer and frozen fresh-seed gates;
then evaluate the next performance hypothesis without weakening those checks.

## Step 12 — nested fixes clear their gates; shared-load experiment measured

Completed 2026-10-24:

- Nested-fix compiler f99ea244... passes 172 previous local regressions plus
  8 reduced, 32 seeded, 4 explicit dynamic and 16 frozen fresh nested cases:
  232 local cases. Hosted: 32 seeded + 4 dynamic + 16 fresh = 52 new checks.
- Four sanitizer tools clean on 12 selected nested cases (8 reduced + 4 dynamic).
  Nine transformer seed/depth checks / 147 stages pass. Dynamic underallocation
  is rejected before PTX emission with the required/provided byte counts.
- All 15 public outputs are byte-identical PTX to cca5bf2, preserving its prior
  code generation; this is not a newly timed speed measurement.
- Tested a separate shared-read-only leaf optimization: prove adjacent aligned
  scalar reads, preload them as ld.shared.v2.f32, and preserve data dependencies
  and original FMA adjacency boundaries. Unlike the rejected earlier GLOBAL
  grouped-access pass, this pairs separate SHARED loads across pure arithmetic.
- Seven alternating paired RTX 5090 trials: m01 66.208 → 64.128 µs (1.032x),
  a01 90.784 → 90.784. Candidate remains isolated in
  artifacts/compiler-paired-shared.py and is NOT merged or generally verified.
  Source-builder and raw report are in artifacts; alignment follows NVIDIA PTX
  access-size requirements. Do not adopt before negative eligibility tests,
  independent cases and sanitizer checks.

Next: accept/archive nested-fix evidence, then validate the shared-pair hypothesis
against aliasing, intervening bindings/uses, weak alignment and shared writes;
keep main compiler as a measured fallback until candidate gates pass.

## Step 13 — nested correctness iteration accepted and preserved

Completed 2026-10-24:

- Accepted compiler SHA-256
  f99ea2445222a65e737a0b75dde99ee73c50d41dd546d30b8390abcec6f135c2.
  Combined verified evidence: validation/2026-10-24-production-iteration3.json.
- 232 local and 52 new hosted checks pass, with explicit 96 KiB dynamic cases;
  all four sanitizers clean on 12 selected cases. Nine full staged transformer
  runs / 147 stages and three no-Torch compilation checks pass.
- Source, case/oracle hashes, generator manifests, prior 6/8 failures, two passing
  controls, byte-identical public PTX proof, fresh freeze and underallocation
  diagnostic are preserved. No fresh public timing claim or final release.
- Preserved the separately measured, unaccepted shared-read pairing patch under
  experiments, with raw target samples and compiler/patch hashes in the record.
  M01 register count increases 40 → 48; PTX bytes fall 187430 → 151531. Its 3.2%
  target timing improvement still requires semantic eligibility testing.
- Updated README, DESIGN, TESTING and the current HANDOFF snapshot.

Next concrete step: validate paired shared reads with positive/negative motion,
mutation, alignment and alias cases; check independent matmul workloads,
regressions/sanitizers, then repeat target timings before any adoption. Main stays
at the accepted nested fix. General shrinking, broader IR/dtype support, target
counter/roofline work and production readiness remain open. Submission deferred.

## Step 14 — motion eligibility probes and isolated candidate tooling

Completed 2026-10-24:

- Added tools/generate_pair_cases.py: 13 motion/alignment variants with renamed
  counterparts and exact independent small-integer NumPy oracles. Includes
  adjacency/cancellation positives, nested weak alignment, destination uses and
  updates, index/axis shadowing, odd/nonadjacent offsets, writes, barriers and
  mixed global reads. All 26 one-trip fixtures pass the accepted baseline.
- tools/check_pair_codegen.py verifies actual emitted vector-read eligibility:
  all 26 candidate checks pass, including zero vectors in negative cases and
  exactly one vector in each positive one-trip case. This prevents a scalar
  fallback from masquerading as validation of the optimization.
- Added --compiler selection to hosted validation, sanitizer and transformer
  tools so candidate verification does not replace the accepted compiler.
- Candidate numerical isolated execution is in progress; codegen checks alone
  are not numerical or memory-safety acceptance. Main remains f99ea244....

Next: candidate runtime checks, loop repetitions, independent matmul shapes,
full regressions, sanitizer gates and fresh frozen cases; then repeat target
performance and assess whether the gain justifies adoption.

## Step 15 — candidate numerical and repeated-loop checks

Verified from current reports, 2026-10-24:

- Candidate da4f12dd4522262b5f6919e132a74725f10e024f3316debfbb7134cde2c5c33a
  passes 26 one-trip isolated local and 26 hosted motion fixtures. Baseline also
  passes all 26. Another 26 seed-71 seven-trip cases pass isolated local checks.
- Both 26-case codegen suites pass: positive cases actually vectorize and
  negative cases stay scalar. Nine staged transformer runs pass.
- Preserved completed reports in validation/2026-10-24-paired-shared-partial.json;
  this is partial evidence, not candidate acceptance.
- Revalidated live existing full-regression and sanitizer processes rather than
  restarting them. Memcheck and racecheck have completed clean on 27 selected
  cases; synccheck/initcheck are not yet recorded complete.
- Started independent four-shape matmul generation via expressions_shared_matmul.py
  and hosted seven-trip fixture validation. Both remain live at this checkpoint.

Next: poll generation session 63514 and hosted session 82961, inspect completed
regression/sanitizer records, validate independent matmuls, then frozen fresh
cases and repeated target timing. Main compiler remains f99ea244; no adoption.

## Step 16 — shared-pair semantic gates completed

Completed 2026-10-24 for frozen candidate da4f12dd...:

- All 232 prior local regressions pass. Motion suites: 26 one-trip local/hosted,
  26 seven-trip local/hosted, 26 fresh seed-1907 three-trip local/hosted pass.
  All three 26-case eligibility/codegen suites pass. Baseline one-trip 26 passes
  and nine transformer runs remain preserved.
- Four new independent matmul shapes pass locally and hosted; float64 NumPy
  references check their results. Actual PTX has 32 vector loads for 384×256×320;
  the other three shapes have zero and are controls, not positive coverage.
- All four sanitizers clean on 27 motion/public cases and separately on all four
  independent matmuls. Sanitizer scope is these cases only.
- Combined evidence validation/2026-10-24-paired-shared-gates.json preserves 16
  verified reports and hashes of IR/metadata/oracle files for 82 fixtures.
- Added reproduction and coverage limits to TESTING.md. Found an older matmul
  generator running separately; both distinct output runs completed successfully.
- Started final seven-pair RTX 5090 timing on m01, a01 and all four independent
  matmuls, session 48325; artifacts/pair-final-benchmark.json/log are incremental.

Next: finish target timing, inspect independent positive schedule regression/win,
verify isolated runtime packaging, then accept or archive the experiment. Current
compile.py is still accepted f99ea244; no submission or candidate adoption.

## Step 17 — aligned shared-read pairing accepted

Completed 2026-10-24:

- Adopted exactly tested compiler da4f12dd4522262b5f6919e132a74725f10e024f3316debfbb7134cde2c5c33a.
  Saved previous f99ea244 source at artifacts/compiler-before-pair.py; reproducible
  from committed 5791ed6. No kernel-name specialization selects the pass.
- Final seven alternating target pairs: public m01 66.176 → 64.160 µs (1.031x);
  independent vectorizing 384×256×320 matmul 35.488 → 31.392 µs (1.130x).
  Attention and three scalar controls differ by at most 0.032 µs, treated as noise.
  Full outputs checked first trial, prefixes thereafter; clocks unlocked.
- Isolated existing Python 3.12 environment has no Torch and emits byte-identical
  PTX from /tmp for public m01, independent vectorizing matmul and nested weak
  alignment scope fixture. This does not rebuild/test the stale release archive.
- Consolidated all semantic gates, fixture hashes, final raw timing and runtime
  evidence in validation/2026-10-24-production-iteration4.json. Updated README,
  DESIGN and TESTING with behavior, timing limits and baseline reconstruction.

Next: broaden adversarial IR correctness and improve unsupported-feature
validation/diagnostics before further address reuse; production readiness remains
unproven. Keep target profiling/roofline and long-running CI gaps explicit.
Submission remains deferred and the existing ZIP stale.

Step 17 publication: source, tools, documentation and acceptance evidence committed
as 2306028 and pushed to origin/master. Remote ref verified equal to the full local
commit. Active compiler hash remains da4f12dd...; active-path codegen rerun passes
all 26 one-trip eligibility checks. This is repository publication, not entry
submission or a rebuilt release archive.

## Step 18 — explicit unsupported features now fail before PTX emission

Completed 2026-10-24:

- Eight typed-reader-valid probes found silent acceptance in da4f12dd: f16
  tensor/Write/Assign/Init/Accum/Let, double shared storage, and B32-swizzled Write.
  All exited zero and emitted f32/unswizzled PTX despite those declarations.
- Added recursive feature validation before lowering with kernel/statement paths.
  Rejects these unsupported explicit types/layouts, preserving f32 storage and
  arithmetic, inferred dtypes, and i32 Let bindings. This is not a full type
  checker or an implementation of reduced precision/swizzles.
- CLI catches expected ValueError/NotImplementedError with concise stderr and
  nonzero exit. All eight final probes reject with zero stdout and no traceback.
- Final compiler hash 5af51dc2b497f8acd42f378eff27ed03c518ed4bd2952aedb940972eb4ac2998.
  255 selected kernels emit byte-identical PTX to da4f12dd. A separate 338-folder
  check covers all 232 prior regression names, including duplicate fixture copies,
  also byte-identical. No fresh timing claim or broad generalization from counts.
- Added reusable CPU-only CLI rejection and two-compiler PTX-equivalence tools.
  Preserved baseline/final results, hashes and equivalence driver in
  validation/2026-10-24-feature-validation.json. Updated DESIGN/TESTING.

Next: strengthen control-flow/index type validation and genuinely new adversarial
IR probes, then systematic failure shrinking and target profiling. Compiler dtype
coverage remains f32; production readiness unproven, submission deferred, ZIP stale.

## Step 19 — signed index division/remainder bug and performance control

Completed discovery/initial experiments, 2026-10-24:

- Installed Emmy 0.3.11 expr.apply_binop defines / and // as floor division and
  % as divisor-signed Python remainder. Existing PTX div/rem truncate toward zero.
- Added tools/generate_signed_index_cases.py: 12 typed-reader-valid signed-index
  probes (three operators, positive/negative varying divisor, renamed pairs),
  independent NumPy int64 floor/remainder references and in-bounds addresses.
  Accepted 5af51dc2 fails all 12 numerically in isolated GPU contexts.
- Generic correction candidate 51fc430f... passes all 12 locally and hosted;
  all four sanitizers clean on these cases. Seven-pair target timing rejects
  unconditional correction for performance: m01 64.160 → 80.544 µs. Attention
  and sum remain within 0.032 µs. This is NOT accepted as the final compiler.
- Separate range candidate conservatively bounds signed-int32 expressions and
  omits correction only for proven nonnegative dividend / positive divisor.
  Frozen v2 is artifacts/compiler-signed-ranged-v2.py; 14/15 public kernels
  already emit identical PTX to accepted baseline, including m01. M02 differs.
- Discovery evidence: validation/2026-10-24-signed-index-discovery.json. An earlier
  ranged timing run overlapped source edits and is excluded from final acceptance.

Live work: generic regression session 31758 writes signed-index-regression.json;
its compile.py input remains the generic candidate (worktree UNACCEPTED).
Frozen ranged-v2 local original/fresh cases session 88887; hosted original suite
30214; target m01/m02 timing 76614. Preserve these handles and inspect reports.
Fresh seed 2049 cases were generated after freezing ranged v2.
Next: finish frozen v2 runtime/performance gates; test range proof under lexical
shadowing and mutable bindings; rerun regression/sanitizer/transformer gates for
that exact hash before adoption. Accepted source remains 5af51dc2 in commit
 aa668e2 and artifacts/compiler-before-signed-index.py. Submission still deferred.

Step 19 continuation: generic regression completed 255/255. Restored compile.py
to accepted 5af51dc2 after its process exited. Frozen v2 completed 24/24 original
and fresh local cases, 12/12 hosted originals; seven target pairs preserve m01
64.160 → 64.160 µs and m02 5.056 → 5.056. These reports now join discovery evidence.
Preserved reconstructible v2 patch in experiments/signed-index-floor.patch.
Exact v2 full regression session 29029, selected sanitizer session 78413, and
transformer session 51894 are now live. Main stays accepted until these gates and
adversarial range-proof scope checks finish.

## Step 20 — adversarial range proofs pass; final gates approaching completion

Completed 2026-10-24 for frozen v2 e034896678a6a6b8ada4ae40c4b40059811e2f81f0993d0853a9ec6abee11073:

- Added tools/generate_index_scope_cases.py with six families and renamed pairs:
  branch/loop axis shadowing, negative strided-loop starts, shadowed divisors,
  int32-add overflow, and ternary indices. Independent NumPy oracles and a second
  output half check restored enclosing bindings. All 12 pass locally and hosted.
- Nine transformer seed/depth checks pass. Three selected no-Torch isolated /tmp
  compilation checks emit byte-identical PTX (m01, negative remainder, overflow).
- Full exact-v2 regression and final sanitizer tools are still live; latest
  observation was 221/255 regression cases passing, three sanitizer tools clean
  on the original selected suite and two on the new scope suite. Do not promote
  these observations into completed gate claims.
- Original/fresh signed suites and prior target timing evidence remain unchanged.
  Main compile.py stays accepted 5af51dc2; candidate is isolated and frozen.

Live handles: 29029 regression, 78413 original selected sanitizer suite,
34022 new scope sanitizers, 47277 fresh hosted signed suite. Original scope hosted
session 26913 and local 70863 completed, runtime 8305 completed. Next: collect final
reports, preserve exact-hash acceptance evidence, then accept only if all pass.

## Step 21 — signed-floor correction accepted after full coverage audit

Completed 2026-10-24:

- Accepted exact frozen e034896678a6a6b8ada4ae40c4b40059811e2f81f0993d0853a9ec6abee11073.
  All 255 selected regression cases pass. Case-name audit against older 232-case
  corpus found 59 omitted cases (schedules/shared tails/body budgets etc.); reran
  all 59 successfully before acceptance. 314 prior fixtures total, correlated.
- 24 original/fresh signed cases and 12 scope/overflow probes pass locally;
  36 signed/scope hosted checks pass. Nine transformer runs pass. Four sanitizers
  clean on 52 selected original/semantic/pair cases and 12 new scope cases.
- Three no-Torch /tmp compilation checks match byte-for-byte. 14/15 public kernels
  retain identical PTX to prior compiler; m02 differs. Seven paired target medians
  remain m01 64.160 µs and m02 5.056 µs, both unchanged. No new speedup claim.
- Preserved final hashes, fixture inventory, coverage audit and all gate reports in
  validation/2026-10-24-production-iteration5.json. Unconditional correction timing
  and excluded nonfrozen run remain disclosed in discovery record.
- Updated DESIGN/TESTING/experiments notes and rewrote HANDOFF's active snapshot
  to remove obsolete live-session instructions. All these sessions are complete.

Next: adversarial strided-loop lexical-binding and index/control-flow typing
checks, systematic shrinking, target profiling and sustained CI. Production
readiness remains unproven. Submission deferred; packaged release still stale.

Active-path follow-up: all eight unsupported-feature CLI probes still reject
cleanly with current e0348966; report included in iteration-five evidence.

## Step 22 — strided end/increment evaluation fixed and accepted

Completed 2026-10-24:

- Inspected installed StridedLoop.render: initialize the axis, capture explicit
  end once, expose axis_end inside the body, and evaluate step after each body.
  The previous compiler evaluated end and step before initializing the axis and
  did not provide axis_end. Same-name start shadowing was not used as the oracle
  because C initialization semantics make that probe less clear.
- Added tools/generate_strided_bound_cases.py: four bounded families / eight
  renamed cases, independent exact Python cursor/small-integer sum references,
  and enclosing-coordinate observations after the loop. Six original probes fail
  numerically on e0348966; two axis_end probes fail compilation with missing names.
- Accepted bb3a72b276f87396b15e9e8e5d0d984b9f1eb86330c43cb6189d6856b85f22fb.
  Captures end after axis initialization, evaluates dependent increments in the
  loop, binds axis_end, and conservatively declines dependent-bound range proofs.
- 16 original/fresh local checks and 16 hosted checks pass. Nine transformer runs
  pass; all four sanitizer tools clean on eight selected new cases. Three no-Torch
  /tmp compilation checks emit identical PTX to the development environment.
- All 350 prior regression kernels emit byte-identical PTX to e0348966, including
  all public kernels. This preserves tested code generation; no new timing claim.
- Final baseline failures, candidate gates and fixture hashes preserved in
  validation/2026-10-24-production-iteration6.json. Updated design/testing/handoff.

Next: control-flow/index typing, systematic shrinking and target profiling/CI.
Production readiness unproven. Submission deferred and release ZIP still stale.
