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

## Step 23 — systematic predicate-preserving statement deletion

Completed 2026-10-24, compiler unchanged bb3a72b2...:

- Added tools/shrink_failure.py: hierarchical chunk/single-statement deletion
  across nested bodies. Both compilers must compile each retained trial; current
  reference must pass the original independent fixture oracle; historical suspect
  must retain a numerical mismatch. Errors/timeouts do not count as the defect.
  Each GPU run uses a separate context, and final independent replay is required.
- Historical scope failure reduced 11 → 10 statements by deleting unused arithmetic.
  Historical strided failure remained 8 → 8. Both completed with no further single
  deletion preserving the predicate, zero timeouts, and successful final replays.
  This is a deletion fixed point, not globally smallest IR or parameter shrinking.
- Verified CLI refuses a passing input without publishing a fixture, and a two-
  trial budget reports budget exhaustion rather than a minimality/fixed-point claim.
  Reports are incremental; changed compiler sources invalidate fixture publication.
- Preserved self-contained IR/meta/oracle fixtures under validation/counterexamples
  and all final reducer/compiler hashes, trial classifications and CLI gate reports
  in validation/2026-10-24-statement-shrinking.json. TESTING documents reproduction
  and the limits of an unchanged-oracle/reference-acceptance validity filter.

Next: sustained CPU CI using committed fixtures/isolated runtime contract, broader
index/control-flow type validation, and target bottleneck profiling. General IR
shrinking and production readiness remain unproven; submission still deferred.

## Step 24 — reproducible CPU contract CI configured

Completed locally 2026-10-24, compiler unchanged bb3a72b2...:

- Added check_cpu_contract.py and a committed snapshot manifest for two numerical
  counterexamples. Verifies input/meta/oracle hashes, copies only required assets,
  compiles twice with Python -I from an unrelated directory, checks deterministic
  GPU-validated PTX snapshots and three unsupported-type/layout CLI diagnostics.
- Local no-Torch checks pass from a clean copied checkout without ignored assets.
  A deliberately corrupted golden is rejected; snapshots never auto-refresh.
- Added requirements-ci.lock from the verified standalone environment and GitHub
  Actions workflow: Python 3.12/3.13/3.14, push/PR/manual/nightly, read-only repository
  permissions, pinned official action commits and retained JSON evidence artifacts.
  Resolved release tags/commit SHAs from official GitHub repository APIs.
- Evidence validation/2026-10-24-cpu-ci.json records local checks. First actual CI
  run is pending publication; do not claim matrix success or sustained history yet.

Next: push, inspect authoritative Actions job results, fix matrix/setup failures,
then record verified outcomes. CPU snapshots do not replace GPU numerical gates.
Production readiness remains unproven; submission deferred and release ZIP stale.

## Step 25 — first real CPU CI matrix verified

Completed 2026-10-24:

- Published CPU workflow/checker as 435355b. GitHub Actions run 40089837329
  completed successfully on Python 3.12.15, 3.13.16 and 3.14.8:
  https://github.com/StarAtNyte/emmy-ptx-compiler/actions/runs/40089837329.
- Downloaded all three report artifacts and verified five checks pass in each,
  Torch is absent, Emmy 0.3.11/NumPy 2.5.4 match constraints, and compiler hash
  exactly matches accepted bb3a72b2. Both fixture PTX snapshots match on each
  Python version; negative CLI probes reject cleanly.
- Saved job/step conclusions, artifact metadata and full hosted reports in
  validation/2026-10-24-cpu-ci.json. Compiler/workflow/checker remain unchanged
  after that successful run. The nightly trigger is configured; one successful
  push matrix is not evidence of sustained history or GPU correctness.

Next: target instruction/occupancy/bottleneck profiling and measured optimization,
plus broader index/control-flow typing probes. CPU CI covers only two committed
fixtures/three diagnostics; expand it with future retained counterexamples.
Submission remains deferred, release archive stale, production readiness unproven.

## Step 26 — conservative integer CSE rejected after target measurement

Completed 2026-10-24; accepted compiler unchanged bb3a72b2...:

- Built isolated instruction-level integer/address CSE keyed by opcode, actual
  operand registers/immediates and type. Raw emission (branches/labels/in-place
  updates) clears reuse; no text-based axis/value-name cache.
- Fifteen public PTX/resource inspections succeed. PTX instructions drop, e.g.
  m01 4547 → 4252 and vector add 61 → 44. This is static PTX, not machine-code
  execution count, utilization or an established bottleneck improvement.
- Seven alternating RTX 5090 pairs: m01 64.160 → 64.160 µs, a01 90.752 → 90.752,
  large sum 21.152 → 21.152, vector add 2.976 → 2.976, independent shared matmul
  31.392 → 31.424. No measured win; reject rather than adopt added compiler logic.
- 80 prior isolated cases and 20 frozen fresh scope/strided cases completed and
  passed. Intentionally stopped remaining 366-case acceptance sweep after timing
  rejection. These partial gates are NOT full candidate acceptance evidence.
- Preserved patch experiments/address-cse.patch and measurements/resources/partial
  gates in validation/2026-10-24-address-cse.json. Main remains unchanged.

Next: development-only driver-linked cubin/SASS inspection to distinguish static
PTX changes from native code. Installed verified NVIDIA cuobjdump/nvdisasm 13.0.85
redistributables under ignored artifacts/vendor. Prototype profile_sass.py uses
cuLink and disassembly without kernel execution; verify it and document link-path
and hardware limits before drawing conclusions. Hosted SASS/counters remain absent.

## Step 27 — local-driver native instruction inspection established

Completed 2026-10-24, compiler unchanged bb3a72b2:

- Added development-only tools/profile_sass.py using documented cuLink APIs and
  hash-verified NVIDIA cuobjdump/nvdisasm 13.0.85 redistributables. Inspects local
  sm86 and explicit sm120 output without executing kernels or changing lowering.
- Five before/after workloads inspected on each architecture. Vector add,
  attention and large sum have identical full disassembly despite smaller CSE
  PTX. Matmul code differs without a target timing win; rejected CSE remains out.
- Captures separate entry/helper counts, resources and full binary/disassembly
  hashes. Static counts include padding and are not dynamic execution counts.
  Raw outputs remain ignored artifacts; compact reports, hashes and official
  download provenance are in validation/2026-10-24-native-instructions.json.
- Local driver 595.99.02, CUDA API 13020. cuLink relocatable compilation can differ
  from direct module JIT, and is not hosted cubin capture. No hardware counters,
  utilization, roofline or target spill-traffic claim. Explicit optimization-level
  four control gives identical vector-add output to the default link settings.

Next: native code highlights expensive integer division/remainder lowering;
measure explicit power-of-two shift/mask lowering with exact signed semantics.

## Step 28 — power-of-two index candidate shows large target gains; gates live

Frozen unaccepted candidate 8da911f26419d9fbbf2f7836c79d9081ae320a83beab330e6f73067a65eb0c32:

- Unsigned constant power-of-two division/remainder emits shifts/masks. Signed
  floor division by a proven positive power of two uses arithmetic shift; Python
  remainder uses a mask, including negative numerators. No kernel-name selection.
- Seven paired RTX 5090 trials: m01 64.160 → 23.232 µs (2.762x), independent
  shared matmul 31.392 → 15.008 (2.092x). Attention/sum/vector-add remain within
  0.032 µs. Full first-trial outputs and later prefixes checked; clocks unlocked.
- Local-driver sm120 m01 entry falls 26960 → 1458 static SASS instructions, REG
  48 → 40. This codegen-path observation is not dynamic work or hosted SASS proof.
- Discovery record validation/2026-10-24-power2-index-discovery.json; reconstructible
  experiments/power2-index.patch applies to current accepted compiler. Main stays
  bb3a72b2 until all exact-candidate correctness/performance gates pass.
- Extended signed-index generator with explicit divisors and manifest parameters.
  Final proof corpora power2-index-proof-d{1,2,8,32} have unique case IDs; generated
  after candidate freeze. Earlier fresh-* directories have duplicate IDs across
  roots and are preliminary, excluded from acceptance provenance.

Live handles: 10753 final 48-case proof local checks; 92307 full 366 prior isolated
regressions; 85282 hosted public/matmul/proof suites; 96497 transformer; 61493
selected sanitizers. Reports use power2-index-* under artifacts. Also preliminary
43011 may finish independently; do not confuse it with final proof corpus.
Next: inspect those exact-hash gates, runtime equivalence, paired timing repeat,
then accept or reject. CPU snapshots will need reviewed updates only if adopted.

Step 28 checkpoint: final proof-local corpus completed 48/48; hosted public 15/15
and independent matmul 4/4; nine transformer checks; three no-Torch /tmp compilation
checks all pass for frozen 8da911f2. These completed reports now join discovery
record. Full prior regression, hosted proof suites and sanitizer gates remain live;
no acceptance. Profiling/experiment checkpoint published as 0d2d442.

## Step 29 — full correctness gates pass; unsigned tile rewrite regresses attention

Completed original-candidate checks, 2026-10-24:

- Original 8da911f2 passes all 366 prior isolated cases, 48 frozen divisor probes,
  42 added INT_MIN/INT_MAX probes, two committed CPU fixtures, 67 hosted checks,
  nine transformer runs, three no-Torch runtime checks and four sanitizers on 45
  selected cases. Extreme oracle observes sign and low bits; excludes INT_MIN/-1.
- Seven-pair repeat across all 15 public and four independent matmuls confirms
  m01 64.160 → 23.232 and second shared matmul 31.392 → 15.008, but a02 attention
  regresses consistently 53.920 → 58.016 (7.6%). Hold original candidate, not adopted.
- Signed-only control keeps a02 at baseline and m01 64.160 → 24.896 (2.577x).
  Eliminating unsigned divisor-one identities does NOT heal a02 (58.016), so that
  alternate control is also not adopted. Local-driver a02 resources remain REG56,
  LOCAL0; fewer native instructions alone do not explain execution latency.
- Freeze signed-only refinement 04b50e69438a15973a590a881562b78dc41f3fd23c0f870352e63ab334019979.
  Preserved experiments/power2-index-signed.patch and control reports in
  validation/2026-10-24-power2-refinement.json. Main still accepted bb3a72b2.

New exact-refinement gates live: 65578 all-19 target timing; 32213 prior 366 isolated
regressions; 83434 90 signed/extreme + 2 committed-fixture local checks; 54301
transformer; 91765 hosted proof/extreme/committed; 80476 sanitizers; 46395 no-Torch
runtime; 94789 rejection probes. Current reports use power2-signed-*.
Next: finish those exact-hash gates, then acceptance decision and reviewed CPU
snapshot update. Do not substitute original-candidate gates for refinement hashes.

## Step 30 — signed power-of-two lowering accepted without attention regression

Completed 2026-10-24:

- Accepted frozen refinement 04b50e69438a15973a590a881562b78dc41f3fd23c0f870352e63ab334019979.
  Exact signed floor/remainder strength reduction applies to positive singleton
  power-of-two divisors. Unsigned Tile rewrite remains rejected/archived.
- Exact-refinement gates: 366 prior isolated local cases + 90 signed/extreme
  probes + two committed fixtures pass (458 executions); 92 hosted direct cases
  plus 19 fully checked first timing trials (111 cases). Nine transformer runs,
  three no-Torch runtime checks, eight CLI rejections pass. Four sanitizers clean
  on 45 selected cases. These correlated cases are not hidden-set estimates.
- Final seven target pairs: public m01 64.160 → 24.928 µs (2.574x); independent
  256×512×384 37.760 → 17.344 (2.177x); 384×256×320 31.392 → 15.008 (2.092x).
  Other 16 workloads differ at most 0.032 µs, treated as noise. Full outputs first
  trial/prefixes later; clocks unlocked. Prior signed-only m01 24.896 corroborates.
- Local-driver sm120 m01 static entry 26960 → 1546 instructions, REG48 → 40;
  attention128 disassembly matches baseline. No hosted SASS/utilization claim.
- Combined acceptance/fixture/oracle hashes: production-iteration7.json. Broader
  unsigned+signed attention regression 53.920 → 58.016 remains disclosed in its
  discovery/refinement records. No kernel-name tuning selects the accepted pass.
- Added a GPU-verified INT32-extreme fixture to CPU CI manifest. Existing two
  goldens unchanged; new six-check no-Torch CPU contract passes locally. Updated
  README/DESIGN/TESTING and rewrote HANDOFF as the current snapshot.

Next: publish and verify actual six-check CI matrix, then refresh/test stale
release archive without submitting. Broader production gaps remain open.

## Step 31 — accepted implementation published; six-check CPU matrix verified

Completed 2026-10-24:

- Published signed-index acceptance source/evidence and third CPU fixture as
  074d4f9. GitHub Actions run 40092577026 passed all six checks on Python
  3.12/3.13/3.14. Downloaded artifacts confirm exact compiler 04b50e69, no Torch,
  and all fixture/diagnostic checks pass. Hosted reports now join iteration7.
- Updated default runtime installation commands to consume the tested version
  constraints in requirements-ci.lock, matching the verified CI install path.

Next: refresh stale archive from this committed checkpoint and run extracted
no-Torch hosted smoke checks. Packaging does not authorize contest submission.

## Step 32 — current code checkpoint archive rebuilt and verified

Completed 2026-10-24:

- Created fresh Python 3.12.13 environment artifacts/power2-release-runtime and
  installed requirements with the verified exact CI constraints. Torch absent.
- Archived committed 00da61e; extracted outside the working tree. Packaged compiler
  SHA 04b50e69 matches accepted source. Hosted scalar add, m01 matmul and attention
  smoke checks pass from the extracted path without Torch.
- Current checkpoint ZIP is artifacts/emmy-ptx-current-checkpoint.zip. Replaced
  stale artifacts/emmy-ptx-submission.zip with the same verified bytes; old archive
  and report preserved as artifacts/emmy-ptx-before-power2.zip/json.
- Archive/environment/requirements hashes and all smoke outputs preserved in
  validation/2026-10-24-power2-package.json. This is a verified code checkpoint,
  not contest entry submission or a claim of complete production readiness.

Next: broader index/control-flow typing probes and target hardware lower-bound /
bottleneck work. General operation/dtype/dynamic-shape support remains limited;
nightly CI has only an initial history. Submission stays deferred.

## Step 33 — predicate-valued indices found invalid PTX; context-aware fix accepted

Completed 2026-10-24:

- Added generate_predicate_index_cases.py: ten renamed families / 20 fixtures,
  independent NumPy boolean/0-or-1 oracles, including nested comparisons/ternaries,
  Let snapshots, truthiness, shared scalar and positively paired shared reads.
- Previous 04b50e69 fails all 20 final fixtures with driver invalid-PTX assembly
  errors. Initial logic fixture's != was changed to equivalent >0 for nonnegative
  remainder to stay within installed apply_binop vocabulary; preliminary reports
  are excluded from final evidence.
- Accepted exact b13ccba603f9e74bef1f65254284d7172a929eddeeba2ec48b22ba0f24028b26:
  separate generic expression result from integer/predicate use. Integer contexts
  materialize booleans as 0/1; conditions keep native predicate registers.
- Final 20 original + 20 frozen fresh cases pass locally and hosted. Four sanitizer
  tools clean on 20 original cases; nine transformer runs and three no-Torch /tmp
  compilation checks pass. Shared-pair fixtures actually emit one vector read.
- All 458 prior kernels emit byte-identical PTX, preserving their prior tested
  codegen and timing evidence. No fresh speedup claim. Eight rejection probes pass.
- Preserved reports/fixture hashes in production-iteration8.json. Added shared
  predicate fixture to CPU manifest; existing three goldens unchanged, seven
  no-Torch contract checks pass locally. Updated design/testing/current handoff.

Next: publish, verify seven-check CPU matrix and refresh code archive for changed
compiler. Broader type/rank/ABI validation and hardware lower-bound work remain.
Production readiness unproven; submission stays deferred.

## Step 34 — seven-check CI and extracted predicate-fix archive verified

Completed 2026-10-24:

- Published accepted predicate fix as 35208cf. Actions run 40093823051 passed seven
  checks on Python 3.12/3.13/3.14. Downloaded reports confirm b13ccba6, no Torch and
  all four fixture goldens/three diagnostics pass; preserved in iteration8.
- Archived committed 35208cf; packaged compiler b13ccba6 matches active source.
  Extracted scalar-add, matmul and attention hosted checks pass in the existing
  isolated no-Torch 3.12.13 environment from the preceding fresh pinned install.
  No dependency changes; this is not a new environment-creation claim.
- Current predicate-checkpoint/current-checkpoint/submission ZIP aliases all hold
  identical verified bytes; previous power2 archive/report retained. Latest package
  report predicate-package.json; older report preserved as power2-package.json.
- Updated current handoff/README. No submission or general production-readiness
  claim. Next: broader type/rank/ABI validation and target bottleneck/lower bounds.

## Step 35 — scalar f32 expressions, predicates and casts corrected

Completed 2026-10-24:

- Installed Let/Cond/Expr contracts permit typed scalar bindings and conditions
  over existing SSA values. Prior b13ccba6 truncates float expression literals,
  compares float registers as integers, and copies bits for typed numeric casts.
- Added generate_float_expression_cases.py: thirteen renamed families / 26 cases
  with independent explicit f32 NumPy formulas. Covers fractional comparisons,
  signed zeros, mixed ternaries/arithmetic, numeric casts/rounding, NaN truth and
  equality, integer-operand FMA and integer values stored into f32 buffers.
  Preliminary equal-zero sentinels were replaced before final corpora so wrong
  predicate selection is observable. All 26 final baseline probes fail.
- Accepted frozen 393d7600af297b365c6653e0b8dca84017a9c9d4dd9c482d845bf6e329cb5483.
  Float expression types follow bindings/literals, mixed operands promote to f32,
  predicates use float comparisons/truthiness, explicit Let conversions emit cvt.
  Floating scalar arguments, Select branches, folded FMA operands and integer
  writes into f32 storage convert numerically. Fold state remains f32.
- Final 26 original + 26 frozen fresh cases pass on both GPUs. Four sanitizers
  clean on 26 selected cases; nine transformer runs and three no-Torch /tmp
  compilation checks pass. All 498 prior kernels emit byte-identical PTX; no new
  performance claim. Eight existing and three scalar-expression diagnostics pass.
- Float indices now reject rather than silently truncate/emit invalid PTX. Floating
  Expr /,//,% explicitly unsupported; typed Assign divide remains supported.
  f32→i32 casts are checked only for finite in-range values. No integer buffers or
  full-type-checker claim. Limits/reports/hashes in production-iteration9.json.
- Added four GPU-checked scalar fixtures and one float-index negative CPU probe.
  Existing four goldens unchanged; all twelve no-Torch CPU checks pass locally.

Next: publish, verify actual twelve-check CI matrix and refresh changed-code
archive. Broader rank/ABI/static-shape validation and target roofline work remain;
production readiness unproven and submission still deferred.

## Step 36 — scalar-fix twelve-check CI and current archive verified

Completed 2026-10-24:

- Published scalar-fix source/fixtures/evidence as 48793dc. Actions run 40096495312
  passes twelve checks on Python 3.12/3.13/3.14. Downloaded reports verify compiler
  393d7600, Torch absent, eight positive snapshots and four negative diagnostics;
  full hosted evidence included in iteration9.
- Archived committed 48793dc and verified packaged compiler matches active source.
  Extracted hosted scalar-add, matmul and attention smoke checks pass using the
  existing isolated no-Torch 3.12.13 runtime; dependencies unchanged.
- Scalar/current/submission ZIP aliases now hold the same verified archive. Old
  predicate archive remains available and its record is preserved separately as
  predicate-package.json. Historical package evidence is scalar-package.json.
- Updated README/current handoff. No contest entry submitted or broad production
  readiness claimed. Next: rank/ABI/static-shape validation, additional operations
  and target hardware lower-bound/bottleneck work.

## Step 37 — index-rank contract corrected; metadata views audited

Completed 2026-10-24:

- Installed render_index explicitly defines empty=zero, one coordinate=flat,
  matched rank=row-major, otherwise sum of additive coordinates. Strictly rejecting
  all rank differences would reject valid IR. Previous address()/flat_linear()
  zipped against shape and dropped coordinates or ignored scalar-view flat indices.
- Added generate_address_cases.py, eight renamed families / 16 cases with independent
  NumPy formulas, including global/shared reads, additive writes, flat/empty/full
  controls and scalar logical view backed by an adequate physical allocation.
  Previous 393d7600 passes six controls and fails ten probes numerically.
- Accepted bf07d517f3a31619ce7f46514def0e8cedfdf1ea4f0f5b1206bb76fd4c85fac3.
  Address emission and shared-pair linear proofs now agree on additive mapping.
- All 32 original/frozen-fresh cases pass on both GPUs. Two reordered-ABI controls
  pass on both compilers locally and candidate hosted. Four sanitizers clean on
  16 originals; nine transformer runs, three no-Torch compilations and eleven
  diagnostic probes pass. Shared cases positively emit vector reads.
- All 550 prior kernels emit identical PTX, preserving prior codegen/performance
  evidence; no fresh speedup claim. Metadata audit found 38 valid scalar logical
  views vs allocation descriptors; no order/coop-width mismatch in that corpus.
  This is not full ABI/static-shape/launch validation or a bounds checker.
- Preserved gates/IR/oracle hashes/audit in production-iteration10.json. Added four
  GPU-verified addressing/reordered-ABI CPU fixtures; existing eight goldens stay
  unchanged. Sixteen local no-Torch contract checks pass.

Next: publish and verify actual sixteen-check CI, refresh changed-code archive;
then broad ABI/static-shape diagnostics that preserve these valid layouts.
Submission deferred and general production readiness unproven.

## Step 38 — address-fix sixteen-check CI and archive verified

Completed 2026-10-24:

- Published address fix as 998f3b0. Actions run 40099484929 passes sixteen checks
  on Python 3.12/3.13/3.14. Downloaded reports verify bf07d517, Torch absent and
  all twelve positive/four negative checks; full evidence joins iteration10.
- Archived committed 998f3b0. Archive SHA256
  79e840c54d289418c909fbf1aa04ec1a6ca807ea39cf606d7c8b1f6c72916b05
  and packaged compiler match verified bytes. Extracted hosted scalar-add, matmul
  and attention checks pass in the existing isolated no-Torch 3.12.13 runtime.
- Address/current/submission ZIP aliases hold identical verified bytes. Preserved
  prior scalar evidence as scalar-package.json; address-package.json preserves
  the address checkpoint after the later launch update. Updated README and current handoff.
- No entry submitted. Broad ABI/static-shape/launch diagnostics, operation/dtype
  coverage and target hardware lower-bound work remain; readiness unproven.

## Step 39 — validate launch geometry before PTX generation

Completed 2026-10-24:

- Baseline bf07d517 emits PTX for 25 malformed or unsupported launch probes,
  including multidimensional launches ignored by the one-dimensional mapping
  and cooperative Tile.block_threads mismatches. Invalid launches never executed.
- Accepted compiler 1dfcf75fe9d2ecb8d1f9aec50517b1583bc183c753bd594e61150997663df882
  checks dimension triples/types/positivity, x limits, one-dimensional y/z,
  nonnegative integer smem_bytes and matching positive cooperative width.
- All 25 probes now fail cleanly with empty stdout; three valid controls pass.
  Matching cooperative and reordered-ABI controls keep identical PTX. Scalar
  logical views remain legal. No argument-order or shape-equality restriction.
- All 594 retained kernels emit identical PTX; twelve committed GPU fixtures pass
  locally. Existing sixteen and new 28 launch checks pass in the isolated no-Torch
  runtime. Added launch suite to all three CPU CI versions. No fresh performance,
  sanitizer or hosted numerical claim. Evidence: production-iteration11.json.
- Initial local GPU command accidentally included cpu-fixtures/README.md after
  ten passes; reran all twelve manifest-selected fixtures successfully.

Next: publish and verify actual expanded CI, refresh archive; complete argument
ABI/static-shape/coverage diagnostics and hardware bottleneck work. Submission
remains deferred; full production readiness unproven.

## Step 40 — expanded CI and launch-validation archive verified

Completed 2026-10-24:

- Published source 0fea2c0. Actions run 40106607610 passes sixteen snapshot/feature
  checks plus 28 launch checks on each of Python 3.12/3.13/3.14. Downloaded reports
  verify exact 1dfcf75f compiler and Torch absent; preserved in iteration11.
- Archived committed 0fea2c0, compiler hash unchanged. Verified archive SHA256
  c15871976ee0cebb5f6e7458a4e37084fa15b6b35b2acc2ed37f63f2819c0662.
  Extracted hosted scalar-add, matmul and attention checks pass in the reused
  isolated no-Torch 3.12.13 runtime. Launch/current/submission aliases identical.
- Preserved prior address package record separately; launch-package.json preserves
  the historical launch checkpoint after the later ABI update. Updated README and current handoff.
- Official submit.py and local runner both pass args descriptors positionally;
  next ABI checks must require agreement with arg_order and permit consistent
  reordering. Logical/allocation rank equality remains an invalid constraint.
- Submission deferred. Full ABI/static-shape/coverage checks, broader operations
  and hardware lower-bound analysis remain; general readiness unproven.

## Step 41 — positional buffer ABI diagnostics

Completed 2026-10-24:

- Audited all 594 retained KernelOp argument sets: metadata matches input/output
  buffers. Official/local runners pass args list positionally, so its order must
  match arg_order; consistent permutations remain legal.
- Added pre-emission checks for a single KernelOp per file, unique metadata entry,
  complete unique buffer names, descriptor order/type, f32 physical allocation
  dtype and nonnegative integer allocation dimensions. Logical/physical shape
  identity deliberately not required; scalar logical views stay supported.
- Accepted 7fbe5ee4ed6c8688929b3892c06903a317bc9a4ca60e9496b0a231174ec35761.
  37 negative/five valid controls pass without Torch; failures emit no PTX and
  provide an ABI diagnostic. Multi-entry metadata with one matching entry is valid.
- All 594 prior kernels retain identical PTX; twelve retained GPU fixtures pass
  locally. Two derived reorder/allocation-reshape controls pass locally and hosted.
  Added 42-case ABI suite to CI alongside existing sixteen and 28 launch checks.
- No full schema/type/bounds/payload/output-role validation or new speedup claim.
  Evidence: production-iteration12.json. Next publish/verify CI and archive, then
  broader static-shape/control validation and target hardware bottleneck work.
  Submission remains deferred.

## Step 42 — 86-check CI and ABI checkpoint archive verified

Completed 2026-10-24:

- Published source b2a1b19. Actions run 40107112159 passes sixteen snapshot/feature,
  28 launch and 42 ABI checks on Python 3.12/3.13/3.14 (258 check results).
  Downloaded reports verify exact 7fbe5ee4 compiler and Torch absent; iteration12
  retains full reports. Baseline had 23 silent PTX emissions/14 tracebacks across
  37 negative ABI/entry probes; all now fail clearly before output.
- Archived b2a1b19 with compiler hash verified. Archive SHA256
  da5716307f46d4d7452e87d668617cbc63980a760fd6ab495ce01cf6070e0000.
  Extracted scalar-add, matmul and attention pass hosted using the reused isolated
  no-Torch 3.12.13 runtime. ABI/current/submission ZIP aliases match byte-for-byte.
- Preserved previous launch evidence separately; current-package.json records ABI
  checkpoint. Updated README and handoff. No entry submission/readiness claim.
- Remaining validation includes static tile shapes, grid coverage, argument roles,
  payloads and graph/schema consistency. Next performance priority is measured
  target launch/bandwidth calibration and workload lower bounds, with explicit
  cache/API limitations. General production coverage remains unproven.

## Step 43 — target hardware calibration and conditional workload models

Completed 2026-10-24:

- Added calibrate_hosted.py: committed typed-IR fixture transformed into copy and
  32-plane strided reads, PTX emitted by our compiler, exact seeded NumPy float32
  references. Only no-work timing control uses hand-written diagnostic PTX.
- Queried actual runner health/OpenAPI. Initial largest request (256+8 MiB)
  rejected with aggregate buffer limit 67108864 elements; preserved evidence and
  terminal partial run, corrected largest case to 224+7 MiB. Return cap 4194304.
- Five forward/reverse hosted trials, eight warmups/75 iterations: no-work median
  2.784 us; copy 1 MiB 3.008 us; read 64 MiB 12.896 us/5366.47 effective GB/s;
  read 224 MiB 146.400 us/1654.52 GB/s. All full first-trial outputs pass exact
  references; later prefixes pass. Raw per-request medians preserved, not raw
  iteration samples. Clocks unlocked, buffers reused and no eviction control.
- Inspected/pinned NVIDIA RTX Blackwell whitepaper Appendix A: 170 SMs,
  104.8 non-Tensor f32 TFLOP/s at boost, 1792 GB/s DRAM, 96 MiB L2. Effective
  64 MiB rate exceeds DRAM spec, so cache-sized workloads cannot use cold-DRAM
  models. Large streaming throughput is not a hardware traffic counter.
- Fresh identical-compiler self-comparison: three matmuls, six request medians
  each, full first-trial outputs/prefixes pass. Medians 25.072/19.024/14.912 us;
  nominal arithmetic model with prescribed 128/96/80 CTA ceilings gives
  3.402/1.701/1.276 us. The 7.37–11.69x gaps guide investigation, not measured
  utilization or rigorously clock-aware bounds. No compiler speedup claimed.
- Evidence: target-calibration.json; interpretation/reproduction TARGET_PERFORMANCE.md.
  Compiler remains 7fbe5ee4; package/86-check accepted-code CI remain valid. Updated
  research, testing, README and handoff. Next examine address/instruction/shared
  traffic and available warp parallelism, then measure any candidate on target.
  Submission deferred; broader validation/production coverage remains open.

## Step 44 — repository made private at user request

Completed 2026-10-24:

- User explicitly requested private visibility. GitHub PATCH returned private=true,
  visibility=private; independent gh repo view confirmed isPrivate=true for
  StarAtNyte/emmy-ptx-compiler. Routine pushes stay authorized; do not republish.
- Updated current handoff and repository/entry draft wording. Submission remains
  deferred. Earlier public-publication records are historical, not current state.

## Step 45 — cross-iteration shared pairing candidate validated

Completed 2026-10-24:

- Frozen candidate 40a1de84 pairs affine adjacent shared scalar reads across even/odd
  iterations of the existing bounded unroll, retaining alignment/immutable-index
  guards, within-body pairs and scalar tails. Main remains 7fbe5ee4 pending adoption.
- Added independent exact generator: 15 families × two naming variants, original
  seed 0 / frozen fresh 9161. Guards cover odd/weak alignment, local/shadowed/
  nonlinear/nonunit indices, writes/global reads, additive/flat layouts and
  4/5/64/65 trip boundaries. All 30 baseline controls and all 60 candidate cases
  pass on both GPUs; 60 positive/negative vector eligibility checks pass.
- 561/594 prior kernels retain identical PTX; all 33 changed kernels pass locally
  and hosted. Four sanitizers clean on 30 new cases and separate exact-allocation
  5/65 boundaries. Nine transformer runs, 86 existing no-Torch CPU checks,
  11 diagnostics and three isolated /tmp compilation comparisons pass.
- Final seven alternating target pairs across 19 workloads, after correctness:
  public m01 25.088 → 16.928 us (1.482x), independent 256×512×384
  19.040 → 12.992 (1.466x), 384×256×320 14.912 → 10.752 (1.387x).
  Other 16 workloads differ by at most 0.032 us. Full first-trial outputs and
  later prefixes pass. Earlier exploratory broad run overlapped correctness;
  acceptance must use final run. Clocks unlocked, no cache eviction/counters.
- Local-driver sm120 static entry counts: public 1546 → 4976; independent 384 shape
  1658 → 1402. Different JIT path, not hosted SASS or dynamic instruction proof.
  Public scalar shared-load PTX count 128 → 0, v2 count 64 → 128;
  PTX bytes 151318 → 115545.
- Preserved patch and complete evidence in cross-iteration-discovery.json. All
  gate jobs terminal. Next review/adopt, add CPU fixtures, verify updated CI and
  refresh archive. Repository private; no entry submission/readiness claim.

## Step 46 — accepted cross-iteration pairing; retained CPU fixtures

Completed 2026-10-24:

- Reviewed guards and revalidated source/patch and all sixty fixture hashes against
  recorded evidence. Adopted exact frozen 40a1de84; no lowering edits after GPU,
  sanitizer, integration and final seven-pair gates. Main now includes the change.
- Retained four GPU-checked CPU fixtures: tail, large tail, write guard and local
  index guard. All twelve existing PTX goldens remain unchanged. Twenty no-Torch
  snapshot/feature checks pass locally; unchanged launch/ABI suites bring CI to 90.
- Added production-iteration13 acceptance record referencing immutable discovery
  evidence. Updated design/testing/README, own-voice outline and entry draft with
  measured 1.482x/1.466x/1.387x gains and limits. No universal speedup claim.
- Next publish, inspect actual 90-check CI reports and refresh compiler archive.
  Prior archive is still 7fbe5ee4. Repository stays private; submission deferred.
