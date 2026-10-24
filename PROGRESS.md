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
