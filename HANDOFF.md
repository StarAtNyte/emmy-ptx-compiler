# Handoff — current state, 2026-10-24

## User instructions and goal

Build a production-level, win-worthy Emmy Kernel IR → PTX compiler and try to win
https://github.com/cloudrift-ai/emmy-hackathon-2026.

- **Do not submit yet.** User deferred submission while we improve the compiler.
- Research Joseph Suarez/PufferAI, GPU MODE and recent papers/repos; turn findings
  into measured experiments. `RESEARCH_PRODUCTION.md` pins inspected sources and
  defines acceptance gates. Continue iteratively; do not claim exhaustive research.
- **Update PROGRESS.md and HANDOFF.md after every meaningful completed step.**
  PROGRESS is the chronological evidence trail; this file is the current snapshot.
- Publishing the public repository and routine project pushes are already approved.
- No outreach/messages to organizers are authorized. Personal entry fields/video
  are missing but are not a blocker to technical work. Do not invent them.

## Workspace and current compiler

- Workspace `/home/nitiz/Projects/Self/Challenges/emmy`; branch `master`, origin
  https://github.com/StarAtNyte/emmy-ptx-compiler. Check git status/log before work.
- Current accepted compiler SHA-256:
  `bb3a72b276f87396b15e9e8e5d0d984b9f1eb86330c43cb6189d6856b85f22fb`.
- Use `.venv/bin/python` (3.12); host default is 3.14. Pinned Emmy 0.3.11,
  NumPy 2.5.4, CPU Torch 2.14.1. Runtime compiler does not need Torch or a GPU.
- Local RTX 3050 Laptop 6 GiB, sm_86; hosted RTX 5090, sm_120:
  `http://runner.riftstack.ai:48701`. Compiler emits PTX 8.0 targeting sm_89;
  local runner substitutes sm_86. Do not choose target optimizations locally.
- Ignored `.venv`, `challenge`, `upstream-emmy`, `artifacts` are local assets.
  Reproduction commands are in TESTING.md; a fresh checkout lacks those assets.
- Challenge pinned `ec8cde1943f88efea3827a95bda6085553d2d407`. Upstream HEAD
  differs from the pinned installed release: inspect installed 0.3.11 contracts.

## Required contract and implementation

`python compile.py path/to/kernel.ir.json > out.ptx` uses typed Graph.from_dict,
neighboring metadata ABI order and prescribed launch geometry. Required lowering
is ours: no Emmy CUDA renderer, nvcc or another compiler as a submission backend.
Development-only reference math is allowed. LLVM passes are organizer-permitted;
compile-time weight is very low. Correctness categories and review precede speed.

Current f32/static-shape compiler covers scalar/grouped accesses, expressions,
tiles, loops, conditional folds, shared memory, async copies, atomics, warp and
ordered tree merges. Observable tree scratch uses per-slot atomic writer tickets
without extra allocation; dead scratch takes the register path with final barrier.
No auxiliary thread bands, shared swizzles, broad dtype support or general IR claim.

Accepted optimizations: aligned adjacent shared-read pairing, adjacent reduction
multiply/add FMA and shared-only scalar
loop expansion by a power of two up to 64, extent and 1024 expanded statements.
Four-way baseline m01 was ~74.3 µs; larger bounded expansion reaches ~66.2 µs.
Global unrolling, explicit vector memory (neutral), and blanket backend nounroll
(regressions) are archived in experiments. Avoid repeating rejected hypotheses
without a new reason.

## Correctness findings and verified fixes

Flat value maps leaked ordinary branch/loop bindings and destroyed shadowed outer
axes. Let(Var(acc)) also aliased a mutable register instead of capturing its value.
Current compiler restores lexical environments; publishes externally carried
reduction states into stable enclosing registers; exports loop folds; snapshots
variable Lets with copies. Both bugs are fixed generally, not by case names.

The next iteration found shared declarations collapsed by source name and a
nested maximum selecting an outer sum seed. Storage now belongs to declaration
identity; lexical exit restores pointers, shapes and address-space membership.
Nested axes and locally seeded folds no longer leak into outer seed discovery.
Dynamic pool alignment follows declarations; undersized launch allocation fails
with required/provided byte counts.

New tools:
- `tools/fuzz_semantics.py`: seven bounded families with exact independent NumPy
  formulas, seeded parameters, renamed counterparts and reduced mode. Covers
  branch float/index bindings, loop float/axis bindings, accumulator snapshots,
  conditional full-warp reduction and conditional named-barrier shared tree.
- `tools/validate_isolated.py`: one process/CUDA context per case; records hashes
  and output. An illegal access poisons a context, so unisolated subsequent failures
  are not independent observations.
- `tools/profile_resources.py`: local JIT resources and occupancy ceiling, hosted
  register counts. No hosted counters/SASS/spill byte query is available.
- `tools/benchmark.py --evict-l2`: optional eviction conditioning outside captured
  events; not a guarantee of cold L2. Hosted cache/clock control is unavailable.

**Prior nested-correctness evidence:**
`validation/2026-10-24-production-iteration3.json`:
- 232 local: 172 prior + 8 reduced + 32 seeded nested + 4 dynamic + 16 fresh.
- 52 new hosted: 32 seeded + 4 dynamic + 16 fresh. All pass.
- Previous compiler fails 6/8 reduced nested probes; both controls pass.
- Nine transformer runs / 147 stages, maximum absolute error 4.292e-6.
- Four sanitizer tools clean on 12 selected nested cases (8 reduced + 4 dynamic).
- All 15 public kernels emit byte-identical PTX to cca5bf2; prior measured
  performance applies to identical generated code, not a new timing claim.
- Three isolated no-Torch byte-identical compilation checks from `/tmp`, including
  static and dynamic nested shared kernels. Underallocation diagnostic checked.

Prior scope corpus/evidence remains in production-iteration2.json and the current
172-case regression run. Counts are correlated development cases, not hidden-test
estimates. Fresh seeds are within existing families; reduced mode is parameter
reduction, not arbitrary program delta debugging.

## Commands and important local files

```sh
.venv/bin/python tools/fuzz_semantics.py --count 7 --minimal --out artifacts/fuzz-reduced
.venv/bin/python tools/fuzz_semantics.py --count 28 --out artifacts/fuzz-scopes
.venv/bin/python tools/validate_isolated.py artifacts/fuzz-scopes/*/ --report artifacts/fuzz-local.json
.venv/bin/python tools/validate.py --problems artifacts/fuzz-scopes --report artifacts/fuzz-hosted.json
.venv/bin/python tools/validate_transformer.py
```

Pre-fix accepted source: `artifacts/compiler-before-scope-fix.py`, hash `a180f7ec...`;
reproducible from public commit `bb8d2c8`. Scope-only intermediate
`artifacts/compiler-scope-only.py`, hash `094c269a...`, still has snapshot aliasing.
Current raw reports use `nested-*`; previous accepted source is
`artifacts/compiler-before-nested-fix.py` from cca5bf2 (7a1a3dfe...). Baseline
resource reports: local a01 48 registers vs hosted 128; do not assume allocation,
spills or performance transfer between GPUs.

Sanitizer binary:
`artifacts/vendor/cuda_sanitizer_api-linux-x86_64-13.0.85-archive/compute-sanitizer/compute-sanitizer`.
Use tools/sanitize.py; TESTING.md documents full suites and reconstruction.

## Current acceptance and next work

Current compiler bb3a72b2... is accepted after strided-loop bound correction.
Latest record: validation/2026-10-24-production-iteration6.json. All 350 prior
regression kernels emit byte-identical PTX to e0348966. New end/step/axis_end
fixtures: 16 original/fresh checks pass locally and hosted, nine transformer runs
pass, four sanitizers clean on eight selected new cases, three no-Torch /tmp
compilation checks match PTX. Installed StridedLoop.render initializes the axis
before capturing end once; step runs after each body; axis_end is body-visible.
The previous compiler failed six numerical fixtures and could not compile the
two axis_end fixtures. No new performance claim. All live sessions completed.

Prior signed-index record: validation/2026-10-24-production-iteration5.json. The original
5af51dc2 compiler failed all 12 signed-division/remainder probes. PTX truncates
signed division; Emmy 0.3.11 expr.apply_binop floors both / and // and uses Python
remainder. Current code corrects signs unless conservative signed-int32 bounds
prove nonnegative dividend and positive divisor. Bounds belong to register
identities; scopes preserve shadowed names. Unknown/overflow bounds take correction.

Gates: 255 selected prior local fixtures plus 59 older cases found by case-name
coverage audit; 24 original/fresh signed cases and 12 new scope/overflow cases
pass locally. 36 signed/scope checks pass hosted. Nine transformer runs pass.
Four sanitizer tools clean on 52 selected original/semantic/pairing fixtures and
12 new scope probes. Three standalone no-Torch /tmp compilation checks match PTX.
14/15 public kernels emit byte-identical PTX to prior compiler (m02 differs).
Seven paired RTX 5090 trials preserve m01 64.160 µs and m02 5.056 µs. Unlocked
clocks; full first output, prefixes thereafter. Candidate is now active exactly.

The unconditional sign-correction control passed correctness but slowed m01
64.160 → 80.544 µs; rejected. Earlier nonfrozen ranged timing is excluded.
Historical candidate patch experiments/signed-index-floor.patch applies to
compiler at aa668e2; do not apply to new main. All prior live handles completed.

Next: broaden control-flow/index typing and add failure shrinking;
add systematic failure shrinking, target instruction/roofline analysis and CI.
Division by zero and signed division overflow are not supported behavior.
Explicit unsupported dtypes/layouts fail before PTX output; this boundary check
is not a full type checker. Production readiness remains unproven.

Entry drafts: SUBMISSION.md and WALKTHROUGH.md. **Existing
artifacts/emmy-ptx-submission.zip is stale (several compiler iterations behind).**
No final release or entry submission has occurred. Deadline is Oct 27 23:59 AoE,
Oct 28 17:44 Nepal time; do not silently submit against the user's deferral.

No final archive rebuild or competition submission has occurred.

## Step 23 — statement reducer available

Compiler remains bb3a72b2... unchanged. tools/shrink_failure.py now removes nested
statement chunks only while the reference compiler passes the original independent
oracle and the historical suspect retains a numerical mismatch. Final isolated
replay required; source changes invalidate publication; budgets/timeouts avoid
unsupported minimality claims. Actual scope failure 11 → 10 statements, strided
failure 8 → 8, both observed deletion fixed points. Passing-input/budget CLI checks
pass. Evidence validation/2026-10-24-statement-shrinking.json and self-contained
fixtures validation/counterexamples; TESTING.md has commands/limits. All reducer
jobs completed. Next: CPU CI on committed fixtures, broader typing, target profiling.

## CPU CI — current verified state

Compiler unchanged bb3a72b2. check_cpu_contract.py uses only committed fixtures,
checks isolated no-Torch CLI execution, two GPU-validated PTX snapshots and three
unsupported-feature diagnostics. Local clean-checkout checks pass; deliberately
corrupted golden fails. requirements-ci.lock pins runtime constraints.
.github/workflows/cpu-contract.yml configures push/PR/manual/nightly across
Python 3.12/3.13/3.14 with pinned official action commits and report artifacts.

First GitHub Actions run 40089837329 succeeded on all three versions (3.12.15,
3.13.16, 3.14.8):
https://github.com/StarAtNyte/emmy-ptx-compiler/actions/runs/40089837329.
Downloaded reports confirm accepted compiler hash, no Torch, Emmy 0.3.11,
NumPy 2.5.4 and 5/5 checks on each. Full evidence:
validation/2026-10-24-cpu-ci.json. Nightly schedule is configured, but sustained
history and GPU coverage are not established by this first matrix. No live jobs
remain from this implementation step. Next: target bottleneck profiling/measured
optimization and broader typing probes. Submission remains deferred.

## Step 26 — address CSE rejected, SASS inspection underway

Main remains bb3a72b2. experiments/address-cse.patch preserves a conservative
register-identity CSE candidate. Static PTX shrinks, but five seven-pair RTX 5090
workloads have no gain (at most 0.032 µs gap). 80 prior/20 fresh cases passed;
remaining acceptance sweep intentionally stopped after rejection. Evidence:
validation/2026-10-24-address-cse.json. Do not adopt or claim complete safety gates.
Development-only profile_sass.py prototype uses local driver cuLink and verified
NVIDIA cuobjdump/nvdisasm 13.0.85 binaries in artifacts/vendor. Baseline/candidate
sm86 inspections sessions 77930/60969 are live. Next verify outputs, test explicit
sm120 code generation if supported, preserve exact instruction/resource evidence.
These are local-driver link outputs, not hosted SASS or measured hardware counters.

## Current work — native profiling and unaccepted power-of-two index candidate

Main stays accepted bb3a72b2. Address-CSE candidate rejected: five target timings
neutral despite smaller PTX; 80 prior/20 fresh checks passed before remaining
acceptance sweep intentionally stopped. Patch/evidence address-cse.* retained.
profile_sass.py now inspects local-driver-linked sm86/sm120 cubins. Native reports
and verified NVIDIA tool provenance: validation/2026-10-24-native-instructions.json.
These are local cuLink outputs, not hosted SASS/counters. TESTING documents limits.

Frozen candidate artifacts/compiler-power2-index.py, hash
8da911f26419d9fbbf2f7836c79d9081ae320a83beab330e6f73067a65eb0c32,
emits exact shifts/masks for power-of-two unsigned and signed-floor indices.
experiments/power2-index.patch reconstructs it from main. First seven-pair target
medians: m01 64.160 → 23.232 µs (2.762x); independent shared matmul 31.392 → 15.008
(2.092x). Attention/sum/vector add within noise. UNACCEPTED: correctness gates live.
Discovery: validation/2026-10-24-power2-index-discovery.json.

Live sessions: 10753 final proof-local; 92307 prior 366 isolated regressions;
85282 hosted public/matmul/four proof suites; 96497 transformer; 61493 selected
sanitizers. Final frozen proof dirs are power2-index-proof-d{1,2,8,32}; earlier
fresh-* preliminary dirs have duplicate case IDs and are excluded from acceptance.
Next collect gates, standalone runtime, repeat target timing, then decide adoption
and update CPU snapshots only with numerical evidence. Submission deferred.

Step 28 latest checkpoint: final proof-local 48/48, hosted public 15/15, independent
matmul 4/4, nine transformer runs and three no-Torch runtime compilation checks
all completed/pass, preserved in discovery record. Remaining live: 92307 prior
regressions, 85282 hosted proof suites, 61493 selected sanitizers. Repeat target
pairs after these complete before deciding adoption. Main unchanged; experiment
publication 0d2d442. Preliminary/finished local handles need no restart.
