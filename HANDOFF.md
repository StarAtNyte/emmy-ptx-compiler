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
  `7a1a3dfe68ada47c57a2ec49da97a40cf752ec88fe3f35480956ea0638d2f94e`.
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

Accepted optimizations: adjacent reduction multiply/add FMA and shared-only scalar
loop expansion by a power of two up to 64, extent and 1024 expanded statements.
Four-way baseline m01 was ~74.3 µs; larger bounded expansion reaches ~66.2 µs.
Global unrolling, explicit vector memory (neutral), and blanket backend nounroll
(regressions) are archived in experiments. Avoid repeating rejected hypotheses
without a new reason.

## Latest correctness findings and verified fix

Flat value maps leaked ordinary branch/loop bindings and destroyed shadowed outer
axes. Let(Var(acc)) also aliased a mutable register instead of capturing its value.
Current compiler restores lexical environments; publishes externally carried
reduction states into stable enclosing registers; exports loop folds; snapshots
variable Lets with copies. Both bugs are fixed generally, not by case names.

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

**Authoritative current evidence:**
`validation/2026-10-24-production-iteration2.json`:
- 172 local: 74 prior + 14 reduced + 56 seeded/renamed + 28 frozen fresh seeds.
- 99 hosted: 15 public + 56 seeded/renamed + 28 fresh. All pass.
- Previous compiler fails all 14 reduced probes, isolated; current passes them.
- Nine full staged transformer runs, 147 stage checks, max absolute error 4.292e-6.
- memcheck/racecheck/synccheck/initcheck clean on 15 selected cases, not all IR.
- Four paired target medians retained: m01 66.208, a01 90.752, softmax 2.976,
  large sum 21.152 µs, unlocked clocks. Three no-Torch identical-PTX smoke checks.

These are correlated development cases, not a hidden-test pass-rate estimate.
Fresh seeds are within existing grammar, not held-out operation families.
`--minimal` is family parameter reduction, not general statement delta debugging.
Earlier `fuzz-final*` marker-0 corpora and context-poisoned runs are excluded.
Use corrected `artifacts/fuzz-proof*`. Freeze compiler+generator before fresh tests.

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
Current raw reports use `scope-final-*` and `fuzz-proof-*` prefixes. Baseline
resource reports: local a01 48 registers vs hosted 128; do not assume allocation,
spills or performance transfer between GPUs.

Sanitizer binary:
`artifacts/vendor/cuda_sanitizer_api-linux-x86_64-13.0.85-archive/compute-sanitizer/compute-sanitizer`.
Use tools/sanitize.py; TESTING.md documents full suites and reconstruction.

## Next concrete work

Broaden semantic generation to nested shared-buffer scopes and carried-state
interactions; retain/shrink new failures with independent oracles. Then experiment
with SSA-aware integer/address expression reuse and loop-invariant addressing,
using the new binding tests as mandatory gates. Do not cache expressions by their
text across mutable axis/accumulator updates. Preserve measured fallback code.

Open production gaps: general shrinking/fuzzing, operation/dtype coverage,
unsupported-feature diagnostics, target instruction/counter/roofline analysis,
and sustained CI. Current tests do not establish production readiness.

Entry drafts: SUBMISSION.md and WALKTHROUGH.md. **Existing
artifacts/emmy-ptx-submission.zip is stale (two compiler iterations behind).**
No final release or entry submission has occurred. Deadline is Oct 27 23:59 AoE,
Oct 28 17:44 Nepal time; do not silently submit against the user's deferral.
