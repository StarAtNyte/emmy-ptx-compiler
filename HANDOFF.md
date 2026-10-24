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
  `5af51dc2b497f8acd42f378eff27ed03c518ed4bd2952aedb940972eb4ac2998`.
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

## Next concrete work

Aligned shared-read pairing is now accepted. Latest combined evidence:
`validation/2026-10-24-production-iteration4.json`. Optimization compiler da4f12dd... preceded the feature-validation change.
Current compiler preserves its PTX on the verified supported inputs. Prior baseline source is reproducible from
5791ed6 and saved as artifacts/compiler-before-pair.py. Historical experiment
patch is preserved; do not apply it to the newly active compiler.

Completed gates: 232 prior local regressions, 26 each one-trip/seven-trip/fresh
three-trip cases on both GPUs, 78 codegen checks, nine transformer runs,
four independent matmuls on both GPUs. Four sanitizer tools clean on 27 selected
motion/public cases and separately four matmuls. Three standalone no-Torch
compilation checks emit byte-identical PTX from /tmp.

Final seven-pair RTX 5090 timings: public m01 66.176 → 64.160 µs (1.031x),
independent 384×256×320 matmul 35.488 → 31.392 µs (1.130x). Attention and three
scalar control matmuls change by at most 0.032 µs (noise). Only the independent
384×256×320 schedule vectorizes (32 static vector instructions). Clocks unlocked;
full output first trial, prefixes thereafter. All generation/validation/timing
sessions listed in old progress entries are complete.

Next: broaden adversarial IR probes and strengthen unsupported-feature
validation/diagnostics before further expression/address reuse. Preserve these
semantic gates and avoid interpreting bounded grammar coverage as general safety.

Afterward consider SSA-aware integer/address reuse and loop-invariant addressing;
never cache expressions by text across mutable axes/state. Broaden fuzzing and
add actual statement shrinking as further correctness work.

Open production gaps: general shrinking/fuzzing, operation/dtype coverage,
unsupported-feature diagnostics, target instruction/counter/roofline analysis,
and sustained CI. Current tests do not establish production readiness.

Entry drafts: SUBMISSION.md and WALKTHROUGH.md. **Existing
artifacts/emmy-ptx-submission.zip is stale (several compiler iterations behind).**
No final release or entry submission has occurred. Deadline is Oct 27 23:59 AoE,
Oct 28 17:44 Nepal time; do not silently submit against the user's deferral.

Latest implementation/evidence publication: 2306028, pushed and remote-ref verified.
Active compile.py hash da4f12dd... matches tested candidate. Active-path 26-case
codegen rerun passes. No final archive rebuild or competition submission.

## Step 18 — current compiler feature validation

Current hash 5af51dc2...; evidence validation/2026-10-24-feature-validation.json.
Eight explicit unsupported declarations previously silently emitted f32/unswizzled
PTX; now reject before output with kernel/statement paths and concise CLI errors.
This covers f16 tensor/Write/Assign/Init/Accum/Let, double shared storage and
B32-swizzled Write. It is a feature boundary check, not a full type checker.
255 selected kernels plus 338 fixture folders covering all 232 prior regression
names emit byte-identical PTX to da4f12dd (overlapping, duplicate copies included).
Tools: check_feature_rejections.py and check_codegen_equivalence.py; TESTING.md
has reproduction. No fresh timing claim. Next: adversarial control-flow/index
typing probes and fuller validation, then systematic shrinking and target profiling.
