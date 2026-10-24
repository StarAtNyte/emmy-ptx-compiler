# Handoff — current state, 2026-10-24

## Goal and authorization

Build a production-level, win-worthy Emmy Kernel IR → PTX compiler and try to win
https://github.com/cloudrift-ai/emmy-hackathon-2026.

- **Do not submit yet.** User deferred submission until production-level quality.
- Research Joseph Suarez/PufferAI, GPU MODE and recent papers/repos; apply measured
  findings. RESEARCH_PRODUCTION.md pins inspected sources and acceptance gates.
- Update PROGRESS.md and HANDOFF.md after each meaningful completed step. This
  file is the current snapshot; PROGRESS retains the chronological evidence trail.
- Public repository publication and routine pushes already approved. Origin:
  https://github.com/StarAtNyte/emmy-ptx-compiler, branch master.
- No organizer outreach/messages authorized. Entrant fields/video remain missing;
  do not invent them. Deadline Oct 27 23:59 AoE / Oct 28 17:44 Nepal time.

## Workspace/runtime

Workspace /home/nitiz/Projects/Self/Challenges/emmy. Use .venv/bin/python (3.12),
not the host default. Emmy 0.3.11, NumPy 2.5.4; development Torch CPU 2.14.1.
Compiler needs neither Torch nor GPU. requirements-ci.lock pins CI constraints.
Ignored challenge/upstream-emmy/.venv/artifacts are local assets, absent from a
fresh checkout. Challenge revision ec8cde1943f88efea3827a95bda6085553d2d407.

Local RTX 3050 Laptop 6 GiB sm86, driver 595.99.02; contest RTX 5090 sm120 at
http://runner.riftstack.ai:48701. Compiler emits PTX 8.0 sm89; local runner changes
that target to sm86. Local performance cannot select contest optimizations.

**Current accepted compile.py SHA-256:**
393d7600af297b365c6653e0b8dca84017a9c9d4dd9c482d845bf6e329cb5483.
Exact source frozen/tested as artifacts/compiler-float-expressions.py. Prior
b13ccba6 is reproducible from 35208cf:compile.py; 04b50e69 from 074d4f9.

## Contract/current implementation

`python compile.py path/to/kernel.ir.json > out.ptx` reads typed Graph.from_dict
and adjacent meta.json, preserves parameter ABI and prescribed launch geometry.
Our lowering must not use Emmy's CUDA renderer, nvcc or another compiler backend.
Development driver JIT/disassembly/reference math do not change that contract.

Static f32 storage/arithmetic, 32-bit indices, global/shared addresses, grouped
accesses, tiles, loops, conditionals, atomics, async copies, warp and ordered tree
merges. Observable tree scratch uses per-slot writer tickets. Unsupported explicit
dtypes/layouts fail before PTX output; this is not a complete IR type checker.
No auxiliary thread bands, swizzles, broad dtype/dynamic-shape support claim.

Accepted optimizations: adjacent fold FMA; bounded shared-only loop expansion
(up to 64 / extent / 1024 expanded statements); aligned adjacent shared-read
pairing; exact signed floor/remainder power-of-two shifts/masks. No kernel-name
selection. Generic signed / and // floor; % follows divisor sign. Lexical scopes,
Let snapshots, declaration-identity shared allocation and nested fold discovery
are fixed. Predicates in integer contexts become numeric 0/1, while conditions
retain predicate registers. Strided end capture happens after axis initialization; dependent step
executes after body; axis_end is body-visible. See DESIGN.md for details.

## Latest acceptance evidence

production-iteration9.json corrects scalar f32 expressions/conditions/casts.
Accepted 393d7600 supports f32 +,-,*, mixed comparisons/ternaries, IEEE float
truthiness, explicit numeric f32/i32 Let conversion, mixed scalar/FMA operands and
integer scalar writes into f32 storage. Floating indices reject; floating Expr
/,//,% unsupported (typed Assign divide still supported). Nonfinite/out-of-range
f32→i32 casts and integer-buffer storage remain outside scope.

26 final baseline cases fail on b13ccba6. Final 26 original + 26 frozen fresh cases
pass on both GPUs; four sanitizers clean on 26 selected cases, nine transformer
runs, three no-Torch compilations and eleven diagnostic probes pass. All 498 prior
kernels emit identical PTX, preserving prior codegen/timing evidence. Current CPU
manifest has eight positive fixtures/four negatives: twelve local no-Torch checks
pass; actual updated Actions matrix pending. Existing four goldens unchanged.

Previous predicate-index acceptance:
validation/2026-10-24-production-iteration8.json fixes predicate-valued indices.
Scalar booleans materialize integer 0/1 for arithmetic, comparisons, ternary
branches, integer Lets and memory offsets. Conditions retain .pred registers.
All 20 corrected original probes fail driver assembly on 04b50e69; final candidate
passes 20 original + 20 frozen fresh cases on both GPUs. Four sanitizers clean on
20 new cases, nine transformer runs and three no-Torch compilations pass. Shared
pairing positively emits a vector load in the predicate-address fixture.
All 458 prior kernels emit byte-identical PTX, preserving earlier codegen; no
fresh speedup claim. Eight rejection diagnostics still pass. New committed shared
predicate fixture adds a fourth CPU golden: all seven checks pass locally without
Torch, existing three goldens unchanged. Actions run 40093823051 passes all seven
checks on Python 3.12/3.13/3.14; downloaded exact-hash reports saved in iteration8.

Prior GPU/timing evidence for identical prior PTX:
validation/2026-10-24-production-iteration7.json:

- 366 prior isolated local cases plus 90 signed/divisor/extreme and two committed
  fixtures pass (458 executions; correlated development cases).
- 92 direct hosted proof/fixture cases plus full first-trial outputs on 19 paired
  timing workloads pass (111 cases, not hidden-test success estimates).
- Nine transformer runs pass; four sanitizers clean on 45 selected cases.
- Three isolated no-Torch /tmp compilations emit byte-identical PTX, and eight
  unsupported-feature CLI probes still reject cleanly.
- Final seven alternating RTX 5090 trials, unlocked clocks, full outputs first
  trial and prefixes later: public m01 64.160 → 24.928 µs (2.574x), independent
  256×512×384 matmul 37.760 → 17.344 (2.177x), 384×256×320 31.392 → 15.008 (2.092x).
  Other 16 workloads differ by at most 0.032 µs; no claimed win for those gaps.
- Local-driver sm120 m01 entry 26960 → 1546 static instructions, REG48 → 40.
  Local cuLink compilation is not hosted SASS, dynamic work or utilization proof.

Original combined unsigned+signed candidate 8da911f2 was NOT adopted: m01 reached
23.232 µs but attention128 regressed 53.920 → 58.016. Signed-only refinement
preserves attention128; local native disassembly matches baseline exactly there.
An unsigned divisor-one identity control did not heal the regression. Evidence:
power2-index-discovery.json and power2-refinement.json under validation.
Historical patches power2-index*.patch apply to 1ad0dbf source, not new main.

Earlier correctness records production-iteration2/3/5/6 preserve historical failures
and semantic gates. Larger bounded expansion/paired reads have separate timing
records. General CSE, global unroll, grouped global vector memory and blanket
nounroll controls are rejected/archived; avoid repeating without new evidence.

## Tools and CI

TESTING.md has commands. Semantic generators fuzz_semantics.py, fuzz_nested.py,
generate_pair_cases.py, generate_signed_index_cases.py, generate_index_scope_cases.py
generate_strided_bound_cases.py and generate_predicate_index_cases.py use
independent mathematical references.
validate_isolated.py uses a fresh process/context per case to avoid poisoned-context
cascades. Shared sanitizer binary is under artifacts/vendor (see TESTING).

shrink_failure.py deletes statements only while the reference passes the unchanged
oracle and the suspect numerically mismatches. Final replay required. Historical
scope 11 → 10 and strided 8 → 8 retained fixtures are committed under
validation/counterexamples. These are deletion fixed points, not globally smallest.

profile_resources.py reports PTX/JIT resources and local occupancy ceilings.
profile_sass.py uses local driver cuLink plus verified NVIDIA cuobjdump/nvdisasm
13.0.85 under artifacts/vendor, supports explicit sm120 output on sm86 hardware.
Compact evidence/provenance: validation/2026-10-24-native-instructions.json.
This differs from remote/direct-module JIT. No hosted counters/SASS capture or
roofline study. Address CSE reduced PTX but five target timings were neutral;
partial safety sweep stopped after rejection, patch/evidence preserved.

CPU Actions run on push/PR/manual/nightly across Python 3.12–3.14. First matrix
40089837329 passed all five checks without Torch; evidence in cpu-ci.json.
Current manifest has four GPU-verified fixtures plus three negative probes.
Accepted predicate fix published as 35208cf; Actions run 40093823051 passed seven
checks on all three versions. Downloaded reports confirm b13ccba6 and no Torch;
saved in iteration8. Existing three goldens unchanged, shared predicate fixture
added. These push matrices are an initial history, not sustained-nightly proof.

## Packaging and next work

Current verified archive built from 35208cf, compiler b13ccba6. Reused the
isolated no-Torch Python 3.12.13 runtime created fresh with pinned constraints in
the preceding phase; dependencies unchanged. Extracted outside the worktree;
hosted scalar-add, matmul and attention smoke checks pass. Latest evidence:
validation/2026-10-24-current-package.json. Paths:
artifacts/emmy-ptx-predicate-checkpoint.zip, artifacts/emmy-ptx-current-checkpoint.zip
and artifacts/emmy-ptx-submission.zip contain identical verified bytes. Previous
power2 archive/report retained as artifacts/emmy-ptx-power2-checkpoint.zip/json;
prior evidence preserved in validation/2026-10-24-power2-package.json.
No rebuild needed for documentation-only changes. No entry submitted.

Next: broaden index/control-flow type validation, operation/dtype coverage and
bounded failure shrinking. Continue target hardware lower-bound/bottleneck work;
no hosted counter/roofline evidence yet. Add future retained cases to CPU CI.
No candidate jobs remain live from this acceptance step. New CI runs may trigger
on documentation pushes; existing accepted-code seven-check matrix is verified.

Entry drafts SUBMISSION.md/WALKTHROUGH.md reflect current measured optimizations.
**No contest entry submitted.** Personal fields/video still missing; don't invent
them or contact organizers. General production readiness remains unproven.

Current packaging note supersedes earlier archive section: current verified ZIP
still contains prior b13ccba6. After committing scalar fix, verify twelve-check CI
and rebuild/check the archive for accepted 393d7600. Keep submission deferred.
All scalar-fix GPU/sanitizer/model/runtime gates completed; no need to restart them.
