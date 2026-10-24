# Handoff — current state, 2026-10-24

## Goal and authorization

Build a production-level, win-worthy Emmy Kernel IR → PTX compiler and try to win
https://github.com/cloudrift-ai/emmy-hackathon-2026.

- **Do not submit yet.** User deferred submission until production-level quality.
- Research Joseph Suarez/PufferAI, GPU MODE and recent papers/repos; apply measured
  findings. RESEARCH_PRODUCTION.md pins inspected sources and acceptance gates.
- Update PROGRESS.md and HANDOFF.md after each meaningful completed step. This
  file is the current snapshot; PROGRESS retains the chronological evidence trail.
- Repository is PRIVATE by user request (2026-10-24), confirmed through GitHub API.
  Routine pushes remain authorized; do not make it public again without instruction. Origin:
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
7ebf4ff2395ddd9bd4f7c2eb2aa213c8cd24c505149c2c757edf159c0cbf5a61.
Exact source frozen/tested as artifacts/compiler-static-shapes.py. Prior
40a1de84 is reproducible from 07f4851:compile.py. Prior
7fbe5ee4 is reproducible from b2a1b19:compile.py. Prior
1dfcf75f is reproducible from 0fea2c0:compile.py. Prior
bf07d517 is reproducible from 998f3b0:compile.py. Prior 393d7600
is reproducible from 48793dc:compile.py; b13ccba6 from 35208cf.

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
(up to 64 / extent / 1024 expanded statements); aligned within- and cross-iteration shared-read
pairing; exact signed floor/remainder power-of-two shifts/masks. No kernel-name
selection. Generic signed / and // floor; % follows divisor sign. Lexical scopes,
Let snapshots, declaration-identity shared allocation and nested fold discovery
are fixed. Predicates in integer contexts become numeric 0/1, while conditions
retain predicate registers. Strided end capture happens after axis initialization; dependent step
executes after body; axis_end is body-visible. See DESIGN.md for details.

## Latest shape correction

production-iteration14.json supports zero-domain tiles as no-ops before coordinate
decoding. Empty axis LIST has product one and still executes one cell. Static
nonnegative typed tensor/Tile/loop extents are validated; symbolic extents fail
clearly. Shared extents must be positive integer dimensions (scalar() is one f32),
alignment zero/natural or power of two. Zero-sized shared arrays unsupported.
Preserves logical/physical allocation views and valid ABI permutations.

Baseline 40a1de84 passes four controls, fails fourteen zero-domain probes. Final 36
original/frozen-fresh cases pass on both GPUs;27 negative/four valid CLI shape
controls pass. Initial generator used float index 0 in the sentinel control;
corrected to integer and retained invalid initial runs separately, excluded from
acceptance. Candidate source unchanged. All 658 prior kernels emit identical PTX;
nine transformer runs pass. No fresh speedup or changed shared-memory motion.

Four GPU-checked fixtures retained; existing sixteen goldens unchanged. Local
matrix 24+28+42+31=125 passes on Python 3.12.13/3.13.14/3.14.4 without Torch.
Next publish, inspect actual hosted job state and refresh archive for 7ebf4ff2.
Current archive still 40a1de84; hosted CPU CI last rejected for account billing.
All semantic/local test jobs terminal. Repo private; submission deferred.

## Previous accepted optimization — cross-iteration shared pairing

This optimization was accepted as 40a1de84. Exact frozen source
artifacts/compiler-cross-pair.py SHA256
40a1de84e3813cb1377d1dcdd0c0e369666c5638332d6cb18155cb31fcfc678c.
Reconstruct via experiments/cross-iteration-pair.patch against prior 7fbe5ee4.
The lowering adds aligned v2 shared reads across even/odd positions in the existing
bounded unroll. Requires affine unit axis coefficient, even remaining terms,
>=8-byte allocation alignment and immutable index dependencies. Skips within-body
pairs, shadowed/local indices, writes/global reads/control; tails stay scalar.

Evidence: validation/2026-10-24-cross-iteration-discovery.json. 30 baseline controls
pass; candidate 30 original +30 frozen fresh probes pass on both GPUs. 60 explicit
vector eligibility checks pass. Of 594 retained kernels, 561 emit identical PTX;
all 33 changed kernels pass locally and hosted. Four sanitizers clean on 30 new
cases plus separate exact-allocation odd-tail 5/65 checks. Nine transformer,
existing 86 no-Torch CPU checks, eleven diagnostic probes and three isolated /tmp
compilations pass. All sessions from these gates are terminal; do not restart them.

Final SEQUENTIAL hosted 7-pair /19-workload timing (after correctness requests
finished): m01 25.088→16.928 us (1.482x), independent 256×512×384
19.040→12.992 (1.466x), 384×256×320 14.912→10.752 (1.387x). Other 16 within 0.032 us.
Full outputs first trial/prefixes later; unlocked clocks. Separate exploratory
broad timing overlapped correctness; retain but use final timing for acceptance.
Local-driver sm120 native diagnostics differ from actual hosted code; public
static entry grows 1546→4976 despite speedup, so static counts are not dynamic work.

Main adopts the exact validated candidate. Four GPU-verified positive/tail/guard
fixtures are committed; existing twelve goldens unchanged. Twenty local no-Torch
snapshot/feature checks pass; CI now totals 90 checks with launch/ABI suites.
Published as 07f4851. Ninety checks pass locally on Python 3.12.13/3.13.14/3.14.4
without Torch. GitHub run 40109402820 did NOT start any jobs: account failed
payments/spending limit annotation; no test logs/results. Do not claim hosted
CI success or retry repeatedly without account state changing. Archive from
07f4851 passes three hosted GPU smoke checks; see current-package.json.
Broader goal/readiness gaps remain. Keep repository private; submission deferred.

## Latest acceptance evidence

Prior baseline performance calibration (before cross-iteration pairing):
validation/2026-10-24-target-calibration.json and TARGET_PERFORMANCE.md. Five hosted
trials, 8 warmups/75 iterations, alternating forward/reverse: no-work median
2.784 us, copy 1 MiB 3.008 us, 64 MiB strided read 5366.47 effective GB/s,
224 MiB read 1654.52 GB/s. Full outputs checked exactly first trial, prefixes
later. Clocks unlocked/reused buffers/no eviction or counters. Initial 256+8 MiB
request rejected by aggregate 67108864-element limit; corrected to 224+7 MiB.
Fresh same-compiler matmul medians: 25.072/19.024/14.912 us for public512,
256x512x384 and 384x256x320. Nominal f32/grid models leave 7.37–11.69x gaps but
are conditional on published boost rates, not measured utilization or rigorous
hardware bounds. Cache-sized tests must not use a cold DRAM-only roofline.
NVIDIA whitepaper pinned: 170 SMs, 104.8 non-Tensor FP32 TFLOP/s, 1792 GB/s,
96 MiB L2. Do not substitute reduced-precision Tensor Core peaks.

Latest compiler acceptance:


production-iteration12.json adds positional buffer ABI diagnostics: exactly one
KernelOp per IR file, a unique matching launch entry, complete unique arg_order,
args descriptors in the same order, f32 allocation dtype and static nonnegative
integer allocation shapes. Preserves consistently reordered parameters, scalar
logical views and differing logical/physical shapes. 37 negatives/five positive
CLI controls pass. All 594 prior kernels emit identical PTX; twelve retained GPU
fixtures pass locally and two new positive layout/order controls pass on both GPUs.
Existing 16 snapshot/feature and 28 launch checks remain; CI adds 42 ABI probes.
Actions run 40107112159 passes all 86 checks on Python 3.12/3.13/3.14; downloaded
reports verify 7fbe5ee4 and no Torch. Archive from b2a1b19 passes three hosted checks.
No full data/role/bounds/type checker.

Previous launch acceptance:


production-iteration11.json adds pre-emission launch geometry diagnostics. Rejects
invalid/missing dimension triples, nonpositive/noninteger dimensions, y/z != 1,
block.x > 1024, grid.x > 2147483647, invalid smem_bytes and cooperative block-width
mismatches. Baseline emits PTX for all 25 invalid/unsupported probes; accepted
compiler rejects them without PTX and passes three valid controls. All 594 retained
kernels emit identical PTX; twelve committed GPU fixtures pass locally. Existing
sixteen no-Torch checks pass; new 28-case launch CLI suite passes without Torch.
Actions run 40106607610 passes both suites on Python 3.12/3.13/3.14; downloaded
reports verify 1dfcf75f and no Torch. Archive from 0fea2c0 passes three hosted checks.
No new timing/sanitizer/hosted numerical claim; retained PTX remains identical.
Argument ABI/schema, grid coverage and general bounds validation remain open.

Previous address acceptance:


production-iteration10.json fixes address rank semantics per installed render_index:
empty=zero, single=flat, full rank=row-major, otherwise additive coordinate sum.
Shared pairing proofs use the same flattening. Previous compiler passes six controls
but fails ten numerical rank probes. Final 32 original/fresh cases pass on both
GPUs, plus two reordered-ABI candidate controls. Four sanitizers clean on 16 cases;
nine transformer runs, three no-Torch compilations and eleven diagnostics pass.
550 prior kernels emit identical PTX; no new timing claim. Metadata audit finds 38
valid scalar logical-view/allocation rank differences, no order/coop mismatch.
Do not impose strict shape-rank equality in future validation.

Current CPU manifest adds four addressing/ABI fixtures; existing eight goldens
unchanged. Sixteen no-Torch checks pass locally and on Python 3.12/3.13/3.14
in Actions run 40099484929; downloaded reports verify bf07d517. Archive verified
from 998f3b0. No full bounds/ABI validator claim.

Previous scalar acceptance:
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
pass. Actions run 40096495312 passes all twelve checks on Python 3.12/3.13/3.14;
downloaded exact-hash reports join iteration9. Existing four goldens unchanged.

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

CPU workflow is configured for push/PR/manual/nightly on Python 3.12–3.14.
Current suite: sixteen GPU-verified snapshots/four feature probes, plus 28 launch
and 42 ABI checks (90 total). Local isolated matrix passes all 90 on 3.12.13,
3.13.14 and 3.14.4 without Torch; 3.13/3.14 environments were freshly installed
from pinned requirements. Dependencies check clean. Full reports in iteration13.
GitHub run 40109402820 on source 07f4851 failed before any job step: annotation
says recent account payments failed or spending limit needs increasing. Hosted
90-check CI remains UNVERIFIED until account state changes. Keep repo private;
do not alter billing/spending or republish it. Prior public-repo 86-check matrix
run 40107112159 passed 7fbe5ee4, not the current compiler. Nightly history unproven.

## Packaging and next work

Prior verified archive built from 07f4851, compiler 40a1de84. Uses the existing
isolated no-Torch Python 3.12.13 runtime from the earlier pinned fresh install;
dependencies unchanged. Extracted scalar-add, matmul and attention hosted checks
pass. Latest evidence: validation/2026-10-24-current-package.json. Archive aliases
artifacts/emmy-ptx-cross-iteration-checkpoint.zip, artifacts/emmy-ptx-current-checkpoint.zip,
and artifacts/emmy-ptx-submission.zip hold identical verified bytes. Prior ABI
archive/report retained; evidence validation/2026-10-24-abi-package.json.
No rebuild needed for documentation-only changes. No entry submitted.

Next: broaden index/control-flow type validation, operation/dtype coverage and
bounded failure shrinking. Continue target hardware lower-bound/bottleneck work;
no hosted counters or rigorous clock-aware roofline yet. First calibration is
recorded above. Add future retained cases to CPU CI.
No candidate jobs remain live from this acceptance step. New CI runs may trigger
on documentation pushes, but current hosted CI is billing-blocked as stated above.

Entry drafts SUBMISSION.md/WALKTHROUGH.md reflect current measured optimizations.
**No contest entry submitted.** Personal fields/video still missing; don't invent
them or contact organizers. General production readiness remains unproven.

Next ABI/static-shape/launch validation must preserve permissive flat/additive
coordinates, physical/logical views and valid parameter-order permutations.

ABI follow-up evidence: challenge/submit.py builds run_args in args descriptor
order, and tools/local_runner.py builds positional parameters the same way.
Future validation should require agreement with arg_order, while permitting
consistent permutations of both lists. Do not require logical/allocation rank
identity. Full ABI/schema validation and required-grid coverage remain open.

Remaining validation needs initialization/output roles, payload sizes, graph/schema
consistency, static tile shapes and required grid coverage. Preserve valid views.
Next performance priority: investigate matmul instruction/address work, shared
traffic/dependencies and available warp parallelism using the calibrated gap.
Prescribed grids have 80–128 CTAs vs 170 SMs. Further optimization still needs
independent safety gates and target paired timing; no counter-proven bottleneck.
Latest static-shape source needs refreshed packaging; prior cross-iteration archive is historical.
