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
04b50e69438a15973a590a881562b78dc41f3fd23c0f870352e63ab334019979.
Exact source was frozen/tested as artifacts/compiler-power2-signed-only.py.
Prior accepted source bb3a72b2 is reproducible from 1ad0dbf:compile.py.

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
are fixed. Strided end capture happens after axis initialization; dependent step
executes after body; axis_end is body-visible. See DESIGN.md for details.

## Latest acceptance evidence

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
and generate_strided_bound_cases.py use independent mathematical references.
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
Current manifest adds a third GPU-verified INT32-boundary fixture under
validation/cpu-fixtures. Existing two goldens unchanged. Accepted source published
as 074d4f9; Actions run 40092577026 passed six checks on all three Python versions.
Downloaded reports confirm compiler 04b50e69 and no Torch; saved in iteration7.
One push run is not sustained CI history.

## Next concrete action / open scope

1. Refresh/test the stale release archive from the accepted compiler in a fresh
   no-Torch environment. Packaging is not competition submission; keep submission
   deferred. Existing artifacts/emmy-ptx-submission.zip is several iterations stale.
2. Broaden index/control-flow type validation and dtype/operation support; continue
   target-appropriate profiling and measured optimization. General shrinking,
   hardware roofline/counters, dynamic shapes and sustained CI remain open.

All candidate correctness, timing, sanitizer, model and runtime jobs for this
acceptance step have completed. No need to restart old handles. Entry drafts:
SUBMISSION.md/WALKTHROUGH.md. No final release or contest entry submitted, and
current evidence still does not establish general production readiness.
