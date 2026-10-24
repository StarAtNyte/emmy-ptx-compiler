# Handoff — start here

## User instructions that persist

- Objective: try to win the Emmy Hackathon with a production-level compiler.
- Latest direction: **do not submit yet**. Push the limits using recent research,
  repositories, Joseph Suarez/PufferAI and GPU MODE. The previous package is a
  baseline, not the new definition of production readiness.
- Update **both `PROGRESS.md` and `HANDOFF.md` after every meaningful step**.
- The user approved publishing the project. The public repo already exists;
  routine project pushes do not need that approval requested again.
- No video, entry email/team information or entrant confirmations have been
  supplied. Do not invent them or claim the competition entry was submitted.

## Workspace and authoritative state

- Workspace: `/home/nitiz/Projects/Self/Challenges/emmy`
- Public repo: https://github.com/StarAtNyte/emmy-ptx-compiler
- Git remote `origin`, branch `master`; checkpoint before continuity work:
  `f58120be07bdd7475748119b91e34a4384f7f05a`.
- Current compiler hash (accepted production optimization iteration 1):
  `a180f7ec3408d0a74d86293159eac8b80eff3253e5bbfbb28c892a50f8c305a3`.
- Start by checking `git status`, `git log`, current files and report hashes.
  This document is a snapshot, not authority over later changes.
- Python defaults to 3.14 on the host. Use `.venv/bin/python` (3.12).
- `.venv` has `emmy-ml==0.3.11`, NumPy 2.5.4, requests and CPU Torch 2.14.1.
- Local GPU: RTX 3050 Laptop 6 GiB, sm_86. Contest runner: RTX 5090, sm_120,
  `http://runner.riftstack.ai:48701`; compiler emits PTX 8.0 / sm_89.
- `gh` is authenticated as `StarAtNyte`; no credentials belong in these files.

## Competition contract

```sh
python compile.py path/to/kernel.ir.json > out.ptx
```

Use Emmy for typed parsing. The IR→PTX translation must be our own. Read neighbor
`meta.json` for positional arguments and the prescribed launch. No public-problem
name matching or hardcoded outputs. AI assistance is allowed, but the entrant's
3–5 minute own-voice video must demonstrate understanding.

Deadline: October 27, 23:59 AoE = October 28, 17:44 Nepal. Submissions may be
replaced until then; the latest entry is reviewed. Source:
https://riftstack.ai/hackathon and `challenge/TASK.md`.

The user supplied organizer clarification: LLVM passes are allowed; compile
time has very low weight; focus on efficient PTX. Do not infer that an existing
compiler may perform the required translation for us. Generated-kernel speed is
the published last tiebreak after correctness and code review.

## Read in order

1. `PROGRESS.md` — evidence and latest completed step.
2. `README.md`, `DESIGN.md` — current implementation and limits.
3. `compile.py` — standalone implementation.
4. `TESTING.md` — reproduce corpora, execution, timing and sanitizers.
5. Latest `validation/*.json` — compiler-specific evidence, including failures.
6. `RESEARCH.md`, `experiments/README.md` — earlier decisions and rejected ideas.

## What exists and how to resume

`challenge/` is the official repository, pinned at
`ec8cde1943f88efea3827a95bda6085553d2d407`. `upstream-emmy/` is a separate
source checkout. Inspect the installed 0.3.11 classes before assuming current
upstream matches the competition release.

Ignored local assets: `.venv/`, `challenge/`, `upstream-emmy/`, `artifacts/`.
They are not in a fresh public clone. See `TESTING.md` to regenerate them.

```sh
git status --short
git log -3 --oneline
.venv/bin/python tools/validate.py --report artifacts/public-results.json
.venv/bin/python tools/local_runner.py challenge/problems/[aprfm]*_*
.venv/bin/python tools/validate_transformer.py
```

Use the hosted benchmark for contest choices. It alternates baseline/candidate
order, checks results, and supports clipped timing returns only after a full
first-trial check. Clocks are not locked. The local benchmark uses CUDA graph
events to avoid Python dispatch gaps and resets atomics outside timed regions.

```sh
git show 7cf92b2:compile.py > artifacts/compiler-baseline.py
.venv/bin/python tools/benchmark_hosted.py challenge/problems/m01_matmul_512 challenge/problems/a01_attention --baseline artifacts/compiler-baseline.py
```

Compute Sanitizer binary currently available at:
`artifacts/vendor/cuda_sanitizer_api-linux-x86_64-13.0.85-archive/compute-sanitizer/compute-sanitizer`.
Use `tools/sanitize.py`; record the selected cases rather than claiming all
kernels are sanitizer-clean.

## Verified evidence and release assets

- `validation/2026-10-23-iteration5.json`: 88 hosted checks, 157 local cases,
  semantic audit, model checks and preserved performance.
- `validation/2026-10-24-holdout.json`: compiler frozen before fresh cases;
  ten valid cases passed both GPUs without source changes. Two upstream trace
  failures and a fixture correction are preserved explicitly.
- Current implementation passes all 15 public problems. This is not proof of
  hidden-set success or production readiness.
- `artifacts/emmy-ptx-submission.zip` was built from commit `49265af`; compiler
  hash matches current code. Its JSON records three no-Torch fresh-runtime
  smoke passes. It predates later documentation/tests and is not a new release.
- `SUBMISSION.md` and ignored `artifacts/submission-entry-fields.json` contain
  entry text. Personal fields and confirmations are unset. `WALKTHROUGH.md`
  contains recording guidance. **Submission is deferred by the user's request.**

## Immediate next action

Research discovery is complete, deeper reading is next. Use the primary links in
`PROGRESS.md`. Pin repository revisions and distinguish f32/RTX 5090-applicable
ideas from B200/Hopper-specific or reduced-precision results.

Create a research-to-experiment matrix: source claim, relevant code, hardware,
applicability to our IR/launch contract, experiment, correctness gate and target
metric. Prioritize profiling/roofline, register pressure/spills, launch geometry,
memory traffic, async-copy scheduling, numerical stability and real integration.
Keep the current compiler as a measured fallback while testing candidates.

After the step, update both continuity files, commit the evidence and push the
approved project. Do not manufacture progress by rerunning green tests without
a changed hypothesis or concern. Do not finish or block the goal merely because
video fields are absent while this technical improvement phase remains active.

## Step 2 checkpoint — 2026-10-24

Primary review and resource baseline completed; see `RESEARCH_PRODUCTION.md`
for pinned revisions, inspected scope, experiment matrix and release gates.
New tool: `tools/profile_resources.py`. Reports in ignored artifacts:
`resources-baseline-local.json` and `resources-baseline-hosted.json`, all 15
public kernels. Local reported local memory is zero throughout. Attention a01
uses 128 hosted registers/thread vs 48 local; m01 uses 38 hosted vs 40 local.
Do not infer hosted spill bytes or occupancy: API exposes register counts only.

**Next action supersedes earlier discovery instructions:** measure ignored
`artifacts/compiler-unroll{8,16,32,64}.py` against
`artifacts/compiler-production-baseline.py` using paired hosted benchmarks.
These change only the existing shared-leaf-loop expansion factor. Check PTX size,
registers, independent tail cases and model/sanitizer gates before acceptance.
`compile.py` remains unchanged. Add cache-conditioned local timing afterward;
keep cache-conditioning operations outside the measured graph events.

## Step 3 checkpoint — 2026-10-24

Sweep completed: factors 8/16/32/64 improve hosted m01 by 5.6/5.7/8.8/12.2%.
Factor 64 is a promising **unmerged** candidate (187,430-byte PTX for m01).
Raw and combined evidence is committed under `validation/2026-10-24-production-sweep.json`.

Local benchmark now accepts `--evict-l2`; allocation/memset/synchronization stays
outside captured event interval. Four workloads, including atomic m02, pass full
output checks in both modes. r02's cache sensitivity is diagnostic only; reports
are not paired and clocks are unlocked. See TESTING for interpretation.

**Immediate next:** bound expansion by extent and a body-size budget, run exact
candidate against public and independent tail/schedule/model cases, sanitize
changed shared-loop behavior and recheck target timings. Only then merge if green.
A factor-16 exploratory local suite may still have a running process; it is not
acceptance for the exact new candidate. Production compiler still `6fb720f...`.

## Step 4 checkpoint — exact candidate under acceptance checks

`compile.py` is now the **uncommitted candidate** `a180f7ec...`, bounded factor
up to 64 / extent / 1024 expanded body statements. It passed 59 local cases,
20 hosted shared cases and the nine complete transformer runs (147 stages).
Exact target paired result: m01 74.304 → 66.176 µs; attention and m02 unchanged
within noise. Compiler source has not changed since freeze recorded in
`artifacts/production-bounded-freeze.json`.

Separate ignored `compiler-nounroll.py` is **rejected**: disabling backend
unrolling regressed every measured workload, especially attention. Never merge.
See `artifacts/production-nounroll-paired.json` for raw timings.

Pending before acceptance: finish sanitizer run at `artifacts/production-bounded-sanitizers`,
fresh seven-extent local/hosted reports, body-budget regressions, active docs,
combined evidence and commit. These supersede Step 3's baseline-current claim.

## Step 5 — accepted current compiler and next action

Current compiler is **accepted** `a180f7ec3408d0a74d86293159eac8b80eff3253e5bbfbb28c892a50f8c305a3`.
This supersedes the baseline hash and the candidate/pending notes above.

Verification: 50 hosted cases, 74 local cases, nine transformer seed/depth runs
(147 stage outputs), four sanitizer tools clean on seven selected cases.
All seven fresh extents and eight body-budget cases pass both GPUs. Combined
record: `validation/2026-10-24-production-iteration1.json`; raw artifacts use
`production-bounded-*`. m01 paired target timing is 74.304 → 66.176 µs, 1.123x;
both attention cases unchanged within noise. No hidden-test or production-ready
claim. Old `artifacts/emmy-ptx-submission.zip` is stale: do not submit it.

**Next concrete work:** build seeded typed-IR differential/metamorphic fuzzing
with independently computed expected values and shrinking of failures. Include
control flow, mutable values/axes, cooperative schedules and observable scratch.
Then test SSA-aware address-expression reuse / loop-invariant addressing using
those semantic gates. Inspect actual generated behavior; blanket backend
nounroll is rejected. Target SASS/counter access and roofline remain unresolved.
No new permission is needed for research, local changes, tests or approved repo
pushes. Submission remains deferred by user direction.

Current runtime smoke: isolated no-Torch Python, cwd `/tmp`, p00/m01/a01 PTX
matches development output byte-for-byte. Recorded in the combined validation.
This is compilation verification, not a new archive smoke execution.
