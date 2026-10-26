# Reproduce the checks

Start from the runtime setup in README and clone the official problem repository
into `challenge`. Development tracing/reference math additionally uses CPU Torch:

```sh
uv pip install --python .venv/bin/python torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu
```

## Generate the development corpora

```sh
.venv/bin/python challenge/tools/generate.py tools/expressions_variations.py --emmy .venv/bin/emmy --out artifacts/variations
.venv/bin/python challenge/tools/generate.py tools/expressions_schedules.py --emmy .venv/bin/emmy --out artifacts/schedules
.venv/bin/python challenge/tools/generate.py tools/expressions_loop_tails.py --emmy .venv/bin/emmy --out artifacts/loop-tails
.venv/bin/python challenge/tools/generate.py tools/expressions_bandwidth.py --emmy .venv/bin/emmy --out artifacts/bandwidth
.venv/bin/python tools/numerical_variations.py
.venv/bin/python tools/generate_alignment_cases.py
.venv/bin/python tools/generate_shared_loop_cases.py
.venv/bin/python tools/generate_audit_cases.py
.venv/bin/python tools/generate_models.py
.venv/bin/python tools/generate_models.py --specs tools/transformer_model.py --out artifacts/transformer-stages
```

The expression generator targets sm_89 with isolated schedule state. The typed
IR generators use Emmy classes and its serialization/parser. Numerical variants
use zero, magnitude-30 alternating-sign and tiny finite inputs. The model packager
restores seeded tensor parameters omitted from JSON dumps; the official generator
already packages scalar constants. Generate corpora fully before starting checks.

## Hosted execution

```sh
.venv/bin/python tools/validate.py --report artifacts/public-results.json
.venv/bin/python tools/validate.py --problems artifacts/variations --report artifacts/variation-results.json
.venv/bin/python tools/validate.py --problems artifacts/audit --report artifacts/audit-results.json
```

Use the same command for each other corpus directory. This invokes the official
checker against the organizers' GPU. The report retains output and a compiler
SHA-256; each final output must satisfy that problem's tolerances. Compiler output
is generated afresh from each IR.

## Local execution and complete models

```sh
.venv/bin/python tools/local_runner.py challenge/problems/[aprfm]*_* artifacts/variations/* artifacts/schedules/* artifacts/numerical/* artifacts/models/* artifacts/transformer-stages/* artifacts/alignment/* artifacts/loop-tails/* artifacts/shared-loops/* artifacts/bandwidth/* artifacts/audit/* --report artifacts/local-results.json
.venv/bin/python tools/validate_transformer.py
```

The driver runner uses libcuda directly, without nvcc or GPU-enabled Torch, and
adapts the PTX target to the local GPU. Local checks complement contest-target
execution; they do not establish identical behavior on every architecture.

The complete transformer check uses actual PyTorch module outputs at every
stage, with three independent input seeds and depths of one, two and four
repeated blocks. It is an untrained numerical test. Host transfers are part of
this integration harness, so it is not a model performance benchmark.

## Seeded lexical-scope differential cases

```sh
.venv/bin/python tools/fuzz_semantics.py --count 7 --minimal --out artifacts/fuzz-reduced
.venv/bin/python tools/fuzz_semantics.py --count 28 --out artifacts/fuzz-scopes
.venv/bin/python tools/validate_isolated.py artifacts/fuzz-reduced/*/ artifacts/fuzz-scopes/*/ --report artifacts/fuzz-local.json
.venv/bin/python tools/validate.py --problems artifacts/fuzz-scopes --report artifacts/fuzz-hosted.json
```

The generator constructs typed IR for seven bounded families: branch float/index
bindings, loop float/axis bindings, captured mutable state, conditional full-warp
reductions, and conditional named-barrier shared trees. Integer-valued f32 inputs
give exact independent NumPy formulas, without using the compiler or an Emmy
renderer as the oracle. Every seed has a renamed counterpart with the same inputs
and expected output. Cooperative cases use full participating warps and explicit
barrier counts. Scalar tiles use the IR's `block_threads=None` contract.

`--minimal` reduces family parameters and removes optional arithmetic. It is
not a general statement minimizer or a proof of the globally smallest failure.
For fresh seeds, freeze both compiler and generator hashes before generating:

```sh
.venv/bin/python tools/fuzz_semantics.py --start 1000 --count 14 --out artifacts/fuzz-fresh
.venv/bin/python tools/validate_isolated.py artifacts/fuzz-fresh/*/ --report artifacts/fuzz-fresh-local.json
```

An illegal access poisons a CUDA context. `validate_isolated.py` runs each case
in a new process/context and records hashes, stdout, stderr and exit status.
Use it when probing unknown IR combinations; failures after the first fault in
an unisolated process are not independent evidence. Compare a different compiler
using `--compiler path/to/compiler.py`; failed cases remain on disk.

The new-seed suite stays within this grammar. It does not measure unseen operator
families, arbitrary IR support or hidden-test success.

## Nested shared storage and fold seeds

```sh
.venv/bin/python tools/fuzz_nested.py --count 4 --minimal --out artifacts/nested-reduced
.venv/bin/python tools/fuzz_nested.py --count 16 --out artifacts/nested-seeded
.venv/bin/python tools/fuzz_nested.py --count 2 --minimal --dynamic --out artifacts/nested-dynamic
.venv/bin/python tools/validate_isolated.py artifacts/nested-reduced/*/ artifacts/nested-seeded/*/ artifacts/nested-dynamic/*/ --report artifacts/nested-local.json
.venv/bin/python tools/validate.py --problems artifacts/nested-seeded --report artifacts/nested-hosted.json
```

The four families test branch/loop shared declarations with shadowed names and
different shapes, a nested maximum whose name/axis shadows an outer sum, and
an outer state updated through an inner loop. Independent small-integer NumPy
formulas give exact f32 expectations; renamed counterparts test name invariance.
All shared participants execute full-block barriers uniformly. Dynamic variants
request exactly 98,304 launch bytes for two simultaneously live 48 KiB arrays.
This is explicit test metadata, not an invented replacement for a supplied launch.

Freeze compiler and generator before adding fresh seeds with `--start 1000`.
These are bounded synthetic cases, not a general program generator or a hidden
test estimate. `--minimal` reduces family parameters, not arbitrary statements.

## Frozen expression follow-up

```sh
.venv/bin/python challenge/tools/generate.py tools/expressions_holdout.py --emmy .venv/bin/emmy --out artifacts/holdout-reproduced
.venv/bin/python tools/local_runner.py artifacts/holdout-reproduced/*
.venv/bin/python tools/validate.py --problems artifacts/holdout-reproduced --report artifacts/holdout-results.json
```

The compiler hash was frozen before creating the original cases. Two expressions
could not be traced upstream, and a LayerNorm fixture needed its full computation
inside the traced callable. The active specification contains the ten runnable
cases. Original attempts, the fixture correction, both GPU results and unchanged
compiler hash are recorded in `validation/2026-10-24-holdout.json`.

## Performance

Compile-only resource inspection (no kernel execution):

```sh
.venv/bin/python tools/profile_resources.py challenge/problems/*/ --report artifacts/resources-local.json
.venv/bin/python tools/profile_resources.py challenge/problems/*/ --hosted --report artifacts/resources-hosted.json
```

The local report queries actual JIT function registers, local-memory bytes,
shared memory and resource-limited occupancy. Occupancy is a theoretical ceiling,
not measured utilization. Hosted mode exposes register counts only; it cannot
establish hosted spill bytes, occupancy or instruction bottlenecks. Static PTX
instruction counts are not dynamic instruction counts or SASS.

Compare local cache conditions explicitly:

```sh
.venv/bin/python tools/benchmark.py artifacts/bandwidth/* --report artifacts/cache-warm.json
.venv/bin/python tools/benchmark.py artifacts/bandwidth/* --evict-l2 --report artifacts/cache-evict.json
```

`--evict-l2` zeros twice the queried device L2 size before each graph replay and
synchronizes before timing. This work and atomic-output resets remain outside
the captured events. It is eviction conditioning, not proof of a fully cold L2.
Both modes check full final outputs after timing. The hosted API does not expose
cache conditioning or clock locking; do not label its timings cold-cache.

```sh
mkdir -p artifacts
git show 4a8c225:compile.py > artifacts/compiler-before-fma.py
.venv/bin/python tools/benchmark_hosted.py challenge/problems/m01_matmul_512 challenge/problems/a01_attention --baseline artifacts/compiler-before-fma.py
.venv/bin/python tools/benchmark.py challenge/problems/m01_matmul_512
```

The hosted benchmark alternates baseline/candidate order for seven paired trials,
with eight warmups and 75 timed repetitions per request. The default checks full
outputs every trial. For large bandwidth cases, `--timing-return-limit 4096`
checks full outputs in the first trial and returned prefixes in later trials.
Intermediate buffers are always returned fully. Raw samples and compiler hashes
are retained; server clocks are not locked.

The local benchmark captures the kernel and external timing events into a CUDA
graph, excluding Python dispatch gaps. Atomic outputs are re-zeroed outside the
timed region and results are checked after repetitions. It uses NVIDIA's
[event API](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__EVENT.html)
and [graph capture](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html).

## Sanitizers

With NVIDIA Compute Sanitizer installed:

```sh
.venv/bin/python tools/sanitize.py artifacts/audit/* --binary /path/to/compute-sanitizer --out artifacts/sanitizers
```

The harness runs memcheck (including leaks), racecheck, synccheck and initcheck,
with a nonzero error exit code. Records identify the selected cases; a clean
subset is not a claim about every possible kernel. The recorded version is
Compute Sanitizer 2025.3.1 from the official CUDA 13.0.2 redistributable.

## Release archive

Commit source changes before packaging:

```sh
uv venv --python 3.12 artifacts/submission-runtime
uv pip install --python artifacts/submission-runtime/bin/python -r requirements.txt -c requirements-ci.lock
.venv/bin/python tools/package_submission.py --python artifacts/submission-runtime/bin/python
```

This archives committed files, checks that development state is excluded, and
compares the packaged compiler hash with the working tree. The extracted program
then runs official hosted pointwise, dynamic-shared matmul and attention checks
from another directory. The companion JSON records the archive/commit/compiler
hashes, environment and results. Torch is absent in this fresh runtime.

The shared-loop generator additionally supports explicit fresh extents and body
sizes, including the original 64-statement eligibility boundary:

```sh
.venv/bin/python tools/generate_shared_loop_cases.py --sizes 47 79 95 130 257 511 767 --out artifacts/fresh-shared
.venv/bin/python tools/generate_shared_loop_cases.py --sizes 65 129 --body-repeats 8 17 32 33 --out artifacts/body-budget
```

Repeated load/fold pairs stress the expansion budget; expected sums come from
float64 NumPy over the seeded input, multiplied by the repeat count. These are
synthetic correctness tests, not representative model performance measurements.

## Accepted paired shared-read optimization

The historical patch in `experiments/paired-shared-loads.patch` targets the nested
scope compiler at 5791ed6. The current compiler includes this optimization. To
reconstruct the tested candidate independently:

```sh
git show 5791ed6:compile.py > artifacts/compiler-before-pair.py
patch --output artifacts/compiler-paired-shared.py artifacts/compiler-before-pair.py < experiments/paired-shared-loads.patch
.venv/bin/python tools/generate_pair_cases.py --out artifacts/pair-cases
.venv/bin/python tools/generate_pair_cases.py --seed 71 --trips 7 --out artifacts/pair-loops
.venv/bin/python tools/check_pair_codegen.py artifacts/pair-cases/*/ --compiler artifacts/compiler-paired-shared.py
.venv/bin/python tools/validate_isolated.py artifacts/pair-cases/*/ --compiler artifacts/compiler-paired-shared.py
.venv/bin/python tools/validate.py --problems artifacts/pair-cases --compiler artifacts/compiler-paired-shared.py
.venv/bin/python challenge/tools/generate.py tools/expressions_shared_matmul.py --emmy .venv/bin/emmy --out artifacts/shared-matmul
.venv/bin/python tools/validate_isolated.py artifacts/shared-matmul/*/ --compiler artifacts/compiler-paired-shared.py
```

The motion generator has 13 bounded variants with renamed counterparts and
independent exact small-integer NumPy references. Codegen checks require actual
vector reads in positive cases and scalar reads in negative cases; numerical
passes alone could otherwise validate only a fallback. The nested weak-alignment
case includes a strong-alignment outer loop, whose vector reads remain expected.
The four matmul expressions have independent float64 NumPy references. Inspect
actual vector instruction counts: only the 384×256×320 generated schedule uses
pairing in this set; the other three are controls, not positive vector coverage.
Use `--compiler` on sanitizer and transformer tools as well. Run fresh frozen-seed
fixtures and paired hosted timings before adopting; these commands alone do not
establish acceptance or production readiness.

## Unsupported-feature diagnostics

After generating `artifacts/pair-cases`, run CPU-only typed-IR probes through the
real CLI. They require rejected inputs to produce no stdout/PTX, a nonzero exit,
and a concise unsupported-feature diagnostic without a traceback:

```sh
.venv/bin/python tools/check_feature_rejections.py --report artifacts/feature-rejections.json
.venv/bin/python tools/check_codegen_equivalence.py challenge/problems/*/ artifacts/pair-cases/*/ --baseline artifacts/compiler-before-feature-validation.py --report artifacts/feature-equivalence.json
```

The baseline compiler for this iteration is reproducible from `2306028:compile.py`.
The probes cover eight explicit unsupported declarations: f16 tensors, f16
Write/Assign/Init/Accum/Let, double shared storage and a B32-swizzled write. They
are typed-reader-valid fixtures, not numerical evidence for those unsupported
features. Supported f32 storage/arithmetic and i32 Let bindings remain accepted.
This validation does not replace a full type checker or add reduced-precision
support. `validation/2026-10-24-feature-validation.json` preserves baseline silent
acceptance, final rejection results and supported PTX-equivalence evidence.

## Signed index division and range proofs

Emmy 0.3.11 `ir/expr.py:apply_binop` defines both `/` and `//` as integer floor
operations and `%` as Python remainder. These differ from PTX truncating signed
operations when operand signs differ. Development fixtures use independent NumPy
references, including negative divisors, with all accessed addresses in bounds:

```sh
.venv/bin/python tools/generate_signed_index_cases.py
.venv/bin/python tools/generate_signed_index_cases.py --seed 2049 --out artifacts/signed-index-fresh
.venv/bin/python tools/generate_index_scope_cases.py
.venv/bin/python tools/validate_isolated.py artifacts/signed-index/*/ artifacts/index-scopes/*/ --compiler artifacts/compiler-signed-ranged-v2.py --report artifacts/signed-range-local.json
```

The candidate can be reconstructed from the accepted compiler at `aa668e2` using
`experiments/signed-index-floor.patch`; the exact candidate is now accepted in
`validation/2026-10-24-production-iteration5.json`. The scope generator
checks branch/loop axis shadowing, divisor shadowing, a negative strided-loop start,
signed-int32 addition overflow, and ternaries. Its second output half observes the
outer binding after the scope. Overflow references explicitly wrap through int32
before applying NumPy remainder. These bounded probes do not define division by
zero or signed division overflow as supported behavior.

## Strided-loop end and increment timing

```sh
.venv/bin/python tools/generate_strided_bound_cases.py
.venv/bin/python tools/generate_strided_bound_cases.py --seed 3719 --out artifacts/strided-bounds-fresh
.venv/bin/python tools/validate_isolated.py artifacts/strided-bounds/*/ artifacts/strided-bounds-fresh/*/ --report artifacts/strided-bounds-local.json
.venv/bin/python tools/validate.py --problems artifacts/strided-bounds --report artifacts/strided-bounds-hosted.json
```

Four bounded families with renamed counterparts check an axis-dependent end,
an axis-dependent increment, both together, and the implicit `<axis>_end` binding.
Independent Python cursor recurrences and small-integer f32 sums provide exact
references; a second output half observes the enclosing coordinate afterward.
Installed Emmy 0.3.11 `ir/stmt/blocks.py:StridedLoop.render` establishes the timing
and implicit-binding contract. Iteration-six evidence includes baseline numerical
failures and missing-binding errors, final original/fresh cases on both GPUs,
selected sanitizer, transformer, standalone-runtime and 350-kernel byte-identical
PTX checks. This changes new-case correctness without a fresh speedup claim.

## Predicate-preserving statement reduction

`tools/shrink_failure.py` attempts hierarchical deletion of statement chunks in
one-kernel fixtures. Every retained trial must compile on both compilers, pass the
fixture's **unchanged** independent numerical oracle on the reference compiler,
and produce a numerical mismatch on the suspect compiler. GPU executions run in
separate processes/contexts. Compilation failures, memory faults and timeouts do
not substitute for the numerical defect. A final independent replay is required;
compiler-source changes invalidate publication. Reports are saved incrementally.

```sh
git show bb8d2c8:compile.py > artifacts/compiler-before-scope-fix.py
.venv/bin/python tools/fuzz_semantics.py --count 7 --out artifacts/shrink-inputs
.venv/bin/python tools/shrink_failure.py artifacts/shrink-inputs/branch_value_s0 --suspect artifacts/compiler-before-scope-fix.py --out artifacts/reduced-scope --max-trials 100
```

Use a new output directory. The tool leaves launch metadata, inputs and expected
outputs unchanged; it does not recompute an oracle from either compiler. Thus it
can delete unused arithmetic but cannot discard an observable output half or
change dimensions. A completed deletion fixed point means no tried single
statement deletion preserved this predicate; it is not global minimality. A
budget-limited search or one containing timeouts explicitly avoids that claim.
This is a bounded deletion reducer, not a general valid-IR generator/type checker.

`validation/2026-10-24-statement-shrinking.json` records an actual historical scope
failure reduced 11 → 10 statements, a strided-loop failure retained at 8 statements,
all trial classifications, and checks for passing-input rejection and trial-budget
reporting. Self-contained retained fixtures are under `validation/counterexamples`.
The numerical oracle remains independent, but reference-compiler acceptance is
only an observed validity filter. The reducer does not prove arbitrary IR valid.

## CPU contract CI

`.github/workflows/cpu-contract.yml` runs on pushes, pull requests, manual dispatch
and nightly at 02:17 UTC on Python 3.12/3.13/3.14. It installs runtime requirements
with exact version constraints from `requirements-ci.lock`, checks dependency
consistency, and requires Torch to be absent. The lock records the previously
verified Python 3.12 standalone environment; matrix compatibility is checked by CI.

```sh
uv venv --python 3.12 artifacts/cpu-ci-runtime
uv pip install --python artifacts/cpu-ci-runtime/bin/python -r requirements.txt -c requirements-ci.lock
artifacts/cpu-ci-runtime/bin/python tools/check_cpu_contract.py --forbid-torch --report artifacts/cpu-contract.json
```

The checker uses only tracked fixtures. It verifies their input/meta/oracle hashes,
copies the compiler and fixtures to a temporary directory, and invokes the CLI
with isolated Python (`-I`) from an unrelated working directory. Two deterministic
compilations must match the committed, previously GPU-validated PTX snapshots.
Three real CLI probes require unsupported dtype/swizzle errors before any output.
The checker fails a deliberately corrupted golden in local verification; it never
automatically refreshes snapshots. Intentional codegen changes require reviewed
snapshot updates backed by numerical GPU gates. CI reports are retained as Actions
artifacts for 14 days. This is compilation/CLI coverage on two fixtures, not GPU
correctness, sanitization, throughput or proof of long-running CI stability.

## Native instruction inspection (development only)

`tools/profile_sass.py` uses CUDA driver `cuLink*` to generate cubins and NVIDIA
`cuobjdump`/`nvdisasm` to inspect them without executing kernels. It does not change
submission lowering. An explicit `--target sm_120` can inspect local-driver codegen
for that architecture even on the local sm_86 GPU. This is **not** a captured cubin
from the hosted runner; the relocatable link path can also differ from the
runner's direct module-loading JIT. Never infer target utilization from it.

NVIDIA's [CUDA 13.0.2 redistribution manifest](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.0.2.json)
pins both linux-x86_64 tools at 13.0.85. Archive URLs and verified SHA-256 values
are preserved in `validation/2026-10-24-native-instructions.json`. Download those
archives, verify hashes and extract under `artifacts/vendor`, then:

```sh
.venv/bin/python tools/profile_sass.py challenge/problems/m01_matmul_512 --target sm_120 --cuobjdump artifacts/vendor/cuda_cuobjdump-linux-x86_64-13.0.85-archive/bin/cuobjdump --nvdisasm artifacts/vendor/cuda_nvdisasm-linux-x86_64-13.0.85-archive/bin/nvdisasm --out artifacts/sass-m01-sm120
```

The output directory retains PTX, cubin, full disassembly, resource text and a JSON
report. Reports include compiler/IR/meta/tool/cubin/disassembly hashes, local driver
version, requested codegen architecture, opcode counts, total static counts and
per-function counts. Counts include padding; cubin totals can include helper
functions. They are not dynamic instruction counts, spill traffic or counters.
Primary contracts: [CUDA module/link APIs](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__MODULE.html),
[binary utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/).

For the rejected address-CSE candidate, vector add, attention and large sum have
identical disassembly despite reduced PTX in both sm86/sm120 local-driver outputs.
Matmul code changes without a measured target win. These observations motivate
examining native code before selecting further PTX-size optimizations.

## Accepted signed power-of-two index lowering

The exact signed-only refinement is recorded in production-iteration7.json.
Historical `power2-index-signed.patch` reconstructs it from compiler at 1ad0dbf.
The broader unsigned+signed patch remains an unaccepted performance control.

```sh
.venv/bin/python tools/generate_signed_index_cases.py --seed 5731 --divisor 32 --out artifacts/power2-proof-d32
.venv/bin/python tools/generate_signed_index_cases.py --seed 9079 --divisor 32 --extremes --out artifacts/power2-extremes-d32
```

Repeat divisors 1/2/8/32 to recreate 48 normal and 42 boundary fixtures. Boundary
oracles observe quotient sign and low bits, so logical shifts cannot silently pass
a modulo-only oracle. INT_MIN/-1 is excluded as unsupported division overflow.
No compiler computes the NumPy reference. The new committed CPU fixture under
validation/cpu-fixtures preserves the independently verified signed-boundary case;
CPU CI now has three PTX fixtures and three negative probes (six checks).

Final seven-pair measurements cover all 15 public kernels and four independent
matmuls. Full outputs are checked on the first trial, prefixes subsequently;
clocks are unlocked. The accepted variant improves three matmuls; other 16 gaps
are at most 0.032 µs and treated as noise. The original unsigned rewrite's 7.6%
attention128 regression and its rejected identity-control follow-up are retained
in discovery/refinement evidence, not hidden by the accepted results.

## Predicates used as integer indices

```sh
.venv/bin/python tools/generate_predicate_index_cases.py
.venv/bin/python tools/generate_predicate_index_cases.py --seed 8231 --out artifacts/predicate-index-fresh
.venv/bin/python tools/validate_isolated.py artifacts/predicate-index/*/ artifacts/predicate-index-fresh/*/ --report artifacts/predicate-index-local.json
```

Ten families with renamed counterparts cover direct comparisons as indices,
boolean arithmetic, nested numeric comparisons, logical arithmetic/truthiness,
boolean ternary branches, integer Let snapshots, nested conditions, scalar shared
loads and positively vectorized paired shared loads. Independent NumPy formulas
explicitly convert booleans to 0/1. All memory indices remain in bounds.

The prior compiler emitted invalid PTX by placing .pred operands in integer
instructions. The final corpus uses only operators supported by installed
0.3.11 apply_binop; preliminary use of != was replaced with equivalent >0 for a
nonnegative remainder before final checks. Iteration8 preserves those final IR
hashes, all 20 baseline assembly failures, 40 local/hosted passes, 458 prior
byte-identical kernels, selected sanitizer/model/runtime checks and vector
eligibility proof. CPU CI adds the committed shared-vector predicate fixture;
four positive fixtures plus three negative probes now make seven checks.

## f32 expression, condition and conversion probes

```sh
.venv/bin/python tools/generate_float_expression_cases.py
.venv/bin/python tools/generate_float_expression_cases.py --seed 12671 --out artifacts/float-expressions-fresh
.venv/bin/python tools/validate_isolated.py artifacts/float-expressions/*/ artifacts/float-expressions-fresh/*/ --report artifacts/float-expressions-local.json
.venv/bin/python tools/check_scalar_expression_rejections.py --report artifacts/scalar-expression-rejections.json
```

Thirteen families with renamed counterparts cover fractional literal/loaded
comparisons, signed-zero truthiness, mixed comparisons/ternaries, separate f32
arithmetic, explicit f32/i32 casts and integer→float rounding near 2^24, NaN truth/
equality, mixed-type folded FMA and integer scalar writes into f32 buffers. NumPy
oracles explicitly round f32 operations and truncate in-range conversions. Finite
input data stays JSON-safe; IR division of zero by zero generates NaN internally.
Distinct sentinel values make wrong predicate selection numerically observable.

The final source is frozen before fresh seeds. Iteration9 records 26 baseline
failures (wrong results, invalid PTX or unsupported integer-value stores), 52
local/hosted passes, four sanitizers on 26 selected cases, nine transformer runs,
three no-Torch compilation checks and 498 prior byte-identical kernels.
Scalar diagnostic probes reject floating indices and floating Expr division with
zero stdout and a concise unsupported-feature error. The Expr division boundary
does not remove supported scalar Assign divide. Nonfinite/out-of-range f32→i32
casts are outside this checked domain; integer-buffer storage remains unsupported.

CPU CI adds NaN truth, numeric casts, f32 arithmetic and mixed-comparison fixtures.
All four prior PTX goldens stay unchanged. Eight positive fixtures plus four
negative probes now make twelve CPU contract checks; these do not execute PTX.

## Flat/additive rank addressing and ABI controls

```sh
.venv/bin/python tools/generate_address_cases.py
.venv/bin/python tools/generate_address_cases.py --seed 8159 --out artifacts/address-ranks-fresh
.venv/bin/python tools/validate_isolated.py artifacts/address-ranks/*/ artifacts/address-ranks-fresh/*/ --report artifacts/address-ranks-local.json
```

Eight families with renamed counterparts check extra/partial Load coordinates,
flat/empty/full-rank controls, additive Write coordinates, paired shared loads and
a scalar logical view with flat addressing into a physically larger buffer.
Independent NumPy formulas check both primary results and a control output half.
The contract is installed `ir/stmt/base.py:render_index`, not an assumed rank-equals-
shape rule. Every access stays within the prescribed 128-element allocation.

Prior compiler passes six controls but fails ten rank probes numerically. Final
16 original/16 frozen fresh cases pass on both GPUs; four sanitizers cover the
16 originals. Two additional controls reverse both arg_order and descriptors;
they pass on both compilers locally and on the candidate hosted GPU. Vector
eligibility checks require one shared vector read in the two shared-pair cases.
550 prior kernels emit byte-identical PTX. Evidence: production-iteration10.json.

The metadata audit also found 38 valid scalar-view/allocation rank differences
in the older corpus, with no descriptor-order/cooperative-width mismatches.
These observations constrain future validation: preserve valid views and parameter
permutations. They do not establish general launch/ABI validation or memory bounds.
CPU CI adds four GPU-verified addressing/ABI fixtures; twelve positive snapshots
and four negative probes make sixteen checks. Existing eight goldens unchanged.

Launch diagnostics use only committed fixtures and run without a GPU:

```sh
python tools/check_launch_contract.py --report artifacts/launch-contract-result.json
```

Twenty-five malformed/unsupported launch probes must fail with a launch diagnostic
and empty stdout. Three valid controls preserve a matching cooperative width,
reordered ABI and scalar logical view. Mutated invalid launches are never executed.
The accepted launch-only change emits identical PTX for 594 retained kernels;
twelve committed numerical fixtures pass locally. CPU CI runs this 28-case suite
in addition to the existing sixteen snapshot/feature checks. Evidence is recorded
in production-iteration11.json; this does not establish full ABI/bounds validation.

The positional ABI suite also runs entirely on the CPU:

```sh
python tools/check_abi_contract.py --report artifacts/abi-contract-result.json
```

It checks 37 negative cases and five valid controls, including consistent
permutations, equal-capacity allocation reshaping, scalar logical views and
selection among metadata entries. Invalid descriptors are never launched.
The accepted ABI change preserves 594 retained PTX outputs. Twelve retained GPU
fixtures pass locally; two derived positive permutation/16x8 allocation controls
pass locally and hosted (same independent oracle values, reshaped allocation
control oracle to 16x8). All existing sixteen snapshot/feature and 28 launch
checks remain unchanged. CI adds the 42 ABI checks. See production-iteration12.json.

Target launch/copy/strided-read calibration:

```sh
python tools/calibrate_hosted.py --out artifacts/target-calibration --trials 5
```

Uses only committed fixtures and the compiler runtime; no local GPU or Torch.
First trial verifies every output; later trials verify prefixes. Repeated buffers,
unlocked clocks and no eviction control are explicit. See TARGET_PERFORMANCE.md
for effective bandwidth definitions, runner allocation/return limits, raw evidence
and the separate current matmul self-comparison. This is a diagnostic calibration,
not a candidate performance acceptance test or measured DRAM traffic.

Cross-iteration shared pairing uses committed generator templates:

```sh
python tools/generate_cross_iteration_cases.py --out artifacts/cross-original --seed 0
python tools/generate_cross_iteration_cases.py --out artifacts/cross-fresh --seed 9161
git show b2a1b19:compile.py > artifacts/compiler-before-cross-pair.py
python tools/check_cross_iteration_codegen.py \
  artifacts/cross-original/manifest.json artifacts/cross-fresh/manifest.json \
  --baseline artifacts/compiler-before-cross-pair.py --candidate compile.py \
  --report artifacts/cross-codegen.json
```

The baseline source is reproducible from b2a1b19:compile.py. Fifteen families with
two naming variants cover eligibility and unsafe-motion guards, scalar tails,
flat/additive layouts and 4/5/64/65 trips. All sixty original/fresh cases pass on
both GPUs; all sixty code-generation expectations pass. Of 594 retained kernels,
561 PTX outputs stay identical; all 33 changed kernels pass on both GPUs. Four
sanitizers cover thirty originals and separate exact-allocation 5/65 boundaries;
nine transformer and three isolated no-Torch compilation comparisons pass.
Final seven alternating pairs cover nineteen workloads after correctness requests
finished; use cross-final-timing, not the exploratory overlapping broad run.
Full evidence: cross-iteration-discovery.json. Acceptance: production-iteration13.json.
Four GPU-checked cross-iteration fixtures add positive/tail and write/local-index
guards to CPU CI; all twelve existing snapshots stay unchanged. Twenty snapshot/
feature, 28 launch and 42 ABI checks make ninety checks per Python version.

Current 90-check local matrix passes without Torch on Python 3.12.13, 3.13.14
and 3.14.4. The latter two environments were freshly installed with requirements.txt
and requirements-ci.lock; all dependency checks pass. GitHub run 40109402820 on
07f4851 produced no test results: all three jobs were rejected before starting
because of account payments/spending-limit state. Local matrix results are not
hosted CI evidence. Reports and GitHub annotation are in production-iteration13.json.
The archive from 07f4851 passes extracted hosted scalar-add/matmul/attention checks;
current-package.json records exact bytes and compiler. Repository remains private.

Typed static dimensions and empty tile domains:

```sh
python tools/check_static_shapes.py --report artifacts/shape-contract-result.json
python tools/generate_zero_tile_cases.py --out artifacts/zero-tile-original --seed 0
python tools/generate_zero_tile_cases.py --out artifacts/zero-tile-fresh --seed 17011
```

The CLI suite has 27 rejection probes and four valid scalar/zero-view/alignment
controls. The numerical generator has nine renamed families, including zero
outer/inner/multiple axes, nested conditional/loop tiles, a zero logical input
view, and nonempty singleton/empty-axis-list controls. Baseline 40a1de84 fails
fourteen zero-domain cases and passes four controls. Final 36 original/frozen-fresh
cases pass locally and hosted, after correcting an initial floating-index mistake
in the generator's sentinel control. Those invalid-fixture runs are excluded.
All 658 retained kernels emit identical PTX; nine transformer runs pass. No new
shared/async instruction motion or performance claim. Full evidence: iteration14.
Four numerical fixtures join CPU snapshots; existing 16 goldens stay unchanged.
Twenty-four snapshot/feature + 28 launch + 42 ABI + 31 shape checks total 125 per Python
version, all passing locally without Torch on 3.12/3.13/3.14. Hosted CPU CI remains
subject to the previously recorded account billing restriction; local results are
not hosted CI success. Packaging state is recorded in current-package.json.

Current source 7a7f518 archive (compiler 7ebf4ff2) passes extracted hosted scalar-add,
matmul and attention smoke checks using the existing isolated no-Torch 3.12.13
runtime. Archive/compiler hashes in current-package.json. GitHub run 40111398281
on that source again produced no steps/results because account payments/spending
state prevented startup. All125 checks pass locally on each of three Python
versions; these are not hosted CPU CI results. No billing or visibility changes.

Local/hosted initialization parity (requires both a local NVIDIA GPU and the
hosted API):

```sh
python tools/probe_runner_initialization.py --require-match
python tools/probe_runner_initialization.py --out artifacts/runner-initialization-fresh \
  --first-value=-5.5 --require-match
python tools/validate_isolated.py validation/runner-fixtures/*/ \
  --report artifacts/runner-fixtures.json
```

The copy probe checks explicit data, returned-input flags and zero_outputs naming
an input. The initial local correctness launch now zeros requested buffers too;
the timed path already did so. Four flag combinations × two exact input values
match hosted outputs and an independent reference in both initial and timed local
execution. Public fifteen and transformer nine checks pass; memcheck/synccheck
cover the two retained numerical fixtures. An audit of 705 retained/model kernel
descriptors finds no non-output zero targets, preserving their prior initialization
scope. Evidence: runner-initialization.json. Compiler/PTX and CPU suite unchanged;
these GPU fixtures are not added as misleading CPU snapshot coverage.

The c95beb5 archive includes the corrected runner and these GPU fixtures. Archive
member hashes are checked against tested source in current-package.json; three
extracted hosted emitter smoke checks pass without Torch using the existing pinned
Python 3.12 runtime. Compiler7ebf4ff2 is unchanged. Prior local 125-check CPU results
remain compiler evidence; they do not test CUDA host initialization or establish
hosted CI success.

## Four-wide shared-read acceptance — 26 October

Current compiler c2f7e2b3 passes 130 checks per Python version (29 snapshot/feature,
28 launch, 42 ABI, 31 shape) in isolated no-Torch 3.12.13, 3.13.14 and 3.14.4.
Two old GPU-verified snapshots were refreshed; eighteen are unchanged and five
new alignment/stride/tail fixtures were added. These are local results; hosted
CPU CI remains unverified under the known account restriction.

Generate structural cases with `tools/generate_cross_iteration_cases.py --wide-guards`
and floating stress with `--wide-guards --floating-stress`, using seeds 0 and 28103
and distinct `--out` directories. Check their manifests with
`tools/check_cross_width_codegen.py MANIFEST... --baseline BASELINE --candidate compile.py --report REPORT`.
Recover the baseline with `git show 60676f3:compile.py`. All 92 codegen cases pass,
including exact PTX preservation for scalar/two-wide fallbacks. All 92 pass on both
local and hosted GPUs. The 59 changed members of 700 retained kernels also pass
both GPUs; 641 remain byte-identical. Nine transformer checks pass.
All four Compute Sanitizer tools are clean on 46 synthetic cases and three actual
pipelined matmuls. Final seven-pair timing covers 19 workloads sequentially.
Full reports and fixture hashes: validation/2026-10-26-production-iteration15.json.

## Signed-wrap shared-address regression coverage — 26 October

Generate with `tools/generate_wrap_shared_cases.py --out DIRECTORY --seed SEED`
for seeds0 and39019. Each generates22 cases: add/subtract/multiply intermediates
wrap at signed32 boundaries while final indices remain in bounds. Flat/additive
views, opaque row expressions, width-two fallback and scalar guards are included
with renamed variants. Original independent scalar-fixture oracles are preserved.
All44 codegen checks and numerical cases pass on both GPUs. All four sanitizer
tools are clean on22 original cases. The generator never forms an out-of-bounds
pointer to test wraparound. This is regression coverage, not a formal proof.

Four GPU-verified fixtures add snapshot coverage without changing existing25
goldens. All33 snapshot/feature checks newly pass in each isolated no-Torch
Python3.12/3.13/3.14 environment. Together with101 diagnostic checks from unchanged
compiler/checker source, the local matrix totals134 per version. Hosted CPU
execution remains unverified. Evidence: validation/2026-10-26-wrap-addresses.json.

## Scalar logarithm and exp2 acceptance — 26 October

Use `tools/generate_scalar_log_cases.py --out DIRECTORY --seed SEED` for seeds0
and48127, then `tools/generate_scalar_log_ieee_cases.py --source DIRECTORY
--out IEEE_DIRECTORY`. Each seed produces36 boundary/composition cases plus16
IEEE-class cases. Special operands are created inside PTX; transported values
remain finite. References use independent NumPy float64 functions with explicit
f32 accumulation where required. The two-ULP exp2 power bounds and result classes
are checked exactly; general approximate math uses recorded tolerances. Initial
bitwise-power tests were too strict and are retained outside acceptance results.

The frontend specs `tools/expressions_scalar_logs.py` produce four natural-log
workloads with the official generator. All104 typed and four frontend workloads
pass on both GPUs. All four sanitizers are clean on56 original/frontend cases.
845 retained kernels emit identical PTX; nine transformer checks pass.
Four snapshots were added without changing the previous29 goldens.
The complete local no-Torch3.12/3.13/3.14 matrix is now37 snapshot/feature +
28 launch +42 ABI +31 shape +4 unsupported-op checks =142 per version.
`tools/check_scalar_op_rejections.py` is included in CPU CI; hosted execution
remains unverified under the known account restriction. Full evidence:
validation/2026-10-26-production-iteration16.json.

The pinned frontend CUDA renderer cannot emit log2/log10/exp2. Four initial
frontend generation failures are recorded separately; these operations are
covered by typed-IR probes, not claimed as frontend-generation successes.
The final four supported specs regenerate identical IR/metadata/oracle bytes.
