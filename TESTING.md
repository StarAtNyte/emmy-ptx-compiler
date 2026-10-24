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
uv pip install --python artifacts/submission-runtime/bin/python -r requirements.txt
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
