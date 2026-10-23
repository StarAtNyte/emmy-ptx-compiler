# Emmy IR to PTX compiler

A direct Python translator for Emmy Kernel IR. It loads typed IR using
`emmy-ml==0.3.11` and emits PTX without invoking another compiler or Emmy's
CUDA renderer.

## Setup

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python compile.py path/to/kernel.ir.json > out.ptx
```

The problem's `meta.json` must be beside the IR. It supplies the prescribed
parameter order. PTX targets sm_89 and version 8.0.

## Validation

Clone https://github.com/cloudrift-ai/emmy-hackathon-2026 into `challenge`, then:

```sh
.venv/bin/python tools/validate.py
```

The script uses the official submit checker and hosted GPU runner, preserves
each check's output in `artifacts/public-results.json`, and returns a failure
status if any problem fails.

On 2026-10-23 all 15 public problems passed on the hosted RTX 5090 across all
five categories. This establishes a public-suite baseline, not hidden-test
correctness. The full suite was checked after adding matmul support and dynamic
shared-memory allocation.

The independent corpus adds 23 generated problems with different sizes and
operations. All 23 passed the hosted runner. It includes odd sizes, max reductions,
short softmax rows, masked matmuls, sigmoid, tanh and smaller attention shapes.
It exposed missing Select lowering and an empty-state softmax NaN, both now fixed.
A seeded three-layer PyTorch MLP with ReLU and SiLU also passed on the hosted GPU.

To reproduce development tests (CPU Torch is used for tracing and reference math):

```sh
uv pip install --python .venv/bin/python torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python challenge/tools/generate.py tools/expressions_variations.py --emmy .venv/bin/emmy --out artifacts/variations
.venv/bin/python tools/numerical_variations.py
.venv/bin/python tools/generate_models.py
.venv/bin/python tools/validate.py --problems artifacts/variations --report artifacts/variation-results.json
.venv/bin/python tools/validate.py --problems artifacts/models --report artifacts/model-results.json
.venv/bin/python tools/local_runner.py artifacts/variations/* artifacts/numerical/* artifacts/models/* challenge/problems/[aprfm]*_* --report artifacts/local-results.json
```

The local runner uses libcuda directly and adapts the PTX target to the local GPU.
It does not need nvcc or GPU-enabled Torch. Local checks complement hosted sm_89
validation; they do not establish identical behavior on every architecture.
The numerical corpus uses zero, alternating magnitude-30, and tiny finite values.
The model packager restores seeded tensor parameters omitted from the JSON dumps;
the official packager already handles scalar constants correctly.

## Current implementation

- Row-major tile decoding with guards and prescribed cooperative block width.
- Explicit scalar loads/stores for grouped values; separate rounded arithmetic.
- Loops and scoped accumulator identities, including persistent initialized state.
- Warp shuffle combines and local folding of shared partial states, including
  coupled online softmax state.
- Shared memory, async copies, barriers and atomic output accumulation.
- Dynamic shared memory when the total exceeds 48 KiB.
- Masked Select, combined conditions and identity-aware reduction merges.

Current limits: f32 computation and buffer addressing; no auxiliary thread bands
or shared-memory swizzles; integer expression and operation coverage is incomplete.
Reduction folding uses a different association from TreeHalve's parallel tree.
Grouped accesses are scalarized. These choices need stress testing, numerical
review and performance measurements before treating the compiler as competition
ready. Branch assignments, repeated tiles, nested accumulator scopes and additional
TreeHalve options need further review.

Next: schedule diversity and a genuinely held-out corpus; additional model families;
sanitizer checks, compiler cleanup and performance. The development cases above
have been used to fix bugs, so they are regression coverage rather than a held-out
estimate. The required submission video and public repository publication are
still outstanding.
