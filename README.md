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

## Current implementation

- Row-major tile decoding with guards and prescribed cooperative block width.
- Explicit scalar loads/stores for grouped values; separate rounded arithmetic.
- Loops and scoped accumulator identities, including persistent initialized state.
- Warp shuffle combines and local folding of shared partial states, including
  coupled online softmax state.
- Shared memory, async copies, barriers and atomic output accumulation.
- Dynamic shared memory when the total exceeds 48 KiB.

Current limits: f32 computation and buffer addressing; no auxiliary thread bands
or shared-memory swizzles; integer expression and operation coverage is incomplete.
Reduction folding uses a different association from TreeHalve's parallel tree.
Grouped accesses are scalarized. These choices need stress testing, numerical
review and performance measurements before treating the compiler as competition
ready. Branch assignments, repeated tiles, nested accumulator scopes and additional
TreeHalve options need further review.

Next: independently generated shapes, schedules and operation combinations;
adversarial numerical inputs; local execution and model integration; code cleanup;
then performance. The required submission video and public repository publication
are still outstanding.
