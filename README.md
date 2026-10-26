# Emmy PTX Compiler

A Python compiler that translates [Emmy Kernel IR](https://github.com/cloudrift-ai/emmy-hackathon-2026) directly into NVIDIA PTX.

Built for the Emmy Hackathon 2026. It supports pointwise operations, reductions,
matrix multiplication, fused normalization, and attention kernels. Compilation
runs on the CPU and requires neither PyTorch nor a CUDA toolkit.

## Getting started

Requires Python 3.12 or later.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -c requirements-ci.lock
```

Compile a kernel:

```sh
python compile.py path/to/kernel.ir.json > out.ptx
```

The input's directory must also contain its `meta.json`. The compiler uses that
metadata for argument order and launch geometry, and emits PTX 8.0 for `sm_89`.
The caller assembles and launches the kernel.

To try a bundled regression input:

```sh
python compile.py tests/fixtures/abi-reordered/k_sum_1_reduce.ir.json > out.ptx
```

## Implementation

The lowering lives in [`compile.py`](compile.py). Emmy reconstructs the typed IR;
the compiler validates its inputs, walks the statement tree, and emits PTX. Each
statement kind has a corresponding lowering method. NVIDIA's assembler handles
physical register allocation.

The compiler implements global and shared memory, loops and conditionals,
asynchronous copies, barriers, atomics, warp reductions, and coupled softmax
state. Optimizations include reduction FMA contraction, bounded loop expansion,
and aligned shared-memory reads. Lowering decisions follow the IR structure.

See [compiler design](docs/design.md) for indexing, numerical behavior,
synchronization, and register budgets.

## Testing

Run the CPU regression suite:

```sh
python tests/run.py
```

To use the official GPU checker:

```sh
git clone https://github.com/cloudrift-ai/emmy-hackathon-2026 challenge
python tools/validate.py
```

All 15 public problems passed on the official RTX 5090 runner. Additional checks
cover numerical edge cases, launch resources, and staged transformer execution.
[Validation results](docs/results.md) distinguish CPU checks, GPU execution, and
timing measurements. [Testing instructions](docs/testing.md) cover reproduction
and optional GPU tools.

## Scope

The current target is static `f32` kernels with a one-dimensional CUDA launch.
Internal integer and boolean values support indexing and masks. Additional
storage types, tensor-core instructions, shared-memory swizzles, and auxiliary
thread bands are unsupported. The compiler validates supported shapes and ABIs;
it does not prove whole-program types or memory bounds.

## Repository

```text
compile.py          IR-to-PTX compiler and command-line entry point
docs/               Design, testing, and validation results
tests/checks/       CPU contract and code-generation checks
tests/fixtures/     Regression inputs, references, and PTX snapshots
tests/generators/   Additional workload generators
tests/integration/  Staged transformer correctness checks
tools/              GPU validation, profiling, benchmarking, and packaging
```
