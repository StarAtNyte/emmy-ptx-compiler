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

## Optimizations and engineering work

| Change | Purpose |
| --- | --- |
| Reduction FMA contraction | Combine adjacent multiply-and-accumulate operations with one rounding step. |
| Bounded loop expansion | Expose independent work in shared-memory loops while preserving ordered scalar tails. |
| Aligned shared-memory reads | Combine eligible reads into two- or four-wide loads, including across expanded loop iterations. |
| Signed index simplification | Use shifts and masks where they preserve the IR's floor-division and remainder semantics. |
| Launch bounds and register budgets | Improve small-block scheduling and keep large blocks within the register allocation limit, including partial warps. |

Correctness work includes lexical scoping and accumulator lifetimes, observable
shared-tree writes, boolean masks and conversions, and empty-lane handling in
online softmax. These checks caught and fixed all-NaN attention outputs and
large-block kernels that compiled successfully but failed at launch.

The compiler also implements accurate compound math directly in PTX. Sampled
normal-range errors reached 2 ULP for exp, tanh, sin, and cos, and 1 ULP for log;
sampled subnormal exp results reached 4 ULP. Power uses `f64` intermediates.
These are measured errors on tested inputs, not global error bounds.

## Results

| Validation | Result |
| --- | --- |
| Public challenge suite | **15/15** across all five categories on the official RTX 5090 |
| Additional generated workloads | 43 shape/op-mix cases, 57 edge shapes, and 13 batched/broadcast layouts passed on the official GPU |
| Launch-resource regressions | 16 frontend schedule probes and five fresh pressure cases passed on the official GPU |
| CPU compatibility | 176 contract checks per version on Python 3.12, 3.13, and 3.14 without PyTorch |
| Register ceilings | 3,072 block-size/state-pressure combinations checked per Python version |
| Transformer integration | Nine staged runs covering 147 output comparisons against CPU PyTorch |
| Memory and synchronization | Selected cases passed memcheck, racecheck, synccheck, and initcheck |

The 57 retained fixtures preserve earlier numerical and code-generation
regressions. Transformer stages exchange data through the host and measure
correctness. Test corpora overlap; their counts do not estimate hidden-set
coverage. See [validation details](docs/results.md).

## Performance

Selected optimization experiments on the official RTX 5090 runner:

| Optimization and workload | Before | After | Speedup |
| --- | ---: | ---: | ---: |
| Four-wide shared reads: 512 × 512 × 512 matmul | 17.024 µs | 15.008 µs | **1.134×** |
| Four-wide shared reads: 256 × 512 × 384 matmul | 13.152 µs | 11.104 µs | **1.184×** |
| Small-block launch bounds: `a02_attention_128` | 53.888 µs | 41.600 µs | **1.295×** |

The shared-read measurements use seven alternating paired trials; the launch-bound
measurement uses five. These are separate comparisons between compiler revisions,
not cumulative or end-to-end model speedups. Clocks were unlocked, buffers were
reused, and outputs were checked in full on the first trial and by prefix thereafter.

Accurate exp added 0.128 µs to the public row-softmax kernel. The latest register
safety fix preserved the measured public matmul and 32 × 16 attention medians;
an independent shared matmul changed by less than 1%. Regressing variants were
discarded. [Measurement details and records](docs/results.md#optimization-milestones)
include the baselines and methodology.

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

[Testing instructions](docs/testing.md) cover fixture generation, local GPU tools,
sanitizers, benchmarking, and packaging.

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
