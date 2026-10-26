# Emmy PTX Compiler

A direct compiler from typed [Emmy](https://github.com/cloudrift-ai/emmy-hackathon-2026)
Kernel IR to NVIDIA PTX. It lowers control flow, reductions and memory operations
while preserving the kernel entry point, buffer ABI and prescribed launch geometry.

The compiler is implemented in Python and emits PTX directly. Compilation requires
neither a GPU nor PyTorch, and does not invoke CUDA C++, nvcc or another compiler
backend. NVIDIA's assembler and driver perform final machine-code generation.

## Getting started

Requires Python 3.12 or later. Install the pinned runtime with `uv`:

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt -c requirements-ci.lock
```

Compile a kernel:

```sh
.venv/bin/python compile.py path/to/kernel.ir.json > kernel.ptx
```

The input directory must contain the IR file and an adjacent `meta.json`:

```text
problem/
├── k_example.ir.json
└── meta.json
```

The metadata specifies positional buffer arguments, allocation descriptors and
launch configuration. Logical tensor views may differ from physical allocation
shapes. The compiler emits PTX 8.0 targeting `sm_89`; the caller launches the entry
using the supplied grid, block and shared-memory settings.

## Supported features

| Area | Implementation |
|---|---|
| Data types | f32 buffers and arithmetic; i32 scalar bindings and numeric casts |
| Indexing | Static shapes, multi-axis tile decoding, 32-bit index arithmetic, 64-bit global addresses |
| Control flow | Loops, strided loops, conditionals, scoped bindings and carried reduction state |
| Memory | Global and shared memory, grouped accesses, aligned vector reads and dynamic shared allocation |
| Synchronization | Barriers, named barriers, asynchronous copies and atomic adds |
| Reductions | Warp operations, ordered tree merges and coupled online-softmax state |
| Diagnostics | ABI, launch and typed static-shape checks before PTX emission |

Optimizations include adjacent reduction FMA contraction, bounded shared-memory
loop expansion, aligned shared reads across iterations and exact signed
power-of-two index lowering. Selection uses IR structure and alignment proofs;
it does not depend on kernel names.

[DESIGN.md](DESIGN.md) describes the lowering, register model, numerical semantics
and synchronization rules.

Natural, base-two and base-ten logarithms and the base-two exponential use PTX
approximate instructions with subnormal handling enabled. Unknown scalar
operations produce a diagnostic before PTX is written.

## Performance

The following measurements compare two versions of this compiler before and after
two-wide and four-wide aligned shared reads across iterations. Each result uses seven alternating
paired trials on the hosted RTX 5090, with eight warmups and 75 timed iterations.

| Matmul dimensions (M × K × N) | Baseline | Optimized | Speedup |
|---|---:|---:|---:|
| 512 × 512 × 512 | 17.024 µs | 15.008 µs | 1.134× |
| 256 × 512 × 384 | 13.152 µs | 11.104 µs | 1.184× |

Both implementations passed full-output checks in the first trial and prefix
checks thereafter. Clocks were unlocked and buffers were reused. These are
kernel-level measurements for the listed workloads, not end-to-end model results.
See the [measurement record](validation/2026-10-26-production-iteration15.json)
and [target performance analysis](TARGET_PERFORMANCE.md) for samples and assumptions.

## Validation

Validation combines numerical references, generated IR regressions, integration
checks, isolated CLI execution and selected NVIDIA Compute Sanitizer runs.

| Check | Verified scope |
|---|---|
| Official public suite | 15 problems across pointwise, reduction, matmul, fused and attention categories |
| Model integration | Nine staged transformer seed/depth checks, covering 147 stage outputs |
| Python compatibility | 142 local checks per version on Python 3.12, 3.13 and 3.14, without PyTorch |
| Memory and synchronization | Selected memcheck, racecheck, synccheck and initcheck runs |
| Packaging | Extracted archive compilation and three hosted GPU smoke checks |

The transformer checks execute separately traced stages with host transfers and
measure correctness. Generated cases use independent numerical references and
include alignment, loop-tail, scope, shape and initialization regressions. Test
corpora overlap; their counts are not estimates of unseen-workload coverage.

To run the official hosted checker:

```sh
git clone https://github.com/cloudrift-ai/emmy-hackathon-2026 challenge
.venv/bin/python tools/validate.py
```

[TESTING.md](TESTING.md) contains reproduction commands and hardware requirements.
The [CPU workflow](.github/workflows/cpu-contract.yml) is configured for Python
3.12–3.14. Current compatibility results are from isolated local environments;
GitHub-hosted execution remains unverified. Exact source hashes, environments and
results are retained in [validation](validation/).

## Current limitations

The implementation targets static f32 kernels. Additional storage dtypes,
auxiliary thread bands, shared-memory swizzles, zero-sized shared allocations and
some expression operators remain unsupported. Grouped accesses are generally
scalarized unless a vectorization rule proves eligibility. Multi-axis tiles map
to a one-dimensional CUDA grid and block.

The compiler does not yet provide complete whole-IR type checking, access-bounds
proofs or cardinality-overflow analysis. Cooperative kernels must satisfy their
barrier participation requirements. Local sm_86 validation uses a test-harness
target adjustment; the compiler's default output target remains sm_89.

## Documentation

- [Compiler design](DESIGN.md)
- [Testing and reproduction](TESTING.md)
- [Hackathon requirements and readiness](HACKATHON_READINESS.md)
- [Target performance and hardware models](TARGET_PERFORMANCE.md)
- [Research and optimization criteria](RESEARCH_PRODUCTION.md)
- [Validation records](validation/)
