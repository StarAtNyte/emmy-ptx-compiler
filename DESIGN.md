# How the compiler works

## Input contract

`compile.py` accepts one Kernel IR JSON path and emits only PTX on stdout.
`Graph.from_dict` reconstructs Emmy's typed statements and expressions. The
neighboring metadata selects the entry and positional buffer argument order.
The submission never invokes a reference renderer, nvcc, or another compiler.

Buffer shapes come from graph tensor records. Buffer arguments are 64-bit device
pointers. A row-major index becomes an element offset, then a byte offset.
Global addresses are 64-bit; shared-memory addresses are 32-bit. The current
storage and arithmetic implementation supports f32.

## Mapping work to threads

A Tile is a Cartesian product of axes. A linear work index is decoded by dividing
by each axis's row-major stride and taking its remainder. Scalar tiles use the
launched block width; cooperative tiles specify their own work width. Bounds
checks exclude unused work items. Raster grouping is currently replaced with
canonical block visitation, which changes locality rather than logical coverage.

The implementation assumes the supplied cooperative geometry keeps participating
threads alive through its barriers. More general partial cooperative blocks need
explicit analysis; arbitrary early exits before barriers are not a safe extension.

## Values, registers and control flow

Fresh PTX virtual registers hold computed values. A map connects IR value names
to registers. ptxas performs physical register allocation. Index trees lower
recursively, including arithmetic, comparisons, logical conditions and selection.
Grouped memory operations currently become individual scalar accesses.

Branches and loop bodies have lexical value environments. Ordinary local
bindings cannot replace an enclosing name after the block, and a nested loop
axis may shadow an enclosing axis without destroying its binding. Each branch
starts from the same enclosing environment. Loop results export their carried
accumulators; other locals remain private, including between expanded bodies.
When a reduction primitive rebinds an externally carried state to a fresh
register, a copy into the enclosing register publishes its runtime update before
restoring the environment. Nested seeded folds and local declarations stay local.
A Let bound to a variable captures its current value with a copy; it must not
alias a register that an accumulator can mutate later.

Loop axis bounds and strides come from IR. A seeded loop initializes the folds
carried by its axis before the loop. A non-seeding loop updates existing state,
which matters for matmul's accumulation across shared-memory chunks. Outer-loop
accumulators and inner-loop accumulators have different reset points.
Discovery respects lexical scope: a nested axis of the same name shadows the
outer axis, and a locally seeded inner fold cannot choose the outer fold's
identity. True updates to an externally seeded state still cross inner loops.

The code deliberately uses separately rounded add and multiply rather than
contracting arbitrary arithmetic into FMA. This preserves exact public pointwise
expectations. An adjacent multiply followed by an additive Accum in the same body
can use fma.rn.f32: adjacency proves the operands have not changed, and reduction
tolerances permit the single-rounding contraction. Ordinary pointwise expressions
remain separate. The discarded multiply is left for ptxas dead-code elimination
if no other statement consumes it.

Straight-line scalar loops reading only shared memory expand by a power-of-two
factor up to 64, capped by the extent and a 1,024 expanded-statement budget.
A scalar epilogue preserves ordered remainders. Eligibility also limits the
original body to 64 statements and the extent to 4–1024. The memory-space guard
is deliberate: an unrestricted experiment
improved matmul but doubled one global-memory attention kernel's runtime. No
kernel-name or public-problem checks select this optimization.

## Reductions

WarpShuffle exchanges each state component with an XOR partner and runs the
IR-provided merge program. This supports coupled states such as online softmax's
running maximum and normalized exponential sum. The merge program's statement
order matters: old state must be consumed before its update.

TreeHalve preserves the binary halving order. A conservative use analysis selects
the cheaper register path only when no explicit read, atomic update, repeated
tree or enclosing loop can observe its scratch mutations. Each thread loads its
partial states, merges the same ordered pairs as the IR and retains the final
required barrier, which also orders unrelated shared-memory accesses.

Observable scratch uses a materialized tree. Logical coordinates follow the IR,
including segmented column subsets and named participant barriers. Multiple lanes
may name one slot. They first cache operands/results in registers, synchronize,
atomically reset that slot's first component to a counter, synchronize, acquire
atomic tickets, synchronize, then let ticket zero write all result components.
Another barrier publishes the restored floating-point state. No extra shared
allocation is needed. This slower fallback is limited to observable scratch.

Short softmax rows exposed an important neutral-state problem: empty lanes carry
maximum=-infinity and sum=0. Combining two empty states naively evaluates
exp(-infinity - -infinity), creating NaN. Merge lowering checks the known identity
tuple and selects the other operand when either operand is neutral. This leaves
ordinary scalar subtraction and exponential semantics unchanged.

## Matmul transport

Smem declares shared buffers and their address-space information. CpAsyncCopy
emits a direct global-to-shared asynchronous copy. Commit, wait and barriers
preserve the supplied dependency order. Atomic writes emit reduction-add
instructions; the launcher handles the metadata's zero-initialization contract.

Static shared allocations above 48 KiB fail assembly. Such kernels use one
aligned external shared pool with computed per-buffer offsets. The prescribed
launch supplies the dynamic allocation. Shared swizzles remain unsupported.
Allocation belongs to each declaration, so nested arrays with equal source names
have distinct live storage and header symbols. Lexical exit restores pointer,
shape and address-space bindings. The dynamic pool honors declaration alignment;
an undersized prescribed dynamic allocation fails compilation with a size error.

## Evidence and limits

The validation tools compare every final output against independently computed
expected arrays using each problem's tolerances. The local runner invokes the
CUDA driver directly, adapting the PTX target to the local sm_86 GPU. Hosted tests
keep the compiler's sm_89 PTX target and run on the organizers' RTX 5090.

Generated shape variations, finite-input stress cases and an actual seeded
PyTorch MLP have caught bugs beyond the public suite. These are development and
regression cases, not a statistical estimate of hidden-test success. The reports
record a SHA-256 of the compiler used for each run.

Tile exits use fresh labels, including repeated tiles. Let literals preserve their
integer/floating type. Ordinary branch-local SSA values do not need phi nodes:
the IR scopes them to their branch, while mutable accumulators use already-bound
registers. Conditional-accumulator regressions and a seeded seven-family
lexical-scope corpus check this boundary, including captured values, nested axes,
conditional full-warp reductions and named-barrier trees with observed scratch.

Remaining priorities include held-out computations, wider dtype/operation
coverage and additional measured performance work. The supported compute/storage
path is f32; this document does not claim every possible IR field combination or
numeric corner case has been tested.
