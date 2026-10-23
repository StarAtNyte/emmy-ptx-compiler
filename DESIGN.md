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

Loop axis bounds and strides come from IR. A seeded loop initializes the folds
carried by its axis before the loop. A non-seeding loop updates existing state,
which matters for matmul's accumulation across shared-memory chunks. Outer-loop
accumulators and inner-loop accumulators have different reset points.

The code deliberately uses separately rounded add and multiply rather than
contracting arbitrary arithmetic into FMA. This preserves exact public pointwise
expectations; optimization requires its own numerical validation.

## Reductions

WarpShuffle exchanges each state component with an XOR partner and runs the
IR-provided merge program. This supports coupled states such as online softmax's
running maximum and normalized exponential sum. The merge program's statement
order matters: old state must be consumed before its update.

The current TreeHalve lowering loads shared partials and folds them locally in
each participating thread. This broadcasts results without shared write races,
but changes association from a parallel halving tree. It is conservative for the
small partial arrays tested so far, not a proven replacement for every possible
TreeHalve use or subsequent observation of its scratch storage.

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

## Evidence and limits

The validation tools compare every final output against independently computed
expected arrays using each problem's tolerances. The local runner invokes the
CUDA driver directly, adapting the PTX target to the local sm_86 GPU. Hosted tests
keep the compiler's sm_89 PTX target and run on the organizers' RTX 5090.

Generated shape variations, finite-input stress cases and an actual seeded
PyTorch MLP have caught bugs beyond the public suite. These are development and
regression cases, not a statistical estimate of hidden-test success. The reports
record a SHA-256 of the compiler used for each run.

Remaining priorities include more schedule families, held-out computations,
dtype coverage, repeated Tile labels, branch value merging, scratch-storage
semantics, sanitizer checks and generated-kernel performance.
