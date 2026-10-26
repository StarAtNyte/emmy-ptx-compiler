# Compiler design

The compiler accepts one Kernel IR JSON file and writes PTX to stdout. Emmy's
`Graph.from_dict` reconstructs typed IR objects. All instruction selection and
IR-to-PTX translation are implemented in `compile.py`.

## Input and launch contract

Each input contains one `KernelOp`. The adjacent `meta.json` supplies the kernel
entry, positional buffer arguments, physical allocations, and launch geometry.
Logical tensor views may differ from allocation shapes. Parameters are 64-bit
pointers; storage is `f32`. Validation rejects unsupported shapes, types, and
launches before printing PTX.

A `Tile` maps its Cartesian iteration space onto the prescribed one-dimensional
CUDA launch. A linear thread index is decoded into axis coordinates. Bounds
guards skip surplus work items. Cooperative kernels must keep the required
threads participating in barriers; arbitrary partial cooperative blocks are
not supported.

## Values and control flow

A binding map connects IR names to PTX virtual registers. Statements dispatch to
individual `lower_*` methods; index expressions compile recursively. Lexical
scopes restore local bindings while preserving updates to external reduction
state. Pure bindings capture values rather than aliasing mutable accumulators.

Loops seed their carried accumulators from the IR operation's identity. The
compiler tracks conservative signed index ranges for safe simplifications.
Signed division and remainder follow the IR's floor semantics, with corrections
when PTX's truncation would differ. Global byte addresses are 64-bit; intermediate
index arithmetic is 32-bit.

## Memory and reductions

Global and shared accesses use row-major offsets. Shared allocations belong to
declarations, including identically named arrays in nested scopes. Allocations
above the static shared-memory threshold use a dynamic pool and must fit the
metadata's shared-memory allowance.

Asynchronous copies, commit groups, waits, and barriers preserve the IR's
ordering. Atomic writes emit floating-point reduction adds. Buffer zeroing is
the launcher's responsibility, as specified by `zero_outputs`.

Warp reductions use XOR shuffles. Shared trees preserve operand order and
observable scratch mutations. When scratch is observable, one lane writes each
logical slot. Coupled online-softmax state treats empty operands as reduction
identities so idle lanes do not introduce NaNs.

## Numerical behavior

Ordinary pointwise arithmetic uses rounded `f32` instructions. Adjacent
multiply/add folds may contract to an FMA. The compiler implements compound math
functions directly: corrected exponential, polynomial tanh and erf, reduced
logarithm, trigonometric argument reduction, and an `f64`-based power operation.
These implementations are tested against numerical references; sampled error
measurements are not global error guarantees.

## Optimization and resources

Straight-line shared-memory loops can expand by a bounded power-of-two factor.
Aligned adjacent reads combine within a body or across expanded iterations only
when dependency and alignment checks permit it. Scalar tails retain their order.
Unsupported patterns take the scalar path.

NVIDIA's assembler allocates physical registers. Blocks of at most 256 threads
receive a `.maxntid` hint. Larger blocks always receive a register cap bounded
by the target's register budget, accounting for complete physical warps and
allocation granularity. A state/math estimate tunes the budget within that
ceiling; underestimation can cause spilling but cannot remove the ceiling.
1024-thread blocks retain a 64-register cap.

The compiler does not change the supplied launch geometry, select by workload
name, or invoke Emmy's renderer, nvcc, or another lowering backend.
