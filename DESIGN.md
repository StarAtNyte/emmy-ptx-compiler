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

Straight-line bodies of at most 64 statements may pair two aligned, adjacent
f32 shared reads into ld.shared.v2.f32 across pure arithmetic. Linearized index
terms prove adjacency and eight-byte alignment; declaration alignment must also
permit eight-byte accesses. Writes, barriers, global reads and control flow
disable pairing. Intervening index mutations and uses or redefinitions of the
second destination prohibit motion. Removing the scalar second load retains its
original arithmetic-adjacency boundary for FMA selection. This pass uses IR
structure and lexical storage bindings, without kernel-name checks.

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

## Unsupported features

Before lowering, explicit tensor dtypes must be f32, shared declarations must use
float/f32 storage, scalar arithmetic declarations must be f32, and Let may also
bind i32 index values. Writes accept f32/i32 scalar values into f32 storage;
integer values are numerically converted. Other explicit write types and swizzled
writes are rejected. Missing/inferred scalar dtypes retain their existing interpretation.
Errors identify the kernel and nested statement location before any PTX reaches
stdout; expected ValueError/NotImplementedError failures print a concise CLI
message. This is a feature boundary check, not a complete IR type checker.

## Signed index expressions

Both `/` and `//` floor signed index division, matching Emmy 0.3.11's expression
contract; `%` returns a remainder with the divisor's sign. Lowering computes the
PTX truncating remainder and corrects quotient/remainder when the remainder is
nonzero and operand signs differ. Conservative signed-int32 interval proofs skip
this work only for a proven nonnegative dividend and positive divisor. Bounds
belong to register identities, with tile/loop bounds and integer Let snapshots;
unknown expressions and possible arithmetic overflow discard the proof. This
retains public matmul code generation while supporting negative index arithmetic.
Division by zero and signed division overflow are outside supported behavior.

## Strided-loop expression timing

A strided loop evaluates its start in the enclosing environment, initializes its
axis, then captures an explicit end bound once. The captured value is visible
inside the body as `<axis>_end`, matching installed Emmy 0.3.11's renderer.
Axis-dependent steps execute after each body at the loop increment; steps using
the captured end binding are also resolved there. Independent immutable steps
retain their existing lowering. Range proofs for axis-dependent bounds/steps
remain conservative rather than treating their initial values as whole-loop
bounds. Exiting the loop restores enclosing bindings and exports carried states.

For a proven singleton positive power-of-two divisor, signed floor division lowers
to arithmetic right shift and remainder to a mask. This is exact for negative
numerators as well. Boundary fixtures observe both sign and low bits near INT_MIN
and INT_MAX, excluding division overflow. The corresponding unsigned Tile rewrite
was not adopted: it regressed one target attention workload despite fewer native
instructions. Current tile-coordinate lowering retains its prior behavior.

## Predicate and integer register classes

Expression lowering distinguishes predicate-valued intermediates from numeric
index values. Integer contexts materialize a predicate as 0/1 with selp.b32 before
arithmetic, numeric comparisons, ternary branches, integer Let bindings or memory
addressing. Condition contexts retain native predicates, preserving existing
branch lowering. This also applies to aligned paired shared-read addresses; no
predicate register can be used directly as an integer byte/element offset.

## Scalar expression types and conversions

Expression inference follows literals and actual bound register types, including
mixed branches. Floating +, -, * and comparisons promote integer/predicate operands
to f32; mixed ternaries select f32 values. Separate .rn operations preserve
pointwise rounding. Float truthiness uses unordered not-equal to zero, so NaN is
true and both signed zeros false. Conditions retain native predicate registers.

Explicit Let f32/i32 bindings numerically convert values (integer→float nearest,
float→integer toward zero) rather than copying their bits. Same-type variable
bindings still snapshot with a move. Floating elementwise arguments, Select
branches, folded FMA operands and integer scalar writes into f32 storage follow
these conversions. Mutable fold state itself remains f32.

Index expressions must be integer-valued; floating offsets reject before PTX
output. Floating Expr /, // and % are explicitly unsupported because this AST's
contract specifies integer index division; use typed scalar Assign divide for
f32 division. Float→integer casts are supported for finite in-range values; no
nonfinite/out-of-range cast guarantee or general type-checker claim is made.

## Address rank and physical allocation views

Addressing follows installed Emmy 0.3.11 `render_index`: empty coordinates select
zero; one coordinate is a flat pointer offset regardless of logical rank; a full
rank tuple uses row-major strides; other multi-coordinate tuples sum as additive
flat offsets. Paired shared-read adjacency/alignment proofs use the same mapping.
No coordinate is silently dropped. Logical view dimensions need not equal the
allocation descriptor's rank; metadata supplies actual buffer capacity/ABI order.
In particular, a scalar logical view can use flat coordinates into a larger
provided allocation. This is not a general bounds checker or launch validator.

Launch geometry is checked before PTX generation: grid and block are three
positive integer dimensions, y/z are one for this compiler's one-dimensional
mapping, block.x is at most 1024 and grid.x at most 2147483647. smem_bytes is a
nonnegative integer. Cooperative Tile.block_threads must be positive and match
block.x when no auxiliary band is present. Auxiliary bands remain unsupported.
These checks do not infer required grid coverage, validate argument descriptors,
or establish dynamic memory safety. Logical tensor/allocation ranks may differ.

The input IR file must contain exactly one KernelOp, selected by a unique matching
metadata entry. Other launch entries may coexist in meta.json for a kernel chain.
arg_order contains each KernelOp input/output buffer once; args descriptors must
name those buffers in the same order because the runner passes them positionally.
Consistent permutations are legal. Allocation descriptors use f32 and a list of
nonnegative integer dimensions (including scalar []); physical shapes need not
match logical tensor shapes. This validates ABI syntax and buffer membership,
not access bounds, initialization/output roles, data payloads or chain provenance.

The bounded shared-only unroll also pairs reads across two iterations. A scalar
load must have affine flattened index coefficient +1 for the loop axis, an even
constant and even coefficients for every other term. Any other term containing
the loop axis is rejected. The physical allocation must be aligned to at least
8 bytes, and no index dependency may be declared or updated in the loop body.
Existing within-body pairs take precedence. The unrolled loop starts at zero
and advances by an even power-of-two factor, proving an even address at each
paired position. The first iteration emits ld.shared.v2.f32; its second register
supplies the next iteration's scalar binding. Pair state is cleared before scalar
tails. Only the existing straight-line shared-read-only expansion is eligible;
there is no motion across writes, global reads, barriers or control flow.

Typed tensor, Tile and loop dimensions must be static nonnegative integers. An
empty tile domain (any axis extent zero) emits no work before coordinate decoding,
which avoids zero-divisor strides. An empty axis LIST instead has product one,
so it still executes one logical cell. Zero logical tensor dimensions and scalar
views remain valid independently of the physical metadata allocation. Shared
allocation extents must be positive integers (scalar shape () allocates one f32);
zero-sized shared arrays remain explicitly unsupported. Shared alignment must be
zero for natural alignment or a positive power of two. Checks run before shared
allocation/address arithmetic, including nested and else bodies. This does not
establish cardinality/address overflow safety, grid coverage or access bounds.
