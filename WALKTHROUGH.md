# Four-minute walkthrough outline

The contest requires a 3–5 minute video in your own voice demonstrating your
understanding. Use this outline in your own words. The code and latest validation
record are the evidence to show; the outline is not a claim that a video exists.

## 0:00–0:30 — The task and approach

Explain the exact contract: one Kernel IR JSON path in, PTX on stdout. Emmy loads
the typed IR. Our visitor translates it directly; it does not invoke Emmy's CUDA
renderer or another compiler. State that AI coding assistance was used.

Show the one-command invocation and explain that the evaluator supplies unseen
shapes and operation combinations, so the compiler cannot recognize problem IDs.

## 0:30–1:15 — Mapping the IR

Show `Compiler.statement`, `integer`, and `address` in `compile.py`.

- Metadata supplies positional parameter order and the prescribed launch.
- Tile axes decode a linear work index; extra work items are guarded.
- Buffers use row-major element offsets and 64-bit global pointers.
- Values map to fresh virtual registers; ptxas handles physical allocation.
- Loop folds reset at their owning axis, while non-seeding loops retain state.

Explain why a wrong argument order or tile decode can produce incorrect results
even when PTX assembles successfully.

## 1:15–2:10 — Reductions and correctness

Show `combine` and the two TreeHalve paths.

Online softmax carries a maximum and a scaled exponential sum together. Empty
lanes start with maximum=-infinity and sum=0. A naive merge of two empty states
evaluates exp(-infinity - -infinity). Identity-aware merging avoids that NaN.

The root-only reduction preserves the binary tree's order and required barrier.
If later code observes scratch, the compiler materializes its mutations. Logical
slots can repeat across lanes, so an atomic ticket chooses one writer per slot.
Operands are already in registers before that scratch word becomes a temporary
counter, and the final result restores the word. This requires no extra shared
allocation and works with named participant barriers.

Show an audit case that failed before the fix and now passes. Also explain why
integer Let bindings and repeated Tile labels needed separate fixes.

## 2:10–3:00 — Measured optimization decisions

Show `body`, `accumulate`, and the shared-memory loop guard.

An adjacent multiply followed by an additive fold uses FMA. Ordinary pointwise
arithmetic keeps separate rounding. Expansion by up to 64 applies only to bounded,
straight-line scalar loops whose loads read shared memory; a scalar epilogue
handles remaining iterations. The factor also respects an expanded-body budget.

Show the paired timing records. FMA improved an attention kernel from about
119 to 91 microseconds. Shared-memory loop expansion improved matmul from about
99 to 74 microseconds; later bounded expansion brought it to about 66.
A broader unroll pass slowed attention, and explicit
vector memory instructions were timing-neutral, so those versions were rejected.
Clocks were not locked; these are measured cases rather than universal speedups.

## 3:00–4:00 — Validation and limits

Show the latest validation summary and one official checker invocation. Explain
the coverage: all public categories, independently generated shapes and schedules,
numerical stress inputs, semantic audit cases, actual PyTorch model comparisons,
and selected Compute Sanitizer runs.

The complete transformer test uses separately traced stages because the pinned
Emmy materializer fails on the fully fused block. It checks the same model math
end to end, with host transfers; it is not a model speed benchmark.

State the current scope honestly: f32 compute, static shapes, and the documented
IR subset. Hidden-test results are unknown. Point to the compiler hash in each
record so the results are attributable to a specific implementation.

## Questions to answer before recording

1. Why does `meta.json` determine parameter order?
2. How does the compiler distinguish a seeded loop from a persistent accumulator?
3. Why must softmax merge its maximum and denominator together?
4. When can TreeHalve scratch writes be omitted, and why is its barrier retained?
5. Why do ticket operations not destroy the reduction's original operands?
6. Why is FMA restricted to reduction steps, and unrolling to shared-memory loops?
7. Which tests establish correctness, and which observations establish performance?
