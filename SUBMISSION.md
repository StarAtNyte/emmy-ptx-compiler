# Submission draft

Submission page: https://riftstack.ai/hackathon/submit

The live form was inspected on 2026-10-23. It asks for entrant name/email,
optional teammates (up to three people total), tech stack, approach, public
repository URL, video URL, and the entrant's age/rules/video confirmations.
The deadline is October 27 at 23:59 AoE, October 28 at 17:44 in Nepal.

## Technical fields ready to use

Private repository: https://github.com/StarAtNyte/emmy-ptx-compiler

Tech stack:

> Python 3.12, Emmy 0.3.11 typed IR loader, direct PTX 8.0 generation, NVIDIA
> ptxas/RTX 5090 runner, NumPy and PyTorch reference tests, CUDA driver API,
> NVIDIA Compute Sanitizer.

Approach:

> A typed visitor translates Emmy Kernel IR directly to PTX, using Emmy only to
> deserialize the IR. Metadata supplies positional buffer arguments and launch
> geometry. Our lowering implements coordinates, global/shared addressing,
> lexical bindings, loop-owned accumulators and virtual register allocation.
>
> Reductions preserve ordered combines and synchronization. Coupled online-softmax
> states handle empty lanes explicitly. Observable shared-tree mutations use one
> writer per logical slot; unobserved scratch can use a register reduction.
>
> Structural optimizations include fold FMA, bounded shared-only loop expansion,
> signed power-of-two index lowering and aligned shared reads across iterations.
> Four-wide reads require matching eligibility proofs at both widths; mixed
> eligibility retains two-wide reads and tails stay scalar. Latest seven-pair
> RTX 5090 measurements compare two-wide with four-wide: public matmul
> 17.024→15.008 microseconds (1.134x), independent 256×512×384
> 13.152→11.104 (1.184x). Clocks were unlocked; these are selected kernel results,
> not a model speedup. Regressing expansion variants were rejected.
>
> Validation covers all public categories, independent shapes and operation mixes,
> renamed and fresh-seed typed IR, model comparisons and selected sanitizer runs.
> New logarithm/exp2 support passes 104 typed and four frontend compound cases on
> both GPUs. Internal boolean SSA/masks pass 128 typed and 13 frontend workloads,
> including full masked attention. Square/erf lowering closes GELU/RMS coverage
> and passes a three-layer GELU MLP for three input seeds. The refined erf
> implementation measures up to3ULP on selected datasets, not a global guarantee.
> 1104 prior programs retain identical PTX. Four sanitizer tools are clean on31
> selected math/model cases. The local no-Torch Python 3.12–3.14 matrix passes
> 162 checks per version. Hosted CPU CI has not executed the current suite.
>
> This is a static f32 compiler with documented unsupported operations and layouts.
> Numerical approximations have explicit tolerances and boundary tests. Hidden-test
> results are unknown. AI coding assistance was used.

## Human fields still required

- Entrant's preferred name and email, and any teammate details.
- A public-link 3–5 minute own-voice video. See `WALKTHROUGH.md`.
- The entrant's own age, rule and video-understanding confirmations.

No entry has been submitted. These fields and confirmations must come from the
entrant. The draft above is below the form's 3000-character approach limit.
