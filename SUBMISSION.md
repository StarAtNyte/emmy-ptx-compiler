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

> A typed visitor lowers Emmy Kernel IR directly to PTX without calling an
> existing IR-to-code compiler. It uses the supplied metadata's parameter order
> and launch contract, and explicitly tracks tile coordinates, loop-owned
> accumulators, address spaces, shared-memory allocation and virtual registers.
> Lexical environments preserve shadowed bindings and captured values while
> explicitly carried reduction states remain writable across control flow.
>
> Reduction lowering supports coupled online-softmax state and neutral-state
> handling for empty lanes. Binary tree order and synchronization are preserved.
> When scratch mutations are observable, an atomic ticket selects one writer per
> logical slot, including repeated lanes and named-barrier groups, without extra
> shared allocation. A cheaper register reduction is used when scratch writes
> are dead.
>
> Optimizations are tested with alternating paired runs on the hosted RTX 5090.
> Adjacent multiply/add reduction steps contract to FMA; bounded shared-memory
> scalar loops expand by up to 64 iterations under extent/body-size limits,
> with scalar tails. FMA measured about 1.32x for one attention kernel. Shared
> expansion improved one matmul step by 1.33x, then a further 1.12x in later
> paired trials; these are separate measurements, not a combined benchmark.
> Aligned shared-read pairing adds a measured 1.03x public matmul gain. Exact
> signed power-of-two index shifts/masks then measure 2.57x on that matmul and
> 2.18x/2.09x on two independent shapes. These remain separate paired observations.
> Cross-iteration aligned shared reads add 1.48x on public matmul and 1.47x/1.39x
> on two independent shapes in seven paired trials; scalar tails and dependency
> guards are retained. The other sixteen workloads differ by at most 0.032 us.
> A broader unsigned tile rewrite regressed attention and was rejected; the
> accepted signed refinement preserves its latency. Broader unrolling and an
> explicit global-vector pass were also rejected after measurement.
>
> Validation includes all public categories, independent shape/operation and
> schedule variations, finite-value stress cases, semantic regressions, actual
> PyTorch MLP and staged transformer comparisons, and selected memory/race/sync/
> initialization checks. Records identify the compiler hash and test scope.
> Seeded typed-IR probes use independent exact numerical formulas, renamed
> counterparts and per-case process isolation; they exposed and fixed lexical
> binding leakage and captured-value aliasing.
> Hidden-test results are unknown. AI coding assistance was used.

## Human fields still required

- Entrant's preferred name and email, and any teammate details.
- A public-link 3–5 minute own-voice video. See `WALKTHROUGH.md`.
- The entrant's own age, rule and video-understanding confirmations.

No entry has been submitted. These fields and confirmations must come from the
entrant. The draft above is below the form's 3000-character approach limit.
