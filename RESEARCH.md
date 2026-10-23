# Emmy competition research and iterative strategy

Research snapshot: 2026-10-23. This is a living plan, not a claim of complete investigation or a working compiler.

Implementation update, 2026-10-23: a first direct compiler now passes all 15 public problems through the official hosted runner. Evidence is saved in `artifacts/public-results.json`; a typed feature inventory is in `artifacts/ir-inventory.json`. Pinned 0.3.11 loading and actual runner execution are verified. Independent shape/schedule variations and model integration remain unverified. See README for current limits.

Second iteration: all 23 independently generated shape/operation variations passed the hosted runner. The corpus exposed missing Select lowering and an online-softmax empty-state NaN; both were fixed generally. A three-layer nn.Module MLP with ReLU/SiLU matches CPU PyTorch on local and hosted GPUs. Local CUDA-driver execution now works without nvcc. CPU Torch 2.14.1+cpu generates test IR. The model packager restores parameter tensors because Emmy's JSON omits their values and the official generator handles scalar constants only. A further 69 finite-input stress variants passed locally. Schedule diversity, broader model families, sanitizer execution and performance remain open.

## Evidence collected

- Challenge checkout: `challenge`, commit `ec8cde1943f88efea3827a95bda6085553d2d407`.
- Upstream source checkout: `upstream-emmy`, commit `8cfa12b85eb63e80168069b507711ffe9d093c44`. Main is reference material; the contest requires emmy-ml 0.3.11, so inspect that installed release before relying on current internals.
- Read challenge README, TASK, MVP, API, generator, expression specifications, provenance, submit runner, and representative pointwise, split-K matmul, softmax and attention IR.
- Local GPU query reports RTX 3050 Laptop, 6 GiB, compute capability 8.6.
- Hosted `/health` responds with compile/run available and RTX 5090, default target sm_120. No actual kernel execution has been verified yet.

## What determines winning

Correctness on hidden problems by category, then compiler code review, then generated kernel speed. No numerical weights or complete aggregation formula are published. Categories 1–2 are the stated minimum target; partial credit is mentioned. Treat category ranking details as unknown, rather than assuming a particular score.

Public source and a 3–5 minute own-voice walkthrough are required. AI assistance is allowed, but the entrant must understand the implementation. Deadline: October 27 23:59 AoE, October 28 17:44 Nepal time.

The submission must implement its own translation. Emmy may load the IR; do not call Emmy's renderer, emmy compile, nvcc, or another compiler as a submission lowering path. Reference CUDA and PyTorch belong in development-only validation.

## Most promising advantage

Implement one faithful, typed IR visitor rather than separate recognizers for public problem names. Public attention already consists of ordinary nested loops, arithmetic, online softmax state, loads and stores. This suggests broad statement support can unlock advanced categories without a bespoke attention algorithm. It remains a hypothesis until tested across generated variants.

Before emitting PTX, analyze accumulator scope and initialization, buffer address spaces, axis mapping, and types. Accumulators can reset inside an outer loop or persist across it. Online softmax reductions combine coupled states; independently summing or maximizing them is incorrect. Preserve update ordering and evaluate combine inputs before overwriting state.

Thread geometry is a critical correctness surface. Derive mappings from typed IR plus prescribed launch. Extra threads must not access buffers, but early returns must not strand participating threads at barriers. Use the supplied parameter order and atomic-output initialization contract.

## Creative validation experiments

1. Build a feature inventory using Graph.from_dict and typed visitors: statements, elementwise operations, dtypes, loop nesting, vector widths, shared memory, axis mappings and combine-state patterns. Do not parse serialized body strings ourselves.
2. Generate a private development corpus with the organizers' public generator, pinned release, sm_89 target and isolated scheduling state. Hold out an entire shape family and operation combination until a milestone is complete.
3. Use metamorphic testing: rename symbols, change runtime constant-buffer values, swap commutative operands, and apply other proven semantics-preserving transformations. Changing a name must not change behavior. Never mutate a shape without rebuilding indexing and launch metadata.
4. Vary schedules as well as shapes through supported Emmy knobs. Record actual structural changes; reject generator failures separately from compiler failures.
5. Use three-way comparisons where possible: high precision NumPy/PyTorch reference, Emmy reference CUDA, our PTX. If both GPU paths disagree with the mathematical reference, investigate source generation or tolerance rather than immediately changing our lowering.
6. Test adversarial finite values: cancellation, repeated values, tiny magnitudes, large softmax offsets, and reduction identities. Add NaN/infinity cases only where semantics are defined and useful; do not let exotic cases displace contest coverage.
7. Keep every failing seed and shrink it to a smaller computation or schedule. Prioritize the feature fixing the greatest number of failures.
8. Compare optimized code against an initially conservative lowering. Enable an optimization only after semantic equivalence tests; avoid automatic FMA contraction where exact rounding is required.

## Real-model validation

Use a small MLP, transformer block and then tiny multi-layer transformer. These are proposed tests, not verified supported workloads. First inspect the pinned release's backend integration and unsupported operations. Model inputs need not require downloaded pretrained weights: deterministic randomly initialized models still exercise compiler correctness.

The hosted runner accepts custom input data and supports kernel-level execution. submit.py chains intermediates by returning values and passing them into subsequent requests. That permits small multi-kernel checks, but transfers and output limits make it unsuitable for representative model performance. Prefer local CUDA execution for integration tests. The local GPU is sm_86, so keep IR generation targeted to contest sm_89 while using a separately validated compatible PTX target for local tests; always recheck the actual contest target on the hosted runner.

Use NVIDIA Compute Sanitizer locally when available to check memory, shared-memory races, uninitialized global memory and synchronization. Hosted timing is indicative and not clock-locked; do not optimize from tiny timing differences.

## Iteration sequence and gates

- Iteration 0: pinned Python environment; typed corpus inventory; known-good example passes hosted runner; local CUDA tools and pinned Emmy interfaces verified.
- Iteration 1: pointwise visitor; all four public cases and independent shape/operation variations pass.
- Iteration 2: reductions, accumulator scope and coupled combines; all eight baseline cases plus independent reduction variants pass.
- Iteration 3: fused and attention loops where existing primitives suffice; prioritize by measured feature overlap rather than category number.
- Iteration 4: shared copies, split-K atomics and matmul; target all 15 public problems and generated held-out cases.
- Iteration 5: model integration, sanitizer checks, numerical robustness and clean code review. Performance only after coverage is secure.

After every iteration record: exact revision, tested case count, pass/fail by category, newly covered features, compile time, numerical error, unresolved bugs, and next highest-value experiment. Passing public examples alone never counts as generalization.

## Research boundaries and open questions

Source inspection of public Emmy and generated artifacts is directly useful; binary reverse engineering is unnecessary at present. Do not seek private hidden tests or exploit the runner. The advantage should come from understanding semantics and testing more effectively.

Still to verify: pinned 0.3.11 classes and axis mapping; supported elementwise operation registry; schedule knobs; local toolkit availability; custom-model tracing; exact hidden-set category aggregation; evaluator dependency and timeout policy. No competitor implementation inspection has been performed.

## Primary sources

- https://github.com/cloudrift-ai/emmy-hackathon-2026
- https://github.com/cloudrift-ai/emmy-hackathon-2026/blob/main/TASK.md
- https://github.com/cloudrift-ai/emmy-hackathon-2026/blob/main/tools/generate.py
- https://github.com/cloudrift-ai/emmy-hackathon-2026/blob/main/problems/PROVENANCE.md
- https://github.com/cloudrift-ai/emmy-hackathon-2026/blob/main/docs/API.md
- https://riftstack.ai/hackathon
- https://github.com/cloudrift-ai/emmy
- https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html
