# Hackathon alignment and readiness

Reviewed and updated 26 October 2026 against the official
[README](https://github.com/cloudrift-ai/emmy-hackathon-2026/blob/ec8cde1943f88efea3827a95bda6085553d2d407/README.md)
and [task specification](https://github.com/cloudrift-ai/emmy-hackathon-2026/blob/ec8cde1943f88efea3827a95bda6085553d2d407/TASK.md).
The repository head was rechecked and remains that revision.

The project is aligned with the engineering task. Submission readiness and
winning competitiveness remain unproven. The judging order is hidden-category
correctness, code review, then kernel speed; further work should follow that order.

| Requirement | Current evidence | Remaining uncertainty or action |
|---|---|---|
| Runnable compiler program | Root `compile.py` implements the exact path-to-IR / PTX-on-stdout CLI. Three fresh isolated invocations per public kernel produce identical PTX. | Local process timings do not establish an unknown judge time limit. |
| Python 3.12+, Emmy 0.3.11, no GPU or Torch needed to compile | Isolated no-Torch 3.12, 3.13 and 3.14 contract checks; dependency constraints retained. | GitHub CPU jobs currently cannot start because of account billing restrictions. |
| Our own IR-to-PTX translation | Typed statement visitor and expression lowering in `compile.py`; imports Emmy's IR classes and `Graph.from_dict`. No invocation of a renderer, nvcc or another lowering backend. | Judges perform their own code review; this is a source inspection, not a review score. |
| Entry, pointer ABI and prescribed launch | Entry comes from KernelOp; arg order comes from metadata. Dedicated ABI and launch suites include consistent permutations and physical/logical views. | Complete schema, bounds, grid-cardinality and overflow validation remain open. |
| Five evaluation categories | All 15 public problems pass on the official RTX 5090. Fresh proxies generated with the organizers' tool pass there too: 43 shape/op-mix cases, 57 edge shapes and 13 batched/broadcast layouts across all five categories. Probing found and fixed two real bugs (large-block launch failure, NaN attention on small shapes) and tight-tolerance math failures. | Hidden tests are inaccessible; proxies are not the hidden set. |
| General compiler, no public-problem hardcoding | Lowering selects by typed operations and structural eligibility, not workload IDs. Fresh shapes, seeds and renamed variables are checked. | There is no formal whole-program equivalence proof. Unsupported operators and layouts remain documented. |
| Compile within seconds per kernel | Cold isolated no-Torch CLI under 0.3 s per public kernel. | This measures the local environment only. |
| Kernel speed | Correctness-checked paired trials on the official RTX 5090; launch bounds made a02 1.295x faster, accurate exp costs f02 0.13 us, others within 1%. | No hidden performance result. Speed is the last tiebreak. |
| Public repository at submission | Repository remains private under the user's explicit instruction. | Must be made public with user authorization before submitting under current rules. |
| Personal 3–5 minute walkthrough | Compiler design, submission draft and walkthrough outline exist. | User's own recorded explanation and entrant details are still missing. Do not invent them. |
| Submit before deadline | Deadline: 27 October 23:59 AoE, or 28 October 17:44 Nepal time. | Submission remains explicitly deferred. No entry or organizer message sent. |

Evidence: [accepted compiler](validation/2026-10-26-production-iteration18.json),
[current package](validation/2026-10-26-current-package.json),
[address-boundary and CLI checks](validation/2026-10-26-wrap-addresses.json),
and [rejected tuning variants](validation/2026-10-26-bounded-tuning.json).

Current compiler and evidence: see HANDOFF.md and
[iteration 19](validation/2026-10-26-production-iteration19.json),
[shape probes](validation/2026-10-26-shape-probes.json) and
[current package](validation/2026-10-26-current-package.json). Elementwise
math matches libm-like accuracy (exp/tanh/sin/cos 2 ULP, log 1 ULP, pow via
f64). The remaining entry work is the user's: own-voice video, public
repository and submission (resubmission allowed until the deadline).

Launch-resource acceptance: [record](validation/2026-10-26-launch-resources.json).
Every block above256 threads now has a safe register ceiling, including partial
warps. Public15/15, recognized frontend16/16 and five fresh pressure cases pass
on the official GPU. Local no-Torch3.12/3.13/3.14 each pass176 contract checks
plus3072 resource invariants. Selected seven-pair matmul/attention timings remain
effectively unchanged. This fixes confirmed launch failures without establishing
hidden-set success or optimal resource allocation for every workload.
