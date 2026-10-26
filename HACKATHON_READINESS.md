# Hackathon alignment and readiness

Reviewed 26 October 2026 against the official
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
| Five evaluation categories | All 15 public problems pass in the accepted compiler's 19-workload target benchmark; retained generated variants cover all five categories. | Hidden tests are inaccessible. Existing development regressions do not prove hidden-set correctness. Broaden scalar op mixes and relevant statement-field coverage. |
| General compiler, no public-problem hardcoding | Lowering selects by typed operations and structural eligibility, not workload IDs. Fresh shapes, seeds and renamed variables are checked. | There is no formal whole-program equivalence proof. Unsupported operators and layouts remain documented. |
| Compile within seconds per kernel | Three isolated cold processes per public kernel: medians 0.854–0.971 seconds under concurrent local CPU checks, without Torch. | This measures the local environment only. |
| Kernel speed | Correctness-checked alternating trials on official RTX 5090; sample distributions and source hashes retained. Regressing tuning variants rejected. | No hidden performance result or counter-proven bottleneck. Speed is the last tiebreak. |
| Public repository at submission | Repository remains private under the user's explicit instruction. | Must be made public with user authorization before submitting under current rules. |
| Personal 3–5 minute walkthrough | Compiler design, submission draft and walkthrough outline exist. | User's own recorded explanation and entrant details are still missing. Do not invent them. |
| Submit before deadline | Deadline: 27 October 23:59 AoE, or 28 October 17:44 Nepal time. | Submission remains explicitly deferred. No entry or organizer message sent. |

Evidence: [accepted compiler](validation/2026-10-26-production-iteration18.json),
[current package](validation/2026-10-26-current-package.json),
[address-boundary and CLI checks](validation/2026-10-26-wrap-addresses.json),
and [rejected tuning variants](validation/2026-10-26-bounded-tuning.json).

Square and own erf lowering now cover the generated GELU/RMS workloads and a
three-layer GELU MLP. The local matrix has162 checks per Python version. Prior
mask coverage and all1104 retained programs preserve their PTX. Remaining
frontend scalar functions and schema/type/bounds coverage still need audit;
this is not a hidden-set success or broad production-readiness claim.
Large dtype expansions and Tensor Core backends are not required by the published
scope. Keep launch, dtype and arithmetic semantics when considering optimization.
