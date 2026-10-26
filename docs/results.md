# Validation results

Measured on 26 October 2026 with the compiler from commit `3504527`.

| Check | Result | Environment |
| --- | --- | --- |
| Public challenge problems | 15/15 | Official RTX 5090 runner |
| Shape/op-mix variations | 43/43 | Official RTX 5090 runner |
| Edge shapes | 57/57 | Official RTX 5090 runner |
| Batched/broadcast layouts | 13/13 | Official RTX 5090 runner |
| Additional frontend schedules | 16/16 | Official RTX 5090 runner |
| Fresh register-pressure cases | 5/5 | Official RTX 5090 runner |
| CPU contracts | 176 per Python version | Isolated Python 3.12, 3.13, 3.14 without PyTorch |
| Register-ceiling invariants | 3,072 per Python version | CPU; every block width 257–1024 at four state pressures |
| Staged transformer | 9 runs, 147 stage comparisons | Local RTX 3050; three seeds and depths 1, 2, 4 |
| Compute Sanitizer | Four tools passed | Selected local attention and shared-matmul cases |

The launch-pressure tests caught real failures: the earlier compiler emitted
kernels that exceeded the register budget for 384-, 512-, and 768-thread blocks.
The retained `launch-pressure-512` fixture reproduces that regression.

## Selected timings

Seven alternating paired trials on the official runner, with eight warmups and
75 timed iterations per trial. Outputs were checked in full on the first trial
and by returned prefix thereafter. Clocks were unlocked.

| Workload | Before register fix | After register fix |
| --- | ---: | ---: |
| Public 512 × 512 × 512 matmul | 15.040 µs | 15.040 µs |
| 32 × 16 attention | 17.024 µs | 17.024 µs |
| 256 × 512 × 384 shared matmul | 11.072 µs | 11.168 µs |

These results support preserving performance on the measured workloads while
fixing launch failures. They do not establish optimal budgets for other kernels.
The hidden evaluation set is unavailable.

## Optimization milestones

These earlier paired experiments compare specific compiler revisions. They are
independent of the register-safety comparison above, and their speedups should
not be multiplied together.

| Change | Workload | Before | After | Speedup | Pairs |
| --- | --- | ---: | ---: | ---: | ---: |
| Two-wide to four-wide aligned shared reads | 512 × 512 × 512 matmul | 17.024 µs | 15.008 µs | 1.134× | 7 |
| Two-wide to four-wide aligned shared reads | 256 × 512 × 384 matmul | 13.152 µs | 11.104 µs | 1.184× | 7 |
| Small-block `.maxntid` hint | `a02_attention_128` | 53.888 µs | 41.600 µs | 1.295× | 5 |

Trials used the official RTX 5090, eight warmups, and 75 timed iterations per
request, alternating compiler order. Buffers were reused, clocks were unlocked,
and outputs were checked in full first and by returned prefix on later trials.
Adding accurate exp changed row-softmax from 2.976 to 3.104 µs, a 0.128 µs cost.

The original samples and compiler fingerprints are preserved in the
[shared-read experiment record](https://github.com/StarAtNyte/emmy-ptx-compiler/blob/de5609d/validation/2026-10-26-production-iteration15.json)
and [public launch-bound timing record](https://github.com/StarAtNyte/emmy-ptx-compiler/blob/de5609d/validation/iteration19-bench-l-public.json).
Shape/layout results are retained in the
[shape probe record](https://github.com/StarAtNyte/emmy-ptx-compiler/blob/de5609d/validation/2026-10-26-shape-probes.json);
the 43 shape/op-mix results are in the
[compiler acceptance record](https://github.com/StarAtNyte/emmy-ptx-compiler/blob/de5609d/validation/2026-10-26-production-iteration19.json).

Detailed experimental reports were removed from the working tree during repository
cleanup. The original records remain in git history at commit `de5609d`, under
`validation/2026-10-26-launch-resources.json` and
`validation/iteration20-equivalence.json`. New test and benchmark reports belong
in ignored `artifacts/`.
