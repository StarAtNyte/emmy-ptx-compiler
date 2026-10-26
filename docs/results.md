# Validation results

Measured on 26 October 2026 with the compiler from commit `3504527`.

| Check | Result | Environment |
| --- | --- | --- |
| Public challenge problems | 15/15 | Official RTX 5090 runner |
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

Detailed experimental reports were removed from the working tree during repository
cleanup. The original records remain in git history at commit `de5609d`, under
`validation/2026-10-26-launch-resources.json` and
`validation/iteration20-equivalence.json`. New test and benchmark reports belong
in ignored `artifacts/`.
