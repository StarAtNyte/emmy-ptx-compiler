# Target calibration and conditional performance models

Measured 2026-10-24 on the hosted RTX 5090, compiler `7fbe5ee4`. No compiler
change or speedup is claimed. Raw responses, sample medians, hashes, references
and assumptions: [target-calibration.json](validation/2026-10-24-target-calibration.json).

[NVIDIA's RTX Blackwell architecture whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf),
Table 3 / Appendix A, specifies 170 SMs, 104.8 TFLOP/s non-Tensor FP32,
1792 GB/s memory bandwidth and 98304 KiB L2. Peak arithmetic rates assume GPU
Boost clock. The downloaded PDF SHA256 is
`906ff2a409d7a7e4cbc56f5d3a179d574120d19aaba99520670e1a0c064595fa`.
These are nominal specifications, not queried target clocks or measured ceilings.

## Calibration

Five trials alternate forward/reverse workload order. Each request uses eight
warmups and 75 timed iterations on reused buffers. Clocks are unlocked; the API
provides no cache eviction or counter controls. Every memory output is checked
exactly against a seeded NumPy float32 reference on the first trial; later trials
check 4096-element prefixes. Times exclude request/host reference work as reported
by the runner. Raw iteration timing samples are unavailable: retained samples are
the runner's per-request medians.

| Workload | Median µs | Effective GB/s | Per-request median range µs |
|---|---:|---:|---:|
| No-work timing control | 2.784 | — | 2.752–2.816 |
| Copy 4 KiB input + 4 KiB output | 2.976 | 2.75 | 2.944–3.552 |
| Copy 1 MiB input + 1 MiB output | 3.008 | 697.19 | 3.008–3.168 |
| Strided read 4 MiB + 128 KiB output | 4.224 | 1024.00 | 3.264–4.384 |
| Strided read 64 MiB + 2 MiB output | 12.896 | 5366.47 | 12.896–13.056 |
| Strided read 224 MiB + 7 MiB output | 146.400 | 1654.52 | 145.696–146.464 |

The read workload sums 32 coalesced planes per output with scalar f32 adds.
Effective bandwidth counts logical input reads plus output writes; it does not
measure memory transactions. The 64 MiB result exceeds nominal DRAM bandwidth,
showing that a DRAM-only model cannot describe all reused-buffer tests. The
224 MiB input exceeds twice published L2 capacity; its effective bandwidth is
consistent with substantial DRAM traffic, but counters would be needed to prove
traffic volume or characterize misses. This reduction is not a pure copy peak.
The no-work time is a timing reference, not a cost to add/subtract from all kernels.

The runner caps total argument buffers at 67108864 elements, beyond the constraints
listed on individual ArgSpec shapes. An initial 256 MiB input plus 8 MiB output
was rejected with HTTP 422; that request and the partial initial run are retained
locally, and the rejection is in the evidence record. The corrected largest case
uses 231 MiB total. Output returns are capped at 4194304 elements. Memory-workload
PTX comes from our compiler; only the no-work timing control is hand-written PTX.

## Baseline matmul models before cross-iteration pairing

Three fresh forward/reverse trials compare the compiler with itself, yielding six
request medians per workload. Both compiler hashes are identical. Full numerical
outputs pass in both first-trial groups; subsequent returned prefixes pass. The
comparison ratios describe variation, not an optimization. In particular, the
256×512×384 group medians differ by about 2% despite identical source.

For dense M×K times K×N, conventional work is `2*M*N*K` f32 operations, with a
minimum input/output footprint of `4*(M*K+K*N+M*N)` bytes. All three footprints fit
published L2; warm repeated timings therefore must not be scored against a cold
DRAM roofline. A conditional cold-transfer time is retained in the JSON separately.

The ideal arithmetic column uses the nominal f32 peak, scaled by
`min(grid_ctas,170)/170`: a CTA occupies one SM and the prescribed grids have fewer
than 170 CTAs. It ignores additional warp, issue, address, synchronization and
shared-memory constraints. Unknown actual clocks prevent treating this as a
rigorous hardware lower bound or a measured utilization figure.

| M×K×N | CTAs | Footprint MiB | Measured µs | Grid-limited ideal f32 µs | Measured / ideal |
|---|---:|---:|---:|---:|---:|
| 512×512×512 | 128 | 3.000 | 25.072 | 3.402 | 7.37 |
| 256×512×384 | 96 | 1.625 | 19.024 | 1.701 | 11.18 |
| 384×256×320 | 80 | 1.156 | 14.912 | 1.276 | 11.69 |

The remaining gap merits inspection of instruction/address work, shared-memory
traffic, dependencies and available warp parallelism. It does not identify a
single bottleneck. Launch geometry remains prescribed. No Tensor Core or reduced
precision peak is used, and no clock-locked or counter-backed roofline is claimed.

## Accepted cross-iteration pairing

Compiler 40a1de84 pairs aligned shared reads across two bounded-unroll iterations.
Final seven alternating paired trials on nineteen workloads, after hosted safety
requests finished, give the results below. Full first-trial outputs and later
prefixes pass; clocks remain unlocked. The unchanged launch geometry means the
conditional nominal arithmetic columns above still apply. These gains do not
identify a counter-proven bottleneck or claim universal speedups.

| M×K×N | Before µs | After µs | Speedup |
|---|---:|---:|---:|
| 512×512×512 | 25.088 | 16.928 | 1.482× |
| 256×512×384 | 19.040 | 12.992 | 1.466× |
| 384×256×320 | 14.912 | 10.752 | 1.387× |

The other sixteen workloads differ by at most 0.032 µs. Safety and timing evidence:
[cross-iteration discovery](validation/2026-10-24-cross-iteration-discovery.json),
[accepted source](validation/2026-10-24-production-iteration13.json). Baseline
launch/bandwidth calibration above uses 7fbe5ee4, not the newer compiler.

## Reproduction

```sh
mkdir -p artifacts
git show b2a1b19:compile.py > artifacts/compiler-abi-validation.py
python tools/calibrate_hosted.py --compiler artifacts/compiler-abi-validation.py \
  --out artifacts/target-calibration --trials 5
```

The tool uses committed fixtures and the pinned compiler runtime, without Torch
or local CUDA. It contacts the hosted compile/run API and can transfer several
MiB of first-trial outputs. The separate baseline matmul self-comparison used compiler 7fbe5ee4 for both
arguments (use the historical source above to reproduce it):

```sh
python tools/benchmark_hosted.py challenge/problems/m01_matmul_512 \
  artifacts/shared-matmul/m_shared_256_512_384 \
  artifacts/shared-matmul/m_shared_384_256_320 \
  --baseline artifacts/compiler-abi-validation.py \
  --candidate artifacts/compiler-abi-validation.py --trials 3 \
  --timing-return-limit 4096 --report artifacts/calibration-current-matmul.json
```

The independent matmul folders are ignored development assets; generate them via
the shared-matmul generator documented in TESTING.md before reproducing that
command. The calibration tool itself needs only committed fixtures.
