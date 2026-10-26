# Testing

## CPU suite

```sh
python tests/run.py
```

The suite runs seven checks for CLI isolation, deterministic PTX, ABI and launch
validation, static shapes, unsupported scalar operations, boolean conversions,
and register ceilings. It needs the pinned runtime, but no GPU, CUDA toolkit,
or PyTorch. To enforce the absence of PyTorch:

```sh
python tests/run.py --forbid-torch
```

Reports go to ignored `artifacts/test-results/`. Use `--report-dir` to change the
location. The CI workflow runs the same command on Python 3.12, 3.13, and 3.14.

`tests/fixtures/manifest.json` records fixture hashes and PTX snapshots. The 57
fixtures cover previous numerical and code-generation regressions. CPU snapshots
do not execute PTX. Refreshing them requires reviewing the emitted code and
checking affected outputs on a GPU.

## Official GPU runner

Clone the challenge tooling once:

```sh
git clone https://github.com/cloudrift-ai/emmy-hackathon-2026 challenge
python tools/validate.py
```

The checker compiles every public problem, assembles PTX remotely, and compares
GPU outputs against each problem's reference and tolerance. Reports are written
to `artifacts/`. To check generated cases, pass `--problems path/to/cases`.

## Additional workloads

With PyTorch installed for IR generation:

```sh
python challenge/tools/generate.py tests/generators/expressions_hidden_proxy.py --emmy .venv/bin/emmy --out artifacts/hidden-proxy
python tools/validate.py --problems artifacts/hidden-proxy
```

Other expression files cover edge shapes, layouts, precision, and large-block
schedules. The large-block suite uses recognized `t256` and `t512` presets;
Emmy 0.3.11 silently falls back to 256 for unrecognized work-width strings.
Explicit typed IR probes exercise additional launch widths:

```sh
python tests/generators/generate_launch_pressure_cases.py --out artifacts/launch-pressure
python tools/validate.py --problems artifacts/launch-pressure
```

For staged transformer checks, generate the stage fixtures and restore weights,
then compare PTX stage outputs against CPU PyTorch:

```sh
python tests/generators/generate_models.py --specs tests/integration/transformer_model.py --out artifacts/transformer-stages
python tests/integration/validate_transformer.py
```

Stages exchange data through the host. These checks establish correctness, not
end-to-end model throughput.

## Local GPU tools

`tools/local_runner.py` executes explicit problem directories through the CUDA
driver. It retargets emitted PTX for the local GPU; the compiler's default stays
`sm_89`. `tools/validate_isolated.py` runs each case in a separate process so a
GPU fault cannot contaminate later cases.

Use `tools/sanitize.py` with a Compute Sanitizer executable for memory and
synchronization checks. `tools/profile_resources.py` inspects local resources or
hosted register counts. Each tool documents its arguments through `--help`.

## Benchmarks and packaging

`tools/benchmark_hosted.py` compares two compiler files with alternating trial
order and output verification. `tools/benchmark.py` measures local CUDA events.
Keep raw samples in `artifacts/`; compare on the target GPU before accepting a
performance change. The hosted API does not provide clock locking.

`tools/package_submission.py` archives committed source and runs three hosted
smoke tests from an extracted copy. It requires a clean tracked working tree and
an installed runtime. Packaging does not publish the repository or submit an entry.
