"""Check register ceilings for every supported large-block thread count.

This checks the hardware resource invariant, including partially occupied warps,
at several IR state pressures. Numerical GPU tests remain separate.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from emmy.compiler.dtype import F32
from emmy.compiler.ir.stmt import Init


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler', type=Path, default=root / 'compile.py')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('resource_compiler', args.compiler)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    failures = []
    checked = 0
    for state_count in (0, 8, 64, 256):
        body = tuple(Init(f'state_{i}', 0.0, dtype=F32) for i in range(state_count))
        for threads in range(257, 1025):
            compiler = object.__new__(module.Compiler)
            compiler.node = SimpleNamespace(op=SimpleNamespace(body=body))
            compiler.launch = {'block': [threads, 1, 1]}
            bounds = compiler.launch_bounds()
            caps = [int(line.split()[1]) for line in bounds if line.startswith('.maxnreg ')]
            # The target allocates registers in groups of 256 per physical
            # warp. Idle lanes in the final warp consume their allocation too.
            if len(caps) == 1:
                registers_per_warp = ((caps[0] * 32 + 255) // 256) * 256
                used = registers_per_warp * len(range(0, threads, 32))
                passed = 1 <= caps[0] <= 255 and used <= 65536
            else:
                passed = False
            checked += 1
            if not passed:
                failures.append({'threads': threads, 'states': state_count, 'bounds': bounds})
    report = {'compiler_sha256': hashlib.sha256(args.compiler.read_bytes()).hexdigest(),
              'scope': 'register-allocation ceiling only; not GPU execution or shared-memory limits',
              'checked': checked, 'passed': not failures, 'failures': failures}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(f'{checked - len(failures)}/{checked} register ceilings satisfy the CTA resource limit')
    return 0 if not failures else 1


if __name__ == '__main__':
    raise SystemExit(main())
