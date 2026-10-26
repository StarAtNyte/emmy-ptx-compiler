"""Run the CPU contract suite without a GPU or CUDA toolkit."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


CHECKS = (
    'check_cpu_contract', 'check_launch_contract', 'check_abi_contract',
    'check_static_shapes', 'check_scalar_op_rejections', 'check_bool_contract',
    'check_launch_resources',
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--forbid-torch', action='store_true',
                        help='also require PyTorch to be absent')
    parser.add_argument('--report-dir', type=Path, default=Path('artifacts/test-results'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    reports = args.report_dir.resolve()
    reports.mkdir(parents=True, exist_ok=True)
    failures = []
    contracts = resources = 0
    for name in CHECKS:
        report = reports / f'{name}.json'
        command = [sys.executable, str(root / 'tests/checks' / f'{name}.py'),
                   '--report', str(report)]
        if name == 'check_cpu_contract' and args.forbid_torch:
            command.append('--forbid-torch')
        result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=180)
        print(f'{"PASS" if result.returncode == 0 else "FAIL"} {name}', flush=True)
        if result.returncode:
            failures.append(name)
            print(result.stdout + result.stderr, flush=True)
        else:
            data = json.loads(report.read_text())
            if name == 'check_launch_resources':
                resources += data['checked']
            else:
                contracts += len(data['results'])
    if failures:
        return 1
    print(f'{contracts} contract checks and {resources} resource invariants passed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
