"""Run each local GPU correctness case in its own process/CUDA context.

An illegal GPU access poisons its context. Isolation prevents subsequent cases
from being mislabeled as independent compiler failures after the first fault.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('problems', nargs='+', type=Path)
    parser.add_argument('--compiler', type=Path, default=Path('compile.py'))
    parser.add_argument('--report', type=Path, default=Path('artifacts/isolated-results.json'))
    parser.add_argument('--timeout', type=float, default=90)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('timeout must be positive')
    root = Path(__file__).resolve().parents[1]
    report = {'compiler_sha256': hashlib.sha256(args.compiler.read_bytes()).hexdigest(),
              'isolation': 'one subprocess and CUDA context per case', 'results': []}
    with tempfile.TemporaryDirectory(prefix='emmy-isolated-') as temporary:
        for index, folder in enumerate(args.problems):
            detail = Path(temporary) / f'{index}.json'
            command = [sys.executable, str(root / 'tools/local_runner.py'), str(folder),
                       '--compiler', str(args.compiler.resolve()), '--report', str(detail)]
            start = time.monotonic()
            row = {'problem': folder.name, 'path': str(folder),
                   'meta_sha256': hashlib.sha256((folder / 'meta.json').read_bytes()).hexdigest(),
                   'ir_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(folder.glob('*.ir.json'))}}
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout)
                row.update(passed=result.returncode == 0, exit_code=result.returncode,
                           stdout=result.stdout, stderr=result.stderr)
                if detail.exists():
                    row['detail'] = json.loads(detail.read_text())
            except subprocess.TimeoutExpired:
                row.update(passed=False, error=f'Process exceeded {args.timeout} seconds')
            row['elapsed_s'] = round(time.monotonic() - start, 3)
            report['results'].append(row)
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + '\n')
            print(f"{'PASS' if row['passed'] else 'FAIL'} {folder.name}", flush=True)
    passed = sum(row['passed'] for row in report['results'])
    print(f'{passed}/{len(report["results"])} passed')
    return 0 if passed == len(report['results']) else 1


if __name__ == '__main__':
    sys.exit(main())
