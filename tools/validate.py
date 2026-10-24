"""Run the official end-to-end checker and retain machine-readable evidence."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--problems', default='challenge/problems')
    parser.add_argument('--report', default='artifacts/public-results.json')
    parser.add_argument('--compiler', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    rows = []
    compiler = (args.compiler or root / 'compile.py').resolve()
    fingerprint = hashlib.sha256(compiler.read_bytes()).hexdigest()
    for folder in sorted(Path(args.problems).iterdir()):
        if not (folder / 'meta.json').exists():
            continue
        start = time.monotonic()
        result = subprocess.run(
            [sys.executable, str(root / 'challenge/submit.py'), str(folder.resolve()),
             '--compiler', str(compiler)],
            capture_output=True, text=True, timeout=360,
        )
        row = {'problem': folder.name, 'passed': result.returncode == 0,
               'elapsed_s': round(time.monotonic() - start, 3),
               'stdout': result.stdout, 'stderr': result.stderr}
        rows.append(row)
        print(f"{'PASS' if row['passed'] else 'FAIL'} {folder.name}", flush=True)
        if not row['passed']:
            print(result.stdout + result.stderr, flush=True)
        report = Path(args.report)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps({'compiler_sha256': fingerprint, 'results': rows}, indent=2) + '\n')
    passed = sum(r['passed'] for r in rows)
    print(f'{passed}/{len(rows)} passed')
    return 0 if rows and passed == len(rows) else 1


if __name__ == '__main__':
    sys.exit(main())
