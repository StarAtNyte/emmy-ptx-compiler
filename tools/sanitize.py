"""Run NVIDIA Compute Sanitizer tools against explicit correctness cases."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('problems',nargs='+',type=Path)
    parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--tools',nargs='+',default=['memcheck','racecheck','synccheck','initcheck'])
    parser.add_argument('--out',type=Path,default=Path('artifacts/sanitizers'))
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[1]
    report={'compiler_sha256':hashlib.sha256((root/'compile.py').read_bytes()).hexdigest(),
            'problems':[p.name for p in args.problems], 'results':[]}
    for tool in args.tools:
        log=args.out/f'{tool}.log'
        command=[str(args.binary),'--tool',tool,'--error-exitcode','9','--log-file',str(log)]
        if tool=='memcheck':
            command.extend(['--leak-check','full'])
        command.extend([sys.executable,str(root/'tools/local_runner.py'),*(str(p) for p in args.problems)])
        result=subprocess.run(command,capture_output=True,text=True,timeout=600)
        (args.out/f'{tool}-checks.log').write_text(result.stdout+result.stderr)
        contents=log.read_text() if log.exists() else ''
        summaries=[line for line in contents.splitlines() if 'SUMMARY:' in line]
        row={'tool':tool,'exit_code':result.returncode,'passed':result.returncode==0,
             'summaries':summaries}
        report['results'].append(row)
        (args.out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
        print(f"{'PASS' if row['passed'] else 'FAIL'} {tool}: {summaries}",flush=True)
        if not row['passed']:
            print(result.stdout+result.stderr,flush=True)
    return 0 if all(r['passed'] for r in report['results']) else 1


if __name__=='__main__':
    sys.exit(main())
