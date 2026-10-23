"""Archive the committed source and smoke-test it outside the working tree."""
import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--python',type=Path,default=Path(sys.executable))
    parser.add_argument('--out',type=Path,default=Path('artifacts/emmy-ptx-submission.zip'))
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    if subprocess.run(['git','diff','--quiet','HEAD'],cwd=root).returncode:
        raise SystemExit('Commit tracked source changes before packaging')
    args.out=args.out.resolve()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(['git','archive','--format=zip','--output',str(args.out),'HEAD'],cwd=root,check=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    report={'commit':commit,'compiler_sha256':hashlib.sha256((root/'compile.py').read_bytes()).hexdigest(),
            'archive_sha256':hashlib.sha256(args.out.read_bytes()).hexdigest(),'checks':[]}
    # Keep a venv's executable path: dereferencing its symlink would bypass
    # pyvenv.cfg and silently select the base interpreter's packages.
    python=str(args.python.absolute())
    with zipfile.ZipFile(args.out) as archive, tempfile.TemporaryDirectory(prefix='emmy-release-') as directory:
        names=archive.namelist()
        required={'compile.py','requirements.txt','README.md','DESIGN.md','SUBMISSION.md','WALKTHROUGH.md'}
        if not required.issubset(names):
            raise RuntimeError('Committed archive is missing submission files')
        if any(n.startswith(('.venv/','artifacts/','challenge/','upstream-emmy/','.git/')) for n in names):
            raise RuntimeError('Development state leaked into the archive')
        archive.extractall(directory)
        packaged=Path(directory)/'compile.py'
        if hashlib.sha256(packaged.read_bytes()).hexdigest()!=report['compiler_sha256']:
            raise RuntimeError('Archived compiler differs from the working tree')
        environment=subprocess.check_output([python,'-c',
            'import importlib.util,json,sys;print(json.dumps({"python":sys.version,"torch_installed":importlib.util.find_spec("torch") is not None}))'],text=True)
        report['environment']=json.loads(environment)
        for problem in ('p00_scalar_add','m01_matmul_512','a01_attention'):
            result=subprocess.run([python,str(root/'challenge/submit.py'),problem,'--compiler',str(packaged)],
                                  cwd=directory,capture_output=True,text=True,timeout=360)
            report['checks'].append({'problem':problem,'passed':result.returncode==0,
                                     'stdout':result.stdout,'stderr':result.stderr})
            print(f"{'PASS' if result.returncode==0 else 'FAIL'} packaged {problem}",flush=True)
            if result.returncode:
                raise RuntimeError(result.stdout+result.stderr)
    report_path=args.out.with_suffix('.json')
    report_path.write_text(json.dumps(report,indent=2)+'\n')
    print(f'Archive: {args.out}',flush=True)


if __name__=='__main__':
    main()
