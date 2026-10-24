"""CPU-only CLI contract checks on committed numerical counterexamples.

PTX snapshots guard previously GPU-validated code generation; they do not execute
PTX or replace numerical GPU/sanitizer/performance gates.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

from emmy.compiler.dtype import F16
from emmy.compiler.graph import Graph
from emmy.compiler.ir.expr import Literal
from emmy.compiler.ir.stmt import Write


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=root/'validation/cpu-contract.json')
    parser.add_argument('--report', type=Path, default=Path('artifacts/cpu-contract.json'))
    parser.add_argument('--forbid-torch', action='store_true')
    args = parser.parse_args()
    torch_present = importlib.util.find_spec('torch') is not None
    if args.forbid_torch and torch_present:
        parser.error('Torch must be absent in the CPU contract environment')
    manifest = json.loads(args.manifest.read_text())
    report = {'compiler_sha256':digest(root/'compile.py'), 'manifest_sha256':digest(args.manifest),
              'python':sys.version, 'torch_present':torch_present,
              'packages':{name:importlib.metadata.version(name) for name in ('emmy-ml','numpy')},
              'scope':'CPU CLI/code-generation contract, not GPU execution', 'results':[]}
    env = dict(os.environ)
    env.pop('PYTHONPATH',None)
    env.pop('PYTHONHOME',None)
    with tempfile.TemporaryDirectory(prefix='emmy-cpu-contract-') as temporary:
        sandbox = Path(temporary)
        compiler = sandbox/'compile.py'
        shutil.copy2(root/'compile.py',compiler)
        cwd = sandbox/'unrelated-working-directory'
        cwd.mkdir()
        def run(path):
            return subprocess.run([sys.executable,'-I',str(compiler),str(path)],cwd=cwd,
                                  env=env,capture_output=True,text=True,timeout=30)
        for item in manifest['fixtures']:
            folder = root/item['folder']
            for name,expected in item['files'].items():
                if digest(folder/name) != expected:
                    raise SystemExit(f'Fixture hash mismatch: {folder/name}')
            copy = sandbox/folder.name
            shutil.copytree(folder,copy)
            ir = copy/item['ir']
            outputs = [run(ir),run(ir)]
            actual = hashlib.sha256(outputs[0].stdout.encode()).hexdigest()
            passed = (all(r.returncode==0 and not r.stderr for r in outputs)
                      and outputs[0].stdout==outputs[1].stdout and actual==item['ptx_sha256'])
            report['results'].append({'case':folder.name,'passed':passed,'ptx_sha256':actual,
                'stderr':[r.stderr for r in outputs]})
            print(('PASS' if passed else 'FAIL'),folder.name,flush=True)
        # Real CLI negative probes on the first committed typed fixture.
        template = root/manifest['fixtures'][0]['folder']
        ir_name = manifest['fixtures'][0]['ir']
        for kind in ('tensor_f16','write_f16','write_swizzle','floating_index'):
            graph = Graph.from_dict(json.loads((template/ir_name).read_text()))
            changed = False
            def mutate(body):
                nonlocal changed
                result = []
                for stmt in body:
                    if not changed and isinstance(stmt,Write):
                        changes = ({'value_dtype':F16} if kind=='write_f16'
                                   else {'swizzle':'B32'} if kind=='write_swizzle'
                                   else {'index':(Literal(.5),)})
                        stmt = replace(stmt,**changes)
                        changed = True
                    for field in ('body','else_body'):
                        if hasattr(stmt,field):
                            stmt = replace(stmt,**{field:mutate(getattr(stmt,field))})
                    result.append(stmt)
                return tuple(result)
            if kind=='tensor_f16':
                node = next(iter(graph.nodes.values()))
                node.outputs = tuple(replace(t,dtype=F16) for t in node.outputs)
                changed = True
            else:
                for node in graph.nodes.values():
                    if type(node.op).__name__=='KernelOp':
                        node.op = replace(node.op,body=mutate(node.op.body))
            assert changed,kind
            folder = sandbox/kind
            folder.mkdir()
            (folder/'meta.json').write_bytes((template/'meta.json').read_bytes())
            path = folder/ir_name
            path.write_text(json.dumps(graph.to_dict(),default=str))
            result = run(path)
            passed = (result.returncode!=0 and not result.stdout
                      and 'unsupported' in result.stderr.lower() and 'Traceback' not in result.stderr)
            report['results'].append({'case':kind,'passed':passed,'stderr':result.stderr})
            print(('PASS' if passed else 'FAIL'),kind,flush=True)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    return 0 if report['results'] and all(r['passed'] for r in report['results']) else 1


if __name__=='__main__':
    raise SystemExit(main())
