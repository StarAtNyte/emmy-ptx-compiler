"""Check that valid but unsupported scalar ops produce clean CLI diagnostics.

These probes are never launched on a GPU. The surrounding retained fixture
supplies an existing f32 value and valid buffer/launch metadata.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

from emmy.compiler.dtype import F32
from emmy.compiler.graph import Graph
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.stmt import Assign, Write


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler',type=Path,default=root/'compile.py')
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--observe',action='store_true')
    args=parser.parse_args();compiler=args.compiler.resolve()
    template=root/'validation/cpu-fixtures/2026-10-24-f32-float-arithmetic'
    source=next(template.glob('*.ir.json'));rows=[]
    for operation in ('log1p','expm1','sin','cos'):
        graph=Graph.from_dict(json.loads(source.read_text()));changed=False
        def walk(body):
            nonlocal changed
            out=[]
            for stmt in body:
                if not changed and isinstance(stmt,Write):
                    out.append(Assign('unsupported_probe',ElementwiseImpl(operation),(stmt.values[0],),F32))
                    stmt=replace(stmt,values=('unsupported_probe',));changed=True
                for field in ('body','else_body'):
                    if hasattr(stmt,field):stmt=replace(stmt,**{field:walk(getattr(stmt,field))})
                out.append(stmt)
            return tuple(out)
        node=next(n for n in graph.nodes.values() if type(n.op).__name__=='KernelOp')
        node.op=replace(node.op,body=walk(node.op.body));assert changed
        raw=json.dumps(graph.to_dict(),indent=2,default=str)+'\n';Graph.from_dict(json.loads(raw))
        with tempfile.TemporaryDirectory(prefix='emmy-op-rejection-') as tmp:
            folder=Path(tmp);ir=folder/source.name;ir.write_text(raw)
            (folder/'meta.json').write_bytes((template/'meta.json').read_bytes())
            result=subprocess.run([sys.executable,'-I',str(compiler),str(ir)],cwd=folder,
                                  capture_output=True,text=True,timeout=30)
        passed=(result.returncode==1 and not result.stdout and 'Traceback' not in result.stderr
                and f'Unsupported scalar operation {operation}' in result.stderr)
        rows.append({'operation':operation,'ir_sha256':hashlib.sha256(raw.encode()).hexdigest(),
                     'passed':passed,'returncode':result.returncode,'stdout_bytes':len(result.stdout),'stderr':result.stderr})
        print(('PASS' if passed else 'GAP'),operation,flush=True)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps({'compiler_sha256':hashlib.sha256(compiler.read_bytes()).hexdigest(),
        'python':sys.version,'scope':'Isolated CPU CLI only; unsupported programs never launched','results':rows},indent=2)+'\n')
    if not args.observe and not all(r['passed'] for r in rows):raise SystemExit(1)


if __name__=='__main__':main()
