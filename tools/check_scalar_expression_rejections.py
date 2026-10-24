"""Check clean diagnostics for unsupported scalar-expression/index contexts."""
import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

from emmy.compiler.graph import Graph
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.stmt import Let, Load, Write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler',type=Path,default=Path('compile.py'))
    parser.add_argument('--template',type=Path,default=Path('artifacts/float-expressions/float_arithmetic_s0'))
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    source=next(args.template.glob('*.ir.json'))
    rows=[]
    for case in ('floating_literal_index','floating_value_index','floating_expr_division'):
        graph=Graph.from_dict(json.loads(source.read_text()));changed=False
        def walk(body):
            nonlocal changed
            result=[]
            for stmt in body:
                if not changed and case=='floating_literal_index' and isinstance(stmt,Load):
                    stmt=replace(stmt,index=(Literal(.5),));changed=True
                elif not changed and case=='floating_value_index' and isinstance(stmt,Write):
                    stmt=replace(stmt,index=(Var('value'),));changed=True
                elif not changed and case=='floating_expr_division' and isinstance(stmt,Let) and stmt.dtype is not None and stmt.dtype.name=='f32':
                    stmt=replace(stmt,value=BinaryExpr('/',Literal(3.),Literal(2.)));changed=True
                for field in ('body','else_body'):
                    if hasattr(stmt,field):stmt=replace(stmt,**{field:walk(getattr(stmt,field))})
                result.append(stmt)
            return tuple(result)
        for node in graph.nodes.values():
            if type(node.op).__name__=='KernelOp':node.op=replace(node.op,body=walk(node.op.body))
        assert changed,case
        raw=json.dumps(graph.to_dict(),default=str);Graph.from_dict(json.loads(raw))
        with tempfile.TemporaryDirectory(prefix='emmy-scalar-rejection-') as temporary:
            folder=Path(temporary);ir=folder/source.name;ir.write_text(raw)
            (folder/'meta.json').write_bytes((args.template/'meta.json').read_bytes())
            run=subprocess.run([sys.executable,str(args.compiler.resolve()),str(ir)],capture_output=True,text=True)
        passed=run.returncode!=0 and not run.stdout and 'unsupported' in run.stderr.lower() and 'Traceback' not in run.stderr
        rows.append({'case':case,'ir_sha256':hashlib.sha256(raw.encode()).hexdigest(),'passed':passed,'exit_code':run.returncode,'stdout_bytes':len(run.stdout),'stderr':run.stderr})
        print('PASS' if passed else 'FAIL',case)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps({'compiler_sha256':hashlib.sha256(args.compiler.read_bytes()).hexdigest(),'results':rows},indent=2)+'\n')
    return 0 if all(r['passed'] for r in rows) else 1


if __name__=='__main__':
    raise SystemExit(main())
