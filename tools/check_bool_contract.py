"""Check bool SSA/conversion controls while rejecting unsupported storage/types.

All probes use isolated CPU CLI processes. Invalid declarations are never
launched on a GPU; valid conversion controls have separate numerical GPU tests.
"""
import argparse
import copy
import hashlib
import json
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

from emmy.compiler.dim import Dim
from emmy.compiler.dtype import BOOL,F32,I32
from emmy.compiler.graph import Graph
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import Literal,Var
from emmy.compiler.ir.kernel.ir import Smem
from emmy.compiler.ir.stmt import Accum,Assign,Init,Let,Load,Loop,Write


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler',type=Path,default=root/'compile.py')
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--observe',action='store_true')
    args=parser.parse_args();compiler=args.compiler.resolve()
    template=root/'validation/cpu-fixtures/2026-10-24-f32-float-arithmetic'
    ir=next(template.glob('*.ir.json'));original=json.loads(ir.read_text());metadata=json.loads((template/'meta.json').read_text())
    rows=[]
    cases=['input_bool','output_bool','argument_bool','shared_bool','load_bool','init_bool','accum_bool','assign_i32',
           'bool_assign_write','bool_let_write']
    for case in cases:
        graph=Graph.from_dict(copy.deepcopy(original));meta=copy.deepcopy(metadata);changed=False
        if case in ('input_bool','output_bool'):
            node=next(n for n in graph.nodes.values() if type(n.op).__name__==('InputOp' if case=='input_bool' else 'KernelOp'))
            node.outputs=tuple(replace(t,dtype=BOOL) for t in node.outputs);changed=True
        elif case=='argument_bool':meta['kernels'][0]['args'][0]['dtype']='bool';changed=True
        else:
            def walk(body):
                nonlocal changed
                out=[]
                for stmt in body:
                    if not changed and case=='load_bool' and isinstance(stmt,Load):
                        stmt=replace(stmt,dtype=BOOL);changed=True
                    elif not changed and isinstance(stmt,Write):
                        value=stmt.values[0]
                        if case=='shared_bool':out.append(Smem('bool_scratch',(4,),dtype='bool',align=16))
                        elif case=='init_bool':out.append(Init('bool_init',0.0,dtype=BOOL))
                        elif case=='assign_i32':out.append(Assign('int_assign',ElementwiseImpl('copy'),(value,),I32))
                        elif case=='accum_bool':
                            out.append(Let('contract_zero',Literal(0.),F32))
                            out.append(Loop(Axis('bool_axis',Dim(2)),(
                                Assign('bool_value',ElementwiseImpl('greater'),(value,'contract_zero'),BOOL),
                                Accum('bool_acc','bool_value',ElementwiseImpl('logical_or'),BOOL,axes=('bool_axis',)))))
                        else:
                            if case=='bool_assign_write':
                                out.extend((Let('contract_zero',Literal(0.),F32),
                                    Assign('bool_value',ElementwiseImpl('greater'),(value,'contract_zero'),BOOL)))
                            else:out.append(Let('bool_value',Var(value),BOOL))
                            stmt=replace(stmt,values=('bool_value',),value_dtype=BOOL)
                        changed=True
                    for field in ('body','else_body'):
                        if hasattr(stmt,field):stmt=replace(stmt,**{field:walk(getattr(stmt,field))})
                    out.append(stmt)
                return tuple(out)
            node=next(n for n in graph.nodes.values() if type(n.op).__name__=='KernelOp');node.op=replace(node.op,body=walk(node.op.body))
        assert changed,case
        raw=json.dumps(graph.to_dict(),indent=2,default=str)+'\n';Graph.from_dict(json.loads(raw))
        with tempfile.TemporaryDirectory(prefix='emmy-bool-contract-') as tmp:
            folder=Path(tmp);path=folder/ir.name;path.write_text(raw);(folder/'meta.json').write_text(json.dumps(meta))
            result=subprocess.run([sys.executable,'-I',str(compiler),str(path)],cwd=folder,capture_output=True,text=True,timeout=30)
        valid=case.startswith('bool_')
        passed=(result.returncode==0 and result.stdout.startswith('.version') and not result.stderr) if valid else (
            result.returncode==1 and not result.stdout and 'Traceback' not in result.stderr and
            ('unsupported' in result.stderr.lower() or 'requires f32 allocation dtype' in result.stderr))
        rows.append({'case':case,'expected_valid':valid,'passed':passed,'returncode':result.returncode,
                     'stderr':result.stderr,'emitted_ptx':result.stdout.startswith('.version'),
                     'ir_sha256':hashlib.sha256(raw.encode()).hexdigest(),'meta_sha256':hashlib.sha256(json.dumps(meta).encode()).hexdigest()})
        print(('PASS' if passed else 'GAP'),case,flush=True)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps({'compiler_sha256':hashlib.sha256(compiler.read_bytes()).hexdigest(),
        'python':sys.version,'scope':'CPU CLI bool SSA/conversion versus unsupported storage/type boundaries; never executes invalid programs','results':rows},indent=2)+'\n')
    if not args.observe and not all(r['passed'] for r in rows):raise SystemExit(1)


if __name__=='__main__':main()
