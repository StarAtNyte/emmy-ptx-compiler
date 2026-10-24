"""Probe static shape diagnostics through the compiler CLI; never launch invalid IR."""
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
from emmy.compiler.graph import Graph
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Literal
from emmy.compiler.ir.kernel.ir import Smem
from emmy.compiler.ir.stmt import Cond, Loop, StridedLoop


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler',type=Path,default=root/'compile.py')
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--observe',action='store_true')
    args=parser.parse_args()
    source=root/'validation/cpu-fixtures/2026-10-24-abi-reordered'
    ir=next(source.glob('*.ir.json'));original=json.loads(ir.read_text())
    meta=(source/'meta.json').read_text();cases=[]
    for field in ('tensor','tile','loop','strided'):
        for label,dim in [('negative',Dim(-1)),('symbolic',Dim('runtime_extent')),('bool',Dim(True))]:
            cases.append((f'{field}_{label}',field,dim,False))
        if field!='tensor':
            cases.append((f'{field}_fractional',field,Dim(Literal(1.5)),False))
            cases.append((f'{field}_composite',field,Dim('s')*Dim(2),False))
    for label,value in [('zero',(0,)),('negative',(-1,)),('fractional',(1.5,)),
                        ('symbolic',('runtime',)),('bool',(True,))]:
        cases.append(('shared_'+label,'shared',value,False))
    for label,value in [('negative',-1),('nonpower',6),('fractional',8.0),('bool',True)]:
        cases.append(('align_'+label,'align',value,False))
    cases.extend([('scalar_tensor','tensor',None,True),('zero_tensor_view','tensor',Dim(0),True),
                  ('natural_alignment','align',0,True),('scalar_shared','shared',(),True)])
    report={'compiler_sha256':hashlib.sha256(args.compiler.read_bytes()).hexdigest(),
            'scope':'CPU typed-IR shape diagnostics and valid view controls, no GPU execution', 'results':[]}
    with tempfile.TemporaryDirectory(prefix='emmy-static-shapes-') as temporary:
        folder=Path(temporary);(folder/'meta.json').write_text(meta);path=folder/ir.name
        for name,field,value,valid in cases:
            graph=Graph.from_dict(copy.deepcopy(original));node=next(n for n in graph.nodes.values() if type(n.op).__name__=='KernelOp')
            tile=node.op.body[0]
            if field=='tensor':
                inp=next(n for n in graph.nodes.values() if type(n.op).__name__=='InputOp')
                inp.outputs=tuple(replace(t,shape=() if value is None else (value,)) for t in inp.outputs)
            elif field=='tile':tile=replace(tile,axes=(replace(tile.axes[0],extent=value),))
            elif field=='loop':tile=replace(tile,body=(Loop(Axis('probe',value),tile.body),))
            elif field=='strided':tile=replace(tile,body=(StridedLoop(Axis('probe',value),Literal(0),Literal(1),tile.body),))
            else:
                smem=Smem('unused',(4,) if field=='align' else value,align=value if field=='align' else 0)
                # Else branches must receive the same validation as the main body.
                tile=replace(tile,body=(*tile.body,Cond(Literal(0),(),(smem,))))
            node.op=replace(node.op,body=(tile,))
            payload=json.dumps(graph.to_dict(),default=str);Graph.from_dict(json.loads(payload));path.write_text(payload)
            result=subprocess.run([sys.executable,'-I',str(args.compiler.resolve()),str(path)],cwd=folder,
                                    capture_output=True,text=True,timeout=30)
            passed=((result.returncode==0 and result.stdout.startswith('.version') and not result.stderr)
                    if valid else (result.returncode==1 and not result.stdout and ', shape ' in result.stderr
                                   and 'Traceback' not in result.stderr))
            report['results'].append({'case':name,'expected_valid':valid,'passed':passed,'returncode':result.returncode,
                'stderr':result.stderr,'emitted_ptx':result.stdout.startswith('.version'),
                'ir_sha256':hashlib.sha256(payload.encode()).hexdigest()})
            print(('PASS' if passed else 'GAP'),name,flush=True)
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2)+'\n')
    passed=sum(r['passed'] for r in report['results']);print(f'{passed}/{len(report["results"])} expectations met')
    if not args.observe and passed!=len(report['results']):raise SystemExit(1)


if __name__=='__main__':main()
