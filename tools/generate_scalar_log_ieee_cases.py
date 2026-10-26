"""Classify log/exp2 results on signed subnormals and in-kernel infinities/NaNs.

All transported inputs and outputs are finite. Exceptional operands are created
inside PTX by dividing signed finite values by zero; reference classes come from
independent NumPy float64 functions. No IEEE payload preservation is asserted.
"""
import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from emmy.compiler.dtype import F32
from emmy.compiler.graph import Graph
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import Literal
from emmy.compiler.ir.stmt import Assign, Let


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();rows=[]
    source_manifest=args.source/'manifest.json'
    for case in json.loads(source_manifest.read_text())['cases']:
        if case['mode']!='special':continue
        folder=Path(case['folder']);ir=next(folder.glob('*.ir.json'))
        for mode in ('negative_subnormal','nonfinite'):
            graph=Graph.from_dict(json.loads(ir.read_text()));changed=False
            if mode=='nonfinite':
                def walk(body):
                    nonlocal changed
                    out=[]
                    for stmt in body:
                        if not changed and isinstance(stmt,Assign) and stmt.op.name==case['op']:
                            out.extend((Let('ieee_zero',Literal(0.),F32),
                                Assign('ieee_input',ElementwiseImpl('divide'),(stmt.args[0],'ieee_zero'),F32)))
                            stmt=replace(stmt,args=('ieee_input',));changed=True
                        for field in ('body','else_body'):
                            if hasattr(stmt,field):stmt=replace(stmt,**{field:walk(getattr(stmt,field))})
                        out.append(stmt)
                    return tuple(out)
                node=next(n for n in graph.nodes.values() if type(n.op).__name__=='KernelOp')
                node.op=replace(node.op,body=walk(node.op.body));assert changed
            meta=json.loads((folder/'meta.json').read_text())
            count=meta['kernels'][0]['args'][0]['shape'][0]
            tiny=np.nextafter(np.float32(0),np.float32(1));normal=np.finfo(np.float32).tiny;large=np.finfo(np.float32).max
            x=np.resize(np.array([-large,-normal,-tiny,-0.,0.,tiny,normal,large],dtype=np.float32),count)
            with np.errstate(all='ignore'):
                operand=x.astype(np.float64)
                if mode=='nonfinite':operand=operand/np.float64(0)
                result=getattr(np,case['op'])(operand).astype(np.float32)
            expected=np.where(np.isnan(result),2,np.where(np.isneginf(result),-1,np.where(np.isposinf(result),1,0))).astype(np.float32)
            label=folder.name.replace('_special_',f'_{mode}_');target=args.out/label;target.mkdir(parents=True,exist_ok=True)
            raw=json.dumps(graph.to_dict(),indent=2,default=str)+'\n';Graph.from_dict(json.loads(raw));(target/ir.name).write_text(raw)
            meta.update(id=label,rtol=0,atol=0)
            for arg in meta['kernels'][0]['args']:
                if not arg.get('output'):arg['data']=x.tolist();arg.pop('seed',None)
            (target/'meta.json').write_text(json.dumps(meta,indent=2)+'\n');np.savez(target/'expected.npz',sum_1=expected)
            rows.append({'folder':str(target),'op':case['op'],'mode':mode,'seed':case['seed'],'renamed':case['renamed'],
                'reference':'Exact finite result-class labels from independent NumPy float64; in-kernel division creates exceptional operands',
                'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(target.iterdir()) if p.is_file()}})
    assert len(rows)==16
    (args.out/'manifest.json').write_text(json.dumps({'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'source_manifest_sha256':hashlib.sha256(source_manifest.read_bytes()).hexdigest(),'cases':rows},indent=2)+'\n')
    print(f'Generated {len(rows)} IEEE-class cases')


if __name__=='__main__':main()
