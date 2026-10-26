"""Independent exact probes for bool SSA, f32 mask conventions and selection."""
import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import BOOL,F32
from emmy.compiler.graph import Graph
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import BinaryExpr,Literal,Var
from emmy.compiler.ir.kernel.ir import Tile
from emmy.compiler.ir.stmt import Assign,Let,Load,Write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=0)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    template=root/'challenge/problems/r01_array_sum';ir=template/'k_sum_1_reduce.ir.json'
    operations=('equal','not_equal','less','less_equal','greater','greater_equal')
    functions=dict(zip(operations,(np.equal,np.not_equal,np.less,np.less_equal,np.greater,np.greater_equal)))
    cases=[(op,mode) for op in operations for mode in ('compare','nan_compare')]
    cases += [(op,mode) for op in ('and','or','not') for mode in ('bool_logic','legacy_logic','logical_alias')]
    cases += [(mode,mode) for mode in ('where_float','where_bool','where_nan_mask','where_nan_branch','where_signed_zero',
                                      'let_truth','let_nan','let_literal','let_literal_zero','bool_index','legacy_expression')]
    count=257;rows=[]
    rng=np.random.default_rng(args.seed)
    x=(rng.integers(-16,17,count).astype(np.float32)/np.float32(4));y=(rng.integers(-16,17,count).astype(np.float32)/np.float32(4))
    x[:8]=np.array([-2,-0.,0.,.25,.5,1,2,-.25],dtype=np.float32)
    y[:8]=np.array([2,0.,-0.,.25,0.,1,-2,.25],dtype=np.float32)
    data=np.concatenate((x,y))
    for operation,mode in cases:
        for renamed in (False,True):
            prefix=f'm{args.seed}_' if renamed else ''
            n=lambda s:prefix+s;v=lambda s:Var(n(s));f=lambda a:Literal(float(a));k=lambda a:Literal(int(a),'int')
            b=lambda op,a,c:BinaryExpr(op,a,c)
            body=[Load(n('x'),'x',(v('thread'),),dtype=F32),Load(n('y'),'x',(b('+',v('thread'),k(count)),),dtype=F32)]
            def assign(name,op,inputs,dtype=BOOL):body.append(Assign(n(name),ElementwiseImpl(op),tuple(n(i) for i in inputs),dtype))
            def nan():
                body.append(Let(n('zero'),f(0),F32));assign('nan','divide',('zero','zero'),F32)
            output=n('result');write_dtype=F32
            if mode in ('compare','nan_compare'):
                source='x';left=x
                if mode=='nan_compare':nan();source='nan';left=np.full_like(x,np.nan)
                assign('result',operation,(source,'y'));expected=functions[operation](left,y).astype(np.float32)
                # Bool source values must convert numerically into f32 storage.
                write_dtype=BOOL
            elif mode in ('bool_logic','legacy_logic','logical_alias'):
                if mode=='legacy_logic':a=x!=0;c=y!=0;inputs=('x','y');dtype=F32
                else:
                    body.append(Let(n('zero'),f(0),F32));assign('a','greater',('x','zero'));assign('b','greater',('y','zero'))
                    a=x>0;c=y>0;inputs=('a','b');dtype=BOOL
                op=('logical_' if mode=='logical_alias' else 'bitwise_')+operation
                assign('result',op,inputs[:1] if operation=='not' else inputs,dtype)
                expected=(np.logical_not(a) if operation=='not' else np.logical_and(a,c) if operation=='and' else np.logical_or(a,c)).astype(np.float32)
            elif mode in ('where_float','where_bool','where_nan_mask','where_nan_branch','where_signed_zero'):
                condition='x';mask=x!=0
                if mode in ('where_bool','where_nan_branch','where_signed_zero'):
                    body.append(Let(n('zero'),f(0),F32));assign('mask','greater',('x','zero'));condition='mask';mask=x>0
                if mode=='where_nan_mask':nan();condition='nan';mask=np.ones(count,dtype=bool)
                if mode=='where_nan_branch':
                    nan();assign('selected','where',(condition,'nan','y'),F32);assign('result','not_equal',('selected','selected'))
                    expected=mask.astype(np.float32)
                elif mode=='where_signed_zero':
                    body.extend((Let(n('negative_zero'),f(-0.),F32),Let(n('positive_zero'),f(0.),F32)))
                    assign('selected','where',(condition,'negative_zero','positive_zero'),F32)
                    assign('inverse','reciprocal',('selected',),F32);assign('result','less',('inverse','zero'));expected=mask.astype(np.float32)
                else:
                    assign('negative_y','negative',('y',),F32);assign('result','where',(condition,'y','negative_y'),F32)
                    expected=np.where(mask,y,-y)
            elif mode in ('let_truth','let_nan','let_literal','let_literal_zero'):
                source=v('x');expected=(x!=0).astype(np.float32)
                if mode=='let_nan':nan();source=v('nan');expected=np.ones(count,dtype=np.float32)
                elif mode=='let_literal':source=f(.25);expected=np.ones(count,dtype=np.float32)
                elif mode=='let_literal_zero':source=f(-0.);expected=np.zeros(count,dtype=np.float32)
                body.append(Let(n('result'),source,BOOL));write_dtype=BOOL
            elif mode=='bool_index':
                body.append(Let(n('zero'),f(0),F32));assign('mask','greater',('x','zero'))
                body.append(Load(n('result'),'x',(v('mask'),),dtype=F32));expected=x[(x>0).astype(np.int64)]
            else:
                body.append(Let(n('zero'),f(0),F32));assign('mask','greater',('x','zero'),F32)
                body.append(Let(n('result'),b('+',v('mask'),k(16777217)),F32))
                expected=(np.float32(16777217)+(x>0).astype(np.float32)).astype(np.float32)
            body.append(Write('sum_1',(v('thread'),),value=output,value_dtype=write_dtype))
            graph=Graph.from_dict(json.loads(ir.read_text()))
            for node in graph.nodes.values():
                size=data.size if type(node.op).__name__=='InputOp' else count
                node.outputs=tuple(replace(t,shape=(Dim(int(size)),)) for t in node.outputs)
                if type(node.op).__name__=='KernelOp':node.op=replace(node.op,body=(Tile((Axis(n('thread'),Dim(count)),),tuple(body)),))
            label=f'{mode}_{operation}_s{args.seed}'+('_renamed' if renamed else '')
            folder=args.out/label;folder.mkdir(parents=True,exist_ok=True)
            raw=json.dumps(graph.to_dict(),indent=2,default=str)+'\n';Graph.from_dict(json.loads(raw));(folder/ir.name).write_text(raw)
            meta=json.loads((template/'meta.json').read_text());meta.update(id=label,rtol=0,atol=0)
            launch=meta['kernels'][0];launch.update(grid=[3,1,1],block=[128,1,1],smem_bytes=0)
            for arg in launch['args']:
                arg['shape']=[count if arg.get('output') else int(data.size)]
                if not arg.get('output'):arg.pop('seed',None);arg['data']=data.tolist()
            (folder/'meta.json').write_text(json.dumps(meta,indent=2)+'\n');np.savez(folder/'expected.npz',sum_1=expected.astype(np.float32))
            rows.append({'folder':str(folder),'operation':operation,'mode':mode,'seed':args.seed,'renamed':renamed,
                'reference':'Independent NumPy comparisons, logical operations, selection and explicit f32 rounding; exact finite outputs',
                'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.iterdir()) if p.is_file()}})
    (args.out/'manifest.json').write_text(json.dumps({'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'cases':rows},indent=2)+'\n')
    print(f'Generated {len(rows)} scalar mask cases')


if __name__=='__main__':main()
