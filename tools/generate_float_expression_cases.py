"""Scalar f32 expression/condition/cast probes with independent NumPy oracles."""
import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F32, I32
from emmy.compiler.graph import Graph
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import BinaryExpr, Literal, TernaryExpr, Var
from emmy.compiler.ir.kernel.ir import Tile
from emmy.compiler.ir.stmt import Accum, Assign, Let, Load, Loop, Write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('artifacts/float-expressions'))
    parser.add_argument('--seed',type=int,default=0)
    args=parser.parse_args()
    template=Path(__file__).resolve().parents[1]/'challenge/problems/r01_array_sum'
    source=template/'k_sum_1_reduce.ir.json'
    x=(np.random.default_rng(args.seed).integers(-31,32,size=128).astype(np.float32)/np.float32(4))
    x[:8]=np.array([-2.5,2.5,-0.,0.,-.25,.25,-.75,.75],dtype=np.float32)
    cases=[]
    variants=('literal_compare','loaded_compare','loaded_truth','mixed_compare','float_arithmetic',
              'int_to_float','float_to_int','mixed_ternary','nan_truth','nan_compare','float_rounding','mixed_fma','integer_write')
    for variant in variants:
        for renamed in (False,True):
            prefix=f'q{args.seed}_' if renamed else ''
            name=lambda n:prefix+n
            v=lambda n:Var(name(n))
            k=lambda n:Literal(n,'int')
            f=lambda n:Literal(float(n))
            b=lambda op,a,c:BinaryExpr(op,a,c)
            axis=name('i');value=name('value');out=name('out')
            prelude=[Load(value,'x',(Var(axis),),dtype=F32)]
            result=None
            tail=()
            write_dtype=F32
            if variant=='literal_compare':
                condition=b('>',f(.75),f(.5));expected=np.full(64,x[1],dtype=np.float32)
            elif variant=='loaded_compare':
                condition=b('>',v('value'),f(.5));expected=x[(x[:64]>.5).astype(np.int64)]
            elif variant=='loaded_truth':
                condition=v('value');expected=x[(x[:64]!=0).astype(np.int64)]
            elif variant=='mixed_compare':
                condition=b('<',v('value'),Var(axis));expected=x[(x[:64]<np.arange(64,dtype=np.float32)).astype(np.int64)]
            elif variant=='float_arithmetic':
                result=b('-',b('*',b('+',v('value'),f(.5)),k(2)),f(1))
                expected=((x[:64]+np.float32(.5))*np.float32(2)-np.float32(1)).astype(np.float32)
            elif variant=='int_to_float':
                result=Var(axis);expected=np.arange(64,dtype=np.float32)
            elif variant=='float_to_int':
                prelude.append(Let(name('integer'),v('value'),I32));result=v('integer')
                expected=np.trunc(x[:64]).astype(np.int32).astype(np.float32)
            elif variant=='mixed_ternary':
                result=TernaryExpr(b('<',Var(axis),k(32)),v('value'),Var(axis))
                expected=np.where(np.arange(64)<32,x[:64],np.arange(64,dtype=np.float32)).astype(np.float32)
            elif variant in ('nan_truth','nan_compare'):
                prelude.extend((Let(name('zero'),f(0),F32),
                    Assign(name('nan'),ElementwiseImpl('divide'),(name('zero'),name('zero')),F32)))
                condition=v('nan') if variant=='nan_truth' else b('==',v('nan'),v('nan'))
                expected=np.full(64,x[1 if variant=='nan_truth' else 0],dtype=np.float32)
            elif variant=='mixed_fma':
                prelude.append(Loop(Axis(name('step'),Dim(4)),(
                    Assign(name('product'),ElementwiseImpl('multiply'),(name('step'),value),F32),
                    Accum(out,name('product'),ElementwiseImpl('add'),F32,axes=(name('step'),)))))
                result='bound'
                expected=np.float32(6)*x[:64]
            elif variant=='integer_write':
                out=axis
                result='bound'
                write_dtype=I32
                expected=np.arange(64,dtype=np.float32)
            else:
                prelude.append(Let(name('large'),b('+',k(16777217),Var(axis)),I32))
                result=v('large');expected=(16777217+np.arange(64,dtype=np.int64)).astype(np.float32)
            if result is None:
                prelude.append(Load(out,'x',(TernaryExpr(condition,k(1),k(0)),),dtype=F32))
            elif result != 'bound':
                prelude.append(Let(out,result,F32))
            body=tuple(prelude)+(Write('sum_1',(Var(axis),),value=out,value_dtype=write_dtype),)
            graph=Graph.from_dict(json.loads(source.read_text()))
            for node in graph.nodes.values():
                node.outputs=tuple(replace(t,shape=(Dim(128 if type(node.op).__name__=='InputOp' else 64),)) for t in node.outputs)
                if type(node.op).__name__=='KernelOp':node.op=replace(node.op,body=(Tile((Axis(axis,Dim(64)),),body),))
            label=variant+f'_s{args.seed}'+('_renamed' if renamed else '')
            folder=args.out/label;folder.mkdir(parents=True,exist_ok=True)
            raw=json.dumps(graph.to_dict(),indent=2,default=str)+'\n';Graph.from_dict(json.loads(raw));(folder/source.name).write_text(raw)
            meta=json.loads((template/'meta.json').read_text());meta.update(id=label,rtol=0,atol=0)
            launch=meta['kernels'][0];launch.update(grid=[1,1,1],block=[64,1,1],smem_bytes=0)
            for arg in launch['args']:
                arg['shape']=[64 if arg.get('output') else 128]
                if not arg.get('output'):arg.pop('seed',None);arg['data']=x.tolist()
            (folder/'meta.json').write_text(json.dumps(meta,indent=2)+'\n');np.savez(folder/'expected.npz',sum_1=expected)
            cases.append({'problem':label,'variant':variant,'seed':args.seed,'renamed':renamed,
                'ir_sha256':hashlib.sha256(raw.encode()).hexdigest(),'oracle':'Independent explicit f32 NumPy arithmetic/conversion and IEEE comparisons; no float division Expr assumption'})
    (args.out/'manifest.json').write_text(json.dumps({'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'cases':cases},indent=2)+'\n')
    print('Generated',len(cases),'float expression fixtures')


if __name__=='__main__':
    main()
