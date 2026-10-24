"""Index-rank addressing probes matching the installed IR's flattening contract."""
import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from emmy.compiler.dim import Dim
from emmy.compiler.graph import Graph
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.kernel.ir import Smem, Sync, Tile
from emmy.compiler.ir.stmt import Accum, Assign, Load, Loop, StridedLoop, Write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('artifacts/address-ranks'))
    parser.add_argument('--seed',type=int,default=0)
    args=parser.parse_args()
    template=Path(__file__).resolve().parents[1]/'challenge/problems/r01_array_sum'
    source=template/'k_sum_1_reduce.ir.json'
    x=np.arange(128,dtype=np.float32)+np.float32(args.seed%17)
    t=np.arange(64,dtype=np.int64)
    cases=[]
    for variant in ('extra_load','partial_load','flat_load','empty_load','additive_write','shared_pair','matching_load','scalar_flat'):
        for renamed in (False,True):
            axis=f'coordinate_{args.seed}' if renamed else 'i';v=Var(axis)
            k=lambda n:Literal(n,'int')
            b=lambda op,a,c:BinaryExpr(op,a,c)
            shape=(8,16);stage=()
            if variant=='extra_load':
                index=(b('%',v,k(4)),b('%',v,k(3)),k(5));coords=t%4+t%3+5
            elif variant=='partial_load':
                shape=(4,4,8);index=(b('%',v,k(4)),b('%',v,k(8)));coords=t%4+t%8
            elif variant=='empty_load':index=();coords=np.zeros(64,dtype=np.int64)
            elif variant=='matching_load':index=(b('/',v,k(16)),b('%',v,k(16)));coords=t
            elif variant=='scalar_flat':shape=();index=(v,);coords=t
            else:index=(b('*',v,k(2)),);coords=t*2
            primary=(Load('value','x',index),Write('sum_1',(v,),value='value'))
            values=x[coords]
            control=(Load('control','x',(b('+',v,k(64)),)),Write('sum_1',(b('+',v,k(64)),),value='control'))
            if variant=='additive_write':
                shape=(128,)
                primary=(Load('control','x',(v,)),Write('sum_1',(v,),value='control'),
                         Load('value','x',(b('+',v,k(64)),)),Write('sum_1',(v,k(64)),value='value'))
                control=();values=x[:64]
            elif variant=='shared_pair':
                copy=(Load('item','x',(Var('copy'),)),Write('scratch',(Var('copy'),),value='item'))
                stage=(Smem('scratch',(8,16),align=16),StridedLoop(Axis('copy',Dim(128)),v,k(64),copy),Sync())
                left=(b('*',b('%',v,k(4)),k(2)),k(2),k(4))
                right=(left[0],k(2),k(5))
                leaf=(Load('a','scratch',left),Load('b','scratch',right),
                      Assign('sum',ElementwiseImpl('add'),('a','b')),
                      Accum('result','sum',ElementwiseImpl('add'),axes=('step',)))
                primary=(Loop(Axis('step',Dim(1)),leaf),Write('sum_1',(v,),value='result'))
                values=x[2*(t%4)+6]+x[2*(t%4)+7]
            graph=Graph.from_dict(json.loads(source.read_text()))
            for node in graph.nodes.values():
                dims=shape if type(node.op).__name__=='InputOp' else (128,)
                node.outputs=tuple(replace(a,shape=tuple(Dim(d) for d in dims)) for a in node.outputs)
                if type(node.op).__name__=='KernelOp':node.op=replace(node.op,body=(Tile((Axis(axis,Dim(64)),),stage+primary+control),))
            label=variant+f'_s{args.seed}'+('_renamed' if renamed else '')
            folder=args.out/label;folder.mkdir(parents=True,exist_ok=True)
            raw=json.dumps(graph.to_dict(),indent=2,default=str)+'\n';Graph.from_dict(json.loads(raw));(folder/source.name).write_text(raw)
            meta=json.loads((template/'meta.json').read_text());meta.update(id=label,rtol=0,atol=0)
            launch=meta['kernels'][0];launch.update(grid=[1,1,1],block=[64,1,1],smem_bytes=0)
            for arg in launch['args']:
                arg['shape']=[128]
                if not arg.get('output'):arg.pop('seed',None);arg['data']=x.tolist()
            (folder/'meta.json').write_text(json.dumps(meta,indent=2)+'\n')
            np.savez(folder/'expected.npz',sum_1=np.concatenate((values,x[64:])))
            cases.append({'problem':label,'variant':variant,'seed':args.seed,'renamed':renamed,
                'ir_sha256':hashlib.sha256(raw.encode()).hexdigest(),
                'oracle':'Independent NumPy row-major / flat / additive coordinate formulas; physical buffer has 128 f32 elements'})
    (args.out/'manifest.json').write_text(json.dumps({'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'cases':cases},indent=2)+'\n')
    print('Generated',len(cases),'address-rank fixtures')


if __name__=='__main__':
    main()
