"""Predicate-to-integer index probes with independent NumPy boolean formulas."""
import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import I32
from emmy.compiler.graph import Graph
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import BinaryExpr, Literal, TernaryExpr, Var
from emmy.compiler.ir.kernel.ir import Smem, Sync, Tile
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.stmt import Accum, Assign, Cond, Let, Load, Loop, StridedLoop, Write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('artifacts/predicate-index'))
    parser.add_argument('--seed',type=int,default=0)
    args = parser.parse_args()
    template = Path(__file__).resolve().parents[1]/'challenge/problems/r01_array_sum'
    source = template/'k_sum_1_reduce.ir.json'
    x = np.random.default_rng(args.seed).integers(-100,101,size=128).astype(np.float32)
    t = np.arange(64,dtype=np.int64)
    cases = []
    for variant in ('direct','arithmetic','nested_compare','logic_arithmetic','ternary','let_snapshot','truthiness','condition','shared_direct','shared_pair'):
        for renamed in (False,True):
            axis = f'coordinate_{args.seed}' if renamed else 'i'
            flag = 'predicate_snapshot' if renamed else 'flag'
            v = Var(axis)
            k = lambda n: Literal(n,'int')
            b = lambda op,a,c: BinaryExpr(op,a,c)
            p = b('<',v,k(32))
            even = b('==',b('%',v,k(2)),k(0))
            prelude = ()
            if variant in ('direct','shared_direct','shared_pair'):
                index,expected = p,(t<32).astype(np.int64)
            elif variant=='arithmetic':
                index=b('+',b('*',p,k(3)),b('%',v,k(3)))
                expected=3*(t<32).astype(np.int64)+t%3
            elif variant=='nested_compare':
                index,expected=b('==',p,even),((t<32)==(t%2==0)).astype(np.int64)
            elif variant=='logic_arithmetic':
                index=b('+',b('&&',p,b('>',b('%',v,k(3)),k(0))),k(64))
                expected=np.logical_and(t<32,t%3!=0).astype(np.int64)+64
            elif variant=='ternary':
                index=TernaryExpr(p,even,b('==',b('%',v,k(3)),k(0)))
                expected=np.where(t<32,t%2==0,t%3==0).astype(np.int64)
            elif variant=='let_snapshot':
                prelude=(Let(flag,p,I32),Cond(b('==',k(1),k(1)),(Let(flag,k(7),I32),)))
                index,expected=b('+',Var(flag),k(64)),(t<32).astype(np.int64)+64
            elif variant=='truthiness':
                index=b('+',b('||',b('-',v,k(32)),b('%',v,k(5))),k(64))
                expected=np.logical_or(t-32,t%5).astype(np.int64)+64
            else:
                index=TernaryExpr(b('==',p,even),k(7),k(9))
                expected=np.where((t<32)==(t%2==0),7,9)
            graph=Graph.from_dict(json.loads(source.read_text()))
            expected_values=x[expected]
            body=prelude+(Load('value','x',(index,)),Write('sum_1',(v,),value='value'))
            if variant.startswith('shared_'):
                stage=(Smem('scratch',(128,),align=16),
                       StridedLoop(Axis('copy',Dim(128)),v,k(64),
                           (Load('copy_value','x',(Var('copy'),)),Write('scratch',(Var('copy'),),value='copy_value'))),Sync())
                if variant=='shared_direct':
                    body=stage+(Load('value','scratch',(index,)),Write('sum_1',(v,),value='value'))
                else:
                    pair_index=b('*',index,k(2))
                    leaf=(Load('a','scratch',(pair_index,)),Load('b','scratch',(b('+',pair_index,k(1)),)),
                          Assign('sum',ElementwiseImpl('add'),('a','b')),
                          Accum('result','sum',ElementwiseImpl('add'),axes=('step',)))
                    body=stage+(Loop(Axis('step',Dim(1)),leaf),Write('sum_1',(v,),value='result'))
                    expected_values=x[2*expected]+x[2*expected+1]
            for node in graph.nodes.values():
                node.outputs=tuple(replace(a,shape=(Dim(128 if type(node.op).__name__=='InputOp' else 64),)) for a in node.outputs)
                if type(node.op).__name__=='KernelOp':node.op=replace(node.op,body=(Tile((Axis(axis,Dim(64)),),body),))
            label=f'{variant}_s{args.seed}'+('_renamed' if renamed else '')
            folder=args.out/label;folder.mkdir(parents=True,exist_ok=True)
            raw=json.dumps(graph.to_dict(),indent=2,default=str)+'\n';Graph.from_dict(json.loads(raw))
            (folder/source.name).write_text(raw)
            meta=json.loads((template/'meta.json').read_text());meta.update(id=label,rtol=0,atol=0)
            launch=meta['kernels'][0];launch.update(grid=[1,1,1],block=[64,1,1],smem_bytes=0)
            for arg in launch['args']:
                arg['shape']=[64 if arg.get('output') else 128]
                if not arg.get('output'):arg.pop('seed',None);arg['data']=x.tolist()
            (folder/'meta.json').write_text(json.dumps(meta,indent=2)+'\n')
            np.savez(folder/'expected.npz',sum_1=expected_values)
            cases.append({'problem':label,'variant':variant,'seed':args.seed,'renamed':renamed,'ir_sha256':hashlib.sha256(raw.encode()).hexdigest(),'oracle':'Independent NumPy bool/0-or-1 arithmetic'})
    (args.out/'manifest.json').write_text(json.dumps({'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'cases':cases},indent=2)+'\n')
    print('Generated',len(cases),'predicate index fixtures')


if __name__=='__main__':
    main()
