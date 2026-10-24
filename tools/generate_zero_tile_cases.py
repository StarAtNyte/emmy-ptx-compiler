"""Exact no-op semantics for empty tile domains, with nonempty controls."""
import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from emmy.compiler.dim import Dim
from emmy.compiler.graph import Graph
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.kernel.ir import Tile
from emmy.compiler.ir.stmt import Cond, Let, Load, Loop, Write


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=0)
    args=parser.parse_args()
    source=root/'validation/cpu-fixtures/2026-10-24-abi-reordered'
    original=json.loads(next(source.glob('*.ir.json')).read_text())
    data=np.random.default_rng(args.seed).integers(-9,10,size=64).astype(np.float32)
    variants=['outer_zero','inner_zero','leading_zero','multi_zero','cond_zero',
              'loop_zero','empty_axes','singleton','zero_input_view']
    manifest=[]
    for variant in variants:
        for renamed in (False,True):
            prefix=f's{args.seed}_' if renamed else ''
            axis=lambda name,size:Axis(prefix+name,Dim(size))
            live=Tile((axis('i',64),),(Load(prefix+'item','x',(Var(prefix+'i'),)),
                Write('sum_1',(Var(prefix+'i'),),value=prefix+'item')))
            sizes=([64,0] if variant=='inner_zero' else [0,64] if variant=='leading_zero'
                   else [0,2,0] if variant=='multi_zero' else [] if variant=='empty_axes'
                   else [1] if variant=='singleton' else [0])
            dead=Tile(tuple(axis(f'z{j}',size) for j,size in enumerate(sizes)),(
                Let(prefix+'sentinel',Literal(-999.0)),Write('sum_1',(Literal(0,'int'),),value=prefix+'sentinel')))
            wrapper=(Cond(Literal(1),(dead,)) if variant=='cond_zero'
                     else Loop(axis('repeat',2),(dead,)) if variant=='loop_zero' else dead)
            graph=Graph.from_dict(original)
            for node in graph.nodes.values():
                dims=(Dim(0),) if variant=='zero_input_view' and type(node.op).__name__=='InputOp' else (Dim(64),)
                node.outputs=tuple(replace(t,shape=dims) for t in node.outputs)
                if type(node.op).__name__=='KernelOp':node.op=replace(node.op,body=(live,wrapper))
            expected=data.copy()
            if variant in ('empty_axes','singleton'):expected[0]=-999.0
            name=variant+f'_s{args.seed}'+('_renamed' if renamed else '')
            folder=args.out/name;folder.mkdir(parents=True,exist_ok=True)
            ir=next(source.glob('*.ir.json')).name;payload=json.dumps(graph.to_dict(),indent=2,default=str)+'\n'
            Graph.from_dict(json.loads(payload));(folder/ir).write_text(payload)
            meta=json.loads((source/'meta.json').read_text());meta.update(id=name,rtol=0,atol=0)
            kernel=meta['kernels'][0];kernel.update(grid=[1,1,1],block=[64,1,1],smem_bytes=0)
            for arg in kernel['args']:
                arg['shape']=[64]
                if not arg.get('output'):arg['data']=data.tolist();arg.pop('seed',None)
            (folder/'meta.json').write_text(json.dumps(meta,indent=2)+'\n');np.savez(folder/'expected.npz',sum_1=expected)
            manifest.append({'folder':str(folder),'variant':variant,'seed':args.seed,'renamed':renamed,
                'ir_sha256':hashlib.sha256(payload.encode()).hexdigest(),
                'oracle':'copy independent seeded integer input; overwrite first element only for empty-axis-list/singleton controls'})
    (args.out/'manifest.json').write_text(json.dumps({'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'cases':manifest},indent=2)+'\n')
    print('Generated',len(manifest),'zero-domain/control cases')


if __name__=='__main__':main()
