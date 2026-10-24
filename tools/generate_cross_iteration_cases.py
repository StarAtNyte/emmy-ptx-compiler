"""Independent exact probes for cross-iteration shared-read pairing."""
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
from emmy.compiler.ir.stmt import Accum, Assign, Let, Load, Loop, StridedLoop, Write
from emmy.compiler.dtype import I32


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=0)
    args = parser.parse_args()
    source = root/'validation/cpu-fixtures/2026-10-24-abi-reordered'
    cases = [('aligned',4),('tail',5),('large',64),('large_tail',65),('odd',5),
             ('weak',5),('local_index',5),('axis_shadow',5),('nonunit',5),
             ('nonlinear',7),('shared_write',5),('global_read',5),('additive',7),
             ('flat_view',5),('odd_stride',5)]
    manifest = []
    k = lambda x:Literal(x,'int')
    b = lambda op,x,y:BinaryExpr(op,x,y)
    for variant,trips in cases:
        for renamed in (False,True):
            prefix = f's{args.seed}_' if renamed else ''
            n = lambda name:prefix+name
            v = lambda name:Var(n(name))
            columns = 129 if variant=='odd_stride' else 128
            size = 64*columns
            data = np.random.default_rng(args.seed).integers(-8,9,size=size).astype(np.float32)
            coords = v('step')
            if variant=='odd':coords=b('+',coords,k(1))
            if variant=='nonunit':coords=b('*',coords,k(2))
            if variant=='nonlinear':coords=b('%',coords,k(3))
            before = ()
            if variant=='local_index':
                before=(Let(n('index'),b('+',v('step'),k(1)),I32),)
                coords=v('index')
            if variant=='axis_shadow':before=(Let(n('step'),k(2),I32),)
            index = (v('thread'),coords)
            shape = (64,columns)
            if variant=='additive':index=(b('*',v('thread'),k(columns)),coords,k(0))
            if variant=='flat_view':
                shape=(size,);index=(b('+',b('*',v('thread'),k(columns)),coords),)
            leaf = before+(Load(n('value'),n('scratch'),index),)
            if variant=='shared_write':
                leaf+=(Write(n('scratch'),(v('thread'),b('+',v('step'),k(1))),value=n('value')),)
            value = n('value')
            if variant=='global_read':
                leaf+=(Load(n('global'),'x',(v('thread'),)),
                       Assign(n('sum'),ElementwiseImpl('add'),(n('value'),n('global'))))
                value=n('sum')
            leaf+=(Accum(n('total'),value,ElementwiseImpl('add'),axes=(n('step'),)),)
            copy = StridedLoop(Axis(n('copy'),Dim(size)),v('thread'),k(64),(
                Load(n('item'),'x',(v('copy'),)),
                Write(n('scratch'),(v('copy'),),value=n('item'))))
            body = (Smem(n('scratch'),shape,align=4 if variant=='weak' else 16),copy,Sync(),
                    Loop(Axis(n('step'),Dim(trips)),leaf),Write('sum_1',(v('thread'),),value=n('total')))
            graph=Graph.from_dict(json.loads(next(source.glob('*.ir.json')).read_text()))
            for node in graph.nodes.values():
                extent=size if type(node.op).__name__=='InputOp' else 64
                node.outputs=tuple(replace(t,shape=(Dim(extent),)) for t in node.outputs)
                if type(node.op).__name__=='KernelOp':node.op=replace(node.op,body=(Tile((Axis(n('thread'),Dim(64)),),body),))
            work=data.reshape(64,columns).copy();expected=np.zeros(64,dtype=np.float32)
            for step in range(trips):
                c=(step+1 if variant in ('odd','local_index') else 2 if variant=='axis_shadow'
                   else 2*step if variant=='nonunit' else step%3 if variant=='nonlinear' else step)
                item=work[:,c].copy()
                if variant=='shared_write':work[:,step+1]=item
                if variant=='global_read':item+=data[:64]
                expected+=item
            label=variant+f'_s{args.seed}'+('_renamed' if renamed else '')
            folder=args.out/label;folder.mkdir(parents=True,exist_ok=True)
            payload=json.dumps(graph.to_dict(),indent=2,default=str)+'\n';Graph.from_dict(json.loads(payload))
            ir_name=next(source.glob('*.ir.json')).name;(folder/ir_name).write_text(payload)
            meta=json.loads((source/'meta.json').read_text());meta.update(id=label,rtol=0,atol=0)
            launch=meta['kernels'][0];launch.update(grid=[1,1,1],block=[64,1,1],smem_bytes=0)
            for arg in launch['args']:
                arg['shape']=[64 if arg.get('output') else size]
                if not arg.get('output'):arg['data']=data.tolist();arg.pop('seed',None)
            (folder/'meta.json').write_text(json.dumps(meta,indent=2)+'\n')
            np.savez(folder/'expected.npz',sum_1=expected)
            manifest.append({'folder':str(folder),'variant':variant,'trips':trips,'seed':args.seed,
                'renamed':renamed,'expect_new_vector':variant in ('aligned','tail','large','large_tail','additive','flat_view'),
                'ir_sha256':hashlib.sha256(payload.encode()).hexdigest(),
                'oracle':'exact independent small-integer NumPy updates over private rows'})
    (args.out/'manifest.json').write_text(json.dumps({'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'cases':manifest},indent=2)+'\n')
    print(f'Generated {len(manifest)} cross-iteration cases')


if __name__=='__main__':main()
