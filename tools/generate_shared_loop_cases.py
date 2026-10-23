"""Shared scalar reductions with every remainder of a four-way unroll."""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F32
from emmy.compiler.graph import Graph
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.kernel.ir import Smem, Sync, Tile
from emmy.compiler.ir.stmt import Accum, Cond, Load, Loop, StridedLoop, Write


def main():
    root=Path(__file__).resolve().parents[1]
    template=root/'challenge/problems/r01_array_sum'
    for size in (1,2,3,4,5,6,7,19,65):
        graph=Graph.from_dict(json.loads((template/'k_sum_1_reduce.ir.json').read_text()))
        copy=StridedLoop(Axis('copy',Dim(size)),Var('thread'),Literal(64,'int'),(
            Load(name='loaded',input='x',index=(Var('copy'),),dtype=F32),
            Write(output='scratch',index=(Var('copy'),),value='loaded',value_dtype=F32)))
        reduce=Loop(Axis('reduce',Dim(size)),(
            Load(name='value',input='scratch',index=(Var('reduce'),),dtype=F32),
            Accum(name='acc',value='value',op=ElementwiseImpl('add'),dtype=F32,axes=('reduce',))))
        tile=Tile((Axis('thread',Dim(64)),),(
            Smem('scratch',(size,),align=16),copy,Sync(),reduce,
            Cond(BinaryExpr('==',Var('thread'),Literal(0,'int')),
                 (Write(output='sum_1',index=(),value='acc',value_dtype=F32),))),block_threads=64)
        for name,node in list(graph.nodes.items()):
            if type(node.op).__name__=='InputOp':
                graph.nodes[name]=replace(node,outputs=tuple(replace(t,shape=(Dim(size),)) for t in node.outputs))
            elif type(node.op).__name__=='KernelOp':
                graph.nodes[name]=replace(node,op=replace(node.op,body=(tile,)))
        folder=root/'artifacts/shared-loops'/f'r_shared_{size}'
        folder.mkdir(parents=True,exist_ok=True)
        (folder/'k_sum_1_reduce.ir.json').write_text(json.dumps(graph.to_dict(),indent=2,default=str)+'\n')
        meta=json.loads((template/'meta.json').read_text())
        meta['id']=folder.name
        meta.pop('code',None)
        meta['kernels'][0]['block']=[64,1,1]
        meta['kernels'][0]['smem_bytes']=0
        meta['kernels'][0]['args'][0]['shape']=[size]
        (folder/'meta.json').write_text(json.dumps(meta,indent=1)+'\n')
        x=np.random.default_rng(0).standard_normal(size).astype(np.float32)
        np.savez(folder/'expected.npz',sum_1=np.array([x.astype(np.float64).sum()],dtype=np.float32))
    print('Generated 9 shared-loop cases')


if __name__=='__main__':
    main()
