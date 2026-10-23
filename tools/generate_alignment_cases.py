"""Typed IR tests for aligned and deliberately misaligned grouped accesses.

Unlike public pointwise inputs, odd row strides and one-element offsets prevent
some grouped accesses from satisfying vector alignment. Every element is checked
against NumPy; scalar tail work runs once per row.
"""
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
from emmy.compiler.ir.kernel.ir import Tile
from emmy.compiler.ir.stmt import Assign, Cond, Load, Write


def main():
    root = Path(__file__).resolve().parents[1]
    template = root / 'challenge/problems/p01_vector_add'
    cases = ((7,16,4,0),(7,17,4,0),(7,17,4,1),(5,7,2,0),
             (5,7,2,1),(3,18,2,0),(9,20,4,0),(9,21,4,1))
    for rows,cols,width,offset in cases:
        groups=(cols-offset)//width
        graph=Graph.from_dict(json.loads((template/'k_add_pointwise.ir.json').read_text()))
        r,c=Var('row'),Var('group')
        begin=BinaryExpr('+',BinaryExpr('*',c,Literal(width,'int')),Literal(offset,'int'))
        index=(r,begin)

        def add_at(index,count,suffix):
            left=tuple(f'left_{suffix}_{i}' for i in range(count))
            right=tuple(f'right_{suffix}_{i}' for i in range(count))
            out=tuple(f'out_{suffix}_{i}' for i in range(count))
            return (Load(names=left,input='x0',index=index,dtype=F32),
                    Load(names=right,input='x1',index=index,dtype=F32),
                    *(Assign(o,ElementwiseImpl('add'),(a,b),F32) for o,a,b in zip(out,left,right)),
                    Write(output='add',index=index,values=out,value_dtype=F32))

        body=list(add_at(index,width,'group'))
        tails=[]
        for column in list(range(offset))+list(range(offset+groups*width,cols)):
            tails.extend(add_at((r,Literal(column,'int')),1,f'tail{column}'))
        if tails:
            body.append(Cond(BinaryExpr('==',c,Literal(0,'int')),tuple(tails)))
        tile=Tile((Axis('row',Dim(rows)),Axis('group',Dim(groups))),tuple(body))
        for name,node in list(graph.nodes.items()):
            outputs=tuple(replace(t,shape=(Dim(rows),Dim(cols))) for t in node.outputs)
            graph.nodes[name]=replace(node,outputs=outputs,
                op=replace(node.op,body=(tile,)) if type(node.op).__name__=='KernelOp' else node.op)
        name=f'p_alignment_{rows}_{cols}_v{width}_offset{offset}'
        folder=root/'artifacts/alignment'/name
        folder.mkdir(parents=True,exist_ok=True)
        (folder/'k_add_pointwise.ir.json').write_text(json.dumps(graph.to_dict(),indent=2,default=str)+'\n')
        meta=json.loads((template/'meta.json').read_text())
        meta['id']=name
        meta.pop('code',None)
        kernel=meta['kernels'][0]
        kernel['grid']=[(rows*groups+127)//128,1,1]
        kernel['block']=[128,1,1]
        for arg in kernel['args']:
            arg['shape']=[rows,cols]
        (folder/'meta.json').write_text(json.dumps(meta,indent=1)+'\n')
        x=np.random.default_rng(0).standard_normal((rows,cols)).astype(np.float32)
        y=np.random.default_rng(1).standard_normal((rows,cols)).astype(np.float32)
        np.savez(folder/'expected.npz',add=x+y)
    print(f'Generated {len(cases)} alignment cases')


if __name__=='__main__':
    main()
