"""Semantic regression cases beyond the expression generator's common paths."""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F32, I32
from emmy.compiler.graph import Graph
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.kernel.ir import Smem, Sync, Tile, TreeHalve
from emmy.compiler.ir.stmt import Accum, Assign, Cond, Init, Let, Load, Loop, Write


def main():
    root=Path(__file__).resolve().parents[1]
    def package(name,body,input_size,expected,two_inputs=False,block=32):
        template=root/'challenge/problems'/('p01_vector_add' if two_inputs else 'r01_array_sum')
        entry='k_add_pointwise' if two_inputs else 'k_sum_1_reduce'
        graph=Graph.from_dict(json.loads((template/f'{entry}.ir.json').read_text()))
        for key,node in list(graph.nodes.items()):
            count=input_size if type(node.op).__name__=='InputOp' else expected.size
            graph.nodes[key]=replace(node,outputs=tuple(replace(t,shape=(Dim(count),)) for t in node.outputs),
                op=replace(node.op,body=body) if type(node.op).__name__=='KernelOp' else node.op)
        folder=root/'artifacts/audit'/name
        folder.mkdir(parents=True,exist_ok=True)
        (folder/f'{entry}.ir.json').write_text(json.dumps(graph.to_dict(),indent=2,default=str)+'\n')
        meta=json.loads((template/'meta.json').read_text())
        meta['id']=name
        meta.pop('code',None)
        kernel=meta['kernels'][0]
        kernel.update(grid=[1,1,1],block=[block,1,1],smem_bytes=0)
        for arg in kernel['args']:
            arg['shape']=[expected.size if arg.get('output') else input_size]
        (folder/'meta.json').write_text(json.dumps(meta,indent=1)+'\n')
        np.savez(folder/'expected.npz',**{meta['graph_outputs'][0]:expected.astype(np.float32)})

    def add(index,suffix):
        return (Load(name='a'+suffix,input='x0',index=(index,),dtype=F32),
                Load(name='b'+suffix,input='x1',index=(index,),dtype=F32),
                Assign('v'+suffix,ElementwiseImpl('add'),('a'+suffix,'b'+suffix),F32),
                Write(output='add',index=(index,),value='v'+suffix,value_dtype=F32))
    x=np.random.default_rng(0).standard_normal(128).astype(np.float32)
    y=np.random.default_rng(1).standard_normal(128).astype(np.float32)
    first=Tile((Axis('first',Dim(64)),),add(Var('first'),'first'))
    second=Tile((Axis('second',Dim(64)),),add(BinaryExpr('+',Var('second'),Literal(64,'int')),'second'))
    package('p_repeated_tiles',(first,second),128,x+y,True,128)
    tile=Tile((Axis('element',Dim(128)),),(
        Let('integer_one',Literal(1,'int'),I32),
        Cond(BinaryExpr('==',Var('integer_one'),Literal(1,'int')),add(Var('element'),'let'))))
    package('p_integer_let',(tile,),128,x+y,True,128)

    def reduced_scratch(values):
        rows=values.reshape(4,-1)
        return np.stack(((rows[0]+rows[2])+(rows[1]+rows[3]),rows[1]+rows[3],rows[2],rows[3])).ravel()
    for name,scale,repetitions in (('r_tree_scratch',1,1),('r_tree_repeat',1,2),('r_tree_segmented_scratch',3,1)):
        size=4*scale
        source=np.random.default_rng(0).standard_normal(size).astype(np.float32)
        expected=source
        for _ in range(repetitions):
            expected=reduced_scratch(expected)
        thread=Var('thread')
        body=[Smem('scratch',(size,),align=16),
              Cond(BinaryExpr('<',thread,Literal(size,'int')),
                   (Load(name='copied',input='x',index=(thread,),dtype=F32),
                    Write(output='scratch',index=(thread,),value='copied',value_dtype=F32))),Sync(),
              Init('acc',0.0,dtype=F32)]
        if scale>1:
            body.extend((Let('group',BinaryExpr('/',thread,Literal(scale,'int')),I32),
                         Let('column',BinaryExpr('%',thread,Literal(scale,'int')),I32)))
        tree=TreeHalve(bufs=('scratch',),state=('acc',),state_b=('other',),
                       combine_states=(Assign('acc',ElementwiseImpl('add'),('acc','other'),F32),),
                       length=4,tid_var='group' if scale>1 else 'thread',dtype=F32,
                       inner=('column',scale) if scale>1 else None)
        body.extend([tree]*repetitions)
        body.append(Cond(BinaryExpr('<',thread,Literal(size,'int')),
            (Load(name='observed',input='scratch',index=(thread,),dtype=F32),
             Write(output='sum_1',index=(thread,),value='observed',value_dtype=F32))))
        package(name,(Tile((Axis('thread',Dim(32)),),tuple(body),block_threads=32),),size,expected)

    # Affine-map composition is associative but not commutative. The prescribed
    # halving tree orders four leaves 0,2,1,3 rather than a linear 0,1,2,3 fold.
    source=np.random.default_rng(0).standard_normal(8).astype(np.float32)
    merge=(Assign('new_a',ElementwiseImpl('multiply'),('a','other_a'),F32),
           Assign('new_b_product',ElementwiseImpl('multiply'),('a','other_b'),F32),
           Assign('new_b',ElementwiseImpl('add'),('new_b_product','b'),F32),
           Assign('a',ElementwiseImpl('copy'),('new_a',),F32),
           Assign('b',ElementwiseImpl('copy'),('new_b',),F32))
    thread=Var('thread')
    body=(Smem('coefficients',(4,),align=16),Smem('offsets',(4,),align=16),
          Cond(BinaryExpr('<',thread,Literal(4,'int')),
               (Load(name='coefficient',input='x',index=(thread,),dtype=F32),
                Load(name='offset',input='x',index=(BinaryExpr('+',thread,Literal(4,'int')),),dtype=F32),
                Write(output='coefficients',index=(thread,),value='coefficient',value_dtype=F32),
                Write(output='offsets',index=(thread,),value='offset',value_dtype=F32))),Sync(),
          Init('a',1.0,dtype=F32),Init('b',0.0,dtype=F32),
          TreeHalve(('coefficients','offsets'),('a','b'),('other_a','other_b'),merge,4,'thread',F32),
          Cond(BinaryExpr('==',thread,Literal(0,'int')),
               (Write(output='sum_1',index=(),values=('a','b'),value_dtype=F32),)))
    a,b=np.float32(1),np.float32(0)
    for leaf in (0,2,1,3):
        b=np.float32(a*source[4+leaf])+b
        a=np.float32(a*source[leaf])
    package('r_tree_ordered_merge',(Tile((Axis('thread',Dim(32)),),body,block_threads=32),),8,np.array([a,b]))

    # The tree's synchronization orders unrelated shared writes too. Warp 0
    # must not observe the initial zero while the other warp is still working.
    body=(Smem('scratch',(4,),align=16),Smem('publication',(1,),align=16),
          Cond(BinaryExpr('<',thread,Literal(4,'int')),
               (Load(name='ready',input='x',index=(thread,),dtype=F32),
                Write(output='scratch',index=(thread,),value='ready',value_dtype=F32))),
          Cond(BinaryExpr('==',thread,Literal(0,'int')),
               (Let('zero',Literal(0.0),F32),Write(output='publication',index=(),value='zero',value_dtype=F32))),Sync(),
          Cond(BinaryExpr('==',thread,Literal(63,'int')),
               (Load(name='increment',input='x',index=(Literal(0,'int'),),dtype=F32),
                Loop(Axis('delay',Dim(1024)),(Accum('delayed','increment',ElementwiseImpl('add'),F32,axes=('delay',)),)),
                Write(output='publication',index=(),value='delayed',value_dtype=F32))),
          Init('tree_acc',0.0,dtype=F32),
          TreeHalve(('scratch',),('tree_acc',),('other',),
                    (Assign('tree_acc',ElementwiseImpl('add'),('tree_acc','other'),F32),),4,'thread',F32),
          Cond(BinaryExpr('==',thread,Literal(0,'int')),
               (Load(name='published',input='publication',index=(),dtype=F32),
                Write(output='sum_1',index=(),value='published',value_dtype=F32))))
    increment=np.random.default_rng(0).standard_normal(4).astype(np.float32)[0]
    delayed=np.add.accumulate(np.full(1024,increment,dtype=np.float32),dtype=np.float32)[-1]
    package('r_tree_orders_other_memory',(Tile((Axis('thread',Dim(64)),),body,block_threads=64),),4,np.array([delayed]),block=64)

    for name,length,scale,columns in (('r_tree_named',4,3,3),
                                      ('r_tree_named_duplicate',2,1,1),
                                      ('r_tree_named_column_subset',4,4,2)):
        size=length*scale
        source=np.random.default_rng(0).standard_normal(size).astype(np.float32)
        if length==2:
            expected=np.array([source[0]+source[1],source[1]])
        else:
            expected=source.reshape(4,scale).copy()
            reduced=reduced_scratch(source).reshape(4,scale)
            expected[:,:columns]=reduced[:,:columns]
            expected=expected.ravel()
        local=Var('local')
        consumer=[Let('local',BinaryExpr('-',thread,Literal(32,'int')),I32)]
        if length==2:
            consumer.append(Let('logical',BinaryExpr('/',local,Literal(32,'int')),I32))
            inner=None
            tid='logical'
        else:
            consumer.extend((Let('group',BinaryExpr('/',local,Literal(columns,'int')),I32),
                             Let('column',BinaryExpr('%',local,Literal(columns,'int')),I32)))
            inner=('column',scale)
            tid='group'
        consumer.extend((Init('acc',0.0,dtype=F32),
            TreeHalve(('scratch',),('acc',),('other',),
                (Assign('acc',ElementwiseImpl('add'),('acc','other'),F32),),length,tid,F32,
                barrier_id=1,barrier_count=32,inner=inner),
            Cond(BinaryExpr('<',local,Literal(size,'int')),
                (Load(name='observed',input='scratch',index=(local,),dtype=F32),
                 Write(output='sum_1',index=(local,),value='observed',value_dtype=F32)))))
        body=(Smem('scratch',(size,),align=16),
            Cond(BinaryExpr('<',thread,Literal(size,'int')),
                (Load(name='copied',input='x',index=(thread,),dtype=F32),
                 Write(output='scratch',index=(thread,),value='copied',value_dtype=F32))),Sync(),
            Cond(BinaryExpr('>=',thread,Literal(32,'int')),tuple(consumer)))
        package(name,(Tile((Axis('thread',Dim(64)),),body,block_threads=64),),size,expected,block=64)

    source=np.random.default_rng(0).standard_normal(19).astype(np.float32)
    reduce=Var('reduce')
    fold=lambda value:Accum('acc',value,ElementwiseImpl('add'),F32,axes=('reduce',))
    loop=Loop(Axis('reduce',Dim(19)),(Cond(BinaryExpr('<',reduce,Literal(9,'int')),
        (Load(name='positive',input='x',index=(reduce,),dtype=F32),fold('positive')),
        (Load(name='negative',input='x',index=(reduce,),dtype=F32),
         Assign('negated',ElementwiseImpl('negative'),('negative',),F32),fold('negated'))),))
    body=(loop,Cond(BinaryExpr('==',Var('thread'),Literal(0,'int')),
                   (Write(output='sum_1',index=(),value='acc',value_dtype=F32),)))
    package('r_conditional_accumulator',(Tile((Axis('thread',Dim(32)),),body,block_threads=32),),
            19,np.array([source[:9].astype(np.float64).sum()-source[9:].astype(np.float64).sum()]))
    print('Generated 11 audit cases')


if __name__=='__main__':
    main()
