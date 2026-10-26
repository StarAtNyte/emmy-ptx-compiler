"""Typed logarithm/exp2 probes with independent float64 NumPy references.

Positive finite bit-pattern inputs cover subnormals, normal range and neighbors
of one. Special results are classified into finite outputs so JSON transport
never has to carry NaN or infinity. Compositions test use beyond one Assign.
"""
import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F32
from emmy.compiler.graph import Graph
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import BinaryExpr, Literal, TernaryExpr, Var
from emmy.compiler.ir.kernel.ir import Tile
from emmy.compiler.ir.stmt import Accum, Assign, Let, Load, Loop, Write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=0)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    source=root/'challenge/problems/r01_array_sum';ir=source/'k_sum_1_reduce.ir.json'
    cases=[(op,mode) for op in ('log','log2','log10','exp2') for mode in ('range','near_one','special')]
    cases += [('exp2','powers'),('exp2','positive'),('log','exp2_composition'),
              ('log2','exp2_roundtrip'),('log10','affine'),('log','loop')]
    rows=[]
    for op,mode in cases:
        rng=np.random.default_rng(args.seed);count=257
        if mode=='range' and op!='exp2':
            x=rng.integers(1,0x7f800000,size=count,dtype=np.uint32).view(np.float32)
            x[:12]=np.array([1,2,3,0x007fffff,0x00800000,0x00800001,0x3f7fffff,
                            0x3f800000,0x3f800001,0x40000000,0x7f7ffffe,0x7f7fffff],dtype=np.uint32).view(np.float32)
        elif mode=='near_one':
            x=(0x3f800000+np.arange(-128,129,dtype=np.int64)).astype(np.uint32).view(np.float32)
        elif mode=='special':
            x=np.resize(np.array([-512,-150,-149,-148,-127,-126,-2,-1,-0.,0.,1.,2.,127.,128.,512.],dtype=np.float32),count)
        elif mode in ('powers','positive'):
            x=np.arange(-149,128,dtype=np.float32);count=x.size
        elif mode in ('exp2_composition','loop'):
            x=rng.uniform(-10,10,count).astype(np.float32)
            if mode=='loop':x=np.abs(x)+np.float32(.125)
        elif mode=='exp2_roundtrip':x=np.exp2(rng.uniform(-40,40,count)).astype(np.float32)
        elif mode=='affine':x=np.exp2(rng.uniform(-40,40,count)).astype(np.float32)
        else:x=rng.uniform(-149,127,count).astype(np.float32)
        for renamed in (False,True):
            prefix=f'r{args.seed}_' if renamed else ''
            n=lambda s:prefix+s
            v=lambda s:Var(n(s))
            scalar=lambda value:Literal(float(value))
            binary=lambda operator,a,b:BinaryExpr(operator,a,b)
            body=[Load(n('x'),'x',(v('thread'),),dtype=F32)]
            def assign(name,operation,inputs):
                body.append(Assign(n(name),ElementwiseImpl(operation),tuple(n(i) for i in inputs),F32))
            output=n('result')
            with np.errstate(all='ignore'):
                function=getattr(np,op)
                if mode=='exp2_composition':
                    assign('intermediate','exp2',('x',));assign('result',op,('intermediate',))
                    expected=np.log(np.exp2(x.astype(np.float64))).astype(np.float32)
                elif mode=='exp2_roundtrip':
                    assign('intermediate',op,('x',));assign('result','exp2',('intermediate',));expected=x.copy()
                elif mode=='affine':
                    assign('logarithm',op,('x',));body.extend((Let(n('scale'),scalar(2),F32),Let(n('bias'),scalar(-.5),F32)))
                    assign('product','multiply',('logarithm','scale'));assign('result','add',('product','bias'))
                    expected=(2*np.log10(x.astype(np.float64))-.5).astype(np.float32)
                elif mode=='loop':
                    body.append(Loop(Axis(n('step'),Dim(5)),(
                        Assign(n('logarithm'),ElementwiseImpl(op),(n('x'),),F32),
                        Accum(n('result'),n('logarithm'),ElementwiseImpl('add'),F32,axes=(n('step'),)))))
                    # The reference rounds each scalar log before ordered f32 accumulation.
                    reference=np.log(x.astype(np.float64)).astype(np.float32);expected=np.zeros_like(x)
                    for _ in range(5):expected+=reference
                else:
                    assign('result',op,('x',));expected=function(x.astype(np.float64)).astype(np.float32)
                if mode=='special':
                    value=v('result');limit=float(np.finfo(np.float32).max)
                    expression=TernaryExpr(binary('!=',value,value),scalar(2),
                        TernaryExpr(binary('<',value,scalar(-limit)),scalar(-1),
                            TernaryExpr(binary('>',value,scalar(limit)),scalar(1),scalar(0))))
                    body.append(Let(n('classified'),expression,F32));output=n('classified')
                    expected=np.where(np.isnan(expected),2,np.where(np.isneginf(expected),-1,np.where(np.isposinf(expected),1,0))).astype(np.float32)
                elif mode=='positive':
                    body.append(Let(n('positive'),TernaryExpr(binary('>',v('result'),scalar(0)),scalar(1),scalar(0)),F32))
                    output=n('positive');expected=(expected>0).astype(np.float32)
            input_values=x
            if mode=='powers':
                # ex2.approx's contract is two ULP, not bitwise identity.
                # Powers of two have exact f32 references, including subnormals.
                lower=expected.copy();upper=expected.copy()
                for _ in range(2):
                    lower=np.nextafter(lower,np.float32(-np.inf))
                    upper=np.nextafter(upper,np.float32(np.inf))
                lower=np.maximum(lower,np.float32(0))
                input_values=np.concatenate((x,lower,upper))
                body.extend((Load(n('lower'),'x',(binary('+',v('thread'),Literal(int(count),'int')),),dtype=F32),
                             Load(n('upper'),'x',(binary('+',v('thread'),Literal(2*int(count),'int')),),dtype=F32),
                             Let(n('within_ulp'),TernaryExpr(binary('&&',binary('>=',v('result'),v('lower')),
                                 binary('<=',v('result'),v('upper'))),scalar(1),scalar(0)),F32)))
                output=n('within_ulp');expected=np.ones(count,dtype=np.float32)
            body.append(Write('sum_1',(v('thread'),),value=output,value_dtype=F32))
            graph=Graph.from_dict(json.loads(ir.read_text()))
            for node in graph.nodes.values():
                node.outputs=tuple(replace(t,shape=(Dim(int(input_values.size) if type(node.op).__name__=='InputOp' else int(count)),)) for t in node.outputs)
                if type(node.op).__name__=='KernelOp':node.op=replace(node.op,body=(Tile((Axis(n('thread'),Dim(int(count))),),tuple(body)),))
            label=f'{op}_{mode}_s{args.seed}'+('_renamed' if renamed else '')
            folder=args.out/label;folder.mkdir(parents=True,exist_ok=True)
            raw=json.dumps(graph.to_dict(),indent=2,default=str)+'\n';Graph.from_dict(json.loads(raw));(folder/ir.name).write_text(raw)
            meta=json.loads((source/'meta.json').read_text());exact=mode in ('special','positive','powers')
            meta.update(id=label,rtol=0 if exact else 1e-5,atol=0 if exact else float(np.float32(2*np.nextafter(np.float32(0),np.float32(1)))) if op=='exp2' and mode=='range' else 2e-7)
            launch=meta['kernels'][0];launch.update(grid=[(int(count)+127)//128,1,1],block=[128,1,1],smem_bytes=0)
            for arg in launch['args']:
                arg['shape']=[int(count) if arg.get('output') else int(input_values.size)]
                if not arg.get('output'):arg.pop('seed',None);arg['data']=input_values.tolist()
            (folder/'meta.json').write_text(json.dumps(meta,indent=2)+'\n');np.savez(folder/'expected.npz',sum_1=expected)
            rows.append({'folder':str(folder),'op':op,'mode':mode,'seed':args.seed,'renamed':renamed,
                'reference':'Independent NumPy float64 functions then float32; ordered f32 loop accumulation; special classes exact; exp2 powers use exact two-ULP-bound classification',
                'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.iterdir()) if p.is_file()}})
    (args.out/'manifest.json').write_text(json.dumps({'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'cases':rows},indent=2)+'\n')
    print(f'Generated {len(rows)} scalar logarithm/exp2 cases')


if __name__=='__main__':main()
