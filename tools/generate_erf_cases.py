"""Independent square/erf boundaries, compositions and exceptional-value classes."""
import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.special import erf
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F32,BOOL
from emmy.compiler.graph import Graph
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import BinaryExpr,Literal,Var
from emmy.compiler.ir.kernel.ir import Tile
from emmy.compiler.ir.stmt import Assign,Let,Load,Write


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--seed',type=int,default=0);parser.add_argument('--transition',type=float,default=1.)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1];source=root/'challenge/problems/r01_array_sum';ir=source/'k_sum_1_reduce.ir.json'
    cases=[('erf',m) for m in ['bits','dense','transition','tiny','sign','positive','exceptional','gelu']]
    cases += [('square',m) for m in ['exact','bits','exceptional','overflow']]
    rows=[]
    for op,mode in cases:
        rng=np.random.default_rng(args.seed)
        if mode=='bits':
            limit=0x7f800000 if op=='erf' else 0x5f800000
            bits=rng.integers(0,limit,8193,dtype=np.uint32);bits|=rng.integers(0,2,bits.size,dtype=np.uint32)<<31;x=bits.view(np.float32)
        elif mode in ('dense','gelu'):x=np.linspace(-8,8,8193,dtype=np.float32)
        elif mode=='transition':
            center=np.float32(args.transition);x=np.linspace(center-.0001,center+.0001,1025,dtype=np.float32);x=np.concatenate((x,-x,np.array([np.nextafter(center,np.float32(0)),center,np.nextafter(center,np.float32(np.inf))],dtype=np.float32)))
        elif mode in ('tiny','sign','positive'):
            bits=rng.integers(0,0x35800000,1025,dtype=np.uint32);bits|=rng.integers(0,2,bits.size,dtype=np.uint32)<<31;x=bits.view(np.float32)
            x[:8]=np.array([0,0x80000000,1,0x80000001,0x007fffff,0x807fffff,0x00800000,0x80800000],dtype=np.uint32).view(np.float32)
        elif mode=='exact':x=np.arange(-128,129,dtype=np.float32)/np.float32(16)
        else:x=np.resize(np.array([-np.finfo(np.float32).max,-1,-0.,0.,1,np.finfo(np.float32).max],dtype=np.float32),257)
        count=int(x.size)
        for renamed in [False,True]:
            prefix=f'e{args.seed}_' if renamed else '';n=lambda s:prefix+s;v=lambda s:Var(n(s));f=lambda a:Literal(float(a));b=lambda o,a,c:BinaryExpr(o,a,c)
            body=[Load(n('x'),'x',(v('thread'),),dtype=F32)];operand='x'
            def assign(name,fn,inputs,dtype=F32):body.append(Assign(n(name),ElementwiseImpl(fn),tuple(n(i) for i in inputs),dtype))
            if mode=='exceptional':
                body.append(Let(n('zero'),f(0),F32));assign('special','divide',('x','zero'));operand='special'
            assign('result',op,(operand,));output=n('result')
            with np.errstate(all='ignore'):
                a=x.astype(np.float64)
                if mode=='exceptional':a=a/np.float64(0)
                expected=(erf(a) if op=='erf' else a*a).astype(np.float32)
                if mode=='gelu':
                    body.extend((Let(n('half'),f(.5),F32),Let(n('scale'),f(1/np.sqrt(2)),F32),Let(n('one'),f(1),F32)))
                    assign('scaled','multiply',('x','scale'));assign('erf_value','erf',('scaled',));assign('plus','add',('one','erf_value'))
                    assign('half_x','multiply',('x','half'));assign('gelu','multiply',('half_x','plus'));output=n('gelu')
                    expected=(a*.5*(1+erf(a/np.sqrt(2)))).astype(np.float32)
                elif mode in ('exceptional','overflow'):
                    value=v('result');limit=float(np.finfo(np.float32).max)
                    from emmy.compiler.ir.expr import TernaryExpr
                    expression=TernaryExpr(b('!=',value,value),f(2),TernaryExpr(b('<',value,f(-limit)),f(-1),TernaryExpr(b('>',value,f(limit)),f(1),f(0))))
                    body.append(Let(n('classified'),expression,F32));output=n('classified')
                    expected=np.where(np.isnan(expected),2,np.where(np.isneginf(expected),-1,np.where(np.isposinf(expected),1,0))).astype(np.float32)
                elif mode=='sign':
                    assign('inverse','reciprocal',('result',));body.append(Let(n('zero'),f(0),F32));assign('sign','less',('inverse','zero'),BOOL);output=n('sign');expected=np.signbit(expected).astype(np.float32)
                elif mode=='positive':
                    body.append(Let(n('zero'),f(0),F32));assign('positive','greater',('result','zero'),BOOL);output=n('positive');expected=(expected>0).astype(np.float32)
            body.append(Write('sum_1',(v('thread'),),value=output,value_dtype=F32))
            graph=Graph.from_dict(json.loads(ir.read_text()))
            for node in graph.nodes.values():
                node.outputs=tuple(replace(t,shape=(Dim(count),)) for t in node.outputs)
                if type(node.op).__name__=='KernelOp':node.op=replace(node.op,body=(Tile((Axis(n('thread'),Dim(count)),),tuple(body)),))
            label=f'{op}_{mode}_s{args.seed}'+('_renamed' if renamed else '');folder=args.out/label;folder.mkdir(parents=True,exist_ok=True)
            raw=json.dumps(graph.to_dict(),indent=2,default=str)+'\n';Graph.from_dict(json.loads(raw));(folder/ir.name).write_text(raw)
            exact=op=='square' or mode in ('sign','positive','exceptional');meta=json.loads((source/'meta.json').read_text());meta.update(id=label,rtol=0 if exact else 2e-6,atol=0 if exact else 4e-7)
            launch=meta['kernels'][0];launch.update(grid=[(count+127)//128,1,1],block=[128,1,1],smem_bytes=0)
            for arg in launch['args']:
                arg['shape']=[count]
                if not arg.get('output'):arg.pop('seed',None);arg['data']=x.tolist()
            (folder/'meta.json').write_text(json.dumps(meta,indent=2)+'\n');np.savez(folder/'expected.npz',sum_1=expected)
            rows.append({'folder':str(folder),'op':op,'mode':mode,'seed':args.seed,'renamed':renamed,'reference':'Independent SciPy float64 erf / float64 square then f32; classes and signed-zero/subnormal predicates exact','files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.iterdir()) if p.is_file()}})
    (args.out/'manifest.json').write_text(json.dumps({'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'cases':rows},indent=2)+'\n');print('Generated',len(rows),'math probes')


if __name__=='__main__':main()
