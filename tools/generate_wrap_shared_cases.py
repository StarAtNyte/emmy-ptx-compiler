"""In-bounds shared reads with overflowing int32 address intermediates.

Final indices and NumPy oracles are unchanged from independent scalar fixtures.
The inserted arithmetic identities hold modulo 2**32; no out-of-bounds pointer
is formed or dereferenced. This is a targeted regression corpus, not a proof of
all possible address expressions.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

from emmy.compiler.graph import Graph
from emmy.compiler.ir.expr import BinaryExpr, Literal
from emmy.compiler.ir.stmt import Load


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=0)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    source=root/'tools/generate_cross_iteration_cases.py'
    literal=lambda x:Literal(x,'int')
    binary=lambda op,x,y:BinaryExpr(op,x,y)
    def identity(expr,kind):
        if kind=='sub':
            return binary('+',binary('-',expr,literal(-2147483648)),literal(-2147483648))
        if kind=='multiply':
            return binary('-',binary('*',expr,literal(1073741825)),binary('*',expr,literal(1073741824)))
        return binary('-',binary('+',expr,literal(2147483647)),literal(2147483647))
    cases=[('signed_add','tail',4),('signed_sub','tail',4),('multiply','tail',4),
           ('row_multiply','tail',4),('additive','additive',4),('flat','flat_view',4),
           ('opaque_row','tail',4),('align8','align8',2),('odd','odd',0),
           ('nonunit','nonunit',0),('nonlinear','nonlinear',0)]
    manifest=[]
    with tempfile.TemporaryDirectory(prefix='emmy-wrap-source-') as temporary:
        subprocess.run([sys.executable,str(source),'--out',temporary,'--seed',str(args.seed),'--wide-guards'],
                       check=True,capture_output=True,text=True)
        for kind,base,width in cases:
            for renamed in (False,True):
                suffix=f'_s{args.seed}'+('_renamed' if renamed else '')
                folder=Path(temporary)/(base+suffix)
                ir=next(folder.glob('*.ir.json'));graph=Graph.from_dict(json.loads(ir.read_text()))
                changed=[]
                def walk(body):
                    result=[]
                    for stmt in body:
                        if isinstance(stmt,Load) and stmt.input.endswith('scratch'):
                            coords=list(stmt.index)
                            if kind in ('row_multiply','opaque_row'):
                                coords[0]=identity(coords[0],'multiply' if kind=='row_multiply' else 'add')
                                if kind=='opaque_row':coords[0]=binary('%',coords[0],literal(64))
                            else:
                                index=0 if kind=='flat' else 1
                                coords[index]=identity(coords[index],'sub' if kind=='signed_sub' else 'multiply' if kind=='multiply' else 'add')
                            stmt=replace(stmt,index=tuple(coords));changed.append(stmt)
                        for field in ('body','else_body'):
                            if hasattr(stmt,field):stmt=replace(stmt,**{field:walk(getattr(stmt,field))})
                        result.append(stmt)
                    return tuple(result)
                node=next(n for n in graph.nodes.values() if type(n.op).__name__=='KernelOp')
                node.op=replace(node.op,body=walk(node.op.body));assert len(changed)==1
                target=args.out/(kind+suffix);target.mkdir(parents=True,exist_ok=True)
                payload=json.dumps(graph.to_dict(),indent=2,default=str)+'\n'
                Graph.from_dict(json.loads(payload));(target/ir.name).write_text(payload)
                meta=json.loads((folder/'meta.json').read_text());meta['id']=target.name
                (target/'meta.json').write_text(json.dumps(meta,indent=2)+'\n')
                (target/'expected.npz').write_bytes((folder/'expected.npz').read_bytes())
                manifest.append({'folder':str(target),'variant':kind,'seed':args.seed,'renamed':renamed,
                    'expected_cross_width':width,'ir_sha256':hashlib.sha256(payload.encode()).hexdigest(),
                    'oracle':'Unchanged independent ordered NumPy float32 scalar fixture; identities preserve final indices modulo2**32',
                    'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(target.iterdir()) if p.is_file()}})
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'manifest.json').write_text(json.dumps({'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'base_generator_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'cases':manifest},indent=2)+'\n')
    print(f'Generated {len(manifest)} in-bounds wraparound cases')


if __name__=='__main__':main()
