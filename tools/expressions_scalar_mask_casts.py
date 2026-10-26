"""Frontend bool-to-f32 casts, with optional finite corner-input packaging.

Run directly to generate both cases and replace seeded inputs with zero,
subnormal and extreme finite values. Compilation itself never uses this tool.
"""
import numpy as np

OPEN=[dict(id='p_numeric_bool_cast',code='torch.randn(257).bool().float()',ref=lambda x:[(x!=0).astype(np.float32)],rtol=0,atol=0),
      dict(id='p_numeric_bool_affine',code='x=torch.randn(129); x.bool().float()*2-1',ref=lambda x:[np.float32(2)*(x!=0).astype(np.float32)-np.float32(1)],rtol=0,atol=0)]


def main():
    import argparse
    import json
    import subprocess
    import sys
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    subprocess.run([sys.executable,str(root/'challenge/tools/generate.py'),'--emmy',str(root/'.venv/bin/emmy'),
                    '--out',str(args.out),str(Path(__file__))],check=True)
    for spec in OPEN:
        folder=args.out/spec['id'];path=folder/'meta.json';meta=json.loads(path.read_text())
        size=next(a['shape'][0] for a in meta['kernels'][0]['args'] if a['name']=='x')
        x=np.random.default_rng(0).standard_normal(size).astype(np.float32)
        tiny=np.nextafter(np.float32(0),np.float32(1))
        x[:8]=[-0.,0.,.25,-.25,np.finfo(np.float32).max,-np.finfo(np.float32).max,tiny,-tiny]
        for kernel in meta['kernels']:
            for arg in kernel['args']:
                if arg['name']=='x':arg.pop('seed',None);arg['data']=x.tolist()
        path.write_text(json.dumps(meta,indent=2)+'\n')
        np.savez(folder/'expected.npz',**{meta['graph_outputs'][0]:spec['ref'](x)[0]})


if __name__=='__main__':main()
