"""Generate the GELU MLP and independent input-seed references before validation."""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from expressions_erf_models import reference


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='emmy-erf-models-') as tmp:
        subprocess.run([sys.executable,str(root/'tools/generate_models.py'),'--specs',str(root/'tools/expressions_erf_models.py'),'--out',tmp],check=True)
        source=Path(tmp)/'model_mlp_gelu'
        for seed in [0,17,613]:
            folder=args.out/f'mlp_gelu_s{seed}';shutil.copytree(source,folder,dirs_exist_ok=True)
            path=folder/'meta.json';meta=json.loads(path.read_text());assert len(meta['graph_inputs'])==1;name=meta['graph_inputs'][0]
            shape=next(a['shape'] for k in meta['kernels'] for a in k['args'] if a['name']==name)
            for kernel in meta['kernels']:
                for arg in kernel['args']:
                    if arg['name']==name:arg['seed']=seed;arg.pop('data',None)
            x=np.random.default_rng(seed).standard_normal(shape).astype(np.float32)
            np.savez(folder/'expected.npz',**{meta['graph_outputs'][0]:reference(x)[0]})
            meta['id']=folder.name;path.write_text(json.dumps(meta,indent=2)+'\n')


if __name__=='__main__':main()
