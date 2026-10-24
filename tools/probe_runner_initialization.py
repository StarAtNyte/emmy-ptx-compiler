"""Compare safe explicit-data/zeroing combinations on local and hosted runners."""
import argparse
import copy
import hashlib
import json
import math
import subprocess
import struct
import sys
import time
from dataclasses import replace
from pathlib import Path

import requests
from emmy.compiler.dim import Dim
from emmy.compiler.graph import Graph
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Literal
from emmy.compiler.ir.kernel.ir import Tile
from emmy.compiler.ir.stmt import Load, Write

from local_runner import Driver


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=root/'artifacts/runner-initialization')
    parser.add_argument('--server',default='http://runner.riftstack.ai:48701')
    parser.add_argument('--first-value',type=float,default=7.)
    parser.add_argument('--require-match',action='store_true')
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    if not math.isfinite(args.first_value) or not args.first_value or abs(args.first_value)>3.402823466e38:
        parser.error('first-value must be finite nonzero f32 to distinguish zeroing')
    if not struct.unpack('f',struct.pack('f',args.first_value))[0]:
        parser.error('first-value underflows to zero in f32')
    source=root/'validation/cpu-fixtures/2026-10-24-abi-reordered'
    graph=Graph.from_dict(json.loads(next(source.glob('*.ir.json')).read_text()))
    body=(Load('value','x',(Literal(0,'int'),)),Write('sum_1',(Literal(0,'int'),),value='value'))
    for node in graph.nodes.values():
        size=4 if type(node.op).__name__=='InputOp' else 1
        node.outputs=tuple(replace(t,shape=(Dim(size),)) for t in node.outputs)
        if type(node.op).__name__=='KernelOp':node.op=replace(node.op,body=(Tile((Axis('i',Dim(1)),),body),))
    kernel=json.loads((source/'meta.json').read_text())['kernels'][0]
    kernel.update(grid=[1,1,1],block=[1,1,1],smem_bytes=0,zero_outputs=[])
    for arg in kernel['args']:
        arg['shape']=[1 if arg.get('output') else 4]
        if not arg.get('output'):arg['data']=[args.first_value,11.,-3.,5.];arg.pop('seed',None)
    ir=args.out/f"{kernel['entry']}.ir.json";ir.write_text(json.dumps(graph.to_dict(),indent=2,default=str)+'\n')
    (args.out/'meta.json').write_text(json.dumps({'kernels':[kernel]},indent=2)+'\n')
    ptx=subprocess.check_output([sys.executable,str(root/'compile.py'),str(ir)],text=True)
    session=requests.Session()
    def post(endpoint,body):
        for _ in range(30):
            response=session.post(args.server+endpoint,json=body,timeout=60);response.raise_for_status()
            result=response.json()
            if result.get('stage')=='busy':time.sleep(2);continue
            if not result.get('ok'):raise RuntimeError(result)
            return result
        raise RuntimeError('runner remained busy')
    compiled=post('/compile',{'ptx':ptx})
    report={'compiler_sha256':hashlib.sha256((root/'compile.py').read_bytes()).hexdigest(),
        'local_runner_sha256':hashlib.sha256((root/'tools/local_runner.py').read_bytes()).hexdigest(),
        'ptx_sha256':hashlib.sha256(ptx.encode()).hexdigest(),'scope':'one safe x[0] read and one y[0] write; no timing comparison','results':[]}
    driver=Driver()
    try:
        for returned in (False,True):
            for zeroed in (False,True):
                descriptor=copy.deepcopy(kernel)
                x=next(a for a in descriptor['args'] if a['name']=='x');x['output']=returned
                descriptor['zero_outputs']=['x'] if zeroed else []
                remote=post('/run',{'cubin_id':compiled['cubin_id'],**descriptor,'warmup':0,'iters':1,'max_return':8})
                hosted={name:out['values'] for name,out in remote['outputs'].items()}
                local={name:array.ravel().tolist() for name,array in driver.run(ptx,descriptor,{}).items()}
                timed={name:array.ravel().tolist() for name,array in driver.run(ptx,descriptor,{},timed_iters=2).items()}
                value=0. if returned or zeroed else struct.unpack('f',struct.pack('f',args.first_value))[0]
                expected={'sum_1':[value]}
                if returned:expected['x']=[0.,0.,0.,0.]
                row={'returned_input':returned,'zeroed_input':zeroed,'descriptor':descriptor,
                     'hosted':hosted,'local':local,'local_timed':timed,
                     'match':hosted==local,'timed_match':hosted==timed,
                     'expected':expected,'oracle_passed':hosted==local==timed==expected}
                report['results'].append(row)
                print(row,flush=True)
                (args.out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    finally:driver.close()
    if args.require_match and not all(r['oracle_passed'] for r in report['results']):
        raise SystemExit(1)


if __name__=='__main__':main()
