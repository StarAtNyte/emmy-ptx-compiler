"""Measure runner launch timing and compiler-generated copy/strided-read throughput.

This is calibration, not a compiler speedup comparison. Repeated launches reuse
allocations, clocks are unlocked, and no cache eviction API is available. Effective
bytes/time is not measured DRAM traffic or a hardware maximum.
"""
import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import requests
from emmy.compiler.dim import Dim
from emmy.compiler.graph import Graph
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.kernel.ir import Tile
from emmy.compiler.ir.stmt import Accum, Load, Loop, Write


def digest(data):
    return hashlib.sha256(data).hexdigest()


def generate(folder, compiler, count, steps):
    root = Path(__file__).resolve().parents[1]
    source = root/'validation/cpu-fixtures/2026-10-24-abi-reordered'
    graph = Graph.from_dict(json.loads(next(source.glob('*.ir.json')).read_text()))
    outputs = count//steps
    if count+outputs > 67108864 or outputs > 4194304:
        raise ValueError('calibration exceeds runner total-buffer or return limits')
    i, j = Var('item'), Var('step')
    if steps == 1:
        body = (Load('value','x',(i,)),Write('sum_1',(i,),value='value'))
    else:
        address = BinaryExpr('+',BinaryExpr('*',j,Literal(outputs,'int')),i)
        leaf = (Load('value','x',(address,)),Accum('total','value',ElementwiseImpl('add'),axes=('step',)))
        body = (Loop(Axis('step',Dim(steps)),leaf),Write('sum_1',(i,),value='total'))
    for node in graph.nodes.values():
        size = count if type(node.op).__name__=='InputOp' else outputs
        node.outputs = tuple(replace(t,shape=(Dim(size),)) for t in node.outputs)
        if type(node.op).__name__=='KernelOp':
            node.op = replace(node.op,body=(Tile((Axis('item',Dim(outputs)),),body),))
    folder.mkdir(parents=True,exist_ok=True)
    kernel = json.loads((source/'meta.json').read_text())['kernels'][0]
    kernel.update(grid=[(outputs+255)//256,1,1],block=[256,1,1],smem_bytes=0)
    for arg in kernel['args']:
        arg['shape'] = [outputs if arg.get('output') else count]
        arg.pop('data',None)
        if not arg.get('output'):arg['seed']=47
    ir = folder/f"{kernel['entry']}.ir.json"
    ir.write_text(json.dumps(graph.to_dict(),indent=2,default=str)+'\n')
    (folder/'meta.json').write_text(json.dumps({'kernels':[kernel]},indent=2)+'\n')
    ptx = subprocess.check_output([sys.executable,str(compiler),str(ir)],text=True)
    (folder/'kernel.ptx').write_text(ptx)
    # Explicit float32 additions in the same strided order. The reference is
    # independent of PTX/codegen and covers every output, not just a checksum.
    data = np.random.default_rng(47).standard_normal(count).astype(np.float32)
    expected = np.zeros(outputs,dtype=np.float32)
    for step in range(steps):
        expected += data[step*outputs:(step+1)*outputs]
    del data
    return {'name':folder.name,'kernel':kernel,'ptx':ptx,'expected':expected,
            'input_bytes':count*4,'output_bytes':outputs*4,'float_adds':0 if steps==1 else count,
            'steps':steps,'ir_sha256':digest(ir.read_bytes()),
            'meta_sha256':digest((folder/'meta.json').read_bytes()),
            'oracle_sha256':digest(expected.tobytes()),'ptx_sha256':digest(ptx.encode())}


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler',type=Path,default=root/'compile.py')
    parser.add_argument('--out',type=Path,default=root/'artifacts/target-calibration')
    parser.add_argument('--trials',type=int,default=5)
    parser.add_argument('--server',default='http://runner.riftstack.ai:48701')
    args = parser.parse_args()
    if args.trials < 1:parser.error('trials must be positive')
    args.out.mkdir(parents=True,exist_ok=True)
    session = requests.Session()
    health = session.get(args.server+'/health',timeout=30);health.raise_for_status()
    report = {'compiler_sha256':digest(args.compiler.read_bytes()),'tool_sha256':digest(Path(__file__).read_bytes()),
        'health':health.json(),'server':args.server,'clock_locked':False,'cache_evicted':False,
        'warmup':8,'iters':75,'order':'forward/reverse on alternating trials',
        'scope':'effective workload bytes/time, not DRAM counters or bandwidth maximum',
        'full_outputs_checked_trial':0,'later_trials':'4096 output prefix checked', 'results':[]}

    def post(endpoint,body):
        for attempt in range(30):
            response = session.post(args.server+endpoint,json=body,timeout=(10,180))
            if not response.ok:
                raise RuntimeError(f'{endpoint}: HTTP {response.status_code}: {response.text[:2000]}')
            value = response.json()
            if value.get('stage')=='busy':
                time.sleep(2)
                continue
            if not value.get('ok'):raise RuntimeError(value)
            return value
        raise RuntimeError('runner remained busy')

    cases = [generate(args.out/name,args.compiler.resolve(),count,steps)
             for name,count,steps in [('copy_4KiB',1024,1),('copy_1MiB',2**18,1),
                ('read_4MiB',2**20,32),('read_64MiB',2**24,32),('read_224MiB',7*2**23,32)]]
    # Standalone hand-written no-work PTX is a timing control, never a submission
    # backend. Every memory workload above uses this repository's compiler.
    noop = '.version 8.0\n.target sm_89\n.address_size 64\n.visible .entry launch_floor() { ret; }\n'
    cases.insert(0,{'name':'launch_floor','kernel':{'entry':'launch_floor','args':[],
        'grid':[1,1,1],'block':[1,1,1],'smem_bytes':0,'zero_outputs':[]},'ptx':noop,
        'expected':np.empty(0,dtype=np.float32),'input_bytes':0,'output_bytes':0,'float_adds':0,
        'ptx_sha256':digest(noop.encode()),'source':'hand-written no-work calibration control'})
    for case in cases:
        compiled = post('/compile',{'ptx':case['ptx']})
        case['compiled'] = compiled
        row = {k:v for k,v in case.items() if k not in ('ptx','expected')}
        row.update(samples=[],validation=[])
        report['results'].append(row)
    output = args.out/'results.json'
    for trial in range(args.trials):
        indices = range(len(cases)) if trial%2==0 else reversed(range(len(cases)))
        for index in indices:
            case,row = cases[index],report['results'][index]
            kernel = case['kernel']
            count = len(case['expected']) if trial==0 else min(4096,len(case['expected']))
            result = post('/run',{'cubin_id':case['compiled']['cubin_id'],**kernel,
                'warmup':report['warmup'],'iters':report['iters'],'max_return':count})
            if count:
                actual = np.asarray(result['outputs']['sum_1']['values'],dtype=np.float32)
                if len(actual)!=count:raise RuntimeError(f'incomplete output: {case["name"]}')
                np.testing.assert_array_equal(actual,case['expected'][:count])
            row['validation'].append({'trial':trial,'elements':count,'full':trial==0,'passed':True})
            row['samples'].append({'trial':trial,'median_us':result['median_us'],
                'runner':{k:v for k,v in result.items() if k!='outputs'}})
            values = [s['median_us'] for s in row['samples']]
            row['median_us'] = statistics.median(values)
            row['effective_GB_s'] = (case['input_bytes']+case['output_bytes'])/row['median_us']/1000
            output.write_text(json.dumps(report,indent=2)+'\n')
            print(f'{case["name"]} trial {trial}: {result["median_us"]:.3f} us, full={trial==0}',flush=True)


if __name__=='__main__':
    main()
