"""Paired correctness-checked kernel timing on the official runner.

Trials alternate compiler order to reduce time-order bias. Clock locking is not
provided by the API, so retain raw samples and avoid conclusions from tiny gaps.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('problems', nargs='+', type=Path)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, default=Path('compile.py'))
    parser.add_argument('--trials', type=int, default=7)
    parser.add_argument('--report', type=Path, default=Path('artifacts/hosted-benchmark.json'))
    parser.add_argument('--server', default='http://runner.riftstack.ai:48701')
    parser.add_argument('--timing-return-limit',type=int,default=0,
                        help='Return full outputs in the first trial, then this many values for final outputs')
    args = parser.parse_args()
    session = requests.Session()

    def post(endpoint, body):
        for attempt in range(30):
            try:
                response = session.post(args.server + endpoint, json=body, timeout=(10, 60))
            except (requests.ConnectionError, requests.Timeout) as error:
                if attempt >= 4:
                    raise
                print(f'Transient runner connection failure ({type(error).__name__}); retrying',file=sys.stderr,flush=True)
                time.sleep(min(2**attempt,8))
                continue
            response.raise_for_status()
            result = response.json()
            if result.get('stage') == 'busy':
                time.sleep(2)
                continue
            if not result.get('ok'):
                raise RuntimeError(result)
            return result
        raise RuntimeError('Runner remained busy')

    compilers = {'baseline': args.baseline, 'candidate': args.candidate}
    report = {'clock_locked':False, 'warmup':8, 'iters':75,
              'compilers': {k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in compilers.items()},
              'timing_return_limit':args.timing_return_limit, 'results':[]}
    for folder in args.problems:
        meta = json.loads((folder / 'meta.json').read_text())
        cubins = {}
        for name, compiler in compilers.items():
            for kernel in meta['kernels']:
                ptx = subprocess.check_output([sys.executable, str(compiler),
                     str(folder / (kernel['entry'] + '.ir.json'))], text=True)
                cubins[name, kernel['entry']] = post('/compile', {'ptx':ptx})
        samples = {name:[] for name in compilers}
        for trial in range(args.trials):
            for name in (('baseline','candidate') if trial % 2 == 0 else ('candidate','baseline')):
                produced = {}
                total = 0
                for kernel in meta['kernels']:
                    arguments = []
                    for arg in kernel['args']:
                        arg = dict(arg)
                        if arg.pop('intermediate',False):
                            arg['data'] = produced[arg['name']].ravel().tolist()
                        arguments.append(arg)
                    count = sum(int(np.prod(a['shape'])) for a in arguments if a.get('output'))
                    final_only=all(a['name'] in meta['graph_outputs'] for a in arguments if a.get('output'))
                    if trial and final_only and args.timing_return_limit:
                        count=min(count,args.timing_return_limit)
                    result = post('/run', {'cubin_id':cubins[name,kernel['entry']]['cubin_id'],
                        'entry':kernel['entry'], 'grid':kernel['grid'], 'block':kernel['block'],
                        'smem_bytes':kernel['smem_bytes'], 'zero_outputs':kernel['zero_outputs'],
                        'args':arguments, 'warmup':report['warmup'], 'iters':report['iters'],
                        'max_return':max(1,count)})
                    total += result['median_us']
                    for output, value in result['outputs'].items():
                        if value.get('truncated') and (not trial or not args.timing_return_limit or not final_only):
                            raise RuntimeError('Truncated benchmark output')
                        produced[output] = np.array(value['values'],dtype=np.float32)
                        if not value.get('truncated'):
                            produced[output]=produced[output].reshape(value['shape'])
                with np.load(folder / 'expected.npz') as expected:
                    for output in expected.files:
                        actual=produced[output].ravel()
                        np.testing.assert_allclose(actual,expected[output].ravel()[:actual.size],rtol=meta['rtol'],atol=meta['atol'])
                samples[name].append(total)
        baseline, candidate = (float(np.median(samples[n])) for n in ('baseline','candidate'))
        row = {'problem':folder.name, 'samples_us':samples, 'baseline_median_us':baseline,
               'candidate_median_us':candidate, 'speedup':baseline/candidate, 'outputs_passed':True,
               'output_check':'full first trial; returned prefix thereafter' if args.timing_return_limit else 'full every trial',
               'registers':{name:{k['entry']:cubins[name,k['entry']].get('registers') for k in meta['kernels']} for name in compilers}}
        report['results'].append(row)
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2)+'\n')
        print(f"{folder.name}: {baseline:.3f} -> {candidate:.3f} us ({baseline/candidate:.3f}x)",flush=True)


if __name__ == '__main__':
    main()
