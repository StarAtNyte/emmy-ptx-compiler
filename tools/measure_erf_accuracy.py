"""Measure selected erf error distributions against independent stored references.

This is development-only execution/measurement, not part of IR lowering.
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

from local_runner import Driver


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--compiler',type=Path,required=True);parser.add_argument('--cases',type=Path,required=True);parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--server',default='http://runner.riftstack.ai:48701');args=parser.parse_args()
    session=requests.Session();driver=Driver();rows=[]
    def post(endpoint,body):
        for _ in range(30):
            response=session.post(args.server+endpoint,json=body,timeout=(10,60));response.raise_for_status();r=response.json()
            if r.get('stage')=='busy':time.sleep(2);continue
            if not r.get('ok'):raise RuntimeError(r)
            return r
        raise RuntimeError('Runner remained busy')
    def ordered(a):
        u=a.astype(np.float32).view(np.uint32).astype(np.int64)
        return np.where(u&0x80000000,0x80000000-(u&0x7fffffff),0x80000000+u)
    try:
        for mode in ['bits','dense','transition','tiny']:
            folder=args.cases/f'erf_{mode}_s0';meta=json.loads((folder/'meta.json').read_text());kernel=meta['kernels'][0]
            ptx=subprocess.check_output([sys.executable,str(args.compiler),str(folder/(kernel['entry']+'.ir.json'))],text=True)
            expected=np.load(folder/'expected.npz')['sum_1'];x=np.array(kernel['args'][0]['data'],dtype=np.float32)
            local=driver.run(ptx,kernel,{})['sum_1'];compiled=post('/compile',{'ptx':ptx})
            remote=post('/run',{'cubin_id':compiled['cubin_id'],'entry':kernel['entry'],'grid':kernel['grid'],'block':kernel['block'],
                'smem_bytes':kernel['smem_bytes'],'zero_outputs':kernel.get('zero_outputs',[]),'args':kernel['args'],'warmup':0,'iters':1,'max_return':int(expected.size)})
            assert not remote['outputs']['sum_1'].get('truncated')
            values={'local':local,'hosted':np.array(remote['outputs']['sum_1']['values'],dtype=np.float32)}
            for target,actual in values.items():
                np.testing.assert_allclose(actual,expected,rtol=meta['rtol'],atol=meta['atol']);assert np.isfinite(actual).all() and (np.abs(actual)<=1).all()
                zeros=expected==0;assert np.array_equal(np.signbit(actual[zeros]),np.signbit(expected[zeros]))
                error=np.abs(actual.astype(np.float64)-expected.astype(np.float64));worst=int(error.argmax());nonzero=expected!=0
                rows.append({'case':mode,'target':target,'elements':int(expected.size),'max_abs_error':float(error.max()),
                    'max_relative_error_nonzero':float(np.max(error[nonzero]/np.abs(expected[nonzero].astype(np.float64)))) if nonzero.any() else 0,
                    'max_ulp_distance':int(np.max(np.abs(ordered(actual)-ordered(expected)))),'worst_abs_input':float(x[worst]),
                    'worst_abs_actual':float(actual[worst]),'worst_abs_reference':float(expected[worst]),'bounded_and_signed_zero_checked':True,'passed':True})
                print(target,mode,rows[-1]['max_abs_error'],rows[-1]['max_ulp_distance'],flush=True)
    finally:driver.close()
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps({'compiler_sha256':hashlib.sha256(args.compiler.read_bytes()).hexdigest(),'scope':'Selected sampled finite values, not a global f32 error proof or correctly-rounded-libm claim','results':rows},indent=2)+'\n')


if __name__=='__main__':main()
