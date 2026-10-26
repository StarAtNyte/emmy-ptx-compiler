"""Measure f32 pointwise accuracy of compiled kernels against float64 references.

Development-only. Each case reuses a generated single-input pointwise problem,
feeds it wide-range sample arrays on the local GPU and reports the worst ULP
distance from the correctly rounded float64 result. CUDA's accurate libm
(expf/tanhf/logf/sinf/powf, what Emmy's reference kernels call) sits at 1-4
ULP, so that is the bar for tight hidden-set tolerances.
"""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

from local_runner import Driver

N = 4096
rng = np.random.default_rng(1234)


def uniform(lo, hi, count=8):
    return [rng.uniform(lo, hi, N) for _ in range(count)]


def logspace(lo, hi, count=8, signed=False):
    out = []
    for _ in range(count):
        x = np.exp2(rng.uniform(lo, hi, N))
        out.append(x * rng.choice([-1, 1], N) if signed else x)
    return out


ERF = np.vectorize(math.erf)
CASES = {
    't_exp': (np.exp, uniform(-5, 5) + uniform(-87, 88) + uniform(-103, -87, 2)),
    't_tanh': (np.tanh, uniform(-4, 4) + uniform(-12, 12) + logspace(-30, 2, 4, True)),
    't_sigmoid': (lambda x: 1/(1+np.exp(-x)), uniform(-6, 6) + uniform(-100, 40)),
    't_silu': (lambda x: x/(1+np.exp(-x)), uniform(-6, 6) + uniform(-100, 40)),
    't_log': (np.log, uniform(0.5, 2) + logspace(-126, 127) + logspace(-149, -126, 2)),
    't_rsqrt': (lambda x: 1/np.sqrt(x), logspace(-126, 127) + logspace(-149, -126, 2)),
    't_sqrt': (np.sqrt, logspace(-149, 127)),
    't_reciprocal': (lambda x: 1/x, logspace(-125, 125, 8, True)),
    't_gelu': (lambda x: 0.5*x*(1+ERF(x/np.sqrt(2))), uniform(-6, 6)),
    't_gelu_tanh': (lambda x: 0.5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x+0.044715*x**3))), uniform(-6, 6)),
    't_softplus': (lambda x: np.log1p(np.exp(x)), uniform(-6, 6) + uniform(-30, 30)),
    't_sin': (np.sin, uniform(-6.3, 6.3) + uniform(-1e4, 1e4) + logspace(-30, 100, 4, True)),
    't_cos': (np.cos, uniform(-6.3, 6.3) + uniform(-1e4, 1e4) + logspace(-30, 100, 4, True)),
    't_pow3': (lambda x: x**3, uniform(-5, 5) + logspace(-40, 40, 4, True)),
    't_pow_half': (np.sqrt, uniform(0, 1) + logspace(-126, 127, 4)),
}


def ordered(a):
    u = a.astype(np.float32).view(np.uint32).astype(np.int64)
    return np.where(u & 0x80000000, 0x80000000 - (u & 0x7fffffff), 0x80000000 + u)


def measure(driver, folder, compiler, fn, samples):
    meta = json.loads((folder / 'meta.json').read_text())
    kernel = meta['kernels'][0]
    name = next(a['name'] for a in kernel['args'] if 'seed' in a)
    out = next(a['name'] for a in kernel['args'] if a.get('output'))
    ptx = subprocess.run([sys.executable, str(compiler), str(folder / f"{kernel['entry']}.ir.json")],
                         capture_output=True, text=True)
    if ptx.returncode:
        return {'error': ptx.stderr.strip().splitlines()[-1][:120]}
    worst = dict(ulp=0, rel=0.0, x=None, got=None, want=None)
    for x in samples:
        x = x.astype(np.float32)
        with np.errstate(all='ignore'):
            want = fn(x.astype(np.float64)).astype(np.float32)
        got = driver.run(ptx.stdout, kernel, {name: x})[out]
        bad = np.isnan(want) != np.isnan(got)
        both = ~np.isnan(want) & ~np.isnan(got)
        ulp = np.zeros(N, np.int64)
        ulp[both] = np.abs(ordered(got[both]) - ordered(want[both]))
        ulp[bad] = 1 << 31
        with np.errstate(all='ignore'):
            rel = np.abs(got.astype(np.float64) - want) / np.maximum(np.abs(want.astype(np.float64)), 1e-30)
        i = int(ulp.argmax())
        if ulp[i] > worst['ulp']:
            worst = dict(ulp=int(ulp[i]), rel=float(rel[i]), x=float(x[i]), got=float(got[i]), want=float(want[i]))
        finite = both & np.isfinite(want) & (np.abs(want) > 1e-30)
        if finite.any():
            worst['rel'] = max(worst['rel'], float(np.nanmax(rel[finite])))
    return worst


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler', type=Path, default=Path(__file__).resolve().parents[1] / 'compile.py')
    parser.add_argument('--cases', type=Path, default=Path('artifacts/precision'))
    parser.add_argument('--only', nargs='*')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    driver = Driver()
    rows = {}
    try:
        for case, (fn, samples) in CASES.items():
            if args.only and case not in args.only:
                continue
            rows[case] = measure(driver, args.cases / case, args.compiler, fn, samples)
            print(f'{case:14s} {rows[case]}', flush=True)
    finally:
        driver.close()
    if args.report:
        args.report.write_text(json.dumps(rows, indent=2) + '\n')


if __name__ == '__main__':
    main()
