"""Run the complete block through separately traced PTX stages.

The pinned Emmy release cannot materialize the fully fused block (unbound-axis
assertion). Stage boundaries preserve the same model math; all stages execute
our PTX. Host transfers make this a correctness test, not a model benchmark.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from local_runner import Driver
from transformer_model import make_block


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stages', type=Path, default=Path('artifacts/transformer-stages'))
    parser.add_argument('--report', type=Path, default=Path('artifacts/transformer-results.json'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    model = make_block()
    driver = Driver()
    cache = {}
    for name in ('norm', 'q', 'k', 'v', 'attention', 'out', 'residual_ff'):
        folder = args.stages / ('transformer_' + name)
        meta = json.loads((folder / 'meta.json').read_text())
        kernels = []
        for kernel in meta['kernels']:
            ptx = subprocess.check_output([sys.executable, str(root / 'compile.py'),
                 str(folder / (kernel['entry'] + '.ir.json'))], text=True)
            kernels.append((kernel, ptx))
        cache[name] = (meta, kernels)

    def stage(name, values):
        meta, kernels = cache[name]
        if len(values) != len(meta['graph_inputs']):
            raise ValueError(f'Wrong input count for {name}')
        produced = dict(zip(meta['graph_inputs'], values))
        for kernel, ptx in kernels:
            produced.update(driver.run(ptx, kernel, produced))
        return produced[meta['graph_outputs'][0]]

    report = {'compiler_sha256': hashlib.sha256((root / 'compile.py').read_bytes()).hexdigest(),
              'arch': driver.arch, 'model': 'seeded RMSNorm + QKV SDPA + output projection + residual SiLU MLP',
              'shape': [1, 4, 16], 'execution': 'seven separately traced stages, host transfers',
              'results': []}
    try:
        with torch.no_grad():
            for seed in (9, 71, 613):
                for depth in (1, 2, 4):
                    gpu = np.random.default_rng(seed).standard_normal((1, 4, 16)).astype(np.float32)
                    reference = torch.from_numpy(gpu.copy())
                    errors = []
                    for _ in range(depth):
                        z = stage('norm', [gpu])
                        zr = model.norm(reference)
                        q, k, v = [stage(name, [z]) for name in ('q', 'k', 'v')]
                        qr, kr, vr = model.q(zr), model.k(zr), model.v(zr)
                        attention = stage('attention', [q, k, v])
                        ar = F.scaled_dot_product_attention(qr, kr, vr)
                        out = stage('out', [attention])
                        outr = model.out(ar)
                        result = stage('residual_ff', [gpu, out])
                        rr = reference + outr
                        rr = rr + model.down(F.silu(model.up(rr)))
                        full_reference = model(reference)
                        torch.testing.assert_close(rr, full_reference, rtol=0, atol=0)
                        for actual, expected in ((z,zr),(q,qr),(k,kr),(v,vr),(attention,ar),(out,outr),(result,rr)):
                            np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-3, atol=1e-4)
                            errors.append(float(np.max(np.abs(actual - expected.numpy()))))
                        gpu, reference = result, full_reference
                    row = {'seed':seed, 'depth':depth, 'passed':True,
                           'stage_checks':7 * depth, 'max_abs_error':max(errors)}
                    report['results'].append(row)
                    args.report.parent.mkdir(parents=True, exist_ok=True)
                    args.report.write_text(json.dumps(report, indent=2) + '\n')
                    print(f"PASS seed={seed} depth={depth}: max abs error {max(errors):.3e}", flush=True)
    finally:
        driver.close()


if __name__ == '__main__':
    main()
