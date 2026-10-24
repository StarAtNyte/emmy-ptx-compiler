"""Signed index division/remainder probes with independent NumPy floor semantics."""
import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from emmy.compiler.dim import Dim
from emmy.compiler.graph import Graph
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import BinaryExpr, Literal, TernaryExpr, Var
from emmy.compiler.ir.kernel.ir import Tile
from emmy.compiler.ir.stmt import Load, Write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('artifacts/signed-index'))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--divisor', type=int, choices=(1,2,4,8,16,32),
                        help='Use a fixed power-of-two divisor instead of a varying divisor')
    parser.add_argument('--extremes', action='store_true',
                        help='Probe INT_MIN/INT_MAX, observing quotient sign and low bits')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    template = root / 'challenge/problems/r01_array_sum'
    source = template / 'k_sum_1_reduce.ir.json'
    rng = np.random.default_rng(args.seed)
    x = rng.integers(-100, 101, size=128).astype(np.float32)
    cases = []
    for op in ('/', '//', '%'):
        for negative in (False, True):
            if args.extremes and args.divisor == 1 and negative:
                continue  # INT_MIN / -1 is outside the supported signed-int32 domain.
            for renamed in (False, True):
                axis = f'coordinate_{args.seed}' if renamed else 'i'
                value = 'loaded_value' if renamed else 'v'
                k = lambda n: Literal(n, 'int')
                b = lambda o, l, r: BinaryExpr(o, l, r)
                numerator = b('-', Var(axis), k(32))
                if args.extremes:
                    numerator = TernaryExpr(b('<', Var(axis), k(32)),
                        b('+', k(-2147483648), Var(axis)), b('+', k(2147483584), Var(axis)))
                divisor = b('+', k(3), b('%', Var(axis), k(5)))
                if args.divisor is not None:
                    divisor = k(args.divisor)
                if negative:
                    divisor = b('-', k(0), divisor)
                index = b('+', b(op, numerator, divisor), k(64))
                if args.extremes:
                    result = b(op, numerator, divisor)
                    index = b('+', TernaryExpr(b('<', result, k(0)), k(0), k(64)), b('%', result, k(64)))
                graph = Graph.from_dict(json.loads(source.read_text()))
                tile = Tile((Axis(axis, Dim(64)),),
                            (Load(value, 'x', (index,)), Write('sum_1', (Var(axis),), value=value)))
                for node in graph.nodes.values():
                    node.outputs = tuple(replace(t, shape=(Dim(128 if type(node.op).__name__ == 'InputOp' else 64),)) for t in node.outputs)
                    if type(node.op).__name__ == 'KernelOp':
                        node.op = replace(node.op, body=(tile,))
                label = f'{dict(zip(("/", "//", "%"), ("divide", "floor", "mod")))[op]}_{"negative" if negative else "positive"}_s{args.seed}' + ('_renamed' if renamed else '')
                if args.divisor is not None:
                    label += f'_d{args.divisor}'
                if args.extremes:
                    label += '_extremes'
                folder = args.out / label
                folder.mkdir(parents=True, exist_ok=True)
                raw = json.dumps(graph.to_dict(), indent=2, default=str)+'\n'
                Graph.from_dict(json.loads(raw))
                (folder / source.name).write_text(raw)
                meta = json.loads((template/'meta.json').read_text())
                meta.update(id=label, rtol=0, atol=0)
                launch = meta['kernels'][0]
                launch.update(grid=[1,1,1], block=[64,1,1], smem_bytes=0)
                for arg in launch['args']:
                    arg['shape'] = [64 if arg.get('output') else 128]
                    if not arg.get('output'):
                        arg.pop('seed',None)
                        arg['data'] = x.tolist()
                (folder/'meta.json').write_text(json.dumps(meta,indent=2)+'\n')
                n = np.arange(64,dtype=np.int64)-32
                if args.extremes:
                    t = np.arange(64,dtype=np.int64)
                    n = np.where(t < 32, -2147483648+t, 2147483584+t)
                d = 3+np.arange(64,dtype=np.int64)%5
                if args.divisor is not None:
                    d = np.full(64,args.divisor,dtype=np.int64)
                if negative:
                    d = -d
                result = np.remainder(n,d) if op=='%' else np.floor_divide(n,d)
                coordinates = np.where(result < 0,0,64)+np.remainder(result,64) if args.extremes else result+64
                np.savez(folder/'expected.npz', sum_1=x[coordinates])
                cases.append({'problem':label, 'seed':args.seed, 'divisor':args.divisor, 'operator':op, 'extremes':args.extremes,
                              'negative_divisor':negative, 'ir_sha256':hashlib.sha256(raw.encode()).hexdigest(), 'oracle':'NumPy floor_divide/remainder on signed int64; all coordinates in bounds'})
    (args.out/'manifest.json').write_text(json.dumps({'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'cases':cases},indent=2)+'\n')
    print('Generated',len(cases),'signed-index fixtures')


if __name__ == '__main__':
    main()
