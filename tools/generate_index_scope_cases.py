"""Adversarial range-proof scopes, negative loops and signed overflow probes."""
import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from emmy.compiler.dtype import I32
from emmy.compiler.dim import Dim
from emmy.compiler.graph import Graph
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import BinaryExpr, Literal, TernaryExpr, Var
from emmy.compiler.ir.kernel.ir import Tile
from emmy.compiler.ir.stmt import Cond, Let, Load, Loop, StridedLoop, Write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('artifacts/index-scopes'))
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    template = root / 'challenge/problems/r01_array_sum'
    source = template / 'k_sum_1_reduce.ir.json'
    rng = np.random.default_rng(args.seed)
    x = rng.integers(-100, 101, size=128).astype(np.float32)
    cases = []
    for variant in ('branch_shadow', 'loop_shadow', 'negative_stride', 'divisor_shadow', 'overflow', 'ternary'):
        for renamed in (False, True):
            axis = f'coordinate_{args.seed}' if renamed else 'i'
            value = 'loaded_value' if renamed else 'v'
            saved = 'saved_coordinate'
            k = lambda n: Literal(n, 'int')
            b = lambda o, l, r: BinaryExpr(o, l, r)
            numerator = b('-', Var(saved), k(32))
            divisor = k(7)
            prelude = (Let(saved, Var(axis), I32),)
            if variant in ('branch_shadow', 'loop_shadow'):
                numerator = Var(axis)
                inner = (Let(axis, b('-', Var(saved), k(32)), I32),)
            elif variant == 'negative_stride':
                numerator = b('+', Var('j'), Var(saved))
            elif variant == 'divisor_shadow':
                divisor = Var('d')
                prelude += (Let('d', k(7), I32),)
                inner = (Let('d', k(-7), I32),)
            elif variant == 'overflow':
                numerator = b('+', Var(saved), k(2147483640))
            elif variant == 'ternary':
                numerator = TernaryExpr(b('<', Var(saved), k(32)), numerator, Var(saved))
            index = b('+', b('%', numerator, divisor), k(64))
            memory = (Load(value, 'x', (index,)), Write('sum_1', (Var(saved),), value=value))
            if variant in ('branch_shadow', 'divisor_shadow'):
                body = (Cond(b('==', k(1), k(1)), inner+memory),)
            elif variant == 'loop_shadow':
                body = (Loop(Axis(axis, Dim(1)), inner+memory),)
            elif variant == 'negative_stride':
                body = (StridedLoop(Axis('j', Dim(64)), k(-1), k(1), memory, end=k(0)),)
            else:
                body = memory
            # The second half observes restored outer bindings after scopes.
            post = (Load('post', 'x', (b('+', b('%', Var(axis), k(7)), k(64)),)),
                    Write('sum_1', (b('+', Var(saved), k(64)),), value='post'))
            graph = Graph.from_dict(json.loads(source.read_text()))
            tile = Tile((Axis(axis, Dim(64)),), prelude+body+post)
            for node in graph.nodes.values():
                node.outputs = tuple(replace(t, shape=(Dim(128),)) for t in node.outputs)
                if type(node.op).__name__ == 'KernelOp':
                    node.op = replace(node.op, body=(tile,))
            label = variant+f'_s{args.seed}'+('_renamed' if renamed else '')
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
                arg['shape'] = [128]
                if not arg.get('output'):
                    arg.pop('seed',None)
                    arg['data'] = x.tolist()
            (folder/'meta.json').write_text(json.dumps(meta,indent=2)+'\n')
            t = np.arange(64, dtype=np.int64)
            n = t-32
            d = 7
            if variant == 'negative_stride':
                n = t-1
            elif variant == 'divisor_shadow':
                d = -7
            elif variant == 'overflow':
                n = (t+2147483640).astype(np.int32).astype(np.int64)
            elif variant == 'ternary':
                n = np.where(t < 32, t-32, t)
            coordinates = np.concatenate((np.remainder(n, d)+64, np.remainder(t, 7)+64))
            np.savez(folder/'expected.npz', sum_1=x[coordinates])
            cases.append({'problem':label, 'ir_sha256':hashlib.sha256(raw.encode()).hexdigest(), 'oracle':'NumPy floor_divide/remainder on signed int64; all coordinates in bounds'})
    (args.out/'manifest.json').write_text(json.dumps({'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'cases':cases},indent=2)+'\n')
    print('Generated',len(cases),'index-scope fixtures')


if __name__ == '__main__':
    main()
