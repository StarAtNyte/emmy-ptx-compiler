"""Strided end/step evaluation probes with bounded independent loop references."""
import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from emmy.compiler.dtype import I32
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.dim import Dim
from emmy.compiler.graph import Graph
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import BinaryExpr, Literal, TernaryExpr, Var
from emmy.compiler.ir.kernel.ir import Tile
from emmy.compiler.ir.stmt import Accum, Let, Load, StridedLoop, Write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('artifacts/strided-bounds'))
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    template = root / 'challenge/problems/r01_array_sum'
    source = template / 'k_sum_1_reduce.ir.json'
    rng = np.random.default_rng(args.seed)
    x = rng.integers(-100, 101, size=128).astype(np.float32)
    cases = []
    for variant in ('end_axis', 'step_axis', 'end_and_step', 'end_alias'):
        for renamed in (False, True):
            axis = f'coordinate_{args.seed}' if renamed else 'i'
            inner = 'cursor_renamed' if renamed else 'j'
            value = 'loaded_value' if renamed else 'v'
            saved = 'saved_coordinate'
            k = lambda n: Literal(n, 'int')
            b = lambda o, l, r: BinaryExpr(o, l, r)
            start = b('%', Var(axis), k(3))
            rem = b('%', Var(inner), k(3))
            dynamic_step = b('+', b('*', rem, rem), k(1))
            step = k(1) if variant == 'end_axis' else dynamic_step
            if variant == 'end_alias':
                step = b('+', b('%', Var(inner+'_end'), k(3)), k(1))
            end = k(9) if variant == 'step_axis' else b('+', rem, k(4))
            load_index = Var(inner+'_end') if variant == 'end_alias' else Var(inner)
            memory = (Load(value, 'x', (load_index,)),
                      Accum('total', value, ElementwiseImpl('add'), axes=(inner,)))
            body = (Let(saved, Var(axis), I32),
                    StridedLoop(Axis(inner, Dim(128)), start, step, memory, end=end),
                    Write('sum_1', (Var(saved),), value='total'))
            post = (Load('post', 'x', (Var(axis),)),
                    Write('sum_1', (b('+', Var(saved), k(64)),), value='post'))
            graph = Graph.from_dict(json.loads(source.read_text()))
            tile = Tile((Axis(axis, Dim(64)),), body+post)
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
            first = []
            for thread in range(64):
                cursor = thread % 3
                bound = 9 if variant == 'step_axis' else cursor % 3 + 4
                total = np.float32(0)
                while cursor < bound:
                    total = np.float32(total+x[bound if variant == 'end_alias' else cursor])
                    cursor += (bound % 3 + 1) if variant == 'end_alias' else 1 if variant == 'end_axis' else (cursor % 3)**2+1
                first.append(total)
            np.savez(folder/'expected.npz', sum_1=np.concatenate((np.array(first,dtype=np.float32),x[:64])))
            cases.append({'problem':label, 'ir_sha256':hashlib.sha256(raw.encode()).hexdigest(), 'oracle':'Independent bounded Python cursor recurrence and exact small-integer f32 sums'})
    (args.out/'manifest.json').write_text(json.dumps({'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'cases':cases},indent=2)+'\n')
    print('Generated',len(cases),'strided-bound fixtures')


if __name__ == '__main__':
    main()
