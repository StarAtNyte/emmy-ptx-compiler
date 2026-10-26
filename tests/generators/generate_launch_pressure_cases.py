"""Stress live values across a barrier without any reduction accumulators.

The independent oracle is an exact copy. All lanes participate in the barrier;
the loads precede it and the stores follow it. This exposes launch resource
limits that counting Init/Accum statements cannot predict.
"""
import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from emmy.compiler.dim import Dim
from emmy.compiler.graph import Graph
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.kernel.ir import Tile, Sync
from emmy.compiler.ir.stmt import Load, Write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--threads', type=int, nargs='+')
    parser.add_argument('--width', type=int, default=137)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    source = root / 'tests/fixtures/abi-reordered'
    original = next(source.glob('*.ir.json'))
    rows = []
    configurations = ([(threads, args.width) for threads in args.threads]
                      if args.threads else [(256, 160), (384, 192), (512, 160), (768, 128)])
    for threads, width in configurations:
        if not 1 <= threads <= 1024 or width < 1:
            parser.error('threads must be in 1..1024 and width must be positive')
        count = threads * width
        data = np.random.default_rng(args.seed).integers(-16, 17, count).astype(np.float32)
        values = tuple(f'value_{i}' for i in range(width))
        index = (BinaryExpr('*', Var('thread'), Literal(width, 'int')),)
        body = (Load(input='x', index=index, names=values), Sync(),
                Write('sum_1', index, values=values))
        graph = Graph.from_dict(json.loads(original.read_text()))
        for node in graph.nodes.values():
            node.outputs = tuple(replace(t, shape=(Dim(count),)) for t in node.outputs)
            if type(node.op).__name__ == 'KernelOp':
                node.op = replace(node.op, body=(Tile((Axis('thread', Dim(threads)),), body,
                                                     block_threads=threads),))
        name = f'live_{width}_threads_{threads}_s{args.seed}'
        folder = args.out / name
        folder.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(graph.to_dict(), indent=2, default=str) + '\n'
        Graph.from_dict(json.loads(payload))
        (folder / original.name).write_text(payload)
        meta = json.loads((source / 'meta.json').read_text())
        meta.update(id=name, rtol=0, atol=0)
        launch = meta['kernels'][0]
        launch.update(grid=[1, 1, 1], block=[threads, 1, 1], smem_bytes=0)
        for arg in launch['args']:
            arg['shape'] = [count]
            if not arg.get('output'):
                arg.pop('seed', None)
                arg['data'] = data.tolist()
        (folder / 'meta.json').write_text(json.dumps(meta, indent=2) + '\n')
        np.savez(folder / 'expected.npz', sum_1=data)
        rows.append({'folder': str(folder), 'threads': threads, 'live_floats': width,
                     'oracle': 'exact independent seeded input copy',
                     'files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(folder.iterdir()) if p.is_file()}})
    (args.out / 'manifest.json').write_text(json.dumps({
        'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'cases': rows}, indent=2) + '\n')
    print(f'Generated {len(rows)} launch-pressure cases')


if __name__ == '__main__':
    main()
