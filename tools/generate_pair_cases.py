"""Counterexample-oriented checks for moving adjacent shared-memory reads.

Small integer inputs have exact independent NumPy oracles. Negative cases keep
the load body valid while violating a motion/alignment precondition. Renaming
must not affect eligibility. Source bindings shadow only in nested scopes.
"""
import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F32, I32
from emmy.compiler.graph import Graph
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.kernel.ir import Smem, Sync, Tile
from emmy.compiler.ir.stmt import Accum, Assign, Cond, Init, Let, Load, Loop, StridedLoop, Write


VARIANTS = ('adjacent', 'cancellation', 'weak_base', 'odd_offset', 'nonadjacent',
            'index_shadow', 'axis_shadow', 'destination_use', 'destination_update',
            'shared_write', 'barrier', 'global_read', 'nested_weak')


def generate(root, output, variant, seed, trips, renamed):
    rng = np.random.default_rng(seed)
    width, columns = 64, 16
    size = width * columns
    x = rng.integers(-8, 9, size=size).astype(np.float32)
    prefix = f'z{seed}_' if renamed else ''
    n = lambda value: prefix + value
    v = lambda value: Var(n(value))
    k = lambda value: Literal(value, 'int')
    b = lambda op, a, c: BinaryExpr(op, a, c)
    assign = lambda target, op, *args: Assign(n(target), ElementwiseImpl(op), tuple(n(a) for a in args), F32)
    fold = lambda target, value, axes=(): Accum(n(target), n(value), ElementwiseImpl('add'), F32,
                                              axes=tuple(n(a) for a in axes))
    def populate(shift=0):
        return StridedLoop(Axis(n('copy'), Dim(size)), v('thread'), k(width),
            (Load(n('item'), 'x', (b('%', b('+', v('copy'), k(shift)), k(size)),), dtype=F32),
             Write(n('scratch'), (b('/', v('copy'), k(columns)), b('%', v('copy'), k(columns))),
                   value=n('item'), value_dtype=F32)))
    row = v('thread')
    if variant == 'cancellation':
        row = b('-', b('+', row, v('noise')), v('noise'))
    col = b('*', v('step'), k(2))
    if variant == 'odd_offset':
        col = b('+', col, k(1))
    elif variant == 'index_shadow':
        col = b('*', v('index'), k(2))
    first = Load(n('a'), n('scratch'), (row, col), dtype=F32)
    second_col = b('+', col, k(2 if variant == 'nonadjacent' else 1))
    middle = [assign('scaled', 'multiply', 'a', 'two')]
    if variant == 'index_shadow':
        middle.append(Let(n('index'), k(2), I32))
    elif variant == 'axis_shadow':
        middle.append(Let(n('step'), k(2), I32))
    elif variant == 'destination_use':
        middle.append(assign('early', 'copy', 'late'))
    elif variant == 'destination_update':
        middle.extend((fold('late', 'a'), assign('early', 'copy', 'late')))
    elif variant == 'shared_write':
        middle.append(Write(n('scratch'), (row, second_col), value=n('a'), value_dtype=F32))
    elif variant == 'barrier':
        middle.append(Sync())
    elif variant == 'global_read':
        middle.append(Load(n('global'), 'x', (v('thread'),), dtype=F32))
    second = Load(n('late'), n('scratch'), (row, second_col), dtype=F32)
    tail = [assign('sum', 'add', 'scaled', 'late')]
    if variant in ('destination_use', 'destination_update'):
        tail.append(assign('sum_extra', 'add', 'sum', 'early'))
    elif variant == 'global_read':
        tail.append(assign('sum_extra', 'add', 'sum', 'global'))
    result = 'sum_extra' if variant in ('destination_use', 'destination_update', 'global_read') else 'sum'
    tail.append(fold('result', result, ('step',)))
    loop = Loop(Axis(n('step'), Dim(trips)), (first, *middle, second, *tail), seed=False)
    body = [Smem(n('scratch'), (width, columns), align=4 if variant == 'weak_base' else 16),
            populate(), Sync(), Let(n('two'), Literal(2.0), F32),
            Let(n('noise'), b('%', v('thread'), k(3)), I32), Let(n('index'), k(0), I32),
            Init(n('late'), 7.0, dtype=F32), Init(n('result'), 0.0, dtype=F32)]
    if variant == 'nested_weak':
        nested = (Smem(n('scratch'), (width, columns), align=4), populate(1), Sync(), loop)
        body.append(Cond(b('==', k(1), k(1)), nested))
    body.extend((loop, Write('sum_1', (v('thread'),), value=n('result'), value_dtype=F32)))
    expected = np.zeros(width, dtype=np.float32)
    def reference(shift):
        total = np.zeros(width, dtype=np.float32)
        carried = np.full(width, 7, dtype=np.float32)
        for step in range(trips):
            c = 0 if variant == 'index_shadow' else 2 * step + int(variant == 'odd_offset')
            a = x[(np.arange(width) * columns + c + shift) % size]
            second_c = 5 if variant in ('index_shadow', 'axis_shadow') else c + (2 if variant == 'nonadjacent' else 1)
            late = a if variant == 'shared_write' else x[(np.arange(width) * columns + second_c + shift) % size]
            value = 2 * a + late
            if variant == 'destination_use':
                value += 7
            elif variant == 'destination_update':
                carried += a
                value += carried
            elif variant == 'global_read':
                value += x[:width]
            total += value
        return total
    expected += reference(0)
    if variant == 'nested_weak':
        expected += reference(1)
    template = root / 'challenge/problems/r01_array_sum'
    graph = Graph.from_dict(json.loads((template / 'k_sum_1_reduce.ir.json').read_text()))
    tile = Tile((Axis(n('thread'), Dim(width)),), tuple(body), block_threads=width)
    for key, node in list(graph.nodes.items()):
        extent = size if type(node.op).__name__ == 'InputOp' else width
        graph.nodes[key] = replace(node, outputs=tuple(replace(t, shape=(Dim(extent),)) for t in node.outputs),
                                  op=replace(node.op, body=(tile,)) if type(node.op).__name__ == 'KernelOp' else node.op)
    label = f'{variant}_s{seed}_t{trips}' + ('_renamed' if renamed else '')
    folder = output / label
    folder.mkdir(parents=True, exist_ok=True)
    ir = json.dumps(graph.to_dict(), indent=2, default=str) + '\n'
    Graph.from_dict(json.loads(ir))
    (folder / 'k_sum_1_reduce.ir.json').write_text(ir)
    meta = json.loads((template / 'meta.json').read_text())
    meta.update(id=label, rtol=0, atol=0)
    meta.pop('code', None)
    kernel = meta['kernels'][0]
    kernel.update(grid=[1, 1, 1], block=[width, 1, 1], smem_bytes=0)
    for arg in kernel['args']:
        arg['shape'] = [width if arg.get('output') else size]
        if not arg.get('output'):
            arg.pop('seed', None)
            arg['data'] = x.tolist()
    (folder / 'meta.json').write_text(json.dumps(meta, indent=2) + '\n')
    np.savez(folder / 'expected.npz', sum_1=expected)
    eligible = variant in ('adjacent', 'cancellation', 'nested_weak')
    spec = {'variant': variant, 'seed': seed, 'trips': trips, 'renamed': renamed,
            'pair_expected': eligible, 'pairs_exact': (1 if eligible else 0) if trips == 1 else None,
            'oracle': 'independent exact small-integer NumPy formula',
            'ir_sha256': hashlib.sha256(ir.encode()).hexdigest()}
    (folder / 'case.json').write_text(json.dumps(spec, indent=2) + '\n')
    return spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('artifacts/pair-cases'))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--trips', type=int, default=1)
    args = parser.parse_args()
    if args.seed < 0 or not 1 <= args.trips <= 7:
        parser.error('seed must be nonnegative and trips must be in 1..7')
    root = Path(__file__).resolve().parents[1]
    specs = [generate(root, args.out, variant, args.seed, args.trips, renamed)
             for variant in VARIANTS for renamed in (False, True)]
    (args.out / 'manifest.json').write_text(json.dumps({
        'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'cases': specs}, indent=2) + '\n')
    print(f'Generated {len(specs)} pair-motion cases in {args.out}')


if __name__ == '__main__':
    main()
