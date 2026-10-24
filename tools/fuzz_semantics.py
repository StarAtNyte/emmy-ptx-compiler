"""Seeded typed-IR lexical-scope tests with independent NumPy formulas.

No Emmy rendering or compiler execution is used to compute expected results.
Each seed has a name-renamed counterpart. --minimal removes optional arithmetic
and reduces extents, retaining the family and lexical binding interaction.
Cases remain on disk for differential runs with --compiler in local_runner.py.
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
from emmy.compiler.ir.kernel.ir import Smem, Sync, Tile, TreeHalve, WarpShuffle
from emmy.compiler.ir.stmt import Accum, Assign, Cond, Init, Let, Load, Loop, Write


FAMILIES = ('branch_value', 'loop_value', 'branch_index', 'loop_axis', 'mutable_snapshot',
            'conditional_warp', 'conditional_tree')


def generate(root, output, seed, renamed=False, minimal=False):
    rng = np.random.default_rng(seed)
    size = 2 if minimal else int(rng.choice([17, 31, 32, 33, 63, 65, 97, 129]))
    trips = 1 if minimal else int(rng.integers(1, 8))
    shift = 1 if minimal else int(rng.integers(1, size))
    divisor = 2 if minimal else int(rng.integers(2, 6))
    family = FAMILIES[seed % len(FAMILIES)]
    if family in ('conditional_warp', 'conditional_tree'):
        size = 64
    # Small integers make these addition-only references exact in f32. This
    # isolates binding mistakes from contraction and reduction-order differences.
    values = rng.integers(-8, 9, size=size).astype(np.float32)
    if minimal:
        values = np.arange(size, dtype=np.float32)
    prefix = f'z{seed}_' if renamed else ''
    name = lambda value: prefix + value
    var = lambda value: Var(name(value))
    lit = lambda value: Literal(value, 'int')
    expr = lambda op, a, b: BinaryExpr(op, a, b)
    read = lambda target, index: Load(name(target), 'x', (index,), dtype=F32)
    fold = lambda value, axes=('step',): Accum(name('acc'), name(value), ElementwiseImpl('add'), F32,
                                             axes=tuple(name(axis) for axis in axes))
    body = [Let(name('destination'), var('thread'), I32)]
    noise = () if minimal else (Assign(name('unused'), ElementwiseImpl('multiply'),
                                      (name('value'), name('value')), F32),)
    indices = np.arange(size)
    if family == 'branch_value':
        body += [read('value', var('thread')), Init(name('acc'), 0.0, dtype=F32),
                 Cond(expr('==', expr('%', var('thread'), lit(divisor)), lit(0)),
                      (read('value', expr('%', expr('+', var('thread'), lit(shift)), lit(size))),
                       *noise, fold('value', ())), (fold('value', ()),)),
                 Assign(name('result'), ElementwiseImpl('add'), (name('value'), name('acc')), F32)]
        expected = values + np.where(indices % divisor == 0, np.roll(values, -shift), values)
    elif family == 'loop_value':
        body += [read('value', var('thread')),
                 Loop(Axis(name('step'), Dim(trips)),
                      (read('value', expr('%', expr('+', expr('+', var('thread'), var('step')), lit(shift)), lit(size))),
                       *noise, fold('value'))),
                 Assign(name('result'), ElementwiseImpl('add'), (name('value'), name('acc')), F32)]
        expected = values + sum((np.roll(values, -step-shift) for step in range(trips)), np.zeros(size, np.float32))
    elif family == 'branch_index':
        body += [Let(name('index'), var('thread'), I32),
                 Cond(expr('==', expr('%', var('thread'), lit(divisor)), lit(0)),
                      (Let(name('index'), expr('%', expr('+', var('thread'), lit(shift)), lit(size)), I32),)),
                 read('result', var('index'))]
        expected = values.copy()
    elif family == 'loop_axis':
        # A nested loop may shadow a tile's axis; the saved destination keeps
        # all GPU output writes disjoint even in the buggy lowering.
        body += [Loop(Axis(name('thread'), Dim(trips)),
                      (read('value', var('thread')),
                       Accum(name('acc'), name('value'), ElementwiseImpl('add'), F32,
                             axes=(name('thread'),)))),
                 read('value', var('thread')),
                 Assign(name('result'), ElementwiseImpl('add'), (name('value'), name('acc')), F32)]
        expected = values + values[:trips].sum(dtype=np.float32)
    elif family == 'mutable_snapshot':
        body += [Init(name('acc'), 3.0, dtype=F32),
                 Let(name('snapshot'), var('acc'), F32),
                 Loop(Axis(name('step'), Dim(trips)),
                      (read('value', expr('%', expr('+', var('step'), lit(1)), lit(size))),
                       fold('value')), seed=False),
                 Assign(name('result'), ElementwiseImpl('copy'), (name('snapshot'),), F32)]
        expected = np.full(size, 3, dtype=np.float32)
    elif family == 'conditional_warp':
        merge = (Assign(name('acc'), ElementwiseImpl('add'), (name('acc'), name('other')), F32),)
        body += [read('acc', var('thread')),
                 Cond(expr('==', expr('/', var('thread'), lit(32)), lit(0)),
                      (WarpShuffle((name('acc'),), (name('other'),), merge, 32, F32),)),
                 Assign(name('result'), ElementwiseImpl('copy'), (name('acc'),), F32)]
        expected = values.copy()
        expected[:32] = values[:32].sum(dtype=np.float32)
    else:
        length = 2 if minimal else int(rng.choice([2, 4, 8, 16, 32]))
        observed = minimal or bool(rng.integers(0, 2))
        merge = (Assign(name('acc'), ElementwiseImpl('add'), (name('acc'), name('other')), F32),)
        body += [Smem(name('scratch'), (length,), align=16),
                 Cond(expr('<', var('thread'), lit(length)),
                      (read('leaf', var('thread')),
                       Write(name('scratch'), (var('thread'),), value=name('leaf'), value_dtype=F32))), Sync(),
                 read('acc', var('thread')),
                 Cond(expr('<', var('thread'), lit(32)),
                      (TreeHalve((name('scratch'),), (name('acc'),), (name('other'),), merge,
                                 length, name('thread'), F32, barrier_id=1, barrier_count=32),)), Sync()]
        expected = values.copy()
        expected[:32] = values[:length].sum(dtype=np.float32)
        if observed:
            body += [Load(name('root'), name('scratch'), (lit(0),), dtype=F32),
                     Assign(name('result'), ElementwiseImpl('add'), (name('acc'), name('root')), F32)]
            expected += values[:length].sum(dtype=np.float32)
        else:
            body.append(Assign(name('result'), ElementwiseImpl('copy'), (name('acc'),), F32))
    body.append(Write('sum_1', (var('destination'),), value=name('result'), value_dtype=F32))
    cooperative = family in ('conditional_warp', 'conditional_tree')
    block = 64 if cooperative else 32
    tile = Tile((Axis(name('thread'), Dim(size)),), tuple(body), block_threads=block if cooperative else None)
    template = root / 'challenge/problems/r01_array_sum'
    graph = Graph.from_dict(json.loads((template / 'k_sum_1_reduce.ir.json').read_text()))
    for key, node in list(graph.nodes.items()):
        graph.nodes[key] = replace(node, outputs=tuple(replace(t, shape=(Dim(size),)) for t in node.outputs),
                                  op=replace(node.op, body=(tile,)) if type(node.op).__name__ == 'KernelOp' else node.op)
    label = f'{family}_s{seed}' + ('_renamed' if renamed else '') + ('_minimal' if minimal else '')
    folder = output / label
    folder.mkdir(parents=True, exist_ok=True)
    ir = json.dumps(graph.to_dict(), indent=2, default=str) + '\n'
    # Require a genuine typed serialization roundtrip before calling this a case.
    Graph.from_dict(json.loads(ir))
    (folder / 'k_sum_1_reduce.ir.json').write_text(ir)
    meta = json.loads((template / 'meta.json').read_text())
    meta.update(id=label, rtol=0, atol=0)
    meta.pop('code', None)
    kernel = meta['kernels'][0]
    kernel.update(grid=[(size + block - 1) // block, 1, 1], block=[block, 1, 1])
    for arg in kernel['args']:
        arg['shape'] = [size]
        if not arg.get('output'):
            arg.pop('seed', None)
            arg['data'] = values.tolist()
    (folder / 'meta.json').write_text(json.dumps(meta, indent=2) + '\n')
    np.savez(folder / 'expected.npz', sum_1=np.asarray(expected, dtype=np.float32))
    spec = {'seed': seed, 'family': family, 'renamed': renamed, 'minimal': minimal,
            'size': size, 'trips': trips, 'shift': shift, 'divisor': divisor,
            'oracle': 'independent NumPy formula; exact small-integer f32 arithmetic',
            'ir_sha256': hashlib.sha256(ir.encode()).hexdigest()}
    (folder / 'case.json').write_text(json.dumps(spec, indent=2) + '\n')
    return spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('artifacts/fuzz-scopes'))
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--count', type=int, default=24)
    parser.add_argument('--minimal', action='store_true')
    args = parser.parse_args()
    if args.start < 0 or args.count < 1:
        parser.error('start must be nonnegative; count must be positive')
    root = Path(__file__).resolve().parents[1]
    specs = [generate(root, args.out, seed, renamed, args.minimal)
             for seed in range(args.start, args.start + args.count) for renamed in (False, True)]
    (args.out / 'manifest.json').write_text(json.dumps({
        'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'cases': specs}, indent=2) + '\n')
    print(f'Generated {len(specs)} typed lexical-scope cases in {args.out}')


if __name__ == '__main__':
    main()
