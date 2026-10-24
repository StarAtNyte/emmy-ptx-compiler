"""Nested shared-buffer and carried-state probes with independent exact oracles.

Names may shadow across lexical blocks. Static and explicit 96 KiB dynamic
shared-memory variants exercise both allocation paths. No renderer/compiler is
used for expected results. Reduced mode shrinks parameters, not arbitrary IR.
"""
import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F32
from emmy.compiler.graph import Graph
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.kernel.ir import Smem, Sync, Tile
from emmy.compiler.ir.stmt import Accum, Assign, Cond, Init, Let, Load, Loop, StridedLoop, Write


FAMILIES = ('shared_branch', 'shared_loop', 'carried_seed', 'outer_through_inner')


def generate(root, output, seed, renamed=False, minimal=False, dynamic=False):
    rng = np.random.default_rng(seed)
    family = FAMILIES[seed % len(FAMILIES)]
    width = 64
    size = 12288 if dynamic and family.startswith('shared_') else 64
    values = rng.integers(-8, 9, size=size).astype(np.float32)
    if minimal:
        values = np.arange(size, dtype=np.float32) % 17 + 1
    trips = 1 if minimal else int(rng.integers(1, 6))
    inner_trips = 1 if minimal else int(rng.integers(1, 5))
    shift = 1 if minimal else int(rng.integers(1, size))
    prefix = f'renamed_{seed}_' if renamed else ''
    n = lambda value: prefix + value
    v = lambda value: Var(n(value))
    k = lambda value: Literal(value, 'int')
    b = lambda op, left, right: BinaryExpr(op, left, right)
    fold = lambda target, value, axis, op='add': Accum(n(target), n(value), ElementwiseImpl(op),
                                                      F32, axes=(n(axis),) if axis else ())
    read = lambda target, index: Load(n(target), 'x', (index,), dtype=F32)
    body = []
    shared_bytes = 0
    if family.startswith('shared_'):
        outer_columns, inner_columns = (64, 128) if dynamic else (8, 16)
        probe_stride = size // width + 1 if dynamic else 1
        probe = b('%', b('*', v('thread'), k(probe_stride)), k(size))
        coordinate = lambda flat, columns: (b('/', flat, k(columns)), b('%', flat, k(columns)))
        def copy(columns, offset):
            axis = v('copy')
            return StridedLoop(Axis(n('copy'), Dim(size)), v('thread'), k(width),
                (read('item', b('%', b('+', axis, offset), k(size))),
                 Write(n('scratch'), coordinate(axis, columns), value=n('item'), value_dtype=F32)))
        body += [Smem(n('scratch'), (size // outer_columns, outer_columns), align=16),
                 copy(outer_columns, k(0)), Sync(), Init(n('captured'), 0.0, dtype=F32)]
        offset = k(shift) if family == 'shared_branch' else b('+', v('phase'), k(shift))
        nested = (Smem(n('scratch'), (size // inner_columns, inner_columns), align=16),
                  copy(inner_columns, offset), Sync(),
                  Load(n('inner_value'), n('scratch'), coordinate(probe, inner_columns), dtype=F32),
                  fold('captured', 'inner_value', None), Sync())
        indices = (np.arange(width) * probe_stride) % size
        if family == 'shared_branch':
            enabled = minimal or bool(rng.integers(0, 2))
            body.append(Cond(b('==', k(int(enabled)), k(1)), nested))
            extra = values[(indices + shift) % size] if enabled else np.zeros(width, np.float32)
        else:
            body.append(Loop(Axis(n('phase'), Dim(trips)), nested, seed=False))
            extra = sum((values[(indices + shift + phase) % size] for phase in range(trips)),
                        np.zeros(width, np.float32))
        body += [Load(n('outer_value'), n('scratch'), coordinate(probe, outer_columns), dtype=F32),
                 Assign(n('result'), ElementwiseImpl('add'), (n('outer_value'), n('captured')), F32)]
        expected = values[indices] + extra
        shared_bytes = size * 8 if dynamic else 0
    elif family == 'carried_seed':
        inner = Loop(Axis(n('phase'), Dim(inner_trips)),
                     (read('inner_value', v('thread')),
                      fold('acc', 'inner_value', 'phase', 'maximum')))
        body += [Loop(Axis(n('phase'), Dim(trips)),
                      (read('outer_value', v('thread')), fold('acc', 'outer_value', 'phase'), inner)),
                 Assign(n('result'), ElementwiseImpl('copy'), (n('acc'),), F32)]
        expected = values * trips
    else:
        # A nested loop whose fold belongs to the OUTER axis updates a live
        # external state; skipping all nested loops during state analysis is wrong.
        body += [Init(n('acc'), 0.0, dtype=F32),
                 Loop(Axis(n('phase'), Dim(trips)),
                      (Loop(Axis(n('inner'), Dim(inner_trips)),
                            (read('value', v('thread')), fold('acc', 'value', 'phase'))),), seed=False),
                 Assign(n('result'), ElementwiseImpl('copy'), (n('acc'),), F32)]
        expected = values * trips * inner_trips
    body.append(Write('sum_1', (v('thread'),), value=n('result'), value_dtype=F32))
    tile = Tile((Axis(n('thread'), Dim(width)),), tuple(body), block_threads=width)
    template = root / 'challenge/problems/r01_array_sum'
    graph = Graph.from_dict(json.loads((template / 'k_sum_1_reduce.ir.json').read_text()))
    for key, node in list(graph.nodes.items()):
        extent = size if type(node.op).__name__ == 'InputOp' else width
        graph.nodes[key] = replace(node, outputs=tuple(replace(t, shape=(Dim(extent),)) for t in node.outputs),
                                  op=replace(node.op, body=(tile,)) if type(node.op).__name__ == 'KernelOp' else node.op)
    label = f'{family}_s{seed}' + ('_renamed' if renamed else '') + ('_minimal' if minimal else '') + ('_dynamic' if dynamic else '')
    folder = output / label
    folder.mkdir(parents=True, exist_ok=True)
    ir = json.dumps(graph.to_dict(), indent=2, default=str) + '\n'
    Graph.from_dict(json.loads(ir))
    (folder / 'k_sum_1_reduce.ir.json').write_text(ir)
    meta = json.loads((template / 'meta.json').read_text())
    meta.update(id=label, rtol=0, atol=0)
    meta.pop('code', None)
    kernel = meta['kernels'][0]
    kernel.update(grid=[1, 1, 1], block=[width, 1, 1], smem_bytes=shared_bytes)
    for arg in kernel['args']:
        arg['shape'] = [width if arg.get('output') else size]
        if not arg.get('output'):
            arg.pop('seed', None)
            arg['data'] = values.tolist()
    (folder / 'meta.json').write_text(json.dumps(meta, indent=2) + '\n')
    np.savez(folder / 'expected.npz', sum_1=np.asarray(expected, dtype=np.float32))
    spec = {'seed': seed, 'family': family, 'renamed': renamed, 'minimal': minimal,
            'dynamic': dynamic, 'size': size, 'trips': trips, 'inner_trips': inner_trips,
            'shift': shift, 'smem_bytes': shared_bytes,
            'oracle': 'independent exact small-integer NumPy formulas',
            'ir_sha256': hashlib.sha256(ir.encode()).hexdigest()}
    (folder / 'case.json').write_text(json.dumps(spec, indent=2) + '\n')
    return spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('artifacts/fuzz-nested'))
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--count', type=int, default=16)
    parser.add_argument('--minimal', action='store_true')
    parser.add_argument('--dynamic', action='store_true')
    args = parser.parse_args()
    if args.start < 0 or args.count < 1:
        parser.error('start must be nonnegative; count must be positive')
    root = Path(__file__).resolve().parents[1]
    specs = [generate(root, args.out, seed, renamed, args.minimal, args.dynamic)
             for seed in range(args.start, args.start + args.count) for renamed in (False, True)]
    (args.out / 'manifest.json').write_text(json.dumps({
        'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'cases': specs}, indent=2) + '\n')
    print(f'Generated {len(specs)} nested cases in {args.out}')


if __name__ == '__main__':
    main()
