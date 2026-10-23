"""Translate typed Emmy Kernel IR directly to PTX.

The compiler never invokes Emmy's CUDA renderer or an existing compiler.
"""
from __future__ import annotations

import json
import math
import struct
import sys
from collections import Counter
from pathlib import Path

from emmy.compiler.graph import Graph
from emmy.compiler.ir.expr import BinaryExpr, Literal, TernaryExpr, Var
from emmy.compiler.ir.kernel.ir import CpAsyncCommit, CpAsyncCopy, CpAsyncWait, Smem, Sync, Tile, TreeHalve, WarpShuffle
from emmy.compiler.ir.stmt import Accum, Assign, Cond, Init, Let, Load, Loop, Select, StridedLoop, Write


class Compiler:
    def __init__(self, graph, launch):
        self.graph = graph
        self.launch = launch
        self.node = next(n for n in graph.nodes.values() if type(n.op).__name__ == 'KernelOp')
        self.lines = []
        self.counts = Counter()
        self.values = {}
        self.pointers = {}
        self.shapes = {}
        self.label_count = 0
        self.shared = {}
        self.shared_offsets = {}
        self.identities = {}
        def collect_shared(body):
            for stmt in body:
                if isinstance(stmt, Smem) and stmt.name not in self.shared_offsets:
                    alignment = max(4, stmt.align)
                    current = self.shared_size
                    offset = (current + alignment - 1) // alignment * alignment
                    self.shared_offsets[stmt.name] = offset
                    self.shared_size = offset + 4 * math.prod(stmt.extents)
                if hasattr(stmt, 'body'):
                    collect_shared(stmt.body)
                if hasattr(stmt, 'else_body'):
                    collect_shared(stmt.else_body)
        self.shared_size = 0
        collect_shared(self.node.op.body)
        self.dynamic_shared = self.shared_size > 48 * 1024
        for node in graph.nodes.values():
            for tensor in node.outputs:
                self.shapes[tensor.name] = tuple(d.value for d in tensor.shape)

    def reg(self, kind='r'):
        self.counts[kind] += 1
        return f'%{kind}{self.counts[kind]}'

    def emit(self, line):
        self.lines.append('    ' + line)

    def label(self, prefix):
        self.label_count += 1
        return f'{prefix}_{self.label_count}'

    @staticmethod
    def floating(value):
        return '0f' + struct.pack('>f', float(value)).hex().upper()

    def accumulators(self, body, axis):
        """Find folds carried by this axis, including those below conditionals."""
        found = {}
        for stmt in body:
            if isinstance(stmt, Accum) and axis in stmt.axes:
                found[stmt.name] = stmt
            for field in ('body', 'else_body'):
                if hasattr(stmt, field):
                    found.update(self.accumulators(getattr(stmt, field), axis))
        return found

    def instruction(self, opcode, *args, kind='r'):
        result = self.reg(kind)
        self.emit(f'{opcode} {result}, ' + ', '.join(map(str, args)) + ';')
        return result

    def integer(self, expr):
        if isinstance(expr, TernaryExpr):
            predicate = self.predicate(expr.cond)
            return self.instruction('selp.b32', self.integer(expr.if_true), self.integer(expr.if_false), predicate)
        if isinstance(expr, Var):
            return self.values[expr.name]
        if isinstance(expr, Literal):
            return str(int(expr.value))
        if isinstance(expr, (int, float)):
            return str(int(expr))
        if isinstance(expr, BinaryExpr):
            if expr.op in ('&&', '||'):
                return self.instruction('and.pred' if expr.op == '&&' else 'or.pred',
                                        self.predicate(expr.left), self.predicate(expr.right), kind='p')
            left, right = self.integer(expr.left), self.integer(expr.right)
            comparisons = {'==': 'eq', '!=': 'ne', '<': 'lt', '<=': 'le', '>': 'gt', '>=': 'ge'}
            if expr.op in comparisons:
                return self.instruction(f'setp.{comparisons[expr.op]}.s32', left, right, kind='p')
            opcode = {'+': 'add.s32', '-': 'sub.s32', '*': 'mul.lo.s32',
                      '/': 'div.s32', '//': 'div.s32', '%': 'rem.s32'}[expr.op]
            return self.instruction(opcode, left, right)
        raise NotImplementedError(f'Index expression: {expr!r}')

    def predicate(self, expr):
        value = self.integer(expr)
        return value if value.startswith('%p') else self.instruction('setp.ne.s32', value, 0, kind='p')

    def address(self, buffer, index, offset=0):
        shape = self.shapes[buffer]
        flat = '0'
        for coordinate, extent in zip(index, shape):
            flat = self.instruction('mul.lo.s32', flat, extent)
            flat = self.instruction('add.s32', flat, self.integer(coordinate))
        if offset:
            flat = self.instruction('add.s32', flat, offset)
        if buffer in self.shared:
            byte_offset = self.instruction('mul.lo.s32', flat, 4)
            return self.instruction('add.u32', self.pointers[buffer], byte_offset)
        byte_offset = self.instruction('mul.wide.s32', flat, 4, kind='rd')
        return self.instruction('add.u64', self.pointers[buffer], byte_offset, kind='rd')

    def arithmetic(self, op, args):
        if op == 'copy':
            return self.instruction('mov.f32', args[0], kind='f')
        opcode = {'add': 'add.rn.f32', 'subtract': 'sub.rn.f32',
                  'multiply': 'mul.rn.f32', 'divide': 'div.rn.f32',
                  'maximum': 'max.f32', 'minimum': 'min.f32',
                  'negative': 'neg.f32', 'sqrt': 'sqrt.rn.f32',
                  'tanh': 'tanh.approx.f32', 'absolute': 'abs.f32', 'abs': 'abs.f32',
                  'rsqrt': 'rsqrt.approx.f32', 'reciprocal': 'rcp.rn.f32'}
        if op == 'relu':
            return self.instruction('max.f32', args[0], '0f00000000', kind='f')
        if op in ('sigmoid', 'silu'):
            negative = self.instruction('neg.f32', args[0], kind='f')
            exponential = self.arithmetic('exp', [negative])
            denominator = self.instruction('add.rn.f32', exponential, self.floating(1), kind='f')
            sigmoid = self.instruction('rcp.rn.f32', denominator, kind='f')
            return sigmoid if op == 'sigmoid' else self.instruction('mul.rn.f32', args[0], sigmoid, kind='f')
        if op in ('exp', 'exp_fast'):
            scaled = self.instruction('mul.rn.f32', args[0], self.floating(math.log2(math.e)), kind='f')
            return self.instruction('ex2.approx.f32', scaled, kind='f')
        return self.instruction(opcode[op], *args, kind='f')

    def combine(self, stmt):
        """Apply a reduction's merge program with its proven neutral state.

        In an online softmax, merging two empty lanes would otherwise evaluate
        exp(-inf - -inf). Returning the other state for an identity operand
        enforces the reduction identity without changing ordinary scalar exp.
        """
        left = [self.values[name] for name in stmt.state]
        right = [self.values[name] for name in stmt.state_b]
        neutral = all(name in self.identities for name in stmt.state)
        empty = []
        if neutral:
            for values in (left, right):
                test = None
                for name, value in zip(stmt.state, values):
                    equality = self.instruction('setp.eq.f32', value, self.floating(self.identities[name]), kind='p')
                    test = equality if test is None else self.instruction('and.pred', test, equality, kind='p')
                empty.append(test)
        for child in stmt.combine_states:
            self.statement(child)
        if neutral:
            for name, old_left, old_right in zip(stmt.state, left, right):
                value = self.instruction('selp.f32', old_right, self.values[name], empty[0], kind='f')
                self.values[name] = self.instruction('selp.f32', old_left, value, empty[1], kind='f')

    def statement(self, stmt):
        if isinstance(stmt, Tile):
            if stmt.aux_threads:
                raise NotImplementedError('Auxiliary threads')
            # Raster grouping permutes block visitation for locality. Canonical
            # row-major visitation covers the same independent logical tiles.
            tid = self.instruction('mov.u32', '%tid.x')
            self.values['lane'] = self.instruction('rem.u32', tid, 32)
            self.values['warp'] = self.instruction('div.u32', tid, 32)
            bid = self.instruction('mov.u32', '%ctaid.x')
            width = str(stmt.block_threads) if stmt.block_threads else self.instruction('mov.u32', '%ntid.x')
            gid = self.instruction('mad.lo.s32', bid, width, tid)
            extents = [a.extent.value for a in stmt.axes]
            guard = self.instruction('setp.ge.u32', gid, math.prod(extents), kind='p')
            self.emit(f'@{guard} bra tile_end;')
            stride = math.prod(extents)
            for axis, extent in zip(stmt.axes, extents):
                stride //= extent
                quotient = self.instruction('div.u32', gid, stride)
                self.values[axis.name] = self.instruction('rem.u32', quotient, extent)
            self.body(stmt.body)
            self.emit('tile_end:')
        elif isinstance(stmt, Init):
            self.values[stmt.name] = self.instruction('mov.f32', self.floating(stmt.identity), kind='f')
            self.identities[stmt.name] = stmt.identity
        elif isinstance(stmt, CpAsyncCopy):
            if stmt.swizzle != 'NONE':
                raise NotImplementedError('Shared-memory swizzle')
            index = stmt.smem_index
            if stmt.lane_index is not None:
                index = tuple(BinaryExpr('+', origin, lane) for origin, lane in zip(index, stmt.lane_index))
            dst = self.address(stmt.smem, index)
            src = self.address(stmt.src, stmt.src_index)
            self.emit(f'cp.async.ca.shared.global [{dst}], [{src}], {stmt.nbytes};')
        elif isinstance(stmt, CpAsyncCommit):
            self.emit('cp.async.commit_group;')
        elif isinstance(stmt, CpAsyncWait):
            self.emit(f'cp.async.wait_group {stmt.group};')
        elif isinstance(stmt, Smem):
            self.shapes[stmt.name] = tuple(stmt.extents)
            self.shared[stmt.name] = f'.shared .align {max(4, stmt.align)} .b8 shared_{len(self.shared)}[{4 * math.prod(stmt.extents)}];'
            symbol = f'shared_{len(self.shared) - 1}'
            if self.dynamic_shared:
                base = self.instruction('mov.u32', 'shared_pool')
                self.pointers[stmt.name] = self.instruction('add.u32', base, self.shared_offsets[stmt.name])
            else:
                self.pointers[stmt.name] = self.instruction('mov.u32', symbol)
        elif isinstance(stmt, Sync):
            if stmt.warp:
                self.emit('bar.warp.sync 0xffffffff;')
            else:
                count = f', {stmt.count}' if stmt.count else ''
                self.emit(f'bar.sync {stmt.barrier_id}{count};')
        elif isinstance(stmt, Cond):
            other, end = self.label('else'), self.label('endif')
            predicate = self.predicate(stmt.cond)
            self.emit(f'@!{predicate} bra {other};')
            self.body(stmt.body)
            self.emit(f'bra {end};')
            self.emit(other + ':')
            self.body(stmt.else_body)
            self.emit(end + ':')
        elif isinstance(stmt, WarpShuffle):
            distance = stmt.length // 2
            while distance:
                for state, other in zip(stmt.state, stmt.state_b):
                    self.values[other] = self.instruction('shfl.sync.bfly.b32', self.values[state], distance, 31, '0xffffffff', kind='f')
                self.combine(stmt)
                distance //= 2
        elif isinstance(stmt, TreeHalve):
            # Each thread folds the small published partial-state array locally.
            # This preserves arbitrary coupled combine programs and broadcasts
            # the result without additional shared writes or synchronization.
            def slot(index):
                if stmt.inner is None:
                    return Literal(index, 'int')
                name, scale = stmt.inner
                return BinaryExpr('+', Literal(index * scale, 'int'), Var(name))
            for state, buffer in zip(stmt.state, stmt.bufs):
                self.values[state] = self.instruction('ld.shared.f32', f'[{self.address(buffer, (slot(0),))}]', kind='f')
            for index in range(1, stmt.length):
                for other, buffer in zip(stmt.state_b, stmt.bufs):
                    self.values[other] = self.instruction('ld.shared.f32', f'[{self.address(buffer, (slot(index),))}]', kind='f')
                self.combine(stmt)
        elif isinstance(stmt, (Loop, StridedLoop)):
            folds = self.accumulators(stmt.body, stmt.axis.name) if stmt.seed else {}
            for name, fold in folds.items():
                register = self.reg('f')
                self.values[name] = register
                identity = {'add': 0.0, 'multiply': 1.0, 'maximum': -math.inf, 'minimum': math.inf}[fold.op.name]
                self.identities[name] = identity
                self.emit(f'mov.f32 {register}, {self.floating(identity)};')
            axis = self.reg()
            self.values[stmt.axis.name] = axis
            start = self.integer(stmt.start) if isinstance(stmt, StridedLoop) else '0'
            step = self.integer(stmt.step) if isinstance(stmt, StridedLoop) else '1'
            end = self.integer(stmt.end) if isinstance(stmt, StridedLoop) and stmt.end is not None else str(stmt.axis.extent.value)
            self.emit(f'mov.u32 {axis}, {start};')
            head, tail = self.label('loop'), self.label('loop_end')
            self.emit(head + ':')
            done = self.instruction('setp.ge.s32', axis, end, kind='p')
            self.emit(f'@{done} bra {tail};')
            self.body(stmt.body)
            self.emit(f'add.s32 {axis}, {axis}, {step};')
            self.emit(f'bra {head};')
            self.emit(tail + ':')
        elif isinstance(stmt, Select):
            value = self.values[stmt.branches[-1].value]
            for branch in reversed(stmt.branches[:-1]):
                condition = self.predicate(branch.select)
                value = self.instruction('selp.f32', self.values[branch.value], value, condition, kind='f')
            self.values[stmt.name] = value
        elif isinstance(stmt, Let):
            if isinstance(stmt.value, Literal):
                self.values[stmt.name] = self.instruction('mov.f32', self.floating(stmt.value.value), kind='f')
            else:
                self.values[stmt.name] = self.integer(stmt.value)
        elif isinstance(stmt, Accum):
            self.accumulate(stmt)
        elif isinstance(stmt, Load):
            if stmt.dtype is not None and stmt.dtype.name != 'f32':
                raise NotImplementedError(f'Load dtype {stmt.dtype}')
            for i, name in enumerate(stmt.names):
                space = 'shared' if stmt.input in self.shared else 'global'
                self.values[name] = self.instruction(f'ld.{space}.f32', f'[{self.address(stmt.input, stmt.index, i)}]', kind='f')
        elif isinstance(stmt, Assign):
            self.values[stmt.name] = self.arithmetic(stmt.op.name, [self.values[a] for a in stmt.args])
        elif isinstance(stmt, Write):
            for i, value in enumerate(stmt.values):
                address = self.address(stmt.output, stmt.index, i)
                space = 'shared' if stmt.output in self.shared else 'global'
                opcode = f'red.{space}.add.f32' if stmt.atomic else f'st.{space}.f32'
                self.emit(f'{opcode} [{address}], {self.values[value]};')
        else:
            raise NotImplementedError(f'Statement {type(stmt).__name__}')

    def accumulate(self, stmt, product=None):
        dest = self.values[stmt.name]
        base = self.values[stmt.base] if stmt.base is not None else dest
        if product is not None:
            result = self.instruction('fma.rn.f32', *(self.values[a] for a in product.args), base, kind='f')
        else:
            result = self.arithmetic(stmt.op.name, [base, self.values[stmt.value]])
        self.emit(f'mov.f32 {dest}, {result};')

    def body(self, statements):
        """Contract an adjacent multiply/fold, never ordinary pointwise math.

        Adjacency within the same body proves that the multiply operands have
        not changed. Reduction tolerances allow this single-rounding contraction.
        ptxas removes the now-unused separate multiply when it has no other use.
        """
        previous = None
        for stmt in statements:
            if (isinstance(stmt, Accum) and stmt.op.name == 'add'
                    and isinstance(previous, Assign) and previous.op.name == 'multiply'
                    and stmt.value == previous.name):
                self.accumulate(stmt, previous)
            else:
                self.statement(stmt)
            previous = stmt

    def compile(self):
        params = self.launch['arg_order']
        for i, name in enumerate(params):
            raw = self.instruction('ld.param.u64', f'[param_{i}]', kind='rd')
            self.pointers[name] = self.instruction('cvta.to.global.u64', raw, kind='rd')
        self.body(self.node.op.body)
        self.emit('ret;')
        header = ['.version 8.0', '.target sm_89', '.address_size 64',
                  f'.visible .entry {self.node.op.name}(',
                  ',\n'.join(f'    .param .u64 param_{i}' for i in range(len(params))), ')', '{']
        types = {'r': '.b32', 'rd': '.b64', 'f': '.f32', 'p': '.pred'}
        for kind, count in self.counts.items():
            header.append(f'    .reg {types[kind]} %{kind}<{count + 1}>;')
        if self.dynamic_shared:
            header.insert(3, '.extern .shared .align 16 .b8 shared_pool[];')
        else:
            header.extend('    ' + declaration for declaration in self.shared.values())
        return '\n'.join(header + self.lines + ['}', ''])


def main():
    path = Path(sys.argv[1])
    graph = Graph.from_dict(json.loads(path.read_text()))
    node = next(n for n in graph.nodes.values() if type(n.op).__name__ == 'KernelOp')
    metadata = json.loads(path.with_name('meta.json').read_text())
    launch = next(k for k in metadata['kernels'] if k['entry'] == node.op.name)
    print(Compiler(graph, launch).compile(), end='')


if __name__ == '__main__':
    main()
