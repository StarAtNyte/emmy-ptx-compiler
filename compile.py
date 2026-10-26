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


def kernel_node(graph):
    nodes = [n for n in graph.nodes.values() if type(n.op).__name__ == 'KernelOp']
    if len(nodes) != 1:
        raise ValueError('ABI: input IR must contain exactly one KernelOp')
    return nodes[0]


def kernel_launch(metadata, entry):
    if not isinstance(metadata, dict) or not isinstance(metadata.get('kernels'), list):
        raise ValueError('ABI: metadata must contain a kernels list')
    if any(not isinstance(k, dict) for k in metadata['kernels']):
        raise ValueError('ABI: kernels must contain launch objects')
    matches = [k for k in metadata['kernels'] if k.get('entry') == entry]
    if len(matches) != 1:
        raise ValueError(f'ABI: expected exactly one metadata launch for entry {entry}')
    return matches[0]


class Compiler:
    def __init__(self, graph, launch):
        self.graph = graph
        self.launch = launch
        self.node = kernel_node(graph)
        self.validate_abi()
        self.validate_launch()
        self.validate_features()
        self.validate_shapes()
        self.cross_loads = set()
        self.cross_cache = {}
        self.cross_phase = 0
        self.cross_width = 2
        self.lines = []
        self.counts = Counter()
        self.values = {}
        self.index_ranges = {}
        self.pointers = {}
        self.shapes = {}
        self.label_count = 0
        self.shared = {}
        self.shared_offsets = {}
        self.shared_symbols = {}
        self.shared_declarations = []
        self.shared_symbol_alignments = {}
        self.shared_alignment = 16
        self.identities = {}
        self.observed_tree_buffers = set()
        tree_counts = Counter()
        read_buffers = set()
        def inspect_tree_uses(body, in_loop=False):
            for stmt in body:
                if isinstance(stmt, Load):
                    read_buffers.add(stmt.input)
                elif isinstance(stmt, Write) and stmt.atomic:
                    read_buffers.add(stmt.output)
                elif isinstance(stmt, TreeHalve):
                    tree_counts.update(stmt.bufs)
                    if in_loop:
                        self.observed_tree_buffers.update(stmt.bufs)
                for field in ('body', 'else_body'):
                    if hasattr(stmt, field):
                        inspect_tree_uses(getattr(stmt, field), in_loop or isinstance(stmt, (Loop, StridedLoop)))
        inspect_tree_uses(self.node.op.body)
        self.observed_tree_buffers.update(b for b,n in tree_counts.items() if n > 1 or b in read_buffers)
        def collect_shared(body):
            for stmt in body:
                if isinstance(stmt, Smem) and id(stmt) not in self.shared_offsets:
                    alignment = max(4, stmt.align)
                    self.shared_alignment = max(self.shared_alignment, alignment)
                    current = self.shared_size
                    offset = (current + alignment - 1) // alignment * alignment
                    # Equal source names may declare distinct live arrays in
                    # nested lexical scopes. Storage belongs to the declaration.
                    self.shared_offsets[id(stmt)] = offset
                    symbol = f'shared_{len(self.shared_symbols)}'
                    self.shared_symbols[id(stmt)] = symbol
                    self.shared_symbol_alignments[symbol] = alignment
                    self.shared_declarations.append(
                        f'.shared .align {alignment} .b8 {symbol}[{4 * math.prod(stmt.extents)}];')
                    self.shared_size = offset + 4 * math.prod(stmt.extents)
                if hasattr(stmt, 'body'):
                    collect_shared(stmt.body)
                if hasattr(stmt, 'else_body'):
                    collect_shared(stmt.else_body)
        self.shared_size = 0
        collect_shared(self.node.op.body)
        self.dynamic_shared = self.shared_size > 48 * 1024
        if self.dynamic_shared and self.shared_size > launch['smem_bytes']:
            raise ValueError(f"Shared allocations need {self.shared_size} dynamic bytes; "
                             f"launch provides {launch['smem_bytes']}")
        for node in graph.nodes.values():
            for tensor in node.outputs:
                self.shapes[tensor.name] = tuple(d.value for d in tensor.shape)

    def validate_abi(self):
        """Check positional buffer descriptors without equating logical views."""
        def invalid(detail):
            raise ValueError(f'Kernel {self.node.op.name}, ABI: {detail}')

        if not isinstance(self.launch, dict) or self.launch.get('entry') != self.node.op.name:
            invalid('launch entry must match KernelOp name')
        order = self.launch.get('arg_order')
        if (not isinstance(order, list)
                or any(not isinstance(name, str) or not name for name in order)):
            invalid('arg_order must be a list of nonempty buffer names')
        if len(set(order)) != len(order):
            invalid('arg_order contains duplicate buffer names')
        expected = set(self.node.inputs) | {tensor.name for tensor in self.node.outputs}
        if set(order) != expected:
            invalid('arg_order must contain exactly the KernelOp input/output buffers')
        descriptors = self.launch.get('args')
        if (not isinstance(descriptors, list)
                or any(not isinstance(arg, dict) for arg in descriptors)):
            invalid('args must be a list of buffer descriptor objects')
        if [arg.get('name') for arg in descriptors] != order:
            invalid('args descriptor order must match arg_order')
        for arg in descriptors:
            if arg.get('dtype') != 'f32':
                invalid(f"buffer {arg['name']} requires f32 allocation dtype")
            shape = arg.get('shape')
            if (not isinstance(shape, list)
                    or any(type(d) is not int or d < 0 for d in shape)):
                invalid(f"buffer {arg['name']} shape must contain nonnegative integer dimensions")

    def validate_launch(self):
        """Validate geometry used by our one-dimensional tile mapping."""
        def invalid(detail):
            raise ValueError(f'Kernel {self.node.op.name}, launch: {detail}')

        for field in ('grid', 'block'):
            dimensions = self.launch.get(field)
            if (not isinstance(dimensions, (list, tuple)) or len(dimensions) != 3
                    or any(type(d) is not int or d <= 0 for d in dimensions)):
                invalid(f'{field} must contain three positive integer dimensions')
            if dimensions[1:] not in ([1, 1], (1, 1)):
                invalid(f'unsupported multidimensional {field}; y and z must be 1')
        if self.launch['block'][0] > 1024:
            invalid('block.x exceeds 1024 threads')
        if self.launch['grid'][0] > 2147483647:
            invalid('grid.x exceeds 2147483647 blocks')
        shared = self.launch.get('smem_bytes')
        if type(shared) is not int or shared < 0:
            invalid('smem_bytes must be a nonnegative integer')

        def walk(body):
            for stmt in body:
                if isinstance(stmt, Tile) and stmt.block_threads is not None:
                    if (type(stmt.block_threads) is not int or stmt.block_threads <= 0):
                        invalid('Tile.block_threads must be a positive integer')
                    # Auxiliary bands have separate unsupported-feature handling.
                    if not stmt.aux_threads and stmt.block_threads != self.launch['block'][0]:
                        invalid('block.x must equal cooperative Tile.block_threads')
                for field in ('body', 'else_body'):
                    if hasattr(stmt, field):
                        walk(getattr(stmt, field))
        walk(self.node.op.body)

    def validate_features(self):
        """Reject unsupported storage/arithmetic types before emitting any PTX."""
        def unsupported(location, detail):
            raise ValueError(f'Kernel {self.node.op.name}, {location}: unsupported {detail}')

        for node in self.graph.nodes.values():
            for tensor in node.outputs:
                if tensor.dtype.name != 'f32':
                    unsupported(f'tensor {tensor.name}', f'tensor dtype {tensor.dtype}')

        def walk(body, location):
            for index, stmt in enumerate(body):
                here = f'{location}[{index}]/{type(stmt).__name__}'
                dtype = getattr(stmt, 'dtype', None)
                name = getattr(dtype, 'name', dtype)
                allowed = ('f32',)
                if isinstance(stmt, Smem):
                    allowed = ('float', 'f32')
                elif isinstance(stmt, Assign):
                    allowed = ('f32', 'bool')
                elif isinstance(stmt, Let):
                    allowed = ('f32', 'i32', 'bool')
                if name is not None and name not in allowed:
                    unsupported(here, f'dtype {name}')
                if isinstance(stmt, Write):
                    if stmt.value_dtype is not None and stmt.value_dtype.name not in ('f32', 'i32', 'bool'):
                        unsupported(here, f'write dtype {stmt.value_dtype}')
                    if stmt.swizzle != 'NONE':
                        unsupported(here, f'write swizzle {stmt.swizzle}')
                for field in ('body', 'else_body'):
                    if hasattr(stmt, field):
                        walk(getattr(stmt, field), f'{here}.{field}')
        walk(self.node.op.body, 'body')

    def validate_shapes(self):
        """Validate supported static extents before layout arithmetic."""
        def invalid(location, detail):
            raise ValueError(f'Kernel {self.node.op.name}, shape {location}: {detail}')

        def extent(dim, location):
            try:
                value = dim.as_static()
            except (TypeError, ValueError):
                invalid(location, 'requires a static integer extent')
            if type(value) is not int or value < 0:
                invalid(location, 'requires a nonnegative integer extent')

        for node in self.graph.nodes.values():
            for tensor in node.outputs:
                for index, dim in enumerate(tensor.shape):
                    extent(dim, f'tensor {tensor.name}[{index}]')

        def walk(body, location):
            for index, stmt in enumerate(body):
                here = f'{location}[{index}]/{type(stmt).__name__}'
                if isinstance(stmt, Tile):
                    for axis in stmt.axes:
                        extent(axis.extent, f'{here}/{axis.name}')
                elif isinstance(stmt, (Loop, StridedLoop)):
                    extent(stmt.axis.extent, f'{here}/{stmt.axis.name}')
                elif isinstance(stmt, Smem):
                    if any(type(d) is not int or d <= 0 for d in stmt.extents):
                        invalid(here, 'shared extents must be positive integers; zero-sized shared allocations unsupported')
                    if (type(stmt.align) is not int or stmt.align < 0
                            or (stmt.align and stmt.align & (stmt.align - 1))):
                        invalid(here, 'shared alignment must be zero or a positive power of two')
                for field in ('body', 'else_body'):
                    if hasattr(stmt, field):
                        walk(getattr(stmt, field), f'{here}.{field}')
        walk(self.node.op.body, 'body')

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
            if isinstance(stmt, (Loop, StridedLoop)):
                if stmt.axis.name == axis:
                    # The nested axis shadows this axis, including its folds.
                    continue
                nested = self.accumulators(stmt.body, axis)
                if stmt.seed:
                    for name in self.accumulators(stmt.body, stmt.axis.name):
                        nested.pop(name, None)
                found.update(nested)
                continue
            for field in ('body', 'else_body'):
                if hasattr(stmt, field):
                    found.update(self.accumulators(getattr(stmt, field), axis))
        return found

    def instruction(self, opcode, *args, kind='r'):
        result = self.reg(kind)
        self.emit(f'{opcode} {result}, ' + ', '.join(map(str, args)) + ';')
        return result

    def index_interval(self, expr):
        """Conservative signed-int32 bounds; unknown/overflow means no proof."""
        if isinstance(expr, Literal):
            if expr.dtype == 'float' or isinstance(expr.value, float):
                return None
            value = int(expr.value)
            return (value, value) if -(1 << 31) <= value < (1 << 31) else None
        if isinstance(expr, Var):
            return self.index_ranges.get(self.values.get(expr.name))
        if isinstance(expr, BinaryExpr):
            a, b = self.index_interval(expr.left), self.index_interval(expr.right)
            if a is None or b is None:
                return None
            if expr.op == '+':
                bounds = (a[0] + b[0], a[1] + b[1])
            elif expr.op == '-':
                bounds = (a[0] - b[1], a[1] - b[0])
            elif expr.op == '*':
                products = [x*y for x in a for y in b]
                bounds = (min(products), max(products))
            elif expr.op in ('/', '//') and b[0] > 0:
                quotients = [x//y for x in a for y in b]
                bounds = (min(quotients), max(quotients))
            elif expr.op == '%' and b[0] > 0:
                bounds = (0, b[1]-1)
            else:
                return None
            if -(1 << 31) <= bounds[0] <= bounds[1] < (1 << 31):
                return bounds
        return None

    def integer(self, expr):
        value = self.expression(expr)
        if self.is_float(value):
            raise ValueError(f'Kernel {self.node.op.name}: unsupported floating index expression {expr!r}')
        return self.as_integer(value)

    @staticmethod
    def is_float(value):
        return value.startswith(('%f', '0f'))

    def as_integer(self, value, *, cast_float=False):
        if value.startswith('%p'):
            return self.instruction('selp.b32', 1, 0, value)
        if self.is_float(value):
            if not cast_float:
                raise ValueError(f'Kernel {self.node.op.name}: unsupported implicit floating index')
            return self.instruction('cvt.rzi.s32.f32', value)
        return value

    def as_float(self, value):
        if self.is_float(value):
            return value
        value = self.as_integer(value)
        return (self.instruction('cvt.rn.f32.s32', value, kind='f') if value.startswith('%')
                else self.floating(int(value)))

    def float_expression(self, expr):
        if isinstance(expr, Var):
            return self.is_float(self.values[expr.name])
        if isinstance(expr, Literal):
            return expr.dtype == 'float' or isinstance(expr.value, float)
        if isinstance(expr, float):
            return True
        if isinstance(expr, TernaryExpr):
            return self.float_expression(expr.if_true) or self.float_expression(expr.if_false)
        if isinstance(expr, BinaryExpr) and expr.op in ('+', '-', '*', '/', '//', '%'):
            return self.float_expression(expr.left) or self.float_expression(expr.right)
        return False

    def expression(self, expr):
        if isinstance(expr, TernaryExpr):
            predicate = self.predicate(expr.cond)
            if self.float_expression(expr):
                return self.instruction('selp.f32', self.as_float(self.expression(expr.if_true)),
                                        self.as_float(self.expression(expr.if_false)), predicate, kind='f')
            return self.instruction('selp.b32', self.integer(expr.if_true), self.integer(expr.if_false), predicate)
        if isinstance(expr, Var):
            return self.values[expr.name]
        if isinstance(expr, Literal):
            if self.float_expression(expr):
                return self.floating(expr.value)
            return str(int(expr.value))
        if isinstance(expr, (int, float)):
            return self.floating(expr) if isinstance(expr, float) else str(expr)
        if isinstance(expr, BinaryExpr):
            if expr.op in ('&&', '||'):
                return self.instruction('and.pred' if expr.op == '&&' else 'or.pred',
                                        self.predicate(expr.left), self.predicate(expr.right), kind='p')
            comparisons = {'==': 'eq', '!=': 'ne', '<': 'lt', '<=': 'le', '>': 'gt', '>=': 'ge'}
            if self.float_expression(expr.left) or self.float_expression(expr.right):
                left = self.as_float(self.expression(expr.left))
                right = self.as_float(self.expression(expr.right))
                if expr.op in comparisons:
                    relation = 'neu' if expr.op == '!=' else comparisons[expr.op]
                    return self.instruction(f'setp.{relation}.f32', left, right, kind='p')
                operations = {'+': 'add', '-': 'subtract', '*': 'multiply'}
                if expr.op not in operations:
                    raise NotImplementedError(f'Unsupported floating Expr operation {expr.op}; use typed scalar Assign')
                return self.arithmetic(operations[expr.op], [left, right])
            left, right = self.integer(expr.left), self.integer(expr.right)
            if expr.op in comparisons:
                return self.instruction(f'setp.{comparisons[expr.op]}.s32', left, right, kind='p')
            if expr.op in ('/', '//', '%'):
                a, b = self.index_interval(expr.left), self.index_interval(expr.right)
                if b is not None and b[0] == b[1] and b[0] > 0 and b[0] & (b[0]-1) == 0:
                    # Arithmetic shift floors signed division, including negative
                    # numerators; masking gives the divisor-signed remainder.
                    return (self.instruction('and.b32', left, b[0]-1) if expr.op == '%'
                            else self.instruction('shr.s32', left, b[0].bit_length()-1))
                if a is not None and b is not None and a[0] >= 0 and b[0] > 0:
                    return self.instruction('rem.s32' if expr.op == '%' else 'div.s32', left, right)
                # PTX rounds signed division toward zero. The IR expression
                # contract floors, with remainder carrying the divisor's sign.
                remainder = self.instruction('rem.s32', left, right)
                nonzero = self.instruction('setp.ne.s32', remainder, 0, kind='p')
                signs = self.instruction('xor.b32', remainder, right)
                different = self.instruction('setp.lt.s32', signs, 0, kind='p')
                adjust = self.instruction('and.pred', nonzero, different, kind='p')
                if expr.op == '%':
                    corrected = self.instruction('add.s32', remainder, right)
                    return self.instruction('selp.b32', corrected, remainder, adjust)
                quotient = self.instruction('div.s32', left, right)
                corrected = self.instruction('sub.s32', quotient, 1)
                return self.instruction('selp.b32', corrected, quotient, adjust)
            opcode = {'+': 'add.s32', '-': 'sub.s32', '*': 'mul.lo.s32',
                      '/': 'div.s32', '//': 'div.s32', '%': 'rem.s32'}[expr.op]
            return self.instruction(opcode, left, right)
        raise NotImplementedError(f'Index expression: {expr!r}')

    def as_predicate(self, value):
        if self.is_float(value):
            return self.instruction('setp.neu.f32', value, self.floating(0), kind='p')
        return value if value.startswith('%p') else self.instruction('setp.ne.s32', value, 0, kind='p')

    def predicate(self, expr):
        return self.as_predicate(self.expression(expr))

    def address(self, buffer, index, offset=0):
        shape = self.shapes[buffer]
        flat = '0'
        if (len(index) > 1 and len(index) != len(shape)) or (index and not shape):
            # IR render_index defines mismatched-rank coordinates as additive
            # flat offsets. A single coordinate is flat even for a scalar view.
            for coordinate in index:
                flat = self.instruction('add.s32', flat, self.integer(coordinate))
        else:
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
        if op == 'where':
            predicate = self.as_predicate(args[0])
            return self.instruction('selp.f32', self.as_float(args[1]), self.as_float(args[2]), predicate, kind='f')
        if op in ('bitwise_and', 'logical_and', 'bitwise_or', 'logical_or'):
            opcode = 'and.pred' if op in ('bitwise_and', 'logical_and') else 'or.pred'
            return self.instruction(opcode, *(self.as_predicate(value) for value in args), kind='p')
        if op in ('bitwise_not', 'logical_not'):
            return self.instruction('not.pred', self.as_predicate(args[0]), kind='p')
        args = [self.as_float(value) for value in args]
        comparisons = {'equal': 'eq', 'not_equal': 'neu', 'less': 'lt', 'less_equal': 'le',
                       'greater': 'gt', 'greater_equal': 'ge'}
        if op in comparisons:
            return self.instruction(f'setp.{comparisons[op]}.f32', *args, kind='p')
        if op == 'square':
            return self.instruction('mul.rn.f32', args[0], args[0], kind='f')
        if op == 'erf':
            return self.error_function(args[0])
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
        if op in ('log', 'log2', 'log10'):
            logarithm = self.instruction('lg2.approx.f32', args[0], kind='f')
            if op == 'log2':
                return logarithm
            scale = math.log(2) if op == 'log' else math.log10(2)
            return self.instruction('mul.rn.f32', logarithm, self.floating(scale), kind='f')
        if op == 'exp2':
            return self.instruction('ex2.approx.f32', args[0], kind='f')
        if op not in opcode:
            raise NotImplementedError(f'Unsupported scalar operation {op}')
        return self.instruction(opcode[op], *args, kind='f')

    def error_function(self, value):
        """Own f32 erf: small-x series and A&S 7.1.26 away from zero."""
        magnitude = self.instruction('abs.f32', value, kind='f')
        square = self.instruction('mul.rn.f32', value, value, kind='f')
        # The odd series avoids cancellation and preserves signed zero/subnormals.
        series = self.floating(-1/(math.factorial(11)*23))
        for order in range(10, -1, -1):
            coefficient = (-1)**order / (math.factorial(order)*(2*order+1))
            series = self.instruction('fma.rn.f32', series, square, self.floating(coefficient), kind='f')
        scaled = self.instruction('mul.rn.f32', value, self.floating(2/math.sqrt(math.pi)), kind='f')
        near_zero = self.instruction('mul.rn.f32', scaled, series, kind='f')
        denominator = self.instruction('fma.rn.f32', magnitude, self.floating(.3275911), self.floating(1), kind='f')
        t = self.instruction('rcp.rn.f32', denominator, kind='f')
        polynomial = self.floating(1.061405429)
        for coefficient in (-1.453152027, 1.421413741, -.284496736, .254829592):
            polynomial = self.instruction('fma.rn.f32', polynomial, t, self.floating(coefficient), kind='f')
        polynomial = self.instruction('mul.rn.f32', polynomial, t, kind='f')
        negative_square = self.instruction('neg.f32', square, kind='f')
        exponential = self.arithmetic('exp', [negative_square])
        positive = self.instruction('fma.rn.f32', self.instruction('neg.f32', polynomial, kind='f'),
                                    exponential, self.floating(1), kind='f')
        negative = self.instruction('neg.f32', positive, kind='f')
        sign = self.instruction('setp.lt.f32', value, self.floating(0), kind='p')
        ordinary = self.instruction('selp.f32', negative, positive, sign, kind='f')
        small = self.instruction('setp.le.f32', magnitude, self.floating(1), kind='p')
        return self.instruction('selp.f32', near_zero, ordinary, small, kind='f')

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

    def mutable_states(self, statements):
        """States assigned across a lexical boundary, excluding nested seeds."""
        names = set()
        for stmt in statements:
            if isinstance(stmt, Accum):
                names.add(stmt.name)
            elif isinstance(stmt, (TreeHalve, WarpShuffle)):
                names.update(stmt.state)
            elif isinstance(stmt, Cond):
                names.update(self.mutable_states(stmt.body))
                names.update(self.mutable_states(stmt.else_body))
            elif isinstance(stmt, (Loop, StridedLoop)):
                nested = self.mutable_states(stmt.body)
                if stmt.seed:
                    nested.difference_update(self.accumulators(stmt.body, stmt.axis.name))
                names.update(nested)
        return names

    def scoped_body(self, statements):
        """Keep block-local SSA bindings private and external state writable.

        Reduction primitives can rebind a carried name to a fresh register.
        Copy their final values into the enclosing register before restoring
        the map, so branches and repeated loop iterations see runtime updates.
        """
        values, identities = self.values.copy(), self.identities.copy()
        pointers, shapes, shared = self.pointers.copy(), self.shapes.copy(), self.shared.copy()
        declarations = set()
        for stmt in statements:
            if isinstance(stmt, Load):
                declarations.update(stmt.names)
            elif isinstance(stmt, (Let, Assign, Select, Init)):
                declarations.add(stmt.name)
            elif isinstance(stmt, (Loop, StridedLoop)) and stmt.seed:
                declarations.update(self.accumulators(stmt.body, stmt.axis.name))
        self.body(statements)
        for name in sorted(self.mutable_states(statements) - declarations):
            if name in values and name in self.values and values[name] != self.values[name]:
                self.emit(f'mov.f32 {values[name]}, {self.values[name]};')
        self.values, self.identities = values, identities
        self.pointers, self.shapes, self.shared = pointers, shapes, shared

    def statement(self, stmt):
        if isinstance(stmt, Tile):
            if stmt.aux_threads:
                raise NotImplementedError('Auxiliary threads')
            extents = [a.extent.value for a in stmt.axes]
            if 0 in extents:
                return
            # Raster grouping permutes block visitation for locality. Canonical
            # row-major visitation covers the same independent logical tiles.
            tid = self.instruction('mov.u32', '%tid.x')
            self.values['lane'] = self.instruction('rem.u32', tid, 32)
            self.values['warp'] = self.instruction('div.u32', tid, 32)
            bid = self.instruction('mov.u32', '%ctaid.x')
            width = str(stmt.block_threads) if stmt.block_threads else self.instruction('mov.u32', '%ntid.x')
            gid = self.instruction('mad.lo.s32', bid, width, tid)
            guard = self.instruction('setp.ge.u32', gid, math.prod(extents), kind='p')
            end = self.label('tile_end')
            self.emit(f'@{guard} bra {end};')
            stride = math.prod(extents)
            for axis, extent in zip(stmt.axes, extents):
                stride //= extent
                quotient = self.instruction('div.u32', gid, stride)
                self.values[axis.name] = self.instruction('rem.u32', quotient, extent)
                self.index_ranges[self.values[axis.name]] = (0, extent-1)
            self.body(stmt.body)
            self.emit(end + ':')
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
            symbol = self.shared_symbols[id(stmt)]
            self.shared[stmt.name] = symbol
            if self.dynamic_shared:
                base = self.instruction('mov.u32', 'shared_pool')
                self.pointers[stmt.name] = self.instruction('add.u32', base, self.shared_offsets[id(stmt)])
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
            self.scoped_body(stmt.body)
            self.emit(f'bra {end};')
            self.emit(other + ':')
            self.scoped_body(stmt.else_body)
            self.emit(end + ':')
        elif isinstance(stmt, WarpShuffle):
            distance = stmt.length // 2
            while distance:
                for state, other in zip(stmt.state, stmt.state_b):
                    self.values[other] = self.instruction('shfl.sync.bfly.b32', self.values[state], distance, 31, '0xffffffff', kind='f')
                self.combine(stmt)
                distance //= 2
        elif isinstance(stmt, TreeHalve):
            if self.observed_tree_buffers.intersection(stmt.bufs):
                self.shared_tree(stmt)
                return
            # Preserve the binary tree's operand order, even for coupled or
            # noncommutative merge programs. Scratch writes are dead on this path.
            def slot(index):
                if stmt.inner is None:
                    return Literal(index, 'int')
                name, scale = stmt.inner
                return BinaryExpr('+', Literal(index * scale, 'int'), Var(name))
            partials = [[self.instruction('ld.shared.f32', f'[{self.address(buffer, (slot(index),))}]',kind='f')
                         for buffer in stmt.bufs] for index in range(stmt.length)]
            half = stmt.length // 2
            while half:
                for index in range(half):
                    self.values.update(zip(stmt.state,partials[index]))
                    self.values.update(zip(stmt.state_b,partials[index+half]))
                    self.combine(stmt)
                    partials[index] = [self.values[name] for name in stmt.state]
                half //= 2
            self.values.update(zip(stmt.state,partials[0]))
            if stmt.length > 1:
                self.statement(Sync(barrier_id=stmt.barrier_id,count=stmt.barrier_count))
        elif isinstance(stmt, (Loop, StridedLoop)):
            enclosing_values, enclosing_identities = self.values.copy(), self.identities.copy()
            carried = self.accumulators(stmt.body, stmt.axis.name)
            folds = carried if stmt.seed else {}
            for name, fold in folds.items():
                register = self.reg('f')
                self.values[name] = register
                identity = {'add': 0.0, 'multiply': 1.0, 'maximum': -math.inf, 'minimum': math.inf}[fold.op.name]
                self.identities[name] = identity
                self.emit(f'mov.f32 {register}, {self.floating(identity)};')
            axis = self.reg()
            start = self.integer(stmt.start) if isinstance(stmt, StridedLoop) else '0'
            start_bounds = self.index_interval(stmt.start) if isinstance(stmt, StridedLoop) else None
            self.values[stmt.axis.name] = axis
            dynamic_step = (isinstance(stmt, StridedLoop) and hasattr(stmt.step, 'free_vars')
                            and bool({stmt.axis.name, stmt.axis.name + '_end'} & stmt.step.free_vars()))
            axis_end = (isinstance(stmt, StridedLoop) and stmt.end is not None
                        and stmt.axis.name in stmt.end.free_vars())
            step = None if dynamic_step else self.integer(stmt.step) if isinstance(stmt, StridedLoop) else '1'
            end = None if axis_end else self.integer(stmt.end) if isinstance(stmt, StridedLoop) and stmt.end is not None else str(stmt.axis.extent.value)
            # Expand only straight-line shared-memory scalar inner loops.
            # Bound expansion by extent and a statement budget to expose ILP
            # without unbounded code growth; preserve ordered scalar tails.
            factor = 1
            remainder = 0
            if (isinstance(stmt, Loop) and 4 <= stmt.axis.extent.value <= 1024
                    and len(stmt.body) <= 64
                    and any(isinstance(s, Load) for s in stmt.body)
                    and all(isinstance(s, (Load, Assign, Accum, Let, Init, Select)) for s in stmt.body)
                    and all(not isinstance(s, Load) or s.input in self.shared for s in stmt.body)):
                limit = min(64, stmt.axis.extent.value, 1024 // len(stmt.body))
                factor = 1 << (limit.bit_length() - 1)
                remainder = stmt.axis.extent.value % factor
                end = str(stmt.axis.extent.value - remainder)
                step = str(factor)
            self.emit(f'mov.u32 {axis}, {start};')
            if axis_end:
                if start_bounds is not None:
                    self.index_ranges[axis] = start_bounds
                end = self.integer(stmt.end)
                self.index_ranges.pop(axis, None)
            if isinstance(stmt, StridedLoop) and stmt.end is not None:
                self.values[stmt.axis.name + '_end'] = end
            if isinstance(stmt, Loop) and 0 < stmt.axis.extent.value < (1 << 31):
                self.index_ranges[axis] = (0, stmt.axis.extent.value-1)
            elif isinstance(stmt, StridedLoop):
                lower = start_bounds
                stride_bounds = None if dynamic_step else self.index_interval(stmt.step)
                upper = None if axis_end else self.index_interval(stmt.end) if stmt.end is not None else (stmt.axis.extent.value,)*2
                if (lower is not None and stride_bounds is not None and upper is not None
                        and lower[0] >= 0 and stride_bounds[0] > 0
                        and lower[1] < upper[1] and upper[1] + stride_bounds[1] < (1 << 31)):
                    self.index_ranges[axis] = (lower[0], upper[1]-1)
            head, tail = self.label('loop'), self.label('loop_end')
            self.emit(head + ':')
            done = self.instruction('setp.ge.s32', axis, end, kind='p')
            self.emit(f'@{done} bra {tail};')
            cross = self.cross_iteration_loads(stmt) if factor > 1 else set()
            width = 4 if (factor >= 4 and cross
                          and cross == self.cross_iteration_loads(stmt, 4)) else 2
            for offset in range(factor):
                self.cross_loads = cross
                self.cross_width = width
                self.cross_phase = offset % width
                if not self.cross_phase:
                    self.cross_cache = {}
                self.values[stmt.axis.name] = axis if offset == 0 else self.instruction('add.s32', axis, offset)
                if axis in self.index_ranges:
                    self.index_ranges[self.values[stmt.axis.name]] = self.index_ranges[axis]
                self.scoped_body(stmt.body)
            self.cross_loads = set()
            self.cross_cache = {}
            self.values[stmt.axis.name] = axis
            if dynamic_step:
                step = self.integer(stmt.step)
            self.emit(f'add.s32 {axis}, {axis}, {step};')
            self.emit(f'bra {head};')
            self.emit(tail + ':')
            for offset in range(remainder):
                self.values[stmt.axis.name] = axis if offset == 0 else self.instruction('add.s32', axis, offset)
                if axis in self.index_ranges:
                    self.index_ranges[self.values[stmt.axis.name]] = self.index_ranges[axis]
                self.scoped_body(stmt.body)
            if remainder:
                self.emit(f'add.s32 {axis}, {axis}, {remainder};')
            self.values[stmt.axis.name] = axis
            results = {name: self.values[name] for name in carried}
            neutrals = {name: self.identities[name] for name in carried if name in self.identities}
            self.values = enclosing_values | results
            self.identities = enclosing_identities | neutrals
        elif isinstance(stmt, Select):
            value = self.as_float(self.values[stmt.branches[-1].value])
            for branch in reversed(stmt.branches[:-1]):
                condition = self.predicate(branch.select)
                value = self.instruction('selp.f32', self.as_float(self.values[branch.value]), value, condition, kind='f')
            self.values[stmt.name] = value
        elif isinstance(stmt, Let):
            bounds = self.index_interval(stmt.value)
            if isinstance(stmt.value, Literal):
                dtype = stmt.dtype.name if stmt.dtype is not None else stmt.value.dtype
                if dtype == 'bool':
                    self.values[stmt.name] = self.as_predicate(self.expression(stmt.value))
                elif dtype in ('int','i32','u32'):
                    literal = int(stmt.value.value)
                    self.values[stmt.name] = self.instruction('mov.b32', literal)
                    bounds = (literal, literal) if -(1 << 31) <= literal < (1 << 31) else None
                else:
                    self.values[stmt.name] = self.instruction('mov.f32', self.floating(stmt.value.value), kind='f')
            else:
                value = self.expression(stmt.value)
                if stmt.dtype is not None:
                    value = (self.as_float(value) if stmt.dtype.name == 'f32'
                             else self.as_predicate(value) if stmt.dtype.name == 'bool'
                             else self.as_integer(value, cast_float=True))
                if isinstance(stmt.value, Var) and value == self.values[stmt.value.name]:
                    # A pure binding captures the current value. Aliasing the
                    # source register would observe later accumulator updates.
                    kind = 'f' if value.startswith('%f') else 'p' if value.startswith('%p') else 'r'
                    opcode = {'f': 'mov.f32', 'p': 'mov.pred', 'r': 'mov.b32'}[kind]
                    value = self.instruction(opcode, value, kind=kind)
                self.values[stmt.name] = value
            if bounds is not None and self.values[stmt.name].startswith('%r'):
                self.index_ranges[self.values[stmt.name]] = bounds
        elif isinstance(stmt, Accum):
            self.accumulate(stmt)
        elif isinstance(stmt, Load):
            if stmt.dtype is not None and stmt.dtype.name != 'f32':
                raise NotImplementedError(f'Load dtype {stmt.dtype}')
            for i, name in enumerate(stmt.names):
                space = 'shared' if stmt.input in self.shared else 'global'
                self.values[name] = self.instruction(f'ld.{space}.f32', f'[{self.address(stmt.input, stmt.index, i)}]', kind='f')
        elif isinstance(stmt, Assign):
            value = self.arithmetic(stmt.op.name, [self.values[a] for a in stmt.args])
            if stmt.dtype is not None:
                if stmt.dtype.name == 'bool':
                    value = self.as_predicate(value)
                elif stmt.dtype.name == 'f32' and value.startswith('%p'):
                    value = self.as_float(value)
            self.values[stmt.name] = value
        elif isinstance(stmt, Write):
            for i, value in enumerate(stmt.values):
                address = self.address(stmt.output, stmt.index, i)
                space = 'shared' if stmt.output in self.shared else 'global'
                opcode = f'red.{space}.add.f32' if stmt.atomic else f'st.{space}.f32'
                self.emit(f'{opcode} [{address}], {self.as_float(self.values[value])};')
        else:
            raise NotImplementedError(f'Statement {type(stmt).__name__}')

    def shared_tree(self, stmt):
        """Materialize observable scratch mutations with one writer per slot.

        Logical coordinates can repeat across lanes. After caching a merge's
        operands, temporarily use its first scratch word for atomic tickets.
        Exactly one lane writes each result. This needs no extra shared memory
        and also preserves partial-CTA barriers and segmented column subsets.
        """
        scale = stmt.inner[1] if stmt.inner else 1
        left = Var(stmt.tid_var)
        if stmt.inner:
            left = BinaryExpr('+',BinaryExpr('*',left,Literal(scale,'int')),Var(stmt.inner[0]))
        half = stmt.length // 2
        while half:
            eligible = self.instruction('setp.lt.s32',self.values[stmt.tid_var],half,kind='p')
            end = self.label('tree_merged')
            self.emit(f'@!{eligible} bra {end};')
            right = BinaryExpr('+',left,Literal(half*scale,'int'))
            for name, other, buffer in zip(stmt.state,stmt.state_b,stmt.bufs):
                self.values[name] = self.instruction('ld.shared.f32',f'[{self.address(buffer,(left,))}]',kind='f')
                self.values[other] = self.instruction('ld.shared.f32',f'[{self.address(buffer,(right,))}]',kind='f')
            self.combine(stmt)
            results = [self.values[name] for name in stmt.state]
            self.emit(end + ':')
            barrier = Sync(barrier_id=stmt.barrier_id,count=stmt.barrier_count)
            self.statement(barrier)
            address = self.address(stmt.bufs[0],(left,))
            self.instruction(f'@{eligible} atom.shared.exch.b32',f'[{address}]',0)
            self.statement(barrier)
            ticket = self.instruction('mov.u32','0xffffffff')
            self.emit(f'@{eligible} atom.shared.add.u32 {ticket}, [{address}], 1;')
            self.statement(barrier)
            writer = self.instruction('setp.eq.u32',ticket,0,kind='p')
            for value,buffer in zip(results,stmt.bufs):
                self.emit(f'@{writer} st.shared.f32 [{self.address(buffer,(left,))}], {value};')
            self.statement(barrier)
            half //= 2
        root = Var(stmt.inner[0]) if stmt.inner else Literal(0,'int')
        for name,buffer in zip(stmt.state,stmt.bufs):
            self.values[name] = self.instruction('ld.shared.f32',f'[{self.address(buffer,(root,))}]',kind='f')

    def accumulate(self, stmt, product=None):
        dest = self.values[stmt.name]
        if not dest.startswith('%f'):
            raise ValueError(f'Kernel {self.node.op.name}: unsupported non-f32 accumulator {stmt.name}')
        base = self.as_float(self.values[stmt.base] if stmt.base is not None else dest)
        if product is not None:
            result = self.instruction('fma.rn.f32', *(self.as_float(self.values[a]) for a in product.args), base, kind='f')
        else:
            result = self.arithmetic(stmt.op.name, [base, self.values[stmt.value]])
        self.emit(f'mov.f32 {dest}, {result};')

    @staticmethod
    def index_linear(expr):
        if isinstance(expr, Literal):
            return {}, int(expr.value)
        if isinstance(expr, BinaryExpr) and expr.op in ('+', '-'):
            left, a = Compiler.index_linear(expr.left)
            right, b = Compiler.index_linear(expr.right)
            sign = 1 if expr.op == '+' else -1
            terms = left.copy()
            for key, value in right.items():
                terms[key] = terms.get(key, 0) + sign * value
            return {k:v for k,v in terms.items() if v}, a + sign*b
        if isinstance(expr, BinaryExpr) and expr.op == '*':
            literal = expr.left if isinstance(expr.left, Literal) else expr.right
            if isinstance(literal, Literal):
                other = expr.right if literal is expr.left else expr.left
                terms, constant = Compiler.index_linear(other)
                factor = int(literal.value)
                return {k:v*factor for k,v in terms.items() if v*factor}, constant*factor
        return {expr:1}, 0

    def flat_linear(self, buffer, index):
        terms, constant = {}, 0
        shape = self.shapes[buffer]
        extents = shape if len(index) == len(shape) else (1,) * len(index)
        for coordinate, extent in zip(index, extents):
            terms = {k:v*extent for k,v in terms.items()}
            constant *= extent
            extra, value = self.index_linear(coordinate)
            for key, coefficient in extra.items():
                terms[key] = terms.get(key, 0) + coefficient
            constant += value
        return {k:v for k,v in terms.items() if v}, constant

    def shared_pairs(self, statements):
        # Moving reads is legal only in straight-line, shared-read-only bodies.
        if not (len(statements) <= 64 and any(isinstance(s, Load) for s in statements)
                and all(isinstance(s,(Load,Assign,Accum,Let,Init,Select)) for s in statements)
                and all(not isinstance(s,Load) or s.input in self.shared for s in statements)):
            return {}, set()
        pairs, removed = {}, set()
        for i, first in enumerate(statements):
            if i in removed or not isinstance(first, Load) or len(first.names) != 1:
                continue
            if first.dtype is not None and first.dtype.name != 'f32':
                continue
            if self.shared_symbol_alignments[self.shared[first.input]] < 8:
                continue
            terms, constant = self.flat_linear(first.input, first.index)
            if constant % 2 or any(v % 2 for v in terms.values()):
                continue
            for j in range(i+1,len(statements)):
                second = statements[j]
                if j in removed or not isinstance(second,Load) or second.input != first.input or len(second.names) != 1:
                    continue
                if second.dtype is not None and second.dtype.name != 'f32':
                    continue
                other, value = self.flat_linear(second.input, second.index)
                if other != terms or value != constant+1:
                    continue
                variables = {v for e in (*first.index,*second.index) for v in e.free_vars()}
                reads, writes = set(), set()
                for middle in statements[i+1:j]:
                    reads.update(middle.deps())
                    for expression in middle.exprs():
                        reads.update(expression.free_vars())
                    if isinstance(middle, Load):
                        writes.update(middle.names)
                    else:
                        writes.add(middle.name)
                if variables & writes or set(second.names) & (reads | writes | variables):
                    continue
                pairs[i] = second
                removed.add(j)
                break
        return pairs, removed

    def cross_iteration_loads(self, loop, width=2):
        """Pair affine adjacent reads at even unrolled-loop positions."""
        writes = set()
        loads = [s for s in loop.body if isinstance(s, Load)]
        if len({id(s) for s in loads}) != len(loads):
            return set()
        for stmt in loop.body:
            writes.update(stmt.names if isinstance(stmt, Load) else (stmt.name,))
        if loop.axis.name in writes:
            return set()
        existing, removed = self.shared_pairs(loop.body)
        occupied = set(existing) | removed
        eligible = set()
        axis = Var(loop.axis.name)
        for index, stmt in enumerate(loop.body):
            if (index in occupied or not isinstance(stmt, Load) or len(stmt.names) != 1
                    or self.shared_symbol_alignments[self.shared[stmt.input]] < 4*width):
                continue
            if any(expr.free_vars() & writes for expr in stmt.index):
                continue
            terms, constant = self.flat_linear(stmt.input, stmt.index)
            if terms.get(axis) != 1 or constant % width:
                continue
            if any((loop.axis.name in term.free_vars() or coefficient % width)
                   for term, coefficient in terms.items() if term != axis):
                continue
            eligible.add(id(stmt))
        return eligible

    def body(self, statements):
        """Contract an adjacent multiply/fold, never ordinary pointwise math.

        Adjacency within the same body proves that the multiply operands have
        not changed. Reduction tolerances allow this single-rounding contraction.
        ptxas removes the now-unused separate multiply when it has no other use.
        """
        previous = None
        pairs, removed = self.shared_pairs(statements)
        for index, stmt in enumerate(statements):
            if id(stmt) in self.cross_loads:
                if self.cross_phase:
                    self.values[stmt.names[0]] = self.cross_cache[id(stmt)][self.cross_phase-1]
                else:
                    registers = [self.reg('f') for _ in range(self.cross_width)]
                    address = self.address(stmt.input, stmt.index)
                    self.emit(f'ld.shared.v{self.cross_width}.f32 {{{", ".join(registers)}}}, [{address}];')
                    self.values[stmt.names[0]] = registers[0]
                    self.cross_cache[id(stmt)] = registers[1:]
                previous = stmt
                continue
            if index in removed:
                previous = stmt
                continue
            if index in pairs:
                other = pairs[index]
                registers = [self.reg('f'), self.reg('f')]
                address = self.address(stmt.input,stmt.index)
                self.emit(f'ld.shared.v2.f32 {{{", ".join(registers)}}}, [{address}];')
                self.values.update(zip((*stmt.names,*other.names),registers))
                previous = stmt
                continue
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
            header.insert(3, f'.extern .shared .align {self.shared_alignment} .b8 shared_pool[];')
        else:
            header.extend('    ' + declaration for declaration in self.shared_declarations)
        return '\n'.join(header + self.lines + ['}', ''])


def main():
    path = Path(sys.argv[1])
    graph = Graph.from_dict(json.loads(path.read_text()))
    node = kernel_node(graph)
    metadata = json.loads(path.with_name('meta.json').read_text())
    launch = kernel_launch(metadata, node.op.name)
    print(Compiler(graph, launch).compile(), end='')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, NotImplementedError) as error:
        print(f'compile.py: {error}', file=sys.stderr)
        sys.exit(1)
