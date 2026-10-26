"""Translate typed Emmy Kernel IR directly to PTX.

    python compile.py kernel.ir.json > kernel.ptx

The compiler never invokes Emmy's CUDA renderer, nvcc or another backend; Emmy
is used only to load the IR (Graph.from_dict). One pass does the work:

1. Load and validate: find the single KernelOp, take its launch (entry, pointer
   order, block size) from the adjacent meta.json, and reject IR features,
   shapes or ABIs this compiler does not lower, with a clear message.
2. Walk the statement tree once, keeping a map from IR value names to virtual
   PTX registers (%r, %rd, %f, %fd, %p). Each statement kind has one lowering
   method; expressions (indices, predicates, bounds) compile recursively.
   ptxas performs the real register allocation.
3. Assemble the module: header, parameters, launch-bound directives, register
   and shared-memory declarations, then the emitted body.

Correctness under the prescribed launch comes first. The optimizations are
structural and checked against the IR: FMA folding of reductions, bounded
unrolling of shared-memory inner loops, and v2/v4 vector loads where alignment
is proven. Elementwise math (exp, tanh, log, sin, cos, pow, erf) is implemented
here to libm-like accuracy rather than with bare hardware approximations.
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

    # -- Input validation -----------------------------------------------------

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

    # -- Registers and instruction emission -----------------------------------

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

    # -- Index, predicate and scalar expressions ------------------------------

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

    # -- Elementwise operations -----------------------------------------------

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
        if op in ('copy', 'pad'):
            return self.instruction('mov.f32', args[0], kind='f')
        if op == 'tanh':
            return self.hyperbolic_tangent(args[0])
        if op in ('sin', 'cos'):
            return self.sine_cosine(args[0], cosine=op == 'cos')
        if op == 'pow':
            return self.power(*args)
        opcode = {'add': 'add.rn.f32', 'subtract': 'sub.rn.f32',
                  'multiply': 'mul.rn.f32', 'divide': 'div.rn.f32',
                  'maximum': 'max.f32', 'minimum': 'min.f32',
                  'negative': 'neg.f32', 'sqrt': 'sqrt.rn.f32',
                  'absolute': 'abs.f32', 'abs': 'abs.f32',
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
            return self.exponential(args[0])
        if op == 'log':
            return self.natural_log(args[0])
        if op in ('log2', 'log10'):
            logarithm = self.instruction('lg2.approx.f32', args[0], kind='f')
            if op == 'log2':
                return logarithm
            return self.instruction('mul.rn.f32', logarithm, self.floating(math.log10(2)), kind='f')
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

    # -- Own f32 math library -------------------------------------------------
    # Emmy's reference kernels call CUDA's accurate libm (expf, tanhf, logf,
    # sinf, cosf, powf: 1-4 ULP), so hidden tolerances may be tight. These
    # routines reach that accuracy with ordinary PTX arithmetic: range
    # reduction, short polynomials evaluated with FMA, and explicit IEEE
    # special cases. Hardware .approx instructions are not used here.

    def f32(self, opcode, *args):
        return self.instruction(opcode, *args, kind='f')

    def exponential(self, value):
        """expf: ex2(t) for t = rn(x*log2(e)), corrected for the rounding of t.

        The product's exact rounding error (one FMA) plus the low part of log2(e)
        give d = x*log2(e) - t; then e**x = ex2(t) * (1 + d*ln2). The correction is
        computed beside ex2, so the dependent chain grows by one FMA only.
        """
        t = self.f32('mul.rn.f32', value, self.floating(1.4426950216293335))
        power = self.f32('ex2.approx.f32', t)
        error = self.f32('fma.rn.f32', value, self.floating(1.4426950216293335), self.f32('neg.f32', t))
        error = self.f32('fma.rn.f32', value, self.floating(1.925963033500011e-08), error)
        correction = self.f32('mul.rn.f32', error, self.floating(math.log(2)))
        corrected = self.f32('fma.rn.f32', power, correction, power)
        # Once ex2 has overflowed to inf or underflowed to 0 that is the answer;
        # the correction would only turn it into NaN.
        saturated = self.instruction('setp.eq.f32', power, self.floating(math.inf), kind='p')
        saturated = self.instruction('setp.eq.or.f32', power, self.floating(0), saturated, kind='p')
        return self.f32('selp.f32', power, corrected, saturated)

    def hyperbolic_tangent(self, value):
        """tanhf: odd minimax polynomial for |x| < 0.55, 1 - 2/(e**2|x| + 1) beyond."""
        square = self.f32('mul.rn.f32', value, value)
        # Fitted (relative-error Lawson iteration) to tanh(x)/x - 1 on [0, 0.55];
        # approximation error 9.4e-10, below half an f32 ULP.
        series = self.floating(-0.006235561333596706)
        for coefficient in (0.021039677783846855, -0.053842708468437195, 0.13332460820674896, -0.3333331048488617):
            series = self.f32('fma.rn.f32', series, square, self.floating(coefficient))
        series = self.f32('mul.rn.f32', series, square)
        # x + x*z*P(z) keeps subnormals exact; zero returns itself to keep its sign.
        small = self.f32('fma.rn.f32', value, series, value)
        small = self.f32('selp.f32', value, small, self.instruction('setp.eq.f32', value, self.floating(0), kind='p'))
        magnitude = self.f32('abs.f32', value)
        growth = self.exponential(self.f32('add.rn.f32', magnitude, magnitude))
        quotient = self.f32('rcp.rn.f32', self.f32('add.rn.f32', growth, self.floating(1)))
        large = self.f32('fma.rn.f32', quotient, self.floating(-2), self.floating(1))
        negative = self.instruction('setp.lt.f32', value, self.floating(0), kind='p')
        large = self.f32('selp.f32', self.f32('neg.f32', large), large, negative)
        # NaN fails the comparison and takes the large path, which propagates it.
        near_zero = self.instruction('setp.lt.f32', magnitude, self.floating(0.55), kind='p')
        return self.f32('selp.f32', small, large, near_zero)

    def natural_log(self, value):
        """logf: fdlibm's argument reduction to [sqrt(1/2), sqrt(2)) and minimax tail."""
        subnormal = self.instruction('setp.lt.f32', value, self.floating(2**-126), kind='p')
        normal = self.f32('selp.f32', self.f32('mul.rn.f32', value, self.floating(2**25)), value, subnormal)
        bias = self.instruction('selp.s32', -25, 0, subnormal)
        bits = self.instruction('mov.b32', normal)
        # Shift so the mantissa boundary falls at sqrt(2) instead of 2.
        bits = self.instruction('add.s32', bits, 0x3f800000 - 0x3f3504f3)
        exponent = self.instruction('shr.s32', bits, 23)
        exponent = self.instruction('add.s32', exponent, bias)
        exponent = self.f32('cvt.rn.f32.s32', self.instruction('sub.s32', exponent, 127))
        mantissa = self.instruction('and.b32', bits, 0x007fffff)
        mantissa = self.f32('mov.b32', self.instruction('add.s32', mantissa, 0x3f3504f3))
        f = self.f32('sub.rn.f32', mantissa, self.floating(1))
        s = self.f32('div.rn.f32', f, self.f32('add.rn.f32', f, self.floating(2)))
        z = self.f32('mul.rn.f32', s, s)
        w = self.f32('mul.rn.f32', z, z)
        odd = self.f32('fma.rn.f32', w, self.floating(0.24279078841), self.floating(0.40000972152))
        even = self.f32('fma.rn.f32', w, self.floating(0.28498786688), self.floating(0.66666662693))
        tail = self.f32('add.rn.f32', self.f32('mul.rn.f32', w, odd), self.f32('mul.rn.f32', z, even))
        half_square = self.f32('mul.rn.f32', self.f32('mul.rn.f32', f, f), self.floating(0.5))
        result = self.f32('mul.rn.f32', s, self.f32('add.rn.f32', half_square, tail))
        result = self.f32('fma.rn.f32', exponent, self.floating(9.0580006145e-06), result)
        result = self.f32('sub.rn.f32', result, half_square)
        result = self.f32('add.rn.f32', result, f)
        result = self.f32('fma.rn.f32', exponent, self.floating(6.9313812256e-01), result)
        # Special operands: +inf and NaN return themselves, zero gives -inf, negatives NaN.
        passthrough = self.instruction('setp.geu.f32', value, self.floating(math.inf), kind='p')
        result = self.f32('selp.f32', value, result, passthrough)
        negative = self.instruction('setp.lt.f32', value, self.floating(0), kind='p')
        result = self.f32('selp.f32', '0f7FFFFFFF', result, negative)
        zero = self.instruction('setp.eq.f32', value, self.floating(0), kind='p')
        return self.f32('selp.f32', '0fFF800000', result, zero)

    # 192 bits of 2/pi, most significant word first, for Payne-Hanek reduction.
    TWO_OVER_PI = (0xa2f9836e, 0x4e441529, 0xfc2757d1, 0xf534ddc0, 0xdb629599, 0x3c439041)

    def sine_cosine(self, value, cosine):
        """sinf/cosf: reduce to r = x - q*pi/2, |r| <= pi/4, then Cephes polynomials."""
        magnitude = self.f32('abs.f32', value)
        q = self.f32('cvt.rni.f32.f32', self.f32('mul.rn.f32', value, self.floating(2/math.pi)))
        # Three-part pi/2 with FMA (Cody-Waite) is accurate while q fits 17 bits.
        reduced = self.f32('fma.rn.f32', q, self.floating(-1.5707963705062866), value)
        reduced = self.f32('fma.rn.f32', q, self.floating(4.371138828673793e-08), reduced)
        reduced = self.f32('fma.rn.f32', q, self.floating(1.7151245100058819e-15), reduced)
        quadrant = self.instruction('cvt.rzi.s32.f32', q)
        huge = self.instruction('setp.gt.f32', magnitude, self.floating(105615), kind='p')
        finite = self.instruction('setp.lt.f32', magnitude, self.floating(math.inf), kind='p')
        huge = self.instruction('and.pred', huge, finite, kind='p')
        done = self.label('reduced')
        self.emit(f'@!{huge} bra {done};')
        self.payne_hanek(value, magnitude, reduced, quadrant)
        self.emit(done + ':')
        if cosine:
            # cos(x) = sin(x + pi/2): advance the quadrant by one.
            quadrant = self.instruction('add.s32', quadrant, 1)
        z = self.f32('mul.rn.f32', reduced, reduced)
        sine = self.f32('fma.rn.f32', z, self.floating(-1.9515295891e-4), self.floating(8.3321608736e-3))
        sine = self.f32('fma.rn.f32', sine, z, self.floating(-1.6666654611e-1))
        sine = self.f32('fma.rn.f32', self.f32('mul.rn.f32', sine, z), reduced, reduced)
        # Reduction turns -0 into +0; sin(+-0) must return the argument itself.
        signed_zero = self.instruction('setp.eq.f32', value, self.floating(0), kind='p')
        sine = self.f32('selp.f32', value, sine, signed_zero)
        cos = self.f32('fma.rn.f32', z, self.floating(2.443315711809948e-5), self.floating(-1.388731625493765e-3))
        cos = self.f32('fma.rn.f32', cos, z, self.floating(4.166664568298827e-2))
        cos = self.f32('mul.rn.f32', self.f32('mul.rn.f32', cos, z), z)
        cos = self.f32('add.rn.f32', self.f32('fma.rn.f32', z, self.floating(-0.5), self.floating(1)), cos)
        odd = self.instruction('and.b32', quadrant, 1)
        swap = self.instruction('setp.ne.s32', odd, 0, kind='p')
        result = self.f32('selp.f32', cos, sine, swap)
        upper = self.instruction('and.b32', quadrant, 2)
        flip = self.instruction('setp.ne.s32', upper, 0, kind='p')
        return self.f32('selp.f32', self.f32('neg.f32', result), result, flip)

    def payne_hanek(self, value, magnitude, reduced, quadrant):
        """Exact reduction for |x| > 105615: multiply the mantissa by 2/pi bits.

        Writes the existing `reduced` and `quadrant` registers. With x = M * 2**(e-31)
        for a 32-bit mantissa M, x*2/pi = (M * TABLE) * 2**(e-223); the 64 bits
        around the binary point give the quadrant and a 62-bit fraction.
        """
        bits = self.instruction('mov.b32', magnitude)
        exponent = self.instruction('sub.s32', self.instruction('shr.u32', bits, 23), 127)
        mantissa = self.instruction('or.b32', self.instruction('shl.b32', bits, 8), '0x80000000')
        # 224-bit product, least significant 32-bit limb first.
        limbs, carry = [], None
        for word in reversed(self.TWO_OVER_PI):
            wide = self.instruction('mul.wide.u32', mantissa, f'0x{word:08x}', kind='rd')
            if carry:
                wide = self.instruction('add.u64', wide, self.instruction('cvt.u64.u32', carry, kind='rd'), kind='rd')
            limbs.append(self.instruction('cvt.u32.u64', wide))
            carry = self.instruction('cvt.u32.u64', self.instruction('shr.u64', wide, 32, kind='rd'))
        limbs.append(carry)
        # The window's lowest bit is 161 - e; for e in [16, 127] its limb is 1..4.
        shift = self.instruction('sub.s32', 161, exponent)
        index = self.instruction('shr.u32', shift, 5)
        offset = self.instruction('and.b32', shift, 31)
        def limb(base):
            chosen = limbs[base + 1]
            for i in (2, 3, 4):
                is_index = self.instruction('setp.eq.u32', index, i, kind='p')
                chosen = self.instruction('selp.b32', limbs[base + i], chosen, is_index)
            return chosen
        low, middle, high = limb(0), limb(1), limb(2)
        upper = self.instruction('mov.b64', f'{{{middle}, {high}}}', kind='rd')
        window = self.instruction('shl.b64', upper, self.instruction('sub.u32', 32, offset), kind='rd')
        tail = self.instruction('cvt.u64.u32', self.instruction('shr.u32', low, offset), kind='rd')
        window = self.instruction('or.b64', window, tail, kind='rd')
        # Top two bits are the quadrant; the remaining 62 bits, read as a signed
        # fraction, round the quadrant to nearest and give r/(pi/2) in [-1/2, 1/2).
        fraction = self.instruction('shl.b64', window, 2, kind='rd')
        rounds_up = self.instruction('setp.lt.s64', fraction, 0, kind='p')
        turns = self.instruction('cvt.u32.u64', self.instruction('shr.u64', window, 62, kind='rd'))
        turns = self.instruction('add.s32', turns, self.instruction('selp.s32', 1, 0, rounds_up))
        angle = self.instruction('cvt.rn.f64.s64', fraction, kind='fd')
        angle = self.instruction('mul.rn.f64', angle, f'0d{struct.pack(">d", math.pi / 2 * 2**-64).hex().upper()}', kind='fd')
        angle = self.f32('cvt.rn.f32.f64', angle)
        # sin is odd: a negative argument negates r and the quadrant.
        negative = self.instruction('setp.lt.f32', value, self.floating(0), kind='p')
        self.emit(f'selp.f32 {reduced}, {self.f32("neg.f32", angle)}, {angle}, {negative};')
        self.emit(f'selp.s32 {quadrant}, {self.instruction("neg.s32", turns)}, {turns}, {negative};')

    def double(self, value):
        return '0d' + struct.pack('>d', float(value)).hex().upper()

    def f64(self, opcode, *args):
        return self.instruction(opcode, *args, kind='fd')

    def power(self, base, exponent):
        """powf: exp(y * log|x|) evaluated in f64, then C99 sign and special cases."""
        magnitude = self.f64('cvt.f64.f32', self.f32('abs.f32', base))
        # log|x| in f64: |x| = m * 2**k with m in [sqrt(1/2), sqrt(2)).
        high, low = self.reg(), self.reg()
        self.emit(f'mov.b64 {{{low}, {high}}}, {magnitude};')
        high = self.instruction('add.s32', high, 0x3ff00000 - 0x3fe6a09e)
        k = self.instruction('sub.s32', self.instruction('shr.s32', high, 20), 1023)
        high = self.instruction('add.s32', self.instruction('and.b32', high, 0x000fffff), 0x3fe6a09e)
        m = self.f64('mov.b64', f'{{{low}, {high}}}')
        f = self.f64('sub.rn.f64', m, self.double(1))
        s = self.f64('div.rn.f64', f, self.f64('add.rn.f64', f, self.double(2)))
        z = self.f64('mul.rn.f64', s, s)
        # log(m) = 2*atanh(s) = 2s * sum z**i/(2i+1); |s| < 0.172 so 9 terms suffice.
        series = self.double(1/17)
        for i in range(7, -1, -1):
            series = self.f64('fma.rn.f64', series, z, self.double(1/(2*i + 1)))
        logarithm = self.f64('mul.rn.f64', self.f64('add.rn.f64', s, s), series)
        logarithm = self.f64('fma.rn.f64', self.f64('cvt.rn.f64.s32', k), self.double(math.log(2)), logarithm)
        zero = self.instruction('setp.eq.f64', magnitude, self.double(0), kind='p')
        logarithm = self.f64('selp.f64', self.double(-math.inf), logarithm, zero)
        infinite = self.instruction('setp.eq.f64', magnitude, self.double(math.inf), kind='p')
        logarithm = self.f64('selp.f64', magnitude, logarithm, infinite)
        one = self.instruction('setp.eq.f64', magnitude, self.double(1), kind='p')
        t = self.f64('mul.rn.f64', self.f64('cvt.f64.f32', exponent), logarithm)
        t = self.f64('selp.f64', self.double(0), t, one)
        # exp(t) in f64, clamped where the f32 result is already inf or zero.
        t = self.f64('max.f64', self.f64('min.f64', t, self.double(100)), self.double(-120))
        n = self.f64('cvt.rni.f64.f64', self.f64('mul.rn.f64', t, self.double(1/math.log(2))))
        r = self.f64('fma.rn.f64', n, self.double(-0.6931471803691238), t)
        r = self.f64('fma.rn.f64', n, self.double(-1.9082149292705877e-10), r)
        series = self.double(1/math.factorial(12))
        for order in range(11, -1, -1):
            series = self.f64('fma.rn.f64', series, r, self.double(1/math.factorial(order)))
        scale_bits = self.instruction('shl.b32', self.instruction('add.s32', self.instruction('cvt.rzi.s32.f64', n), 1023), 20)
        scale = self.f64('mov.b64', f"{{{self.instruction('mov.b32', 0)}, {scale_bits}}}")
        result = self.f32('cvt.rn.f32.f64', self.f64('mul.rn.f64', series, scale))
        # Sign: a negative base with an odd integer exponent gives a negative result.
        whole = self.f32('cvt.rzi.f32.f32', exponent)
        integral = self.instruction('setp.eq.f32', whole, exponent, kind='p')
        half = self.f32('mul.rn.f32', exponent, self.floating(0.5))
        odd = self.instruction('setp.ne.f32', self.f32('cvt.rzi.f32.f32', half), half, kind='p')
        odd = self.instruction('and.pred', odd, integral, kind='p')
        sign = self.instruction('setp.lt.s32', self.instruction('mov.b32', base), 0, kind='p')
        flip = self.instruction('and.pred', sign, odd, kind='p')
        result = self.f32('selp.f32', self.f32('neg.f32', result), result, flip)
        # Only finite negative bases are invalid: pow(-inf, 0.5) is +inf, like C99.
        negative = self.instruction('setp.lt.f32', base, self.floating(0), kind='p')
        finite = self.instruction('setp.gt.f32', base, self.floating(-math.inf), kind='p')
        negative = self.instruction('and.pred', negative, finite, kind='p')
        fractional = self.instruction('and.pred', negative, self.instruction('not.pred', integral, kind='p'), kind='p')
        nan = self.instruction('setp.nan.f32', base, exponent, kind='p')
        invalid = self.instruction('or.pred', fractional, nan, kind='p')
        result = self.f32('selp.f32', '0f7FFFFFFF', result, invalid)
        # pow(x, 0) = 1 and pow(1, y) = 1 even for NaN operands.
        unit = self.instruction('setp.eq.f32', exponent, self.floating(0), kind='p')
        unit = self.instruction('or.pred', unit, self.instruction('setp.eq.f32', base, self.floating(1), kind='p'), kind='p')
        return self.f32('selp.f32', self.floating(1), result, unit)

    # -- Statements: scopes, reductions and lowering --------------------------

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
        """Lower one IR statement by dispatching on its kind."""
        for kinds, method in self.LOWERINGS:
            if isinstance(stmt, kinds):
                return method(self, stmt)
        raise NotImplementedError(f'Statement {type(stmt).__name__}')

    def lower_tile(self, stmt):
        """Map the Tile domain onto the 1-D launch, guard extra threads and decode axes."""
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

    def lower_init(self, stmt):
        """Seed carried state with its literal identity."""
        self.values[stmt.name] = self.instruction('mov.f32', self.floating(stmt.identity), kind='f')
        self.identities[stmt.name] = stmt.identity

    def lower_cp_async_copy(self, stmt):
        """Copy 4, 8 or 16 bytes global to shared with one cp.async."""
        if stmt.swizzle != 'NONE':
            raise NotImplementedError('Shared-memory swizzle')
        index = stmt.smem_index
        if stmt.lane_index is not None:
            index = tuple(BinaryExpr('+', origin, lane) for origin, lane in zip(index, stmt.lane_index))
        dst = self.address(stmt.smem, index)
        src = self.address(stmt.src, stmt.src_index)
        self.emit(f'cp.async.ca.shared.global [{dst}], [{src}], {stmt.nbytes};')

    def lower_cp_async_commit(self, stmt):
        self.emit('cp.async.commit_group;')

    def lower_cp_async_wait(self, stmt):
        self.emit(f'cp.async.wait_group {stmt.group};')

    def lower_smem(self, stmt):
        """Bind a shared array to its declaration-owned static symbol or pool offset."""
        self.shapes[stmt.name] = tuple(stmt.extents)
        symbol = self.shared_symbols[id(stmt)]
        self.shared[stmt.name] = symbol
        if self.dynamic_shared:
            base = self.instruction('mov.u32', 'shared_pool')
            self.pointers[stmt.name] = self.instruction('add.u32', base, self.shared_offsets[id(stmt)])
        else:
            self.pointers[stmt.name] = self.instruction('mov.u32', symbol)

    def lower_sync(self, stmt):
        """Warp, CTA or named-barrier synchronization."""
        if stmt.warp:
            self.emit('bar.warp.sync 0xffffffff;')
        else:
            count = f', {stmt.count}' if stmt.count else ''
            self.emit(f'bar.sync {stmt.barrier_id}{count};')

    def lower_cond(self, stmt):
        """Predicated branch; each side gets its own lexical scope."""
        other, end = self.label('else'), self.label('endif')
        predicate = self.predicate(stmt.cond)
        self.emit(f'@!{predicate} bra {other};')
        self.scoped_body(stmt.body)
        self.emit(f'bra {end};')
        self.emit(other + ':')
        self.scoped_body(stmt.else_body)
        self.emit(end + ':')

    def lower_warp_shuffle(self, stmt):
        """XOR butterfly over the lanes, merging states at each distance."""
        distance = stmt.length // 2
        while distance:
            for state, other in zip(stmt.state, stmt.state_b):
                self.values[other] = self.instruction('shfl.sync.bfly.b32', self.values[state], distance, 31, '0xffffffff', kind='f')
            self.combine(stmt)
            distance //= 2

    def lower_tree_halve(self, stmt):
        """Combine per-warp partials from shared memory as a binary tree."""
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

    def lower_loop(self, stmt):
        """Counted or strided loop with folds, bounded unrolling and range facts."""
        enclosing_values, enclosing_identities = self.values.copy(), self.identities.copy()
        carried = self.accumulators(stmt.body, stmt.axis.name)
        folds = carried if stmt.seed else {}
        for name, fold in folds.items():
            register = self.reg('f')
            self.values[name] = register
            # The IR's own reduce identity: Emmy uses -1e30, not -inf, for max, so
            # a lane that folds nothing merges as exp(0) * 0 instead of NaN.
            identity = fold.op.identity
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

    def lower_select(self, stmt):
        """Coordinate-predicated choice between SSA values."""
        value = self.as_float(self.values[stmt.branches[-1].value])
        for branch in reversed(stmt.branches[:-1]):
            condition = self.predicate(branch.select)
            value = self.instruction('selp.f32', self.as_float(self.values[branch.value]), value, condition, kind='f')
        self.values[stmt.name] = value

    def lower_let(self, stmt):
        """Bind a literal or pure expression; capture values instead of aliasing them."""
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

    def lower_accum(self, stmt):
        self.accumulate(stmt)

    def lower_load(self, stmt):
        """Scalar or consecutive loads from global or shared memory."""
        if stmt.dtype is not None and stmt.dtype.name != 'f32':
            raise NotImplementedError(f'Load dtype {stmt.dtype}')
        for i, name in enumerate(stmt.names):
            space = 'shared' if stmt.input in self.shared else 'global'
            self.values[name] = self.instruction(f'ld.{space}.f32', f'[{self.address(stmt.input, stmt.index, i)}]', kind='f')

    def lower_assign(self, stmt):
        """One elementwise operation over SSA values."""
        value = self.arithmetic(stmt.op.name, [self.values[a] for a in stmt.args])
        if stmt.dtype is not None:
            if stmt.dtype.name == 'bool':
                value = self.as_predicate(value)
            elif stmt.dtype.name == 'f32' and value.startswith('%p'):
                value = self.as_float(value)
        self.values[stmt.name] = value

    def lower_write(self, stmt):
        """Plain or atomic-add stores, one per value."""
        for i, value in enumerate(stmt.values):
            address = self.address(stmt.output, stmt.index, i)
            space = 'shared' if stmt.output in self.shared else 'global'
            opcode = f'red.{space}.add.f32' if stmt.atomic else f'st.{space}.f32'
            self.emit(f'{opcode} [{address}], {self.as_float(self.values[value])};')

    # Checked in order; the first matching kind lowers the statement.
    LOWERINGS = (
        (Tile, lower_tile),
        (Init, lower_init),
        (CpAsyncCopy, lower_cp_async_copy),
        (CpAsyncCommit, lower_cp_async_commit),
        (CpAsyncWait, lower_cp_async_wait),
        (Smem, lower_smem),
        (Sync, lower_sync),
        (Cond, lower_cond),
        (WarpShuffle, lower_warp_shuffle),
        (TreeHalve, lower_tree_halve),
        ((Loop, StridedLoop), lower_loop),
        (Select, lower_select),
        (Let, lower_let),
        (Accum, lower_accum),
        (Load, lower_load),
        (Assign, lower_assign),
        (Write, lower_write),
    )

    # -- Shared-memory trees, accumulation and vectorized reads ---------------

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

    # -- Module assembly ------------------------------------------------------

    def launch_bounds(self):
        """Performance directives derived from the prescribed block size.

        Without a register cap, ptxas may use more than 65536/block registers per
        thread and the launch fails. A cap also changes ptxas's allocation (it
        fills the budget), which slowed a 512-thread matmul ~12%, so it applies
        only to 1024-thread blocks, where the budget is the whole register file
        anyway, or when carried accumulators suggest real pressure. Small blocks
        get .maxntid, like __launch_bounds__ in Emmy's reference kernels; it let
        ptxas speed up a single-warp attention kernel ~1.3x.
        """
        threads = self.launch['block'][0]
        accumulators = 0
        def count(body):
            nonlocal accumulators
            for stmt in body:
                accumulators += isinstance(stmt, (Init, Accum))
                for field in ('body', 'else_body'):
                    count(getattr(stmt, field, ()) or ())
        count(self.node.op.body)
        estimate = 32 + 3 * accumulators
        if threads >= 1024 or estimate * threads > 65536:
            return [f'.maxnreg {65536 // threads // 8 * 8}']
        return [f'.maxntid {threads}, 1, 1'] if threads <= 256 else []

    def compile(self):
        params = self.launch['arg_order']
        for i, name in enumerate(params):
            raw = self.instruction('ld.param.u64', f'[param_{i}]', kind='rd')
            self.pointers[name] = self.instruction('cvta.to.global.u64', raw, kind='rd')
        self.body(self.node.op.body)
        self.emit('ret;')
        header = ['.version 8.0', '.target sm_89', '.address_size 64',
                  f'.visible .entry {self.node.op.name}(',
                  ',\n'.join(f'    .param .u64 param_{i}' for i in range(len(params))), ')',
                  *self.launch_bounds(), '{']
        types = {'r': '.b32', 'rd': '.b64', 'f': '.f32', 'fd': '.f64', 'p': '.pred'}
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
