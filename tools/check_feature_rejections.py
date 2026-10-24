"""Probe unsupported typed IR through the real compiler CLI, without a GPU."""
import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

from emmy.compiler.dtype import F16
from emmy.compiler.graph import Graph
from emmy.compiler.ir.kernel.ir import Smem
from emmy.compiler.ir.stmt import Accum, Assign, Init, Let, Write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler', type=Path, default=Path('compile.py'))
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    template = root / 'artifacts/pair-cases/adjacent_s0_t1'
    source = next(template.glob('*.ir.json'))
    cases = [('write_f16', Write, {'value_dtype': F16}),
             ('assign_f16', Assign, {'dtype': F16}),
             ('init_f16', Init, {'dtype': F16}),
             ('accum_f16', Accum, {'dtype': F16}),
             ('let_f16', Let, {'dtype': F16}),
             ('shared_double', Smem, {'dtype': 'double'}),
             ('write_swizzle', Write, {'swizzle': 'B32'}),
             ('tensor_f16', None, {})]
    report = {'compiler_sha256': hashlib.sha256(args.compiler.read_bytes()).hexdigest(), 'results': []}
    for name, cls, changes in cases:
        graph = Graph.from_dict(json.loads(source.read_text()))
        changed = False
        def mutate(body):
            nonlocal changed
            result = []
            for stmt in body:
                if not changed and cls is not None and isinstance(stmt, cls):
                    stmt = replace(stmt, **changes)
                    changed = True
                for field in ('body', 'else_body'):
                    if hasattr(stmt, field):
                        stmt = replace(stmt, **{field: mutate(getattr(stmt, field))})
                result.append(stmt)
            return tuple(result)
        for node in graph.nodes.values():
            if cls is None and not changed:
                node.outputs = tuple(replace(t, dtype=F16) for t in node.outputs)
                changed = True
            if type(node.op).__name__ == 'KernelOp':
                node.op = replace(node.op, body=mutate(node.op.body))
        assert changed, name
        payload = json.dumps(graph.to_dict(), default=str)
        Graph.from_dict(json.loads(payload))  # Require typed-reader-valid fixtures.
        with tempfile.TemporaryDirectory(prefix='emmy-rejection-') as directory:
            folder = Path(directory)
            path = folder / source.name
            path.write_text(payload)
            (folder / 'meta.json').write_bytes((template / 'meta.json').read_bytes())
            run = subprocess.run([sys.executable, str(args.compiler.resolve()), str(path)], capture_output=True, text=True)
        passed = run.returncode != 0 and not run.stdout and ('unsupported' in run.stderr.lower()) and 'Traceback' not in run.stderr
        report['results'].append({'case': name, 'ir_sha256': hashlib.sha256(payload.encode()).hexdigest(),
                                  'exit_code': run.returncode, 'stdout_bytes': len(run.stdout),
                                  'stderr': run.stderr, 'passed': passed})
        print(('PASS' if passed else 'FAIL'), name)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    return 0 if all(r['passed'] for r in report['results']) else 1


if __name__ == '__main__':
    sys.exit(main())
