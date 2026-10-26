"""Exercise launch geometry diagnostics through the isolated compiler CLI.

Mutations are never launched on a GPU. Valid controls include scalar logical
views, reordered parameters and cooperative tiles matching their physical block.
"""
import argparse
import copy
import hashlib
import json
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

from emmy.compiler.graph import Graph
from emmy.compiler.ir.kernel.ir import Tile


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler', type=Path, default=root/'compile.py')
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--observe', action='store_true', help='record baseline gaps without failing')
    args = parser.parse_args()
    compiler = args.compiler.resolve()
    template = root/'tests/fixtures/abi-reordered'
    source = next(template.glob('*.ir.json'))
    original = json.loads(source.read_text())
    metadata = json.loads((template/'meta.json').read_text())
    probes = []
    for field in ('grid', 'block'):
        for label, value in [('missing', None), ('short', [1, 1]), ('zero', [0, 1, 1]),
                             ('negative', [-1, 1, 1]), ('float', [1.5, 1, 1]),
                             ('bool', [True, 1, 1]), ('y', [1, 2, 1]), ('z', [1, 1, 2])]:
            probes.append((f'{field}_{label}', field, value, None, False))
    probes += [('block_limit', 'block', [1025, 1, 1], None, False),
               ('grid_limit', 'grid', [2147483648, 1, 1], None, False)]
    for label, value in [('missing', None), ('negative', -1), ('float', 0.5), ('bool', False)]:
        probes.append((f'smem_{label}', 'smem_bytes', value, None, False))
    probes += [('cooperative_mismatch', None, None, 32, False),
               ('cooperative_zero', None, None, 0, False),
               ('cooperative_negative', None, None, -32, False),
               ('cooperative_match', None, None, 64, True),
               ('reordered_abi', None, None, None, True)]
    report = {'compiler_sha256': hashlib.sha256(compiler.read_bytes()).hexdigest(),
              'scope': 'CPU CLI diagnostics and valid codegen controls; mutated launches never executed',
              'results': []}
    with tempfile.TemporaryDirectory(prefix='emmy-launch-contract-') as temporary:
        folder = Path(temporary)
        for name, field, value, cooperative, valid in probes:
            graph = Graph.from_dict(copy.deepcopy(original))
            if cooperative is not None:
                node = next(n for n in graph.nodes.values() if type(n.op).__name__ == 'KernelOp')
                assert isinstance(node.op.body[0], Tile)
                node.op = replace(node.op, body=(replace(node.op.body[0], block_threads=cooperative),))
            payload = graph.to_dict()
            Graph.from_dict(payload)
            meta = copy.deepcopy(metadata)
            if field is not None:
                if value is None:
                    del meta['kernels'][0][field]
                else:
                    meta['kernels'][0][field] = value
            path = folder/source.name
            path.write_text(json.dumps(payload, default=str))
            (folder/'meta.json').write_text(json.dumps(meta))
            result = subprocess.run([sys.executable, '-I', str(compiler), str(path)],
                                    cwd=folder, capture_output=True, text=True, timeout=30)
            passed = ((result.returncode == 0 and result.stdout.startswith('.version') and not result.stderr)
                      if valid else (result.returncode == 1 and not result.stdout
                                     and ', launch:' in result.stderr and 'Traceback' not in result.stderr))
            report['results'].append({'case':name, 'expected_valid':valid, 'passed':passed,
                'returncode':result.returncode, 'stderr':result.stderr,
                'emitted_ptx':result.stdout.startswith('.version'),
                'stdout_sha256':hashlib.sha256(result.stdout.encode()).hexdigest(),
                'ir_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'meta_sha256':hashlib.sha256((folder/'meta.json').read_bytes()).hexdigest()})
            print(('PASS' if passed else 'GAP'), name, flush=True)
        # Keep the established scalar-view control in the same independent CLI path.
        view = root/'tests/fixtures/address-scalar-flat'
        result = subprocess.run([sys.executable, '-I', str(compiler), str(next(view.glob('*.ir.json')))],
                                cwd=folder, capture_output=True, text=True, timeout=30)
        report['results'].append({'case':'scalar_logical_view', 'expected_valid':True,
            'passed':result.returncode == 0 and result.stdout.startswith('.version') and not result.stderr,
            'returncode':result.returncode, 'stderr':result.stderr})
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    count = sum(r['passed'] for r in report['results'])
    print(f'{count}/{len(report["results"])} contract expectations met')
    if not args.observe and count != len(report['results']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
