"""Check intended vector eligibility independently of numerical execution."""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

from emmy.compiler.graph import Graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('problems', nargs='+', type=Path)
    parser.add_argument('--compiler', type=Path, required=True)
    parser.add_argument('--report', type=Path, default=Path('artifacts/pair-codegen.json'))
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('pair_candidate', args.compiler)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = {'compiler_sha256': hashlib.sha256(args.compiler.read_bytes()).hexdigest(), 'results': []}
    for folder in args.problems:
        case = json.loads((folder / 'case.json').read_text())
        meta = json.loads((folder / 'meta.json').read_text())
        count = 0
        hashes = {}
        for kernel in meta['kernels']:
            path = folder / (kernel['entry'] + '.ir.json')
            data = path.read_bytes()
            hashes[path.name] = hashlib.sha256(data).hexdigest()
            ptx = module.Compiler(Graph.from_dict(json.loads(data)), kernel).compile()
            count += ptx.count('ld.shared.v2.f32')
        passed = (count > 0) == case['pair_expected']
        if case['pairs_exact'] is not None:
            passed = passed and count == case['pairs_exact']
        row = {'problem': folder.name, 'ir_sha256': hashes, 'paired_loads': count,
               'expected_positive': case['pair_expected'], 'expected_exact': case['pairs_exact'],
               'passed': passed}
        report['results'].append(row)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + '\n')
        print(f"{'PASS' if passed else 'FAIL'} {folder.name}: {count} paired reads", flush=True)
    return 0 if all(row['passed'] for row in report['results']) else 1


if __name__ == '__main__':
    sys.exit(main())
