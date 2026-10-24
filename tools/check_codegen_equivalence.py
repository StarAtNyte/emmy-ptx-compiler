"""Compare complete emitted PTX for two compilers on existing problem folders."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

from emmy.compiler.graph import Graph


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('problems', nargs='+', type=Path)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--candidate', default=Path('compile.py'), type=Path)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    baseline = load('baseline_compiler', args.baseline)
    candidate = load('candidate_compiler', args.candidate)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    report = {'baseline_sha256': digest(args.baseline), 'compiler_sha256': digest(args.candidate), 'results': []}
    for folder in args.problems:
        meta_path = folder / 'meta.json'
        meta = json.loads(meta_path.read_text())
        for kernel in meta['kernels']:
            ir = folder / (kernel['entry'] + '.ir.json')
            data = json.loads(ir.read_text())
            # Separate graphs avoid relying on either compiler leaving IR untouched.
            a = baseline.Compiler(Graph.from_dict(data), kernel).compile()
            b = candidate.Compiler(Graph.from_dict(data), kernel).compile()
            report['results'].append({'ir': str(ir), 'ir_sha256': digest(ir),
                'meta_sha256': digest(meta_path), 'identical_ptx': a == b,
                'baseline_ptx_sha256': hashlib.sha256(a.encode()).hexdigest(),
                'candidate_ptx_sha256': hashlib.sha256(b.encode()).hexdigest()})
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    passed = sum(r['identical_ptx'] for r in report['results'])
    print(f'{passed}/{len(report["results"])} byte-identical kernels')
    return 0 if passed == len(report['results']) and passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
