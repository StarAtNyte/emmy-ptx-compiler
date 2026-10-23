"""Package real models, restoring array constants omitted from Emmy JSON dumps.

The official generator packages scalar ConstantOps only. Parameter tensors have
value=None in its JSON, so development models need their seeded weights supplied
explicitly. This tool changes test data only, never the compiler or its IR.
"""
import json
import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('--specs', type=Path, default=root / 'tools/expressions_models.py')
    parser.add_argument('--out', type=Path, default=root / 'artifacts/models')
    args = parser.parse_args()
    source = importlib.util.spec_from_file_location('model_specifications', args.specs)
    specifications = importlib.util.module_from_spec(source)
    source.loader.exec_module(specifications)
    subprocess.run([sys.executable, str(root / 'challenge/tools/generate.py'),
                    str(args.specs),
                    '--emmy', str(root / '.venv/bin/emmy'),
                    '--out', str(args.out)], check=True)
    for spec in specifications.OPEN:
        path = args.out / spec['id'] / 'meta.json'
        meta = json.loads(path.read_text())
        constants = spec['constants']()
        restored = set()
        for kernel in meta['kernels']:
            for arg in kernel['args']:
                if arg.get('data') == [None]:
                    array = constants[arg['name']]
                    if list(array.shape) != arg['shape']:
                        raise ValueError(f"Wrong constant shape for {arg['name']}")
                    arg['data'] = array.ravel().tolist()
                    restored.add(arg['name'])
        path.write_text(json.dumps(meta, indent=1) + '\n')
        print(f"Restored {len(restored)} parameter tensors for {spec['id']}", flush=True)


if __name__ == '__main__':
    main()
