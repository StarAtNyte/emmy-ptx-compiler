"""Package real models, restoring array constants omitted from Emmy JSON dumps.

The official generator packages scalar ConstantOps only. Parameter tensors have
value=None in its JSON, so development models need their seeded weights supplied
explicitly. This tool changes test data only, never the compiler or its IR.
"""
import json
import subprocess
import sys
from pathlib import Path

from expressions_models import OPEN


def main():
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, str(root / 'challenge/tools/generate.py'),
                    str(root / 'tools/expressions_models.py'),
                    '--emmy', str(root / '.venv/bin/emmy'),
                    '--out', str(root / 'artifacts/models')], check=True)
    for spec in OPEN:
        path = root / 'artifacts/models' / spec['id'] / 'meta.json'
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
