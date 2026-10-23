"""Repackage generated problems with adversarial finite input distributions."""
import argparse
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path('artifacts/variations'))
    parser.add_argument('--specs', type=Path, default=Path('tools/expressions_variations.py'))
    parser.add_argument('--out', type=Path, default=Path('artifacts/numerical'))
    args = parser.parse_args()
    module_spec = importlib.util.spec_from_file_location('specifications', args.specs)
    specifications = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(specifications)
    specs = {s['id']: s for s in specifications.OPEN}
    count = 0
    for folder in sorted(args.source.iterdir()):
        if not (folder / 'meta.json').exists():
            continue
        for distribution in ('zeros', 'alternating', 'tiny'):
            meta = json.loads((folder / 'meta.json').read_text())
            info = {a['name']: a for k in meta['kernels'] for a in k['args']}
            arrays = {}
            for index, name in enumerate(meta['graph_inputs']):
                shape = info[name]['shape']
                size = int(np.prod(shape))
                if distribution == 'zeros':
                    data = np.zeros(shape, dtype=np.float32)
                else:
                    rng = np.random.default_rng(1009 + index)
                    magnitude = 30 if distribution == 'alternating' else 1e-8
                    data = (rng.choice([-1.0, 1.0], size=size) * magnitude).astype(np.float32).reshape(shape)
                arrays[name] = data
            for kernel in meta['kernels']:
                for arg in kernel['args']:
                    if arg['name'] in arrays:
                        arg.pop('seed', None)
                        arg['data'] = arrays[arg['name']].ravel().tolist()
            expected = specs[folder.name]['ref'](*[arrays[n] for n in meta['graph_inputs']])
            destination = args.out / f'{folder.name}_{distribution}'
            destination.mkdir(parents=True, exist_ok=True)
            for ir in folder.glob('*.ir.json'):
                shutil.copyfile(ir, destination / ir.name)
            meta['id'] = destination.name
            (destination / 'meta.json').write_text(json.dumps(meta, indent=1) + '\n')
            np.savez(destination / 'expected.npz', **dict(zip(meta['graph_outputs'], expected)))
            count += 1
    print(f'Generated {count} numerical variations')


if __name__ == '__main__':
    main()
