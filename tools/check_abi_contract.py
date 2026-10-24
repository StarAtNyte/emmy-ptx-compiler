"""Check positional buffer ABI diagnostics using committed numerical fixtures.

Invalid metadata is compiled only, never launched. Shape checks validate physical
descriptor syntax; they neither infer access bounds nor equate logical views.
"""
import argparse
import copy
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from emmy.compiler.graph import Graph


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler', type=Path, default=root/'compile.py')
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--observe', action='store_true')
    args = parser.parse_args()
    compiler = args.compiler.resolve()
    folder = root/'validation/cpu-fixtures/2026-10-24-abi-reordered'
    ir = next(folder.glob('*.ir.json'))
    original = json.loads((folder/'meta.json').read_text())
    probes = []

    def add(name, change, valid=False):
        metadata = copy.deepcopy(original)
        change(metadata)
        probes.append((name, metadata, valid))

    for field in ('arg_order', 'args'):
        add(f'{field}_missing', lambda m, f=field:m['kernels'][0].pop(f))
        for name, value in [('null', None), ('string', 'x'), ('empty', [])]:
            add(f'{field}_{name}', lambda m, f=field,v=value:m['kernels'][0].__setitem__(f,v))
    for name, value in [('duplicate', ['x','x']), ('unknown', ['x','unknown']),
                        ('nonstring', ['x',1]), ('empty_name', ['x','']),
                        ('unhashable', ['x',[]])]:
        add('order_'+name, lambda m,v=value:m['kernels'][0].__setitem__('arg_order',v))
    add('descriptor_order', lambda m:m['kernels'][0]['args'].reverse())
    add('descriptor_missing', lambda m:m['kernels'][0]['args'].pop())
    add('descriptor_extra', lambda m:m['kernels'][0]['args'].append({'name':'extra','dtype':'f32','shape':[1]}))
    add('descriptor_not_object', lambda m:m['kernels'][0]['args'].__setitem__(0,None))
    add('descriptor_name_missing', lambda m:m['kernels'][0]['args'][0].pop('name'))
    add('dtype_missing', lambda m:m['kernels'][0]['args'][0].pop('dtype'))
    for dtype in ('f16','i32','f64'):
        add('dtype_'+dtype, lambda m,v=dtype:m['kernels'][0]['args'][0].__setitem__('dtype',v))
    add('shape_missing', lambda m:m['kernels'][0]['args'][0].pop('shape'))
    for name, shape in [('null',None), ('string','128'), ('negative',[-1]),
                        ('fractional',[128.0]), ('bool',[True]), ('nested',[[128]])]:
        add('shape_'+name, lambda m,v=shape:m['kernels'][0]['args'][0].__setitem__('shape',v))
    add('entry_missing', lambda m:m['kernels'][0].pop('entry'))
    add('entry_wrong', lambda m:m['kernels'][0].__setitem__('entry','unknown'))
    add('entry_duplicate', lambda m:m['kernels'].append(copy.deepcopy(m['kernels'][0])))
    add('kernels_missing', lambda m:m.pop('kernels'))
    add('kernels_not_list', lambda m:m.__setitem__('kernels',{}))
    add('kernel_not_object', lambda m:m['kernels'].append(None))
    add('consistent_reorder', lambda m:(m['kernels'][0]['args'].reverse(),m['kernels'][0]['arg_order'].reverse()), True)
    add('allocation_reshape', lambda m:[a.__setitem__('shape',[16,8]) for a in m['kernels'][0]['args']], True)
    add('other_entry', lambda m:m['kernels'].append({'entry':'unrelated_kernel'}), True)
    add('reordered_fixture', lambda m:None, True)
    report = {'compiler_sha256':hashlib.sha256(compiler.read_bytes()).hexdigest(),
              'scope':'CPU ABI diagnostics; invalid metadata never executed on GPU', 'results':[]}
    with tempfile.TemporaryDirectory(prefix='emmy-abi-contract-') as temporary:
        sandbox = Path(temporary)
        path = sandbox/ir.name
        path.write_bytes(ir.read_bytes())
        for name, metadata, valid in probes:
            payload = json.dumps(metadata)
            (sandbox/'meta.json').write_text(payload)
            result = subprocess.run([sys.executable,'-I',str(compiler),str(path)],cwd=sandbox,
                                    capture_output=True,text=True,timeout=30)
            passed = ((result.returncode==0 and result.stdout.startswith('.version') and not result.stderr)
                      if valid else (result.returncode==1 and not result.stdout and 'ABI:' in result.stderr
                                     and 'Traceback' not in result.stderr))
            report['results'].append({'case':name,'expected_valid':valid,'passed':passed,
                'returncode':result.returncode,'stderr':result.stderr,
                'emitted_ptx':result.stdout.startswith('.version'),
                'stdout_sha256':hashlib.sha256(result.stdout.encode()).hexdigest(),
                'ir_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'meta_sha256':hashlib.sha256(payload.encode()).hexdigest()})
            print(('PASS' if passed else 'GAP'),name,flush=True)
        # Kernel files must select one entry unambiguously. Both inputs below
        # deserialize as typed graphs; a multi-kernel graph needs separate files.
        original_ir = json.loads(ir.read_text())
        no_kernel = copy.deepcopy(original_ir)
        no_kernel['nodes'] = {name:node for name,node in no_kernel['nodes'].items() if node['op']!='KernelOp'}
        no_kernel['outputs'] = list(no_kernel['inputs'])
        multiple = copy.deepcopy(original_ir)
        kernel = next(n for n in multiple['nodes'].values() if n['op']=='KernelOp')
        extra = json.loads(json.dumps(kernel).replace('sum_1','extra_output'))
        multiple['nodes']['extra_output'] = extra
        multiple['outputs'].append('extra_output')
        for name, graph in [('no_kernel',no_kernel),('multiple_kernels',multiple)]:
            Graph.from_dict(graph)
            payload = json.dumps(graph)
            path.write_text(payload)
            (sandbox/'meta.json').write_text(json.dumps(original))
            result = subprocess.run([sys.executable,'-I',str(compiler),str(path)],cwd=sandbox,
                                    capture_output=True,text=True,timeout=30)
            passed = (result.returncode==1 and not result.stdout and 'ABI:' in result.stderr
                      and 'Traceback' not in result.stderr)
            report['results'].append({'case':name,'expected_valid':False,'passed':passed,
                'returncode':result.returncode,'stderr':result.stderr,
                'emitted_ptx':result.stdout.startswith('.version'),
                'ir_sha256':hashlib.sha256(payload.encode()).hexdigest()})
            print(('PASS' if passed else 'GAP'),name,flush=True)
        view = root/'validation/cpu-fixtures/2026-10-24-address-scalar-flat'
        result = subprocess.run([sys.executable,'-I',str(compiler),str(next(view.glob('*.ir.json')))],
                                cwd=sandbox,capture_output=True,text=True,timeout=30)
        report['results'].append({'case':'scalar_logical_view','expected_valid':True,
            'passed':result.returncode==0 and result.stdout.startswith('.version') and not result.stderr,
            'returncode':result.returncode,'stderr':result.stderr})
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    count = sum(r['passed'] for r in report['results'])
    print(f'{count}/{len(report["results"])} contract expectations met')
    if not args.observe and count!=len(report['results']):
        raise SystemExit(1)


if __name__=='__main__':
    main()
