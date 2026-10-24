"""Inspect emitted PTX and driver-JIT resources without executing kernels.

Local occupancy is a resource ceiling, not measured occupancy or an RTX 5090
prediction. Hosted mode records target register counts; the runner exposes no
hardware counters, SASS, cache control or occupancy query.
"""
from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import json
import math
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import requests

from local_runner import Driver


def digest(data):
    return hashlib.sha256(data).hexdigest()


def instruction_counts(ptx):
    counts = Counter()
    for line in ptx.splitlines():
        line = re.sub(r'^\s*@!?%\w+\s+', '', line.strip())
        if re.match(r'^[a-z][\w.]*\s', line) and line.endswith(';'):
            counts[line.split()[0]] += 1
    return dict(sorted(counts.items()))


def local_resources(driver, ptx, kernel):
    module, function = ct.c_void_p(), ct.c_void_p()
    try:
        source = ct.create_string_buffer(ptx.replace('.target sm_89', f'.target {driver.arch}').encode())
        driver.call('cuModuleLoadData', ct.byref(module), ct.cast(source, ct.c_void_p))
        driver.call('cuModuleGetFunction', ct.byref(function), module, kernel['entry'].encode())
        if kernel['smem_bytes']:
            driver.call('cuFuncSetAttribute', function, 8, kernel['smem_bytes'])
        result = {}
        # CUfunction_attribute, CUDA driver API. LOCAL_SIZE is all local memory,
        # not an exact spill-byte counter; zero rules out local-memory spills.
        for name, attribute in [('max_threads_per_block', 0), ('static_shared_bytes', 1),
                                ('local_bytes_per_thread', 3), ('registers_per_thread', 4),
                                ('ptx_version', 5), ('binary_version', 6)]:
            value = ct.c_int()
            driver.call('cuFuncGetAttribute', ct.byref(value), attribute, function)
            result[name] = value.value
        blocks = ct.c_int()
        threads = math.prod(kernel['block'])
        driver.call('cuOccupancyMaxActiveBlocksPerMultiprocessor', ct.byref(blocks),
                    function, threads, kernel['smem_bytes'])
        result['max_active_blocks_per_sm'] = blocks.value
        result['resource_occupancy_ceiling'] = blocks.value * math.ceil(threads / 32) / (driver.max_threads_sm // 32)
        result['grid_blocks'] = math.prod(kernel['grid'])
        result['sm_count'] = driver.sm_count
        result['grid_blocks_per_sm'] = result['grid_blocks'] / driver.sm_count
        return result
    finally:
        if module.value:
            driver.call('cuModuleUnload', module)


def prepare_driver():
    driver = Driver()
    signatures = {
        'cuFuncGetAttribute': [ct.POINTER(ct.c_int), ct.c_int, ct.c_void_p],
        'cuDeviceGetAttribute': [ct.POINTER(ct.c_int), ct.c_int, ct.c_int],
        'cuOccupancyMaxActiveBlocksPerMultiprocessor':
            [ct.POINTER(ct.c_int), ct.c_void_p, ct.c_int, ct.c_size_t],
    }
    for name, signature in signatures.items():
        function = getattr(driver.lib, name)
        function.argtypes, function.restype = signature, ct.c_int
    for name, attribute in [('sm_count', 16), ('max_threads_sm', 39)]:
        value = ct.c_int()
        driver.call('cuDeviceGetAttribute', ct.byref(value), attribute, 0)
        setattr(driver, name, value.value)
    return driver


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('problems', nargs='+', type=Path)
    parser.add_argument('--compiler', type=Path, default=Path('compile.py'))
    parser.add_argument('--hosted', action='store_true')
    parser.add_argument('--server', default='http://runner.riftstack.ai:48701')
    parser.add_argument('--report', type=Path, default=Path('artifacts/resources.json'))
    args = parser.parse_args()
    driver = None if args.hosted else prepare_driver()
    report = {'compiler_sha256': digest(args.compiler.read_bytes()),
              'mode': 'hosted' if args.hosted else 'local',
              'executes_kernels': False, 'results': []}
    session = requests.Session()
    try:
        if args.hosted:
            response = session.get(args.server + '/health', timeout=(10, 30))
            response.raise_for_status()
            report['health'] = response.json()
        else:
            report['arch'] = driver.arch
        for folder in args.problems:
            metadata = (folder / 'meta.json').read_bytes()
            for kernel in json.loads(metadata)['kernels']:
                ir = folder / (kernel['entry'] + '.ir.json')
                ptx = subprocess.check_output([sys.executable, str(args.compiler), str(ir)], text=True)
                row = {'problem': folder.name, 'entry': kernel['entry'],
                       'ir_sha256': digest(ir.read_bytes()), 'meta_sha256': digest(metadata),
                       'ptx_sha256': digest(ptx.encode()), 'ptx_bytes': len(ptx.encode()),
                       'grid': kernel['grid'], 'block': kernel['block'],
                       'launch_smem_bytes': kernel['smem_bytes'],
                       'static_ptx_instructions': instruction_counts(ptx)}
                if args.hosted:
                    response = session.post(args.server + '/compile', json={'ptx': ptx}, timeout=(10, 60))
                    response.raise_for_status()
                    result = response.json()
                    if not result.get('ok'):
                        raise RuntimeError(result)
                    row['hosted_compile'] = result
                else:
                    row['local_resources'] = local_resources(driver, ptx, kernel)
                report['results'].append(row)
                print(json.dumps({k: row[k] for k in ('problem', 'entry',
                      'hosted_compile' if args.hosted else 'local_resources')}), flush=True)
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_text(json.dumps(report, indent=2) + '\n')
    finally:
        session.close()
        if driver:
            driver.close()


if __name__ == '__main__':
    main()
