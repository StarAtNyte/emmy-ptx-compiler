"""Measure local device event times and check outputs after timed repetitions.

Numbers are diagnostic: clocks are not locked, and the local GPU is not the
contest GPU. Compilation, transfers and output zeroing are outside event times.
Events and the kernel are captured together to exclude Python dispatch gaps.
"""
import argparse
import ctypes as ct
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from local_runner import Driver


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('problems', nargs='+', type=Path)
    parser.add_argument('--compiler', type=Path, default=Path('compile.py'))
    parser.add_argument('--iters', type=int, default=31)
    parser.add_argument('--evict-l2', action='store_true',
                        help='Zero twice queried device L2 bytes before each replay, outside events')
    parser.add_argument('--report', type=Path, default=Path('artifacts/benchmark.json'))
    args = parser.parse_args()
    if args.iters < 1:
        parser.error('--iters must be positive')
    driver = Driver()
    eviction_bytes = 0
    if args.evict_l2:
        function = driver.lib.cuDeviceGetAttribute
        function.argtypes, function.restype = [ct.POINTER(ct.c_int), ct.c_int, ct.c_int], ct.c_int
        l2_size = ct.c_int()
        driver.call('cuDeviceGetAttribute', ct.byref(l2_size), 38, 0)
        if l2_size.value <= 0:
            driver.close()
            raise RuntimeError('Device did not report a positive L2 size')
        eviction_bytes = 2 * l2_size.value
    report = {'compiler_sha256': hashlib.sha256(args.compiler.read_bytes()).hexdigest(),
              'arch': driver.arch, 'clock_locked': False,
              'cache_condition': 'eviction-conditioned' if args.evict_l2 else 'warm-reuse',
              'eviction_bytes': eviction_bytes, 'results': []}
    try:
        for folder in args.problems:
            meta = json.loads((folder / 'meta.json').read_text())
            produced = {}
            for kernel in meta['kernels']:
                start = time.monotonic()
                compile_result = subprocess.run([sys.executable, str(args.compiler),
                    str(folder / (kernel['entry'] + '.ir.json'))], capture_output=True, text=True, check=True)
                compile_seconds = time.monotonic() - start
                produced.update(driver.run(compile_result.stdout, kernel, produced,
                                           timed_iters=args.iters, eviction_bytes=eviction_bytes))
                times = driver.last_timings_us
                row = {'problem': folder.name, 'entry': kernel['entry'],
                       'compile_s': compile_seconds, 'ptx_bytes': len(compile_result.stdout),
                       'median_us': float(np.median(times)), 'p10_us': float(np.percentile(times, 10)),
                       'p90_us': float(np.percentile(times, 90)), 'samples_us': times}
                report['results'].append(row)
                print(f"{folder.name}: {row['median_us']:.3f} us; compile {compile_seconds:.3f} s", flush=True)
            with np.load(folder / 'expected.npz') as expected:
                for name in expected.files:
                    np.testing.assert_allclose(produced[name], expected[name], rtol=meta['rtol'], atol=meta['atol'])
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + '\n')
    finally:
        driver.close()


if __name__ == '__main__':
    main()
