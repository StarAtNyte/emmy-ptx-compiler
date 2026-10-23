"""Development-only CUDA driver runner; no toolkit or CUDA-enabled Torch needed.

JIT the same emitted PTX for the local device. Contest-target validation remains
the official hosted checker. All allocations and the context are released.
"""
from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


class Driver:
    def __init__(self):
        self.lib = ct.CDLL('libcuda.so.1')
        signatures = {
            'cuInit': [ct.c_uint], 'cuDeviceGet': [ct.POINTER(ct.c_int), ct.c_int],
            'cuDeviceComputeCapability': [ct.POINTER(ct.c_int), ct.POINTER(ct.c_int), ct.c_int],
            'cuCtxCreate_v2': [ct.POINTER(ct.c_void_p), ct.c_uint, ct.c_int],
            'cuCtxDestroy_v2': [ct.c_void_p], 'cuCtxSynchronize': [],
            'cuModuleLoadData': [ct.POINTER(ct.c_void_p), ct.c_void_p],
            'cuModuleUnload': [ct.c_void_p],
            'cuModuleGetFunction': [ct.POINTER(ct.c_void_p), ct.c_void_p, ct.c_char_p],
            'cuFuncSetAttribute': [ct.c_void_p, ct.c_int, ct.c_int],
            'cuMemAlloc_v2': [ct.POINTER(ct.c_uint64), ct.c_size_t],
            'cuMemFree_v2': [ct.c_uint64],
            'cuMemcpyHtoD_v2': [ct.c_uint64, ct.c_void_p, ct.c_size_t],
            'cuMemcpyDtoH_v2': [ct.c_void_p, ct.c_uint64, ct.c_size_t],
            'cuMemsetD8_v2': [ct.c_uint64, ct.c_ubyte, ct.c_size_t],
            'cuEventCreate': [ct.POINTER(ct.c_void_p), ct.c_uint],
            'cuEventRecord': [ct.c_void_p, ct.c_void_p],
            'cuEventRecordWithFlags': [ct.c_void_p, ct.c_void_p, ct.c_uint],
            'cuEventSynchronize': [ct.c_void_p],
            'cuEventElapsedTime': [ct.POINTER(ct.c_float), ct.c_void_p, ct.c_void_p],
            'cuEventDestroy_v2': [ct.c_void_p],
            'cuStreamCreate': [ct.POINTER(ct.c_void_p), ct.c_uint],
            'cuStreamDestroy_v2': [ct.c_void_p],
            'cuStreamBeginCapture_v2': [ct.c_void_p, ct.c_int],
            'cuStreamEndCapture': [ct.c_void_p, ct.POINTER(ct.c_void_p)],
            'cuStreamSynchronize': [ct.c_void_p],
            'cuGraphInstantiateWithFlags': [ct.POINTER(ct.c_void_p), ct.c_void_p, ct.c_uint64],
            'cuGraphLaunch': [ct.c_void_p, ct.c_void_p],
            'cuGraphExecDestroy': [ct.c_void_p],
            'cuGraphDestroy': [ct.c_void_p],
            'cuLaunchKernel': [ct.c_void_p, ct.c_uint, ct.c_uint, ct.c_uint,
                               ct.c_uint, ct.c_uint, ct.c_uint, ct.c_uint,
                               ct.c_void_p, ct.POINTER(ct.c_void_p), ct.c_void_p],
            'cuGetErrorName': [ct.c_int, ct.POINTER(ct.c_char_p)],
        }
        for name, signature in signatures.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = signature, ct.c_int
        self.call('cuInit', 0)
        device = ct.c_int()
        self.call('cuDeviceGet', ct.byref(device), 0)
        major, minor = ct.c_int(), ct.c_int()
        self.call('cuDeviceComputeCapability', ct.byref(major), ct.byref(minor), device)
        self.arch = f'sm_{major.value}{minor.value}'
        self.context = ct.c_void_p()
        self.call('cuCtxCreate_v2', ct.byref(self.context), 0, device)

    def call(self, name, *args):
        status = getattr(self.lib, name)(*args)
        if status:
            message = ct.c_char_p()
            self.lib.cuGetErrorName(status, ct.byref(message))
            raise RuntimeError(f'{name}: {message.value.decode() if message.value else status}')

    def close(self):
        self.call('cuCtxDestroy_v2', self.context)

    def run(self, ptx, kernel, inputs, timed_iters=0):
        module = ct.c_void_p()
        allocations = []
        events = []
        stream, graph, executable = ct.c_void_p(), ct.c_void_p(), ct.c_void_p()
        self.last_timings_us = []
        arrays, parameters = {}, []
        types = {'f32': np.float32, 'f16': np.float16, 'f64': np.float64,
                 'i32': np.int32, 'i64': np.int64}
        try:
            source = ct.create_string_buffer(ptx.replace('.target sm_89', f'.target {self.arch}').encode())
            self.call('cuModuleLoadData', ct.byref(module), ct.cast(source, ct.c_void_p))
            function = ct.c_void_p()
            self.call('cuModuleGetFunction', ct.byref(function), module, kernel['entry'].encode())
            if kernel['smem_bytes']:
                self.call('cuFuncSetAttribute', function, 8, kernel['smem_bytes'])
            for arg in kernel['args']:
                dtype, shape = types[arg['dtype']], arg['shape']
                if arg.get('output'):
                    array = np.zeros(shape, dtype=dtype)
                elif arg['name'] in inputs:
                    array = np.asarray(inputs[arg['name']], dtype=dtype).reshape(shape)
                elif 'data' in arg:
                    array = np.asarray(arg['data'], dtype=dtype).reshape(shape)
                else:
                    array = np.random.default_rng(arg['seed']).standard_normal(shape).astype(dtype)
                array = np.ascontiguousarray(array)
                pointer = ct.c_uint64()
                self.call('cuMemAlloc_v2', ct.byref(pointer), max(1, array.nbytes))
                allocations.append(pointer)
                self.call('cuMemcpyHtoD_v2', pointer, array.ctypes.data, array.nbytes)
                arrays[arg['name']] = (array, pointer)
                parameters.append(pointer)
            addresses = (ct.c_void_p * len(parameters))(*(ct.addressof(p) for p in parameters))
            self.call('cuLaunchKernel', function, *kernel['grid'], *kernel['block'],
                      kernel['smem_bytes'], None, addresses, None)
            self.call('cuCtxSynchronize')
            if timed_iters:
                for _ in range(2):
                    event = ct.c_void_p()
                    self.call('cuEventCreate', ct.byref(event), 0)
                    events.append(event)
                # Capture events with the kernel so Python dispatch cannot add
                # an idle host gap between the measured start and launch.
                self.call('cuStreamCreate', ct.byref(stream), 1)
                self.call('cuStreamBeginCapture_v2', stream, 0)
                self.call('cuEventRecordWithFlags', events[0], stream, 1)
                self.call('cuLaunchKernel', function, *kernel['grid'], *kernel['block'],
                          kernel['smem_bytes'], stream, addresses, None)
                self.call('cuEventRecordWithFlags', events[1], stream, 1)
                self.call('cuStreamEndCapture', stream, ct.byref(graph))
                self.call('cuGraphInstantiateWithFlags', ct.byref(executable), graph, 0)
                for iteration in range(timed_iters + 5):
                    for name in kernel['zero_outputs']:
                        array, pointer = arrays[name]
                        self.call('cuMemsetD8_v2', pointer, 0, array.nbytes)
                    if kernel['zero_outputs']:
                        # The graph stream is nonblocking and does not inherit
                        # legacy default-stream ordering from these memsets.
                        self.call('cuCtxSynchronize')
                    self.call('cuGraphLaunch', executable, stream)
                    self.call('cuStreamSynchronize', stream)
                    milliseconds = ct.c_float()
                    self.call('cuEventElapsedTime', ct.byref(milliseconds), *events)
                    if iteration >= 5:
                        self.last_timings_us.append(milliseconds.value * 1000)
            outputs = {}
            for arg in kernel['args']:
                if arg.get('output'):
                    array, pointer = arrays[arg['name']]
                    self.call('cuMemcpyDtoH_v2', array.ctypes.data, pointer, array.nbytes)
                    outputs[arg['name']] = array
            return outputs
        finally:
            if executable.value:
                self.call('cuGraphExecDestroy', executable)
            if graph.value:
                self.call('cuGraphDestroy', graph)
            if stream.value:
                self.call('cuStreamDestroy_v2', stream)
            for event in events:
                self.call('cuEventDestroy_v2', event)
            for pointer in allocations:
                self.call('cuMemFree_v2', pointer)
            if module.value:
                self.call('cuModuleUnload', module)


def check_problem(driver, folder):
    root = Path(__file__).resolve().parents[1]
    meta = json.loads((folder / 'meta.json').read_text())
    with np.load(folder / 'expected.npz') as expected:
        produced = {}
        for kernel in meta['kernels']:
            result = subprocess.run([sys.executable, str(root / 'compile.py'),
                                     str(folder / f"{kernel['entry']}.ir.json")],
                                    capture_output=True, text=True, check=True)
            produced.update(driver.run(result.stdout, kernel, produced))
        for name in expected.files:
            np.testing.assert_allclose(produced[name], expected[name], rtol=meta['rtol'], atol=meta['atol'])
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('problems', nargs='+', type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    driver = Driver()
    failures = []
    rows = []
    root = Path(__file__).resolve().parents[1]
    fingerprint = hashlib.sha256((root / 'compile.py').read_bytes()).hexdigest()
    try:
        print(f'Local PTX target: {driver.arch}', flush=True)
        for folder in args.problems:
            start = time.monotonic()
            row = {'problem': folder.name}
            try:
                check_problem(driver, folder)
                row['passed'] = True
                print(f'PASS {folder.name}', flush=True)
            except Exception as error:
                failures.append(folder.name)
                row.update(passed=False, error=str(error))
                print(f'FAIL {folder.name}: {error}', flush=True)
                if isinstance(error, subprocess.CalledProcessError):
                    print(error.stderr, flush=True)
                    row['compiler_stderr'] = error.stderr
            row['elapsed_s'] = round(time.monotonic() - start, 3)
            rows.append(row)
            if args.report:
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_text(json.dumps({'compiler_sha256': fingerprint,
                                                   'arch': driver.arch, 'results': rows}, indent=2) + '\n')
    finally:
        driver.close()
    print(f'{len(args.problems)-len(failures)}/{len(args.problems)} passed')
    return bool(failures)


if __name__ == '__main__':
    sys.exit(main())
