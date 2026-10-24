"""Inspect driver-linked cubins using NVIDIA cuobjdump, without executing kernels.

This development-only path uses cuLink*, not the submission compiler's lowering
and not the hosted runner's backend. Cross-target output is local-driver codegen
for the requested architecture, not captured SASS from the remote GPU.
"""
import argparse
import ctypes as ct
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

from local_runner import Driver


def sha(data):
    return hashlib.sha256(data).hexdigest()


def link(driver, ptx, target):
    state, binary, size = ct.c_void_p(), ct.c_void_p(), ct.c_size_t()
    options = (ct.c_int*1)(9)  # CU_JIT_TARGET
    values = (ct.c_void_p*1)(target)
    source = ct.create_string_buffer(ptx.encode())
    try:
        driver.call('cuLinkCreate_v2',1,options,values,ct.byref(state))
        driver.call('cuLinkAddData_v2',state,1,ct.cast(source,ct.c_void_p),ct.sizeof(source),
                    b'kernel.ptx',0,None,None)  # CU_JIT_INPUT_PTX
        driver.call('cuLinkComplete',state,ct.byref(binary),ct.byref(size))
        return ct.string_at(binary,size.value)  # copy before destroying link state
    finally:
        if state.value:
            driver.call('cuLinkDestroy',state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('problems',nargs='+',type=Path)
    parser.add_argument('--compiler',type=Path,default=Path('compile.py'))
    parser.add_argument('--cuobjdump',type=Path,required=True)
    parser.add_argument('--nvdisasm',type=Path,required=True)
    parser.add_argument('--target',help='sm_86 or sm_120; defaults to local device')
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    env = dict(os.environ)
    env['PATH'] = str(args.nvdisasm.resolve().parent)+os.pathsep+env.get('PATH','')
    driver = Driver()
    signatures = {
        'cuLinkCreate_v2':[ct.c_uint,ct.POINTER(ct.c_int),ct.POINTER(ct.c_void_p),ct.POINTER(ct.c_void_p)],
        'cuLinkAddData_v2':[ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_size_t,ct.c_char_p,ct.c_uint,ct.POINTER(ct.c_int),ct.POINTER(ct.c_void_p)],
        'cuLinkComplete':[ct.c_void_p,ct.POINTER(ct.c_void_p),ct.POINTER(ct.c_size_t)],
        'cuLinkDestroy':[ct.c_void_p], 'cuDriverGetVersion':[ct.POINTER(ct.c_int)],
    }
    for name,signature in signatures.items():
        function=getattr(driver.lib,name)
        function.argtypes,function.restype=signature,ct.c_int
    target = int((args.target or driver.arch).removeprefix('sm_'))
    version=ct.c_int()
    driver.call('cuDriverGetVersion',ct.byref(version))
    args.out.mkdir(parents=True,exist_ok=True)
    report={'profiler_sha256':sha(Path(__file__).read_bytes()),
            'compiler_sha256':sha(args.compiler.read_bytes()),'local_arch':driver.arch,
            'requested_codegen_target':f'sm_{target}','driver_version':version.value,
            'cuobjdump_sha256':sha(args.cuobjdump.read_bytes()),
            'nvdisasm_sha256':sha(args.nvdisasm.read_bytes()),
            'cuobjdump_version':subprocess.check_output([str(args.cuobjdump),'--version'],text=True),
            'executes_kernels':False,'path':'local driver cuLink relocatable PTX compilation; not hosted cubin capture', 'results':[]}
    try:
        for folder in args.problems:
            metadata=(folder/'meta.json').read_bytes()
            for kernel in json.loads(metadata)['kernels']:
                ir=folder/(kernel['entry']+'.ir.json')
                emitted=subprocess.check_output([sys.executable,str(args.compiler),str(ir)],text=True)
                effective=emitted.replace('.target sm_89',f'.target sm_{target}') if target<89 else emitted
                data=link(driver,effective,target)
                stem=args.out/(folder.name+'--'+kernel['entry'])
                binary=stem.with_suffix('.cubin')
                binary.write_bytes(data)
                sass=subprocess.check_output([str(args.cuobjdump),'--dump-sass',str(binary)],text=True,env=env)
                resources=subprocess.check_output([str(args.cuobjdump),'--dump-resource-usage',str(binary)],text=True,env=env)
                instructions=[]
                functions={}
                function=None
                for line in sass.splitlines():
                    declaration=re.search(r'Function\s*:\s*(\S+)',line)
                    if declaration:
                        function=declaration.group(1)
                        functions[function]=[]
                    match=re.search(r'/\*[0-9a-f]+\*/\s*((?:@!?P\d+\s+)?[A-Z][A-Z0-9_.]*(?:\s+[^;]*)?);',line)
                    if match:
                        if function is None:raise RuntimeError('SASS instruction outside a function')
                        instruction=match.group(1).strip()
                        instructions.append(instruction)
                        functions[function].append(instruction)
                if not instructions or kernel['entry'] not in functions:
                    raise RuntimeError('Expected entry missing from parsed SASS')
                if f'code for sm_{target}' not in sass:
                    raise RuntimeError('Disassembly architecture does not match requested target')
                opcodes=Counter(re.sub(r'^@!?P\d+\s+','',line).split()[0] for line in instructions)
                stem.with_suffix('.sass.txt').write_text(sass)
                stem.with_suffix('.resources.txt').write_text(resources)
                stem.with_suffix('.ptx').write_text(effective)
                row={'problem':folder.name,'entry':kernel['entry'],'ir_sha256':sha(ir.read_bytes()),
                     'meta_sha256':sha(metadata),'emitted_ptx_sha256':sha(emitted.encode()),
                     'effective_ptx_sha256':sha(effective.encode()),'cubin_sha256':sha(data),
                     'sass_sha256':sha(sass.encode()),'instruction_sequence_sha256':sha('\n'.join(instructions).encode()),
                     'static_instruction_count':len(instructions),
                     'entry_instruction_count':len(functions[kernel['entry']]),
                     'function_instruction_counts':{name:len(body) for name,body in functions.items()},
                     'counts_include':'all functions and padding, not dynamic execution counts',
                     'opcodes':dict(sorted(opcodes.items())),
                     'resources':resources,'disassembly':sass}
                report['results'].append(row)
                (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
                print(folder.name,len(instructions),'SASS instructions',flush=True)
    finally:
        driver.close()


if __name__=='__main__':
    main()
