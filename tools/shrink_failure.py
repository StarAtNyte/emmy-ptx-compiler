"""Delete IR statement chunks while preserving an independently checked GPU failure.

A retained trial must still pass the original fixture's numerical oracle on the
reference compiler, compile on both compilers, and fail numerically on the suspect.
Each GPU execution gets a fresh process/context. No timeouts or compilation
errors count as the numerical defect. This is statement deletion, not a general
IR minimizer or an independent reference implementation.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

from emmy.compiler.graph import Graph


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def blocks(body, path=()):
    yield path, body
    for index, stmt in enumerate(body):
        for field in ('body', 'else_body'):
            if hasattr(stmt, field):
                yield from blocks(getattr(stmt, field), path+((index, field),))


def edit(body, path, replacement):
    if not path:
        return tuple(replacement)
    index, field = path[0]
    result = list(body)
    result[index] = replace(result[index], **{field: edit(getattr(result[index], field), path[1:], replacement)})
    return tuple(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('problem', type=Path)
    parser.add_argument('--suspect', type=Path, required=True)
    parser.add_argument('--reference', type=Path, default=Path('compile.py'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-trials', type=int, default=150)
    parser.add_argument('--timeout', type=float, default=30)
    args = parser.parse_args()
    if args.max_trials < 2 or args.timeout <= 0:
        parser.error('max-trials must be >= 2 and timeout positive')
    if args.out.exists():
        parser.error('--out must not already exist')
    root = Path(__file__).resolve().parents[1]
    meta = json.loads((args.problem/'meta.json').read_text())
    if len(meta['kernels']) != 1:
        parser.error('statement reduction currently requires a single kernel')
    launch = meta['kernels'][0]
    ir_name = launch['entry']+'.ir.json'
    graph = Graph.from_dict(json.loads((args.problem/ir_name).read_text()))
    node = next(n for n in graph.nodes.values() if type(n.op).__name__ == 'KernelOp')
    original = tuple(node.op.body)
    modules = {name: load(name+'_compiler', path) for name, path in [('reference',args.reference),('suspect',args.suspect)]}
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    report = {'reducer_sha256':digest(Path(__file__)), 'reference_sha256':digest(args.reference), 'suspect_sha256':digest(args.suspect),
              'input_ir_sha256':digest(args.problem/ir_name),
              'meta_sha256':digest(args.problem/'meta.json'),
              'oracle_sha256':digest(args.problem/'expected.npz'),
              'predicate':'reference passes unchanged fixture oracle; suspect numerical mismatch; both compile',
              'trials':[], 'accepted_deletions':[]}
    def serialize(body):
        node.op = replace(node.op, body=body)
        return json.dumps(graph.to_dict(), indent=2, default=str)+'\n'
    def count(body):
        return sum(len(b) for _,b in blocks(body))
    args.out.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix='emmy-shrink-') as temporary:
        folder = Path(temporary)/'fixture'
        folder.mkdir()
        for name in ('meta.json','expected.npz'):
            shutil.copy2(args.problem/name, folder/name)
        def evaluate(body, label):
            raw = serialize(body)
            row = {'label':label, 'statements':count(body), 'ir_sha256':hashlib.sha256(raw.encode()).hexdigest()}
            report['trials'].append(row)
            for name,module in modules.items():
                try:
                    module.Compiler(Graph.from_dict(json.loads(raw)),launch).compile()
                except (ValueError, NotImplementedError, KeyError, AssertionError) as error:
                    row.update(retained=False, reason=f'{name} compile rejection', error=str(error))
                    return False
            (folder/ir_name).write_text(raw)
            for name,compiler in [('reference',args.reference),('suspect',args.suspect)]:
                detail = Path(temporary)/(name+'.json')
                detail.unlink(missing_ok=True)
                try:
                    run = subprocess.run([sys.executable,str(root/'tools/local_runner.py'),str(folder),
                        '--compiler',str(compiler.resolve()),'--report',str(detail)],
                        capture_output=True,text=True,timeout=args.timeout)
                except subprocess.TimeoutExpired:
                    row.update(retained=False,reason=f'{name} timeout')
                    return False
                result = json.loads(detail.read_text()) if detail.exists() else {}
                row[name] = {'exit_code':run.returncode,'detail':result,'stderr':run.stderr}
                entries = result.get('results',[])
                passed = run.returncode == 0 and len(entries)==1 and entries[0].get('passed')
                mismatch = (run.returncode != 0 and len(entries)==1
                            and 'Not equal to tolerance' in entries[0].get('error',''))
                if (name=='reference' and not passed) or (name=='suspect' and not mismatch):
                    row.update(retained=False,reason=f'{name} predicate not satisfied')
                    return False
            row.update(retained=True,reason='preserved numerical failure')
            return True
        def probe(body, label):
            retained = evaluate(body, label)
            (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
            return retained
        current = original
        if not probe(current,'initial'):
            (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
            raise SystemExit('Input does not satisfy the numerical-failure predicate')
        exhaustive = False
        while len(report['trials']) < args.max_trials-1:
            changed = False
            for path,body in blocks(current):
                if not body:
                    continue
                chunk = max(1,math.ceil(len(body)/2))
                while chunk:
                    for offset in range(0,len(body),chunk):
                        if len(report['trials']) >= args.max_trials-1:
                            break
                        replacement = tuple(body[:offset])+tuple(body[offset+chunk:])
                        trial = edit(current,path,replacement)
                        label = {'path':path,'offset':offset,'size':min(chunk,len(body)-offset)}
                        if probe(trial,label):
                            report['accepted_deletions'].append(label)
                            current = trial
                            changed = True
                            print(f'Retained {count(current)} statements',flush=True)
                            break
                    if changed or len(report['trials']) >= args.max_trials-1:
                        break
                    chunk = 0 if chunk==1 else max(1,chunk//2)
                if changed or len(report['trials']) >= args.max_trials-1:
                    break
            if not changed:
                exhaustive = len(report['trials']) < args.max_trials-1
                break
        if not probe(current,'final independent replay'):
            report['status'] = 'final replay failed; no reduced fixture published'
            (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
            raise SystemExit(report['status'])
        if digest(args.reference) != report['reference_sha256'] or digest(args.suspect) != report['suspect_sha256']:
            report['status'] = 'compiler source changed; no reduced fixture published'
            (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
            raise SystemExit(report['status'])
        (args.out/ir_name).write_text(serialize(current))
        for name in ('meta.json','expected.npz'):
            shutil.copy2(args.problem/name,args.out/name)
    inconclusive = sum('timeout' in trial['reason'] for trial in report['trials'])
    report.update(initial_statements=count(original), final_statements=count(current),
                  inconclusive_probes=inconclusive, deletion_fixed_point=exhaustive and not inconclusive,
                  status=('search included timeouts; not a verified deletion fixed point' if inconclusive
                          else 'no further single-statement deletion satisfies predicate' if exhaustive
                          else 'trial budget exhausted; not a deletion fixed point'))
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'{count(original)} -> {count(current)} statements; {report["status"]}')


if __name__ == '__main__':
    main()
