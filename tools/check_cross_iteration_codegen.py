"""Verify explicit positive/negative vector-load expectations from case manifests."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

from emmy.compiler.graph import Graph


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifests',nargs='+',type=Path)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    modules=[]
    for label,path in [('baseline',args.baseline),('candidate',args.candidate)]:
        spec=importlib.util.spec_from_file_location(label,path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        modules.append(module)
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    report={'baseline_sha256':digest(args.baseline),'candidate_sha256':digest(args.candidate),'results':[]}
    for manifest in args.manifests:
        for case in json.loads(manifest.read_text())['cases']:
            folder=Path(case['folder']);ir=next(folder.glob('*.ir.json'))
            assert digest(ir)==case['ir_sha256'],f'Fixture changed: {ir}'
            launch=json.loads((folder/'meta.json').read_text())['kernels'][0]
            emitted=[module.Compiler(Graph.from_dict(json.loads(ir.read_text())),launch).compile() for module in modules]
            counts=[ptx.count('ld.shared.v2.f32')+ptx.count('ld.shared.v4.f32') for ptx in emitted]
            report['results'].append(dict(case,baseline_vectors=counts[0],candidate_vectors=counts[1],
                passed=(counts[1]>counts[0])==case['expect_new_vector']))
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    passed=sum(r['passed'] for r in report['results'])
    print(f'{passed}/{len(report["results"])} code-generation expectations met')
    if passed!=len(report['results']):raise SystemExit(1)


if __name__=='__main__':main()
