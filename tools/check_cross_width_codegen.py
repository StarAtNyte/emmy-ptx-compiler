"""Check wide-read selection and exact preservation of two-wide/scalar fallback."""
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
    args=parser.parse_args();modules=[]
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for name,path in [('baseline',args.baseline),('candidate',args.candidate)]:
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);modules.append(module)
    report={'baseline_sha256':digest(args.baseline),'candidate_sha256':digest(args.candidate),'results':[]}
    for manifest in args.manifests:
        for case in json.loads(manifest.read_text())['cases']:
            folder=Path(case['folder']);ir=next(folder.glob('*.ir.json'));assert digest(ir)==case['ir_sha256']
            launch=json.loads((folder/'meta.json').read_text())['kernels'][0]
            before,after=[m.Compiler(Graph.from_dict(json.loads(ir.read_text())),launch).compile() for m in modules]
            width=case['expected_cross_width'];v4=after.count('ld.shared.v4.f32');v2=after.count('ld.shared.v2.f32')
            selected=(v4>0 if width==4 else v4==0 and (v2>0 if width==2 else v2==0))
            report['results'].append(dict(case,v4=v4,v2=v2,unchanged=before==after,
                                         passed=selected and (width==4 or before==after)))
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2)+'\n')
    passed=sum(x['passed'] for x in report['results']);print(f'{passed}/{len(report["results"])} expectations met')
    if passed!=len(report['results']):raise SystemExit(1)


if __name__=='__main__':main()
