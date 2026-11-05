"""Prepare fixed recipes and run case workers sequentially on the one GPU."""
import argparse
import datetime
import json
import os
import pathlib
import subprocess
import sys

from collision_refinement_common import sha256
from multicase_closed_common import REPO,fit_case,prepare_case,recipes,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--stage',choices=['prepare','fit','all'],default='all');p.add_argument('--case');a=p.parse_args()
    root=pathlib.Path(a.root).resolve()
    if a.case:
        fit_case(root/a.case);return
    if a.stage in ['prepare','all']:
        root.mkdir(parents=True,exist_ok=False)
        r=recipes();write_json(root/'recipes.json',r)
        write_json(root/'protocol.json',{'cases':list(r),'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   'training':'One annotated closed image each; 100 registration + 240 geometry steps, final iterate',
                   'methods':['original','prior','optimized'],
                   'evaluation':'Fitting image, fixed unseen open-state cameras, type/independent-instance coverage, GT unsigned axis angle and normalized hinge position, 3D surface distances',
                   'axis_correct_threshold':'5 degrees direction, and 5% GT static diagonal pivot-to-axis for hinges. Missing/wrong-type/duplicate-owned joints count as failure.',
                   'mapping':'Post-seal type-blind geometric assignment on pre-optimization semantic groups; fixed original-static similarity for all variants; no axis-driven permutations',
                   'case_failure_policy':'Keep every selected case and report failures; no removal based on metrics',
                   'explicit_priors':'Manual instance recipes and visible front rectangles, not a learned semantic detector',
                   'code_sha256':{n:sha256(REPO/'repro/tools'/n) for n in ['multicase_closed_common.py','run_multicase_closed.py']}})
        for c,recipe in r.items():
            prepare_case(root,c,recipe);print('PREPARED',c,flush=True)
        (REPO/'repro/current_closed_multicase.txt').write_text(str(root)+'\n')
    if a.stage in ['fit','all']:
        protocol=json.loads((root/'protocol.json').read_text());states={}
        for case in protocol['cases']:
            out=root/case;seal=out/'fit_results.json'
            if seal.exists():states[case]='complete';continue
            log=out/'fit.log'
            env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4')
            with log.open('w') as f:
                proc=subprocess.run([sys.executable,__file__,'--root',str(root),'--case',case],stdout=f,stderr=subprocess.STDOUT,env=env,cwd=REPO)
            states[case]='complete' if proc.returncode==0 and seal.exists() else 'failed'
            write_json(root/'controller_status.json',{'state':'fitting','cases':states})
            print('CASE FIT',case,states[case],flush=True)
        write_json(root/'controller_status.json',{'state':'fit_complete','cases':states})


if __name__=='__main__':main()
