"""Sequential tmux controller for both single-variable paired experiments."""
import argparse
import datetime
import json
import os
import pathlib
import subprocess
import sys

from axis_mode_common import atomic_json,sha256
from run_axis_fixed_routes import compare

REPO=pathlib.Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--phase',choices=['smoke','formal','all'],default='all');a=p.parse_args();root=pathlib.Path(a.root);top=json.loads((root/'config.json').read_text());parent=pathlib.Path(top['parent_root']);assert sha256(parent/'integrity.json')==top['parent_integrity_sha256']
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    def status(state,**extra):atomic_json(root/'controller_status.json',{'state':state,'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra})
    def run(sub,label,args,stop=None):
        path=sub/label
        if (path/'seal.json').exists():return
        cmd=[sys.executable,str(REPO/'repro/tools/run_axis_mode_trial.py'),'--root',str(sub),'--tag',label,*args]
        if (path/'checkpoint.pt').exists():cmd+=['--resume']
        if stop:cmd+=['--stop-after',str(stop)]
        with (root/'commands.jsonl').open('a') as f:f.write(json.dumps(cmd)+'\n')
        status('running',mode=sub.name,job=label)
        with (sub/f'{label}.log').open('ab') as log:subprocess.run(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        if not stop:assert (path/'seal.json').exists()
    try:
        if a.phase in ['smoke','all']:
            for mode in top['modes']:
                sub=root/mode;cfg=json.loads((sub/'config.json').read_text())
                if not (sub/'gradient_and_freeze_checks.json').exists():
                    status('checking_gradients',mode=mode)
                    with (sub/'preflight.log').open('ab') as log:subprocess.run([sys.executable,str(REPO/'repro/tools/check_axis_mode_routes.py'),'--root',str(sub)],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                assert json.loads((sub/'gradient_and_freeze_checks.json').read_text())['status']=='passed'
                for case in cfg['test_cases']:
                    for arm in ['A','B']:run(sub,f'direct_{case}_{arm}_smoke',['--route','direct','--case',case,'--arm',arm,'--steps','30'])
                for arm in ['A','B']:run(sub,f'head_{arm}_smoke',['--route','head','--arm',arm,'--steps','30'])
                run(sub,'direct_zero_control',['--route','direct','--case','8961','--arm','B','--steps','30','--zero-weight'])
                run(sub,'head_zero_control',['--route','head','--arm','B','--steps','30','--zero-weight'])
                compare(sub/'direct_8961_A_smoke/checkpoint.pt',sub/'direct_zero_control/checkpoint.pt',sub/'direct_zero_control.json')
                compare(sub/'head_A_smoke/checkpoint.pt',sub/'head_zero_control/checkpoint.pt',sub/'head_zero_control.json')
                args=['--route','head','--arm','B','--steps','30']
                if not (sub/'resume_head/checkpoint.pt').exists():run(sub,'resume_head',args,stop=10)
                run(sub,'resume_head',args)
                compare(sub/'head_B_smoke/checkpoint.pt',sub/'resume_head/checkpoint.pt',sub/'resume_control.json')
            status('smoke_complete')
        if a.phase in ['formal','all']:
            for mode in top['modes']:
                sub=root/mode;cfg=json.loads((sub/'config.json').read_text())
                assert json.loads((sub/'gradient_and_freeze_checks.json').read_text())['status']=='passed'
                for name in ['direct_zero_control.json','head_zero_control.json','resume_control.json']:assert json.loads((sub/name).read_text())['passed']
                for case in cfg['test_cases']:
                    for arm in ['A','B']:run(sub,f'direct_{case}_{arm}',['--route','direct','--case',case,'--arm',arm])
                for arm in ['A','B']:run(sub,f'head_shared_{arm}',['--route','head','--arm',arm])
                frozen=json.loads((sub/'frozen_assets.json').read_text());assert all(sha256(f['path'])==f['sha256'] for f in frozen)
                atomic_json(sub/'controller_status.json',{'state':'optimization_complete','mode':mode})
            status('optimization_complete')
    except BaseException as exc:status('failed',error=f'{type(exc).__name__}: {exc}');raise


if __name__=='__main__':main()
