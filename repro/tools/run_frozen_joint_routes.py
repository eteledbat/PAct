"""Resumable sequential tmux runner: real-data checks, matched smokes, formal runs."""
import argparse
import datetime
import json
import os
import pathlib
import subprocess
import sys
import time

import torch

from frozen_joint_common import atomic_json,sha256

REPO=pathlib.Path(__file__).resolve().parents[2]


def compare(a,b,out):
    x=torch.load(a,map_location='cpu',weights_only=False)['model'];y=torch.load(b,map_location='cpu',weights_only=False)['model']
    assert x.keys()==y.keys()
    errors={k:float((x[k]-y[k]).abs().max()) for k in x}
    result={'first':str(a),'second':str(b),'max_parameter_difference':max(errors.values()),'per_parameter':errors,'tolerance':1e-5,'passed':max(errors.values())<1e-5}
    atomic_json(out,result);assert result['passed'],result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--phase',choices=['smoke','formal','all'],default='all');a=p.parse_args()
    root=pathlib.Path(a.root);cfg=json.loads((root/'config.json').read_text());started=time.monotonic();env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    def status(state,**kw):atomic_json(root/'controller_status.json',{'state':state,'elapsed_seconds':time.monotonic()-started,'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw})
    def run(label,args,stop=None):
        path=root/label
        if (path/'seal.json').exists():return
        cmd=[sys.executable,str(REPO/'repro/tools/run_frozen_joint_trial.py'),'--root',str(root),'--tag',label,*args]
        if (path/'checkpoint.pt').exists():cmd+=['--resume']
        if stop:cmd+=['--stop-after',str(stop)]
        atomic_json(root/f'{label}_command.json',cmd);status('running',job=label)
        with (root/f'{label}.log').open('ab') as log:subprocess.run(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        if not stop:assert (path/'seal.json').exists()
    try:
        assert 'baseline_association' in cfg,'Same-run frozen feature/assets not finalized'
        for c in cfg['runs']:assert json.loads((root/'features'/c/'status.json').read_text())['state']=='complete'
        if not (root/'gradient_and_rigidity_checks.json').exists():
            status('checking_gradients')
            with (root/'gradient_checks.log').open('ab') as log:subprocess.run([sys.executable,str(REPO/'repro/tools/check_frozen_joint_routes.py'),'--root',str(root)],env=env,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
        assert json.loads((root/'gradient_and_rigidity_checks.json').read_text())['status']=='passed'
        if a.phase in ['smoke','all']:
            for c in cfg['direct_cases']:
                for arm in ['A','B']:run(f'direct_{c}_{arm}_smoke',['--route','direct','--case',c,'--arm',arm,'--steps',str(cfg['smoke_steps'])])
            for arm in ['A','B']:run(f'head_{arm}_smoke',['--route','head','--arm',arm,'--steps',str(cfg['smoke_steps'])])
            run('direct_zero_control',['--route','direct','--case','8961','--arm','B','--steps','30','--zero-weight'])
            run('head_zero_control',['--route','head','--arm','B','--steps','30','--zero-weight'])
            compare(root/'direct_8961_A_smoke/checkpoint.pt',root/'direct_zero_control/checkpoint.pt',root/'direct_zero_control.json')
            compare(root/'head_A_smoke/checkpoint.pt',root/'head_zero_control/checkpoint.pt',root/'head_zero_control.json')
            args=['--route','head','--arm','B','--steps','30']
            if not (root/'resume_head/seal.json').exists():
                if not (root/'resume_head/checkpoint.pt').exists():run('resume_head',args,stop=10)
                run('resume_head',args)
            compare(root/'head_B_smoke/checkpoint.pt',root/'resume_head/checkpoint.pt',root/'resume_control.json')
            status('smoke_complete')
        if a.phase in ['formal','all']:
            for name in ['direct_zero_control.json','head_zero_control.json','resume_control.json']:assert json.loads((root/name).read_text())['passed']
            for c in cfg['direct_cases']:
                for arm in ['A','B']:run(f'direct_{c}_{arm}',['--route','direct','--case',c,'--arm',arm])
            for arm in ['A','B']:run(f'head_shared_{arm}',['--route','head','--arm',arm])
            frozen=json.loads((root/'frozen_assets.json').read_text());assert all(sha256(f['path'])==f['sha256'] for f in frozen)
            status('optimization_complete',evaluation='pending')
    except BaseException as exc:status('failed',error=f'{type(exc).__name__}: {exc}');raise


if __name__=='__main__':main()
