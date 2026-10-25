"""Sequential tmux controller: safety gates then six fixed-budget paired trials."""
import argparse
import datetime
import json
import os
import pathlib
import subprocess
import sys
import time

import torch

REPO=pathlib.Path(__file__).resolve().parents[2]


def compare(first,second,path):
    a=torch.load(first,map_location='cpu',weights_only=False)['model'];b=torch.load(second,map_location='cpu',weights_only=False)['model'];assert a.keys()==b.keys();errors={k:float((a[k]-b[k]).abs().max()) for k in a};maximum=max(errors.values());result={'first':str(first),'second':str(second),'max_absolute_parameter_difference':maximum,'per_parameter_max_difference':errors,'tolerance':1e-5,'passed':maximum<1e-5};path.write_text(json.dumps(result,indent=2));assert result['passed'],result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=pathlib.Path(a.root).resolve();cfg=json.loads((root/'config.json').read_text());start=time.monotonic();env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    def status(state,**kw):
        path=root/'controller_status.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps({'state':state,'elapsed_seconds':time.monotonic()-start,'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw},indent=2));tmp.replace(path)
    def job(label,args):
        out=root/label
        if (out/'seal.json').exists():print('REUSE SEALED',label,flush=True);return
        command=[sys.executable,str(REPO/'repro/tools/run_reconstruction_trial.py'),'--root',str(root),*args]
        if (out/'checkpoint.pt').exists():command+=['--resume']
        (root/f'{label}_command.json').write_text(json.dumps(command,indent=2));status('running',job=label)
        with (root/f'{label}.log').open('ab') as log:subprocess.run(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        assert (out/'seal.json').exists();print('COMPLETE',label,flush=True)
    try:
        for case in cfg['cases']:
            assert json.loads((root/f'gradient_{case}.json').read_text())['status']=='passed'
            assert json.loads((root/f'prealign_{case}/status.json').read_text())['state']=='complete'
        for experiment in cfg['experiments']:
            for arm in ['A','B']:
                name=experiment['name'];job(f'{name}_{arm}_smoke',['--stage','smoke','--experiment',name,'--arm',arm,'--steps','30'])
        job('zero_weight_control_8961',['--stage','smoke','--experiment','8961_native','--arm','B','--steps','30','--weight-override','0','--tag','zero_weight_control_8961'])
        compare(root/'8961_native_A_smoke/checkpoint.pt',root/'zero_weight_control_8961/checkpoint.pt',root/'zero_weight_control.json')
        status('smoke_passed')
        for experiment in cfg['experiments']:
            for arm in ['A','B']:
                name=experiment['name'];job(f'{name}_{arm}_train',['--stage','train','--experiment',name,'--arm',arm])
        compare(root/'8961_native_A_train/checkpoint.pt',root/'8961_sweep90_A_train/checkpoint.pt',root/'native_vs_sweep_A_control.json')
        status('optimization_complete',evaluation='pending')
    except BaseException as exc:status('failed',error=f'{type(exc).__name__}: {exc}');raise


if __name__=='__main__':main()
