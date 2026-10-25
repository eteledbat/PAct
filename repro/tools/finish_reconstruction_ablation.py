"""Evaluate sealed pairs automatically and finish report, without a Codex session."""
import argparse
import datetime
import json
import os
import pathlib
import subprocess
import sys
import time

import torch

from collision_refinement_common import sha256

REPO=pathlib.Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=pathlib.Path(a.root).resolve();cfg=json.loads((root/'config.json').read_text());env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    def status(state,**kw):(root/'evaluation_status.json').write_text(json.dumps({'state':state,'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw},indent=2))
    def job(label,tool,args):
        command=[sys.executable,str(REPO/'repro/tools'/tool),'--root',str(root),*args];(root/f'{label}_command.json').write_text(json.dumps(command,indent=2))
        with (root/f'{label}.log').open('ab') as log:subprocess.run(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    try:
        while True:
            state=json.loads((root/'controller_status.json').read_text())
            if state['state']=='failed':
                zero_path=root/'zero_weight_control.json';repeat_path=root/'native_vs_sweep_A_control.json'
                all_sealed=all((root/f'{e["name"]}_{arm}_train/seal.json').exists() for e in cfg['experiments'] for arm in ['A','B'])
                expected=all_sealed and zero_path.exists() and repeat_path.exists() and json.loads(zero_path.read_text())['passed'] and not json.loads(repeat_path.read_text())['passed'] and 'max_absolute_parameter_difference' in state.get('error','')
                assert expected,state
                archive=root/'controller_strict_replication_failure.json'
                if not archive.exists():archive.write_text(json.dumps({'failed_state':state,'action':'Resume orchestration only with long-run replication kept as a failed diagnostic; all six sealed optimization checkpoints preserved.'},indent=2))
                before={str(p):sha256(p) for p in root.glob('*_train/checkpoint.pt')}
                job('resume_orchestration','run_reconstruction_ablation.py',[])
                assert all(sha256(p)==h for p,h in before.items());state=json.loads((root/'controller_status.json').read_text());assert state['state']=='optimization_complete'
            for experiment in cfg['experiments']:
                name=experiment['name']
                if (root/f'results_{name}.json').exists():continue
                if all((root/f'{name}_{arm}_train/seal.json').exists() for arm in ['A','B']):
                    status('evaluating',experiment=name);job(f'evaluation_{name}','evaluate_reconstruction_ablation.py',['--experiment',name])
            if all((root/f'results_{x["name"]}.json').exists() for x in cfg['experiments']) and state['state']=='optimization_complete':break
            status('waiting_for_sealed_pair',optimization=state);time.sleep(10)
        status('figures_and_report');job('GT_sections','plot_reconstruction_sections.py',[]);job('deformation_audit','audit_reconstruction_deformation.py',[]);job('report','build_reconstruction_ablation_report.py',[])
        audit={}
        for e in cfg['experiments']:
            name=e['name'];states=[]
            for arm in ['A','B']:
                out=root/f'{name}_{arm}_train';seal=json.loads((out/'seal.json').read_text());saved=torch.load(out/'checkpoint.pt',map_location='cpu',weights_only=False)
                assert saved['step']==cfg['steps'] and len(saved['history'])==cfg['steps'];assert seal['checkpoint_sha256']==sha256(out/'checkpoint.pt') and not seal['GT_reads_blocked'];assert all(sha256(x['path'])==x['sha256'] for x in seal['frozen_files']);assert all(torch.isfinite(p).all() for p in saved['model'].values())
                states.append(saved['model']);audit[f'{name}_{arm}']={'step':saved['step'],'checkpoint_hash_verified':True,'original_files_unchanged':True,'GT_reads':seal['GT_reads_blocked']}
            for key in ['log_scale','translation','input_q','joints.d0','joints.o0','joints.basis','joints.pivot_mask']:assert torch.equal(states[0][key],states[1][key])
        (root/'integrity.json').write_text(json.dumps(audit,indent=2));status('complete',report=str(root/'REPORT.md'))
    except BaseException as exc:status('failed',error=f'{type(exc).__name__}: {exc}');raise


if __name__=='__main__':main()
