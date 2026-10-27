"""Verify four new optimizations, transferred head predictions and all-fixed case."""
import argparse
import copy
import datetime
import json
import math
import os
import pathlib
import subprocess

import imageio
import numpy as np
import torch

from axis_fixed_common import atomic_json,sha256

REPO=pathlib.Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=pathlib.Path(a.root);cfg=json.loads((root/'config.json').read_text());parent=pathlib.Path(cfg['parent_root'])
    assert json.loads((root/'controller_status.json').read_text())['state']=='optimization_complete'
    assert json.loads((root/'evaluation_status.json').read_text())['state']=='complete'
    assert sha256(parent/'integrity.json')==cfg['parent_integrity_sha256'];assert sha256(parent/'config.json')==cfg['parent_config_sha256']
    checks={}
    for name in [*[f'{c}_zero_control' for c in cfg['test_cases']],'resume_control']:
        data=json.loads((root/f'{name}.json').read_text());assert data['passed'];checks[name]=data['max_parameter_difference']
    commands=[json.loads(l) for l in (root/'commands.jsonl').read_text().splitlines()]
    assert any('--stop-after' in c and c[c.index('--stop-after')+1]=='10' for c in commands)
    assert any('--resume' in c and 'resume_control' in c for c in commands)
    manifests={name:json.loads((root/name).read_text()) for name in ['frozen_assets.json','older_source_assets.json']}
    for files in manifests.values():assert all(sha256(f['path'])==f['sha256'] for f in files)
    sources=set();checkpoints={};exports=[];sign_reversals={}
    labels=[*[f'direct_{c}_{arm}' for c in cfg['test_cases'] for arm in ['A','B']],*[f'head_shared_{arm}' for arm in ['A','B']]]
    for label in labels:
        out=root/label;status=json.loads((out/'status.json').read_text());seal=json.loads((out/'seal.json').read_text());assert status['state']=='complete' and status['step']==200
        assert sha256(out/'checkpoint.pt')==seal['checkpoint_sha256'];assert seal['max_endpoint_delta']==0 and seal['endpoints_frozen_all_steps'];assert not seal['GT_reads_blocked'];checkpoints[label]=seal['checkpoint_sha256']
        protocol=json.loads((out/'protocol.json').read_text());assert protocol['config_sha256']==sha256(root/'config.json')
        if label.startswith('direct'):
            history=json.loads((out/'history.json').read_text());assert len(history)==200
            assert all(row['step']==i+1 and row['range_grad_none'] and row['max_endpoint_delta']==0 for i,row in enumerate(history))
            assert all(math.isfinite(row[k]) for row in history for k in ['loss','base_loss','motion_proxy'])
            assert all(math.isfinite(v) for row in history for v in row['gradient_norms'].values())
            for name,digest in protocol['sources'].items():
                file=REPO/'repro/tools'/name;assert sha256(file)==digest;sources.add(file)
            checkpoint=torch.load(out/'checkpoint.pt',map_location='cpu',weights_only=False);assert not torch.count_nonzero(checkpoint['model']['range_delta'])
            assert sum(len(g['params']) for g in checkpoint['optimizer']['param_groups'])==2
        else:
            assert status['new_optimizer_steps']==0 and protocol['new_optimizer_steps']==0
            src=pathlib.Path(protocol['checkpoint_reused_from']);assert sha256(src/'checkpoint.pt')==seal['checkpoint_sha256']
            assert sha256(src/'protocol.json')==protocol['original_training_protocol_sha256']
        for file in [*out.glob('*_candidate.json'),*out.glob('*_official.json')]:
            case=file.stem.split('_')[0];kind='official' if file.stem.endswith('_official') else 'continuous';assert sha256(file)==seal['candidates'][case+'_'+kind]
            obj=json.loads(file.read_text());original=pathlib.Path(obj['geometry_source_object']);oldobj=json.loads(original.read_text());byid={n['id']:n for n in oldobj['diffuse_tree']};assert len(byid)==len(obj['diffuse_tree']);flips=[]
            for n in obj['diffuse_tree']:
                old=byid[n['id']];restored=copy.deepcopy(n)
                if old['joint']['type']!='fixed':
                    d=np.array(n['joint']['axis']['direction']);d0=np.array(old['joint']['axis']['direction']);assert np.isfinite(d).all() and np.linalg.norm(d)>1e-8
                    if d@d0<0:flips.append(n['id'])
                    restored['joint']['axis']['direction']=old['joint']['axis']['direction']
                    if old['joint']['type'] in ['revolute','continuous']:restored['joint']['axis']['origin']=old['joint']['axis']['origin']
                assert restored==old,'Non-axis part field changed'
            sign_reversals[str(file.relative_to(root))]=flips
            if label.startswith('head'):
                cache=torch.load(root/'features'/case/'cache.pt',map_location='cpu',weights_only=False);raw=np.load(out/f'{case}_raw.npy');base=cache['raw'].numpy();free=np.zeros_like(raw,dtype=bool)
                for n in oldobj['diffuse_tree']:
                    if n['joint']['type']=='fixed':continue
                    free[n['id'],6:9]=True
                    if n['joint']['type'] in ['revolute','continuous']:free[n['id'],9:12]=True
                assert np.array_equal(raw[~free],base[~free])
            directory=out/f'export_{case}_{kind}';directory.mkdir(exist_ok=True)
            for name in ['gs','ply']:
                target=directory/name
                if not target.exists():target.symlink_to(os.path.relpath(original.parent/name,directory),target_is_directory=True)
            exported=directory/'object.json'
            if exported.exists():assert exported.read_bytes()==file.read_bytes()
            else:exported.write_bytes(file.read_bytes())
            assert all((directory/rel).is_file() for n in obj['diffuse_tree'] for key in ['gs','plys'] for rel in n[key]);exports.append(str(exported))
    failed=json.loads((root/'ineligible_cases.json').read_text())['101284'];assert sha256(failed['object'])==failed['object_sha256'];assert all(n['joint']['type']=='fixed' for n in json.loads(pathlib.Path(failed['object']).read_text())['diffuse_tree'])
    results=json.loads((root/'results.json').read_text());max_static=0.;movies={}
    for route,cases in results.items():
        for case,methods in cases.items():
            baseline=methods['R0']['fixed_range_motion']['paths']
            for method,result in methods.items():
                assert result['geometry_frozen'] and result['endpoints_frozen'] and result['max_numeric_endpoint_change']==0
                for initial,row in zip(baseline,result['fixed_range_motion']['paths']):
                    assert all(len(row[k])==41 and all(math.isfinite(x) for x in row[k]) for k in ['proxy','minimum_sampled_center_distance','minimum_sampled_one_sigma_gap'])
                    max_static=max(max_static,abs(initial['proxy'][0]-row['proxy'][0]))
            folder=root/f'eval_{route}_{case}'
            for name in ['axes_vs_GT.png','motion_overlap.png','rigid_motion_state00.png','rigid_motion_state10.png','rigid_motion_state20.png']:assert (folder/name).stat().st_size>0
            reader=imageio.get_reader(folder/'rigid_motion.mp4');count=reader.count_frames();reader.close();assert count==21;movies[f'{route}_{case}']=count
    assert max_static<1e-6
    sources.update(REPO/'repro/tools'/n for n in ['run_axis_extra_cases.py','cache_axis_extra_features.py','report_axis_extra_cases.py','seal_axis_extra_cases.py','build_axis_fixed_report.py','evaluate_axis_fixed_routes.py','evaluate_frozen_joint_routes.py','evaluate_gt_joints.py','evaluate_freeart_partnet_case.py'])
    snapshot=root/'final_sources';snapshot.mkdir(exist_ok=True)
    for file in sources:(snapshot/file.name).write_bytes(file.read_bytes())
    state={'state':'complete','completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'root':str(root),
        'source_git_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),'branch':subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip(),
        'formal_direct_trials':4,'formal_direct_steps_each':200,'head_new_optimizer_steps':0,'head_source_steps_each':200,'test_objects':cfg['test_cases'],
        'checkpoint_sha256':checkpoints,'frozen_files_verified':len(manifests['frozen_assets.json']),'older_files_verified':len(manifests['older_source_assets.json']),
        'control_max_parameter_difference':checks,'max_numeric_endpoint_change':0.,'max_common_zero_pose_proxy_difference':max_static,
        'all_export_axis_sign_reversals':sign_reversals,'movie_frame_counts':movies,'linked_exports':exports,'ineligible_cases':{'101284':failed},
        'config_sha256':sha256(root/'config.json'),'results_sha256':sha256(root/'results.json'),'report_sha256':sha256(root/'REPORT.md'),
        'parent_integrity_sha256':cfg['parent_integrity_sha256'],'final_source_sha256':{f.name:sha256(f) for f in snapshot.iterdir()}}
    atomic_json(root/'integrity.json',state);evidence=dict(state);evidence['axis_summary']=json.loads((root/'axis_summary.json').read_text());atomic_json(REPO/'repro/evidence/axis-extra-cases.json',evidence)
    print(json.dumps({k:v for k,v in state.items() if k not in ['final_source_sha256','linked_exports','all_export_axis_sign_reversals']},indent=2),flush=True)


if __name__=='__main__':main()
