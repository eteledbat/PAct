"""Audit completed axis-only artifacts and preserve an independent evidence record."""
import argparse
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
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=pathlib.Path(a.root)
    cfg=json.loads((root/'config.json').read_text());parent=pathlib.Path(cfg['parent_root'])
    assert cfg['endpoints_frozen'] and not cfg['GT_ranges_in_training_target']
    assert json.loads((root/'controller_status.json').read_text())['state']=='optimization_complete'
    assert json.loads((root/'evaluation_status.json').read_text())['state']=='complete'
    assert sha256(parent/'integrity.json')==cfg['parent_integrity_sha256']
    assert sha256(parent/'config.json')==cfg['parent_config_sha256']
    checks={}
    for name in ['direct_zero_control','head_zero_control','resume_control']:
        data=json.loads((root/f'{name}.json').read_text());assert data['passed'];checks[name]=data['max_parameter_difference']
    commands=[json.loads(s) for s in (root/'commands.jsonl').read_text().splitlines()]
    resumed=[c for c in commands if 'resume_head' in c]
    assert any('--stop-after' in c and c[c.index('--stop-after')+1]=='10' for c in resumed)
    assert any('--resume' in c and '--stop-after' not in c for c in resumed)
    frozen=json.loads((root/'frozen_assets.json').read_text());older=json.loads((root/'older_source_assets.json').read_text())
    assert all(sha256(f['path'])==f['sha256'] for f in frozen+older)
    for case in cfg['train_cases']:
        pack=torch.load(root/'training_targets'/f'{case}.pt',weights_only=False,map_location='cpu')
        for key in ['target','mask']:
            value=pack[key].clone();value[:,6:12]=0;assert not torch.count_nonzero(value)
    labels=[*[f'direct_{c}_{arm}' for c in cfg['direct_cases'] for arm in ['A','B']],*[f'head_shared_{arm}' for arm in ['A','B']]]
    seals={};exports=[];max_range=0.
    for label in labels:
        out=root/label;status=json.loads((out/'status.json').read_text());assert status['state']=='complete' and status['step']==cfg['steps']
        history=json.loads((out/'history.json').read_text());assert [h['step'] for h in history]==list(range(1,cfg['steps']+1))
        assert all(math.isfinite(h[k]) for h in history for k in ['loss','motion_proxy','base_loss'])
        assert all(math.isfinite(v) for h in history for v in h['gradient_norms'].values())
        assert all(h['range_grad_none'] and h['max_endpoint_delta']==0 for h in history)
        seal=json.loads((out/'seal.json').read_text());assert seal['checkpoint_sha256']==sha256(out/'checkpoint.pt')
        assert seal['frozen_assets_unchanged'] and not seal['GT_reads_blocked']
        assert seal['endpoints_frozen_all_steps'] and seal['max_endpoint_delta']==0
        seals[label]=seal['checkpoint_sha256']
        protocol=json.loads((out/'protocol.json').read_text());assert protocol['config_sha256']==sha256(root/'config.json')
        checkpoint=torch.load(out/'checkpoint.pt',map_location='cpu',weights_only=False)
        if label.startswith('direct'):
            assert not torch.count_nonzero(checkpoint['model']['range_delta'])
            assert sum(len(g['params']) for g in checkpoint['optimizer']['param_groups'])==2
        else:
            for case in cfg['runs']:
                cache=torch.load(root/'features'/case/'cache.pt',map_location='cpu',weights_only=False)
                raw=np.load(out/f'{case}_raw.npy');base=cache['raw'].numpy()
                assert np.array_equal(raw[:,12:18],base[:,12:18])
        for file in [*out.glob('*_candidate.json'),*out.glob('*_official.json')]:
            case=file.stem.split('_')[0];kind='official' if file.stem.endswith('_official') else 'continuous';assert sha256(file)==seal['candidates'][case+'_'+kind]
            obj=json.loads(file.read_text());source=pathlib.Path(obj['geometry_source_object']);original=json.loads(source.read_text())
            byid={n['id']:n for n in original['diffuse_tree']}
            assert len(byid)==len(obj['diffuse_tree'])
            for n in obj['diffuse_tree']:
                old=byid[n['id']];assert n['joint']['range']==old['joint']['range'];assert n['joint']['type']==old['joint']['type'];assert n.get('parent')==old.get('parent')
                for key in ['plys','gs']:assert n[key]==old[key]
            directory=out/f'export_{case}_{kind}';directory.mkdir(exist_ok=True)
            for name in ['ply','gs']:
                target=directory/name
                if not target.exists():target.symlink_to(os.path.relpath(source.parent/name,directory),target_is_directory=True)
            object_file=directory/'object.json'
            if object_file.exists():assert object_file.read_bytes()==file.read_bytes()
            else:object_file.write_bytes(file.read_bytes())
            assert all((directory/rel).is_file() for n in obj['diffuse_tree'] for key in ['plys','gs'] for rel in n[key])
            exports.append(str(object_file))
    results=json.loads((root/'results.json').read_text());max_static=0.;movies={}
    for route,cases in results.items():
        for case,methods in cases.items():
            original=methods['R0']['fixed_range_motion']['paths']
            for method,result in methods.items():
                assert result['geometry_frozen'] and result['endpoints_frozen']
                assert result['max_numeric_endpoint_change']==0
                assert result['fixed_range_motion']==result['own_range_motion']==result['common_original_range_motion']
                for row in result['fixed_range_motion']['paths']:
                    assert all(len(row[k])==41 for k in ['proxy','minimum_sampled_center_distance','minimum_sampled_one_sigma_gap'])
                for joint in result['joints']:max_range=max(max_range,joint['max_numeric_endpoint_change'])
                for first,row in zip(original,result['fixed_range_motion']['paths']):
                    max_static=max(max_static,abs(first['proxy'][0]-row['proxy'][0]))
            directory=root/f'eval_{route}_{case}'
            for name in ['axes_vs_GT.png','motion_overlap.png','rigid_motion_state00.png','rigid_motion_state10.png','rigid_motion_state20.png']:
                assert (directory/name).stat().st_size>0
            reader=imageio.get_reader(directory/'rigid_motion.mp4');count=reader.count_frames();reader.close();assert count==21;movies[f'{route}_{case}']=count
    assert max_static<1e-6 and max_range==0
    sources=root/'final_sources';sources.mkdir(exist_ok=True)
    files=set((REPO/'repro/tools').glob('*axis_fixed*.py'))
    files.update(REPO/'repro/tools'/name for name in ['frozen_joint_common.py','run_frozen_joint_trial.py','collision_refinement_common.py','reconstruction_ablation_common.py','evaluate_frozen_joint_routes.py'])
    for file in files:(sources/file.name).write_bytes(file.read_bytes())
    inventory={'state':'complete','completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'branch':subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip(),
        'source_git_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        'formal_trials':len(labels),'test_objects':cfg['test_cases'],'head_train_objects':cfg['train_cases'],
        'checkpoint_sha256':seals,'frozen_files_verified':len(frozen),'older_source_files_verified':len(older),
        'control_max_parameter_difference':checks,'max_numeric_endpoint_change':max_range,
        'max_common_zero_pose_proxy_difference':max_static,'movie_frame_counts':movies,'linked_exports':exports,
        'config_sha256':sha256(root/'config.json'),'results_sha256':sha256(root/'results.json'),'report_sha256':sha256(root/'REPORT.md'),
        'parent_integrity_sha256':cfg['parent_integrity_sha256'],
        'final_source_sha256':{f.name:sha256(f) for f in sources.iterdir()},'root':str(root)}
    atomic_json(root/'integrity.json',inventory)
    evidence=dict(inventory);evidence['axis_summary']=json.loads((root/'axis_summary.json').read_text());atomic_json(REPO/'repro/evidence/axis-only-fixed-ranges.json',evidence)
    print(json.dumps({k:v for k,v in inventory.items() if k not in ['linked_exports','final_source_sha256']},indent=2),flush=True)


if __name__=='__main__':main()
