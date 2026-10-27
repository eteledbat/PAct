"""Audit single-variable optimizer, FK histories, raw outputs and both exports."""
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

from axis_mode_common import atomic_json,sha256

REPO=pathlib.Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=pathlib.Path(a.root);top=json.loads((root/'config.json').read_text());parent=pathlib.Path(top['parent_root'])
    assert json.loads((root/'controller_status.json').read_text())['state']=='optimization_complete'
    assert json.loads((root/'evaluation_status.json').read_text())['state']=='complete'
    assert sha256(parent/'integrity.json')==top['parent_integrity_sha256'] and sha256(parent/'config.json')==top['parent_config_sha256']
    hashes=json.loads((root/'feature_cache_sha256.json').read_text());assert all(sha256(parent/'features'/c/'cache.pt')==digest for c,digest in hashes.items())
    results=json.loads((root/'results.json').read_text());commands=[json.loads(l) for l in (root/'commands.jsonl').read_text().splitlines()];checkpoints={};controls={};movies={};frozen_files={};sources=set();exports=[];flips={};max_static=0.
    for mode in top['modes']:
        sub=root/mode;cfg=json.loads((sub/'config.json').read_text());assert cfg['mode']==mode;assert json.loads((sub/'gradient_and_freeze_checks.json').read_text())['status']=='passed'
        for name in ['direct_zero_control','head_zero_control','resume_control']:
            d=json.loads((sub/f'{name}.json').read_text());assert d['passed'];controls[mode+'/'+name]=d['max_parameter_difference']
        calls=[c for c in commands if str(sub) in c and 'resume_head' in c]
        assert any('--stop-after' in c and c[c.index('--stop-after')+1]=='10' for c in calls);assert any('--resume' in c and '--stop-after' not in c for c in calls)
        for name in ['frozen_assets.json','older_source_assets.json']:
            for f in json.loads((sub/name).read_text()):assert sha256(f['path'])==f['sha256'];frozen_files[f['path']]=f['sha256']
        lo,hi=(6,9) if mode=='direction' else (9,12)
        for case in cfg['train_cases']:
            pack=torch.load(sub/'training_targets'/f'{case}.pt',map_location='cpu',weights_only=False)
            for key in ['target','mask']:
                check=pack[key].clone();check[:,lo:hi]=0;assert not torch.count_nonzero(check)
        labels=[*[f'direct_{c}_{arm}' for c in cfg['test_cases'] for arm in ['A','B']],*[f'head_shared_{arm}' for arm in ['A','B']]]
        for label in labels:
            out=sub/label;status=json.loads((out/'status.json').read_text());assert status['state']=='complete' and status['step']==200 and status['mode']==mode
            h=json.loads((out/'history.json').read_text());assert len(h)==200 and [r['step'] for r in h]==list(range(1,201))
            assert all(r['max_endpoint_delta']==0 and r['max_frozen_counterpart_delta']==0 and r['range_grad_none'] and r['frozen_counterpart_grad_none'] for r in h)
            assert all(math.isfinite(r[k]) for r in h for k in ['loss','base_loss','motion_proxy']);assert all(math.isfinite(v) for r in h for v in r['gradient_norms'].values())
            seal=json.loads((out/'seal.json').read_text());assert seal['checkpoint_sha256']==sha256(out/'checkpoint.pt');assert seal['frozen_assets_unchanged'] and not seal['GT_reads_blocked'];assert seal['frozen_counterpart_all_steps'] and seal['endpoints_frozen_all_steps'];checkpoints[mode+'/'+label]=seal['checkpoint_sha256']
            protocol=json.loads((out/'protocol.json').read_text());assert protocol['config_sha256']==sha256(sub/'config.json') and protocol['mode']==mode
            assert seal['protocol_sha256']==sha256(out/'protocol.json')
            for name,digest in protocol['sources'].items():
                source=REPO/'repro/tools'/name;assert sha256(source)==digest;sources.add(source)
            checkpoint=torch.load(out/'checkpoint.pt',map_location='cpu',weights_only=False)
            if label.startswith('direct'):
                frozen='phi' if mode=='direction' else 'theta';assert not torch.count_nonzero(checkpoint['model'][frozen]);assert not torch.count_nonzero(checkpoint['model']['range_delta']);assert sum(len(g['params']) for g in checkpoint['optimizer']['param_groups'])==1
                assert all(set(r['gradient_norms'])==({'theta'} if mode=='direction' else {'phi'}) for r in h)
            for file in [*out.glob('*_candidate.json'),*out.glob('*_official.json')]:
                case=file.stem.split('_')[0];kind='official' if file.stem.endswith('_official') else 'continuous';assert sha256(file)==seal['candidates'][case+'_'+kind]
                obj=json.loads(file.read_text());src=pathlib.Path(obj['geometry_source_object']);original=json.loads(src.read_text());byid={n['id']:n for n in original['diffuse_tree']};assert len(byid)==len(obj['diffuse_tree']);reversed_nodes=[]
                for n in obj['diffuse_tree']:
                    old=byid[n['id']];restored=copy.deepcopy(n);active=old['joint']['type'] in ['revolute','continuous','prismatic','screw']
                    if active:
                        d,d0=np.array(n['joint']['axis']['direction']),np.array(old['joint']['axis']['direction']);assert np.isfinite(d).all() and np.linalg.norm(d)>1e-8
                        if d@d0<0:reversed_nodes.append(n['id'])
                        if mode=='direction':restored['joint']['axis']['direction']=old['joint']['axis']['direction']
                        elif old['joint']['type'] in ['revolute','continuous']:restored['joint']['axis']['origin']=old['joint']['axis']['origin']
                    assert restored==old,('Non-selected field changed',mode,file,n['id'])
                flips[str(file.relative_to(root))]=reversed_nodes
                if label.startswith('head'):
                    raw=np.load(out/f'{case}_raw.npy');base=torch.load(sub/'features'/case/'cache.pt',map_location='cpu',weights_only=False)['raw'].numpy();mask=np.zeros_like(raw,dtype=bool)
                    for n in original['diffuse_tree']:
                        if n['joint']['type']=='fixed':continue
                        if mode=='direction':mask[n['id'],6:9]=True
                        elif n['joint']['type'] in ['revolute','continuous']:mask[n['id'],9:12]=True
                    assert np.array_equal(raw[~mask],base[~mask])
                directory=out/f'export_{case}_{kind}';directory.mkdir(exist_ok=True)
                for name in ['gs','ply']:
                    target=directory/name
                    if not target.exists():target.symlink_to(os.path.relpath(src.parent/name,directory),target_is_directory=True)
                exported=directory/'object.json'
                if exported.exists():assert exported.read_bytes()==file.read_bytes()
                else:exported.write_bytes(file.read_bytes())
                assert all((directory/rel).is_file() for n in obj['diffuse_tree'] for key in ['gs','plys'] for rel in n[key]);exports.append(str(exported))
        for route,cases in results[mode].items():
            for case,methods in cases.items():
                base=methods['R0'];metric='pivot_error_percent_GT_D' if mode=='direction' else 'axis_error_degrees'
                for method,result in methods.items():
                    assert result['geometry_frozen'] and result['endpoints_frozen'] and result['max_frozen_counterpart_change']==0 and result['max_numeric_endpoint_change']==0
                    for a,b in zip(base['joints'],result['joints']):assert a[metric]==b[metric]
                    for first,row in zip(base['fixed_range_motion']['paths'],result['fixed_range_motion']['paths']):
                        assert all(len(row[k])==41 and all(math.isfinite(v) for v in row[k]) for k in ['proxy','minimum_sampled_center_distance','minimum_sampled_one_sigma_gap']);max_static=max(max_static,abs(first['proxy'][0]-row['proxy'][0]))
                folder=sub/f'eval_{route}_{case}'
                for name in ['axes_vs_GT.png','motion_overlap.png','rigid_motion_state00.png','rigid_motion_state10.png','rigid_motion_state20.png']:assert (folder/name).stat().st_size>0
                reader=imageio.get_reader(folder/'rigid_motion.mp4');count=reader.count_frames();reader.close();assert count==21;movies[f'{mode}/{route}/{case}']=count
    assert max_static<1e-6
    for name in ['axis_parameter_changes.csv','axis_parameter_changes.json','selected_axis_errors.png','selected_axis_errors.pdf','REPORT.md']:assert (root/name).stat().st_size>0
    sources.update(REPO/'repro/tools'/name for name in ['prepare_axis_mode_experiment.py','check_axis_mode_routes.py','run_axis_mode_routes.py','monitor_axis_mode_routes.py','evaluate_axis_mode_routes.py','report_axis_mode_routes.py','seal_axis_mode_routes.py','evaluate_axis_fixed_routes.py','evaluate_frozen_joint_routes.py','build_axis_fixed_report.py'])
    snapshot=root/'final_sources';snapshot.mkdir(exist_ok=True)
    for source in sources:(snapshot/source.name).write_bytes(source.read_bytes())
    inventory={'state':'complete','completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'root':str(root),
        'branch':subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip(),'source_git_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        'formal_trials':12,'steps_each':200,'test_objects':top['test_cases'],'head_train_objects':top['train_cases'],'modes':top['modes'],
        'checkpoint_sha256':checkpoints,'unique_unchanged_asset_files_verified':len(frozen_files),'control_max_parameter_difference':controls,
        'max_numeric_endpoint_change':0.,'max_non_selected_field_change':0.,'unselected_GT_metric_unchanged':True,'max_zero_pose_proxy_difference':max_static,
        'all_export_axis_sign_reversals':flips,'movie_frame_counts':movies,'linked_exports':exports,
        'parent_integrity_sha256':top['parent_integrity_sha256'],'config_sha256':sha256(root/'config.json'),'results_sha256':sha256(root/'results.json'),'report_sha256':sha256(root/'REPORT.md'),
        'axis_parameter_changes_csv_sha256':sha256(root/'axis_parameter_changes.csv'),'final_source_sha256':{f.name:sha256(f) for f in snapshot.iterdir()}}
    if (root/'interpretation.json').exists():inventory['interpretation_sha256']=sha256(root/'interpretation.json')
    atomic_json(root/'integrity.json',inventory);evidence=dict(inventory);evidence['axis_summary']=json.loads((root/'axis_summary.json').read_text());atomic_json(REPO/'repro/evidence/axis-direction-position.json',evidence)
    print(json.dumps({k:v for k,v in inventory.items() if k not in ['linked_exports','final_source_sha256','all_export_axis_sign_reversals']},indent=2),flush=True)


if __name__=='__main__':main()
