"""Verify finished artifacts and make linked exports; never changes predictions."""
import argparse
import datetime
import json
import math
import os
import pathlib
import subprocess

import imageio

from collision_refinement_common import sha256
from frozen_joint_common import atomic_json

REPO=pathlib.Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=pathlib.Path(a.root);cfg=json.loads((root/'config.json').read_text());assert json.loads((root/'controller_status.json').read_text())['state']=='optimization_complete';assert json.loads((root/'evaluation_status.json').read_text())['state']=='complete'
    checks={}
    for name in ['direct_zero_control','head_zero_control','resume_control']:
        data=json.loads((root/f'{name}.json').read_text());assert data['passed'];checks[name]=data['max_parameter_difference']
    frozen=json.loads((root/'frozen_assets.json').read_text());assert all(sha256(f['path'])==f['sha256'] for f in frozen)
    older=json.loads((root/'older_source_assets.json').read_text());assert all(sha256(f['path'])==f['sha256'] for f in older)
    labels=[*[f'direct_{c}_{arm}' for c in cfg['direct_cases'] for arm in ['A','B']],*[f'head_shared_{arm}' for arm in ['A','B']]]
    seals={};exports=[]
    for label in labels:
        out=root/label;status=json.loads((out/'status.json').read_text());assert status['state']=='complete' and status['step']==cfg['steps']
        history=json.loads((out/'history.json').read_text());assert [h['step'] for h in history]==list(range(1,cfg['steps']+1))
        assert all(math.isfinite(h[k]) for h in history for k in ['loss','motion_proxy','base_loss'])
        assert all(math.isfinite(v) for h in history for v in h['gradient_norms'].values())
        seal=json.loads((out/'seal.json').read_text());assert seal['checkpoint_sha256']==sha256(out/'checkpoint.pt') and seal['frozen_assets_unchanged'] and not seal['GT_reads_blocked'];seals[label]=seal['checkpoint_sha256']
        assert json.loads((out/'protocol.json').read_text())['config_sha256']==sha256(root/'config.json')
        for file in [*out.glob('*_candidate.json'),*out.glob('*_official.json')]:
            case=file.stem.split('_')[0];kind='official' if file.stem.endswith('_official') else 'continuous';assert sha256(file)==seal['candidates'][case+'_'+kind]
            obj=json.loads(file.read_text());source=pathlib.Path(obj['geometry_source_object']).parent;directory=out/f'export_{case}_{kind}';directory.mkdir(exist_ok=True)
            for name in ['ply','gs']:
                target=directory/name
                if not target.exists():target.symlink_to(os.path.relpath(source/name,directory),target_is_directory=True)
            object_file=directory/'object.json'
            if object_file.exists():assert object_file.read_bytes()==file.read_bytes()
            else:object_file.write_bytes(file.read_bytes())
            assert all((directory/rel).is_file() for n in obj['diffuse_tree'] for key in ['plys','gs'] for rel in n[key])
            exports.append(str(object_file))
    results=json.loads((root/'results.json').read_text());max_static=0.;movies={}
    for route,cases in results.items():
        for case,methods in cases.items():
            original=methods['R0']['common_original_range_motion']['paths']
            for method,result in methods.items():
                assert result['geometry_frozen']
                for key in ['own_range_motion','common_original_range_motion']:
                    for row in result[key]['paths']:
                        assert all(len(row[k])==41 for k in ['proxy','minimum_sampled_center_distance','minimum_sampled_one_sigma_gap'])
                for first,row in zip(original,result['common_original_range_motion']['paths']):max_static=max(max_static,abs(first['proxy'][0]-row['proxy'][0]))
            directory=root/f'eval_{route}_{case}'
            for name in ['axes_vs_GT.png','motion_overlap.png','rigid_motion_state00.png','rigid_motion_state10.png','rigid_motion_state20.png']:assert (directory/name).stat().st_size>0
            reader=imageio.get_reader(directory/'rigid_motion.mp4');count=reader.count_frames();reader.close();assert count==21;movies[f'{route}_{case}']=count
    assert max_static<1e-6,'Common zero pose changed: inspect geometry/FK'
    sources=root/'final_sources';sources.mkdir(exist_ok=True)
    for file in (REPO/'repro/tools').glob('*frozen*.py'):(sources/file.name).write_bytes(file.read_bytes())
    inventory={'state':'complete','completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'branch':subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip(),
               'source_git_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),'formal_trials':len(labels),'test_objects':cfg['test_cases'],'head_train_objects':cfg['train_cases'],
               'checkpoint_sha256':seals,'frozen_files_verified':len(frozen),'older_source_files_verified':len(older),'control_max_parameter_difference':checks,
               'max_common_zero_pose_proxy_difference':max_static,'movie_frame_counts':movies,'linked_exports':exports,
               'config_sha256':sha256(root/'config.json'),'results_sha256':sha256(root/'results.json'),'report_sha256':sha256(root/'REPORT.md'),
               'final_source_sha256':{f.name:sha256(f) for f in sources.iterdir()},'root':str(root)}
    atomic_json(root/'integrity.json',inventory)
    evidence=dict(inventory);evidence['axis_summary']=json.loads((root/'axis_summary.json').read_text());atomic_json(REPO/'repro/evidence/frozen-joint-routes.json',evidence)
    print(json.dumps({k:v for k,v in inventory.items() if k not in ['linked_exports','final_source_sha256']},indent=2),flush=True)


if __name__=='__main__':main()
