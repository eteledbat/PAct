"""Select a shared direction for independently moving handle axes from RGB masks.

Only generated Origin geometry, two observed endpoint masks and a generic
parallel-peer prior are used. Each handle keeps its own contact pivot and q.
The angle grid chooses the best projected instance Dice for each candidate.
"""
import argparse
import copy
import json
import pathlib
import sys

import numpy as np
import torch

REPO=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'repro/tools'))
from auto_part_semantics import sha256,write_json
from fit_observed_drawer_pose import load
from fit_two_extrema import load_observations,set_observation


def read(path):return json.loads(pathlib.Path(path).read_text())


@torch.no_grad()
def score(model,groups,observation,directions):
    set_observation(model,observation)
    original={g['id']:copy.deepcopy(g['joint']['axis']['direction']) for g in groups}
    rows=[]
    for group,direction in zip(groups,directions):
        gid=group['id'];group['joint']['axis']['direction']=direction
        target=(model.targets==gid).float();index=model.visible_owners.index(gid)
        grid=np.linspace(-220,220,45);best=(2.,None)
        for q in grid:
            model.set_pose({gid:float(q)})
            probability=model.probabilities()[index]
            value=float(1-(2*(probability*target).sum()+1)/(probability.sum()+target.sum()+1))
            if value<best[0]:best=(value,float(q))
        rows.append(dict(group=gid,dice_loss=best[0],best_q=best[1]))
        group['joint']['axis']['direction']=original[gid]
    return float(sum(x['dice_loss'] for x in rows)),rows


def select(batch,case):
    axis=batch/'axis'/case;root=batch/case
    if not (root/'two_endpoint_prepared.json').exists():return None
    manifest=read(axis/'motion_groups.json');assignments=read(axis/'component_assignment.json')
    semantics=read(batch/'axis_input'/case/'semantics.json')
    owners={x['instance_id'] for x in semantics['parts'] if x['semantic']=='handle'}
    groups=[g for g in manifest['groups'] if g.get('semantic_owner') in owners and g['joint']['type'] in ('revolute','continuous')]
    if len(groups)<2:return None
    model=load(root);model.requires_grad_(False)
    model_groups=[next(g for g in model.groups if g['id']==item['id']) for item in groups]
    observation=load_observations(root)[1]
    current_directions=[g['joint']['axis']['direction'] for g in model_groups]
    original_score,original_rows=score(model,model_groups,observation,current_directions)
    candidates=[]
    for direction in [[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]:
        total,rows=score(model,model_groups,observation,[direction]*len(model_groups))
        candidates.append(dict(direction=direction,total_dice_loss=total,rows=rows))
    choice=min(candidates,key=lambda x:x['total_dice_loss'])
    apply=choice['total_dice_loss']+.02<original_score
    if apply:
        for g in groups:g['joint']['axis']['direction']=choice['direction']
        for a in assignments['assignments']:
            if a['instance_id'] not in owners or a.get('joint',{}).get('type') not in ('revolute','continuous'):continue
            a['joint']['axis']['direction']=choice['direction']
            a['axis_evidence']['two_extreme_parallel_peer_axis_direction']=dict(chosen=choice['direction'],
                original_sum_dice_loss=original_score,chosen_sum_dice_loss=choice['total_dice_loss'],
                rule='One common handle-axis direction with independent pivots and motion values; 2D observed-instance Dice selection')
        write_json(axis/'motion_groups.json',manifest)
        write_json(axis/'component_assignment.json',assignments)
        cfg=read(axis/'config.json');cfg['prior_manifest_sha256']=sha256(axis/'motion_groups.json');write_json(axis/'config.json',cfg)
    result=dict(case=case,original_directions=current_directions,original_sum_dice_loss=original_score,
        original_rows=original_rows,candidates=candidates,chosen=choice['direction'],
        applied=apply,GT_numeric_joint_inputs=False)
    write_json(axis/'two_extreme_handle_direction_selection.json',result)
    print('HANDLE IMAGE AXIS',case,'chosen',choice['direction'],'applied',apply,flush=True)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);a=p.parse_args();batch=a.root.resolve()
    torch.set_num_threads(4)
    summary={case:select(batch,case) for case in read(batch/'protocol.json')['cases']}
    write_json(batch/'axis'/'two_extreme_handle_direction_summary.json',summary)


if __name__=='__main__':main()
