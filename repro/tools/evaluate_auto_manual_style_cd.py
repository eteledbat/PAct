"""Sealed, equal-protocol closed-state Chamfer comparison against manual fit."""
import argparse
import copy
import csv
import pathlib

import numpy as np
import torch
import trimesh

from auto_part_semantics import write_json
from collision_refinement_common import surface
from evaluate_freeart_partnet_case import gt_parts
from evaluate_multicase_closed import distance_scores, joint_scores
from evaluate_observed_drawer_pose import build_mesh
from evaluate_two_extrema import gt_mesh
from fit_closed_extrema_geometry import CASES
from fit_observed_drawer_pose import load
from fit_single_origin_extrema_travel import source
from fit_auto_closed_geometry import read
from multicase_closed_common import Model


def fixed_pose(groups, pose):
    tree=copy.deepcopy(groups)
    for group in tree:
        q=pose.get(str(group['id']),pose.get(group['id'],0.))
        group['joint']['range']=[float(q),float(q)]
    return tree


def evaluate_case(batch,case):
    root=batch/case
    fit=read(root/'manual_style_fit.json')
    _,published=source(case)
    previous=batch.parent/'closed_travel_reference_geometry_v011_20261031_v2'/case
    manual=batch.parent/'closed_multicase_20261028_v1'/case
    torch.set_num_threads(4)
    native=load(published)
    auto_old=load(previous,checkpoint='observed_pose_geometry.pt')
    auto_new=load(root,checkpoint='manual_style_geometry.pt')
    human=Model(read(manual/'config.json'),read(manual/'motion_groups.json'),manual)
    human.load_state_dict(torch.load(manual/'checkpoint.pt',map_location='cuda',weights_only=False)['model'])
    for model in (native,auto_old,auto_new,human):model.requires_grad_(False)
    A=joint_scores(native)['alignment']
    R=np.asarray(A['R_export_to_GT']);scale=float(A['scale']);shift=np.asarray(A['translation']);D=float(A['GT_static_diagonal'])
    frame=int(read(root/'prepared.json')['closed_frame'])
    scene=pathlib.Path(native.config['source_scene'])
    if case=='20411':
        truth=trimesh.util.concatenate(gt_parts(scene)[0][frame])
    else:truth=gt_mesh(scene,case,frame)
    gt_cloud=surface(truth,16000,177)
    methods={
        'previous_auto':(auto_old,read(previous/'fit_seal.json')['observed_pose']),
        'manual_style_auto':(auto_new,fit['pose']),
        'manual_semantic':(human,{str(g['id']):g['observed_q'] for g in human.groups}),
    }
    result={}
    for name,(model,pose) in methods.items():
        mesh=build_mesh(model,optimized=True,tree=fixed_pose(model.groups,pose),progress=0.)
        pred_cloud=scale*(surface(mesh,16000,178)@R.T)+shift
        result[name]=distance_scores(pred_cloud,gt_cloud,D)
    write_json(root/'cd_evaluation.json',dict(case=case,closed_frame=frame,
        CD_percent_D=result,alignment=A,
        protocol='16000 area-uniform samples per whole-object mesh, seeds 177 GT / 178 predicted, symmetric unsquared Euclidean Chamfer mean normalized by the same original static GT diagonal. Shared original fixed-body alignment; each method at its fitted closed pose. GT accessed only in this post-fit evaluator.'))
    print(case,{k:round(v['CD_percent_D'],4) for k,v in result.items()},flush=True)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=pathlib.Path,required=True)
    p.add_argument('--case')
    a=p.parse_args();batch=a.root.resolve()
    cases=[a.case] if a.case else CASES
    rows=[]
    for case in cases:
        result=evaluate_case(batch,case)
        rows.append(dict(case=case,**{k:result[k]['CD_percent_D'] for k in result}))
    if not a.case:
        with (batch/'CD_COMPARISON.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['case','previous_auto','manual_style_auto','manual_semantic'])
            writer.writeheader();writer.writerows(rows)
        summary={k:float(np.mean([r[k] for r in rows])) for k in ['previous_auto','manual_style_auto','manual_semantic']}
        write_json(batch/'CD_SUMMARY.json',dict(cases=rows,mean=summary,
            protocol='Closed-state CD percent GT static diagonal, same GT frame and fixed-body alignment for all methods.'))
        print('MEAN',summary,flush=True)


if __name__=='__main__':main()
