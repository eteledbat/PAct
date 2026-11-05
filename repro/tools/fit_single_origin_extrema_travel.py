"""Fit only joint travel on the published single-image PAct reconstruction.

The old geometry, component ownership, axes, camera registration, and original
four-panel videos are immutable inputs. The two selected reference images and
their existing 2D instance masks choose a pose at each endpoint.
"""
import argparse
import copy
import json
import pathlib
import sys
import time

import numpy as np
import torch

REPO=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'repro/tools'))
from auto_part_semantics import sha256,write_json
from fit_observed_drawer_pose import load,read
from fit_two_extrema import observations,set_observation,estimate,scores
from fit_vlm_component_geometry import automatic_owners

OLD_BATCH=REPO/'repro/outputs/github_v010_multicase_20261030_v1'
REPAIRED=REPO/'repro/outputs/handle_attachment_four_versions_20261030_v1'
EXTREMES=REPO/'repro/outputs/two_extrema_origin_v011_20261031_v1/extremes'


def source(case):
    root=(REPAIRED if case=='100481' else OLD_BATCH)/case
    return root,root/'vlm'


def fit(batch,case):
    out=batch/case;out.mkdir(parents=True,exist_ok=True)
    source_root,model_root=source(case)
    video=source_root/f'{case}_GT_Origin_VLMAxis_VLMPoseGeo.mp4'
    assert video.exists() and (model_root/'observed_pose_geometry.pt').exists()
    torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42)
    model=load(model_root,checkpoint='observed_pose_geometry.pt');model.requires_grad_(False)
    source_groups=copy.deepcopy(model.groups)
    owners=automatic_owners(source_groups)
    data=observations(EXTREMES/case,owners)
    frames=[int(x['frame']) for x in data]
    assert frames==[int(x['frame']) for x in read(EXTREMES/case/'manifest.json')['endpoints']]
    movable=[g for g in source_groups if g['joint']['type']!='fixed' and g.get('semantic_owner') is not None]
    old_poses=[{str(g['id']):float(g['joint']['range'][i]) for g in movable} for i in range(2)]
    start=time.monotonic()
    with torch.no_grad():
        before=[scores(model,o,p) for o,p in zip(data,old_poses)]
        poses,trace=estimate(model,data)
        after=[scores(model,o,p) for o,p in zip(data,poses)]
    groups=copy.deepcopy(source_groups)
    for g in groups:
        if g['joint']['type']=='fixed' or g.get('semantic_owner') is None:continue
        q=[float(poses[i][str(g['id'])]) for i in range(2)]
        g['joint']['range']=q
    assert len(groups)==len(source_groups)
    for old,new in zip(source_groups,groups):
        assert old['id']==new['id'] and old['joint']['type']==new['joint']['type']
        assert old['joint']['axis']==new['joint']['axis']
        assert old['parent']==new['parent']
    write_json(out/'travel_motion_groups.json',dict(groups=groups,source_motion_groups=str(model_root/'motion_groups.json'),
        source_motion_groups_sha256=sha256(model_root/'motion_groups.json'),
        policy='Only the numeric range endpoints of independently matched, nonfixed source joints change. Geometry, axes, joint types, parents, and original four-column video remain unchanged.'))
    write_json(out/'travel_fit.json',dict(case=case,source_four_panel=str(video),source_four_panel_sha256=sha256(video),
        source_model=str(model_root),source_checkpoint='observed_pose_geometry.pt',
        source_checkpoint_sha256=sha256(model_root/'observed_pose_geometry.pt'),
        endpoint_manifest=str(EXTREMES/case/'manifest.json'),endpoint_manifest_sha256=sha256(EXTREMES/case/'manifest.json'),
        frames=frames,old_poses=old_poses,estimated_poses=poses,search_trace=trace,
        before=before,after=after,matched_movable_groups=[g['id'] for g in movable],
        original_active_joint_count=sum(g['joint']['type']!='fixed' for g in source_groups),
        seconds=time.monotonic()-start,
        supervision='Two selected RGB frames and projected 2D instance masks. The known GT joint trajectory selected frames only; no GT numeric joint values used in pose search.'))
    print('TRAVEL FIT',case,frames,'IoU',[[round(x['image']['IoU'],3) for x in rows] for rows in (before,after)],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--case',required=True);a=p.parse_args()
    fit(a.root.resolve(),a.case)
