"""Refit single-image v0.1.1 geometry from the closed travel-reference frame.

The published Origin and automatic-axis model are immutable. The selected RGB,
projected instance mask and camera calibrate an observed pose, then the same
published 240-step geometry optimizer runs with axes and native ranges frozen.
"""
import argparse, copy, json, pathlib, shutil, sys, time
import numpy as np
import torch
from PIL import Image

REPO=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'repro/tools'))
from auto_part_semantics import sha256,write_json
from fit_single_origin_extrema_travel import source
from fit_observed_drawer_pose import load,geometry,read
from fit_two_extrema import observations,estimate,scores
from fit_vlm_component_geometry import automatic_owners

CASES=['100481','101808','10489','10655','20411','31249','45194','45503','8961','9016']

def selected_index(case):return 0

def prepare(batch,case):
    root=batch/case;root.mkdir(parents=True,exist_ok=True)
    _,old=source(case)
    groups=read(old/'motion_groups.json');owners=automatic_owners(groups['groups'])
    closed=batch/'closed_references'/case
    obs=observations(closed,owners)[0]
    manifest=read(closed/'manifest.json');endpoint=manifest['endpoints'][selected_index(case)]
    assert int(obs['frame'])==int(endpoint['frame'])
    for name in ['membership.npz','registration.pt','original_instance_ids.npy']:
        dest=root/name
        if not dest.exists():dest.symlink_to(old/name)
    shutil.copyfile(old/'motion_groups.json',root/'motion_groups.json')
    cfg=copy.deepcopy(read(old/'config.json'))
    cfg.update(prior_root=str(root),source_image=str(endpoint['rgb']),source_image_sha256=sha256(endpoint['rgb']),
               reference_frame=int(obs['frame']),geometry_policy='Published 240-step observed-pose geometry optimizer; closed extreme RGB and projected instance mask; axes and native ranges frozen',
               pose_policy='Independent automatic semantic joints searched against closed-frame instance masks')
    labels=obs['labels'];semantic=np.ones((*labels.shape,3),np.float32)
    for gid in set(owners.values()):semantic[labels==gid]=cfg['palette'][str(gid)]
    np.savez_compressed(root/'observation.npz',rgb=obs['rgb'],alpha=obs['alpha'],labels=labels,semantic=semantic,
                        extrinsics=obs['extrinsics'],intrinsics=obs['intrinsics'])
    cfg['prior_manifest_sha256']=sha256(root/'motion_groups.json');cfg['observations_sha256']=sha256(root/'observation.npz')
    write_json(root/'config.json',cfg)
    Image.open(endpoint['rgb']).save(root/'closed_reference.png')
    write_json(root/'prepared.json',dict(case=case,published_model=str(old),published_axis_sha256=sha256(old/'motion_groups.json'),
        closed_endpoint_index=selected_index(case),closed_frame=int(obs['frame']),closed_rgb=str(endpoint['rgb']),
        closed_rgb_sha256=sha256(endpoint['rgb']),closed_mask=str(endpoint['mask']),closed_mask_sha256=sha256(endpoint['mask']),
        endpoint_manifest_sha256=sha256(closed/'manifest.json'),registration_sha256=sha256(old/'registration.pt'),
        note='Established all-leaves-closed reference frame from the travel-source scene; the global farthest pair may not contain a fully closed state.'))
    print('PREPARED',case,'closed frame',obs['frame'],flush=True)

def fit(root):
    case=root.name;torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42)
    torch.backends.cuda.matmul.allow_tf32=False
    old=pathlib.Path(read(root/'prepared.json')['published_model'])
    assert sha256(old/'motion_groups.json')==sha256(root/'motion_groups.json')
    start=time.monotonic();model=load(root);model.requires_grad_(False)
    owners=automatic_owners(model.groups);obs=observations(root.parent/'closed_references'/case,owners)[0]
    pose_list,trace=estimate(model,[obs]);pose=pose_list[0]
    with torch.no_grad():
        before=scores(model,obs,pose)
        original=scores(model,obs,pose,original=True,progress=float(0 if case=='20411' else 1))
    write_json(root/'position_check.json',dict(case=case,closed_frame=int(obs['frame']),observed_pose=pose,
        search_trace=trace,origin_at_native_endpoint=original,axis_only_at_checked_pose=before,
        axis_and_range_sha256=sha256(root/'motion_groups.json'),policy='Per-instance mask Dice search on the selected closed image; all axis types, directions, origins, parents and native ranges remain unchanged.'))
    result=geometry(model,root,'observed_pose_geometry',pose)
    with torch.no_grad():after=scores(model,obs,pose)
    write_json(root/'fit_seal.json',dict(case=case,state='complete',closed_frame=int(obs['frame']),published_model=str(old),
        published_axis_sha256=sha256(old/'motion_groups.json'),frozen_axis_sha256=sha256(root/'motion_groups.json'),
        closed_rgb_sha256=sha256(read(root/'prepared.json')['closed_rgb']),closed_mask_sha256=sha256(read(root/'prepared.json')['closed_mask']),
        observed_pose=pose,position_check=before,geometry_fit=after,origin_at_native_endpoint=original,
        geometry_checkpoint_sha256=sha256(root/'observed_pose_geometry.pt'),selected_fit=result,seconds=time.monotonic()-start))
    print('FIT COMPLETE',case,'2D IoU',round(before['image']['IoU'],3),'->',round(after['image']['IoU'],3),'seconds',round(time.monotonic()-start,1),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--case',required=True);p.add_argument('--stage',choices=['prepare','fit'],required=True);a=p.parse_args()
    if a.stage=='prepare':prepare(a.root.resolve(),a.case)
    else:fit(a.root.resolve()/a.case)
