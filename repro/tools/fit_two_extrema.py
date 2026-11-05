"""Fit independent semantic joints and shared geometry to two RGB/mask endpoints.

Reuses v0.1.1 automatic component ownership, original PAct geometry and the
frozen source-view registration. Endpoint supervision is the extracted 2D
RGB/instance masks; GT 3D joints/meshes are not read during this fitting stage.
"""
import argparse
import copy
import json
import pathlib
import sys
import time

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_erosion
from scipy.optimize import minimize_scalar

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'repro/tools'))
from auto_part_semantics import sha256, write_json
from collision_refinement_common import transforms
from fit_vlm_component_geometry import prepare_case as prepare_geometry, automatic_owners, semantic_target
from fit_observed_drawer_pose import PoseModel, read, load
from multicase_closed_common import tensor, image_metrics


def observations(root, owners):
    manifest=read(root/'manifest.json')
    result=[]
    for end in manifest['endpoints']:
        rgba=np.asarray(Image.open(end['rgb']).convert('RGBA').resize((256,256),Image.Resampling.LANCZOS)).astype(np.float32)/255
        alpha=rgba[:,:,3]
        rgb=rgba[:,:,:3]*alpha[:,:,None]+1-alpha[:,:,None]
        ids=np.array(Image.fromarray(np.load(end['mask']).astype(np.int32),mode='I').resize((256,256),Image.Resampling.NEAREST))
        labels=semantic_target(ids,owners)
        result.append(dict(rgb=rgb,alpha=alpha,labels=labels,extrinsics=end['extrinsics'],intrinsics=end['intrinsics'],frame=end['frame']))
    return result


def set_observation(model, obs):
    model.rgb=tensor(obs['rgb']).permute(2,0,1)
    model.alpha=tensor(obs['alpha'])
    model.weight=1+3*model.alpha
    model.targets=tensor(obs['labels'])
    model.supervised=tensor(obs['labels']>=0)>.5
    model.interior={gid:tensor(binary_erosion(obs['labels']==gid,iterations=1))>.5 for gid in model.visible_owners}
    model.extrinsics=tensor(obs['extrinsics'])
    model.intrinsics=tensor(obs['intrinsics'])


def prepare(batch,case):
    prepare_geometry(batch,case)
    root=batch/case
    m=read(root/'motion_groups.json')
    owners=automatic_owners(m['groups'])
    observations_root=batch/'extremes'/case
    observations_data=observations(observations_root,owners)
    np.savez_compressed(root/'two_endpoint_observations.npz',**{f'{k}_{i}':v for i,o in enumerate(observations_data) for k,v in o.items()})
    write_json(root/'two_endpoint_prepared.json',dict(case=case,endpoint_frames=[o['frame'] for o in observations_data],
        endpoint_input_sha256=sha256(observations_root/'manifest.json'),
        observation_sha256=sha256(root/'two_endpoint_observations.npz'),
        origin_source_run=m['source_run'],semantic_input=m['semantic_input'],
        independent_movable_owners={str(g['semantic_owner']):g['id'] for g in m['groups'] if g['joint']['type']!='fixed' and g.get('semantic_owner') is not None},
        policy='Original PAct model and registration. Both extreme 2D RGBs and projected instance masks guide observed poses, motion intervals and geometry.'))
    print('PREPARED',case,flush=True)


def load_observations(root):
    data=np.load(root/'two_endpoint_observations.npz')
    return [{k:data[f'{k}_{i}'] for k in ['rgb','alpha','labels','extrinsics','intrinsics','frame']} for i in range(2)]


def search_bounds(group,diagonal):
    native=abs(float(np.diff(group['joint']['range'])[0]))
    if group['joint']['type'] in ('revolute','continuous'):
        # Origin's generated interval can be far too short, especially for a
        # carrying handle observed at widely separated states. Keep a generic
        # semantic bound; the endpoint mask, never GT angle, selects q.
        span=min(220.,max(220. if 'handle' in group['name'] else 180.,1.5*native))
    else:span=max(1.5*native,.35*diagonal)
    return -span,span


@torch.no_grad()
def estimate(model, obs, sweeps=2):
    model.requires_grad_(False)
    joints=[g for g in model.groups if g['joint']['type'] in ('revolute','continuous','prismatic') and g.get('semantic_owner') is not None]
    positions=[];traces=[]
    for endpoint, target in enumerate(obs):
        set_observation(model,target)
        pose={g['id']:0. for g in joints}
        for sweep in range(sweeps):
            for group in joints:
                gid=group['id'];owner_index=model.visible_owners.index(gid)
                target_mask=(model.targets==gid).float()
                lo,hi=search_bounds(group,model.asset['diagonal'])
                grid=np.linspace(lo,hi,17)
                def objective(q):
                    pose[gid]=float(q);model.set_pose(pose)
                    probability=model.probabilities()[owner_index]
                    return float(1-(2*(probability*target_mask).sum()+1)/(probability.sum()+target_mask.sum()+1))
                values=[objective(q) for q in grid]
                choice=int(np.argmin(values));left=grid[max(0,choice-1)];right=grid[min(16,choice+1)]
                result=minimize_scalar(objective,bounds=(left,right),method='bounded',options={'maxiter':22,'xatol':.01}) if right>left else None
                selected=(values[choice],float(grid[choice]))
                if result is not None and result.fun<selected[0]:selected=(float(result.fun),float(result.x))
                pose[gid]=selected[1]
                traces.append(dict(endpoint=endpoint,sweep=sweep,group=gid,instance=group['semantic_owner'],
                                   bounds=[lo,hi],selected_q=selected[1],selected_part_dice_loss=selected[0]))
        positions.append({str(k):v for k,v in pose.items()})
        print('POSE',model.config['case'],endpoint,positions[-1],flush=True)
    return positions,traces


@torch.no_grad()
def scores(model,obs,pose,original=False,progress=0.):
    set_observation(model,obs)
    if original:
        model.original_closed_transforms=transforms(model.asset['nodes'],progress)
        rgb=model.render(model.geometry(original=True),method='original')
        alpha=model.render(model.geometry(original=True),'silhouette',method='original')
        return dict(image=image_metrics(rgb,alpha,model.rgb,model.alpha),part_mean_IoU=None)
    model.set_pose(pose)
    geom=model.geometry()
    rgb=model.render(geom)
    alpha=model.render(geom,'silhouette')
    probs=model.probabilities(geom).cpu().numpy()
    background=np.clip(1-probs.sum(0),0,1)
    predicted=np.array([0,*model.visible_owners])[np.concatenate([background[None],probs]).argmax(0)]
    labels=obs['labels'];part=[]
    for gid in model.visible_owners:
        a=predicted==gid;b=labels==gid
        part.append(float((a&b).sum()/max((a|b).sum(),1)))
    return dict(image=image_metrics(rgb,alpha,model.rgb,model.alpha),part_mean_IoU=float(np.mean(part)) if part else None,
                per_part_IoU={str(k):v for k,v in zip(model.visible_owners,part)})


def fit(batch,case,steps=240):
    root=batch/case
    torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42)
    torch.backends.cuda.matmul.allow_tf32=False
    start=time.monotonic();model=load(root)
    observations_data=load_observations(root)
    poses,trace=estimate(model,observations_data)
    write_json(root/'two_endpoint_pose_estimation.json',dict(poses=poses,trace=trace,
        policy='Per independently matched semantic joint, derivative-free 2D instance Dice search at both observed endpoint frames; no GT numeric joint used.'))
    model.requires_grad_(True)
    model.log_global_scale.requires_grad_(False);model.screen_translation.requires_grad_(False)
    with torch.no_grad():model.log_shape.zero_();model.vertical_shift.zero_()
    before=[scores(model,o,p) for o,p in zip(observations_data,poses)]
    optimizer=torch.optim.Adam([{'params':[model.log_shape],'lr':.012},{'params':[model.vertical_shift],'lr':.004}])
    best=float('inf');best_state=None;best_step=0;history=[]
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        losses=[];values=[]
        for observation,pose in zip(observations_data,poses):
            set_observation(model,observation);model.set_pose(pose)
            loss,row=model.losses();losses.append(loss);values.append(row)
        loss=torch.stack(losses).mean();val=float(loss.detach());assert torch.isfinite(loss)
        if val<best:best=val;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};best_step=step
        loss.backward();optimizer.step();model.project()
        history.append(dict(update=step,loss=val,endpoints=values))
        if (step+1)%60==0:print('GEOMETRY',case,step+1,round(val,5),flush=True)
    with torch.no_grad():
        final=[]
        for o,p in zip(observations_data,poses):
            set_observation(model,o);model.set_pose(p);final.append(model.losses()[0])
        last=float(torch.stack(final).mean())
        if last<best:best=last;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};best_step=steps
    model.load_state_dict(best_state);after=[scores(model,o,p) for o,p in zip(observations_data,poses)]
    torch.save(dict(model=best_state,poses=poses,selected_completed_updates=best_step),root/'two_endpoint_geometry.pt')
    write_json(root/'two_endpoint_geometry_history.json',history)
    groups=copy.deepcopy(model.groups)
    for g in groups:
        if g['joint']['type']=='fixed':continue
        q=[pose.get(str(g['id']),0.) for pose in poses]
        g['joint']['range']=[float(q[0]),float(q[1])]
    write_json(root/'two_endpoint_motion_groups.json',dict(groups=groups,
        source_motion_groups_sha256=sha256(root/'motion_groups.json'),
        policy='Axis type, direction and pivot from v0.1.1 automatic semantic prior; observed interval from the two input images.'))
    write_json(root/'two_endpoint_fit.json',dict(case=case,endpoint_frames=[int(o['frame']) for o in observations_data],
        before=before,after=after,poses=poses,best_training_loss=best,selected_completed_updates=best_step,
        geometry_checkpoint_sha256=sha256(root/'two_endpoint_geometry.pt'),
        source_motion_groups_sha256=sha256(root/'motion_groups.json'),seconds=time.monotonic()-start))
    print('FIT COMPLETE',case,'seconds',round(time.monotonic()-start,1),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=pathlib.Path,required=True)
    p.add_argument('--case',required=True)
    p.add_argument('--stage',choices=['prepare','fit'],required=True)
    p.add_argument('--steps',type=int,default=240)
    a=p.parse_args();batch=a.root.resolve()
    if a.stage=='prepare':prepare(batch,a.case)
    else:fit(batch,a.case,a.steps)


if __name__=='__main__':main()
