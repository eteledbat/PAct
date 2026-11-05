"""Automatic closed-view correction of the published single-image PAct assets.

Uses the existing automatic semantic instances and the selected closed RGB/mask.
No case recipe, hand-drawn box, old fitted checkpoint, or GT 3D joint is an input.
"""
import argparse
import copy
import json
import pathlib
import sys
import time

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scipy.spatial import cKDTree

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from auto_part_semantics import sha256, write_json
from collision_refinement_common import load_asset
from fit_auto_semantic_initial_frame import basis_for
from fit_closed_extrema_geometry import CASES, prepare as prepare_base
from fit_observed_drawer_pose import load, preview
from fit_two_extrema import observations, scores, set_observation
from fit_vlm_component_geometry import automatic_owners, semantic_target
from multicase_closed_common import tensor
from semantic_axis_router import side_hinge
from semantic_motion_rules import top_hinged_door


def read(path):
    return json.loads(pathlib.Path(path).read_text())


def recover_unmatched_instances(root):
    """Assign a distinct generated static node to one unmatched movable mask.

    This conservative rule handles a generated movable leaf mislabeled as base:
    require a single unmatched semantic leaf and a single non-body native node
    that is wholly assigned to the base. Nothing is selected by case number.
    """
    manifest=read(root/'motion_groups.json')
    groups=manifest['groups']
    owners=automatic_owners(groups)
    semantic=read(pathlib.Path(manifest['semantic_input']))
    movable=[p for p in semantic['parts'] if p['semantic'] in ('door','drawer','handle') and p['instance_id'] not in owners]
    body_source=next(g['source_node'] for g in groups if g['parent']==-1)
    candidates=[]
    for source in sorted({g['source_node'] for g in groups}-{body_source}):
        fragments=[g for g in groups if g['source_node']==source]
        if all(g['joint']['type']=='fixed' and g['semantic_owner'] == 1 for g in fragments):
            candidates.append((source,fragments))
    if len(movable)!=1 or len(candidates)!=1:
        return dict(recovered=False,unmatched=[p['instance_id'] for p in movable],candidate_sources=[c[0] for c in candidates])
    part=movable[0];source,fragments=candidates[0]
    if part['semantic']!='door':
        return dict(recovered=False,reason='Only a door hinge can be inferred from this rule')
    asset=load_asset(manifest['source_run'])
    meshes={p['node']['id']:p['mesh'] for p in asset['parts']}
    v=np.asarray(meshes[source].vertices)
    body=np.asarray(meshes[body_source].vertices)
    # The long horizontal panel with this source geometry uses its upper edge;
    # the semantic class comes from the automatic input VLM result.
    joint,_,_,axis_evidence=top_hinged_door(v,body,{'type':'fixed','range':[0.,0.]})
    primary=min(fragments,key=lambda g:g['id'])
    gid=primary['id']
    primary.update(name='door_recovered_'+str(part['instance_id']),joint=joint,kind='hinge_x',
                   semantic_owner=part['instance_id'],shape_owner=None)
    B,anchor,length,kind=basis_for(primary,v)
    primary.update(deformation_basis=B.tolist(),deformation_anchor=anchor.tolist(),length=length,kind=kind)
    for g in fragments:
        g['semantic_owner']=part['instance_id']
        if g is not primary:
            g['parent']=gid
            g['shape_owner']=gid
    owners=automatic_owners(groups)
    manifest['automatic_instance_to_primary_group']={str(k):v for k,v in owners.items()}
    manifest['unresolved_geometry_frozen_groups']=[g['id'] for g in groups if g['semantic_owner'] is None]
    manifest['recovered_semantic_leaf']={'instance_id':part['instance_id'],'source_node':source,'primary_group':gid,'evidence':axis_evidence}
    write_json(root/'motion_groups.json',manifest)
    cfg=read(root/'config.json')
    cfg['prior_manifest_sha256']=sha256(root/'motion_groups.json')
    for g in fragments:
        cfg['palette'][str(g['id'])]=cfg['palette'][str(gid)]
    data=np.load(root/'observation.npz')
    original_ids=np.load(root/'original_instance_ids.npy')
    # The closed label image has the same physical instance IDs as the input.
    endpoint=read(root/'prepared.json')
    mask=np.load(endpoint['closed_mask'])
    labels=semantic_target(mask,owners)
    palette=cfg['palette']
    colors=np.ones((*labels.shape,3),np.float32)
    for owner,group_id in owners.items():colors[labels==group_id]=palette[str(group_id)]
    np.savez_compressed(root/'observation.npz',rgb=data['rgb'],alpha=data['alpha'],labels=labels,
                        semantic=colors,extrinsics=data['extrinsics'],intrinsics=data['intrinsics'])
    cfg['observations_sha256']=sha256(root/'observation.npz')
    write_json(root/'config.json',cfg)
    return dict(recovered=True,semantic_instance=part['instance_id'],source_node=source,primary_group=gid,
                input_mask_pixels=int((original_ids==part['instance_id']).sum()),closed_mask_pixels=int((mask==part['instance_id']).sum()))


def prepare(batch,case):
    root=batch/case
    if not (root/'prepared.json').exists():prepare_base(batch,case)
    if (root/'automatic_recovery.json').exists():recovery=read(root/'automatic_recovery.json')
    else:
        recovery=recover_unmatched_instances(root)
        write_json(root/'automatic_recovery.json',recovery)
    manifest=read(root/'motion_groups.json')
    owners=automatic_owners(manifest['groups'])
    cfg=read(root/'config.json')
    for g in manifest['groups']:
        owner=g.get('semantic_owner')
        if owner in owners:cfg['palette'][str(g['id'])]=cfg['palette'][str(owners[owner])]
    data=np.load(root/'observation.npz')
    labels=data['labels']
    colors=np.ones((*labels.shape,3),np.float32)
    for owner,gid in owners.items():colors[labels==gid]=cfg['palette'][str(gid)]
    np.savez_compressed(root/'observation.npz',rgb=data['rgb'],alpha=data['alpha'],labels=labels,
                        semantic=colors,extrinsics=data['extrinsics'],intrinsics=data['intrinsics'])
    cfg['observations_sha256']=sha256(root/'observation.npz')
    write_json(root/'config.json',cfg)
    print('AUTO RECOVERY',case,recovery,flush=True)


def pivot_candidates(model,group):
    original=copy.deepcopy(group['joint']['axis'])
    if group['joint']['type']!='revolute' or 'door' not in group['name']:
        return [('published',original)]
    direction=np.asarray(original['direction'])
    if abs(direction[1])<.9:
        return [('published',original)]
    asset=model.asset
    mesh={p['node']['id']:p['mesh'] for p in asset['parts']}
    v=np.asarray(mesh[group['source_node']].vertices)
    body_source=next(g['source_node'] for g in model.groups if g['parent']==-1)
    body=np.asarray(mesh[body_source].vertices)
    result=[('published',original)]
    for edge in ('left','right'):
        joint,_=side_hinge(v,body,edge,group['joint'])
        axis=joint['axis']
        if all(np.linalg.norm(np.asarray(axis['origin'])-np.asarray(row[1]['origin']))>1e-3 for row in result):
            result.append((edge,axis))
    if getattr(model,'unconstrained_pivots',False):return result
    # The hinge must be attached to the cabinet. Reject an image-only pivot
    # swap unless its body contact is substantially stronger than published.
    tree=cKDTree(body)
    y=np.linspace(*np.quantile(v[:,1],[.1,.9]),21)
    distances=[]
    for _,axis in result:
        origin=np.asarray(axis['origin'])
        line=np.column_stack([np.full_like(y,origin[0]),y,np.full_like(y,origin[2])])
        distances.append(float(np.median(tree.query(line)[0])))
    best=int(np.argmin(distances))
    if best and distances[0]-distances[best]>.015*model.asset['diagonal']:
        return [result[best]]
    return [result[0]]


def side_door_closed_q(model,group):
    """Angle that brings the generated door's width into the body front plane."""
    meshes={p['node']['id']:p['mesh'] for p in model.asset['parts']}
    ids=model.membership[group['membership_prefix']+'_vertices']
    vertices=np.asarray(meshes[group['source_node']].vertices)[ids]
    xz=vertices[:,[0,2]]
    center=xz.mean(0)
    _,eig=np.linalg.eigh(np.cov(xz.T))
    width=eig[:,-1]
    origin=np.asarray(group['joint']['axis']['origin'])
    if width@(center-origin[[0,2]])<0:width=-width
    body=next(g for g in model.groups if g['parent']==-1)
    body_vertices=np.asarray(meshes[body['source_node']].vertices)[model.membership[body['membership_prefix']+'_vertices']]
    body_x=float(body_vertices[:,0].mean())
    target=0. if origin[0]<body_x else np.pi
    q=float(np.degrees(np.arctan2(width[1],width[0])-target))
    q=(q+180)%360-180
    if group['joint']['axis']['direction'][1]<0:q=-q
    return q


@torch.no_grad()
def pose_and_axis(model,obs,sweeps=2,closed_drawers=False,geometry_closed_pose=False):
    set_observation(model,obs)
    model.requires_grad_(False)
    joints=[g for g in model.groups if g['joint']['type'] in ('revolute','continuous','prismatic') and g.get('semantic_owner') is not None]
    pose={g['id']:0. for g in joints}
    trace=[]
    for sweep in range(sweeps):
        for g in joints:
            gid=g['id']
            if closed_drawers and g['joint']['type']=='prismatic':
                # The selected observation is explicitly the closed extreme.
                # A sliding leaf rests at its retracted canonical state.
                pose[gid]=0.
                model.set_pose(pose)
                trace.append(dict(sweep=sweep,group=gid,selected_axis='published',selected_q=0.,
                                  policy='automatic closed-state prismatic prior'))
                continue
            if (geometry_closed_pose and g['joint']['type']=='revolute' and 'door' in g['name']
                    and abs(g['joint']['axis']['direction'][1])>.9):
                label,axis=pivot_candidates(model,g)[0]
                g['joint']['axis']=axis
                q=side_door_closed_q(model,g)
                pose[gid]=q
                model.set_pose(pose)
                trace.append(dict(sweep=sweep,group=gid,selected_axis=label,selected_q=q,
                                  policy='door width aligned to cabinet front plane using automatic source component'))
                continue
            target=(model.targets==gid).float()
            choices=[]
            for label,axis in pivot_candidates(model,g):
                g['joint']['axis']=axis
                if g['joint']['type']=='prismatic':
                    span=max(.35*model.asset['diagonal'],1.5*abs(float(np.diff(g['joint']['range'])[0])))
                elif 'handle' in g['name']:
                    span=220.
                else:span=min(135.,max(90.,1.3*abs(float(np.diff(g['joint']['range'])[0]))))
                grid=np.linspace(-span,span,13)
                def objective(q):
                    pose[gid]=float(q)
                    model.set_pose(pose)
                    alpha=model.render(kind='silhouette')
                    sil=(alpha-model.alpha).abs().mean()
                    if target.sum()<32:return float(sil)
                    p=model.probabilities()[model.visible_owners.index(gid)]
                    dice=1-(2*(p*target).sum()+1)/(p.sum()+target.sum()+1)
                    if 'handle' in g['name']:
                        return float(.5*sil+2*dice+.0002*(q/span)**2)
                    return float(2*sil+.25*dice+.0002*(q/span)**2)
                values=[objective(q) for q in grid]
                ix=int(np.argmin(values))
                left,right=grid[max(ix-1,0)],grid[min(ix+1,len(grid)-1)]
                sol=minimize_scalar(objective,bounds=(left,right),method='bounded',options={'maxiter':12,'xatol':.1}) if right>left else None
                best=(values[ix],float(grid[ix]))
                if sol is not None and sol.fun<best[0]:best=(float(sol.fun),float(sol.x))
                choices.append((best[0],best[1],label,copy.deepcopy(axis)))
            value,q,label,axis=min(choices,key=lambda x:x[0])
            g['joint']['axis']=axis
            pose[gid]=q
            model.set_pose(pose)
            trace.append(dict(sweep=sweep,group=gid,selected_axis=label,selected_q=q,objective=value,
                              candidates=[dict(loss=v,q=q0,axis=name) for v,q0,name,_ in choices]))
    return pose,trace


def align(model,steps=100):
    model.log_shape.requires_grad_(False)
    model.vertical_shift.requires_grad_(False)
    model.log_global_scale.requires_grad_(True)
    model.screen_translation.requires_grad_(True)
    optimizer=torch.optim.Adam([{'params':[model.log_global_scale],'lr':.008},
                                {'params':[model.screen_translation],'lr':.006}])
    history=[]
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss,detail=model.losses(align=True)
        loss.backward()
        optimizer.step()
        model.project()
        history.append(dict(step=step+1,loss=float(loss.detach()),**detail))
    return history


def rebuild_with_selected_axes(model,root,pose):
    """Re-anchor each deformation basis at its automatically selected hinge."""
    membership=model.membership
    mesh={p['node']['id']:p['mesh'] for p in model.asset['parts']}
    for g in model.groups:
        if g['joint']['type']!='revolute' or g.get('shape_owner') is not None:continue
        fragments=[p for p in model.groups if p.get('semantic_owner')==g.get('semantic_owner')]
        vertices=np.concatenate([np.asarray(mesh[p['source_node']].vertices)[membership[p['membership_prefix']+'_vertices']] for p in fragments])
        B,anchor,length,kind=basis_for(g,vertices)
        g.update(deformation_basis=B.tolist(),deformation_anchor=anchor.tolist(),length=length,kind=kind)
    scale=model.log_global_scale.detach().clone()
    translation=model.screen_translation.detach().clone()
    write_json(root/'motion_groups.json',model.manifest)
    cfg=read(root/'config.json')
    cfg['prior_manifest_sha256']=sha256(root/'motion_groups.json')
    write_json(root/'config.json',cfg)
    replacement=load(root)
    replacement.unconstrained_pivots=getattr(model,'unconstrained_pivots',False)
    with torch.no_grad():
        replacement.log_global_scale.copy_(scale)
        replacement.screen_translation.copy_(translation)
    replacement.set_pose(pose)
    return replacement


def auto_box_loss(model):
    """Instance bounding boxes from the input mask, without drawn rectangles."""
    boxes=[]
    labels=model.targets.detach().cpu().numpy()
    for gid in model.visible_owners:
        yy,xx=np.where(labels==gid)
        if len(xx)<32:continue
        # Full vertex union makes fragments and attached pulls one physical leaf.
        coords=[]
        for i,p in enumerate(model.parts):
            if model.owner_map[p['group']['id']]!=gid:continue
            verts=p['vertex_local']
            if len(verts)==0:continue
            local=tensor(verts[::max(1,len(verts)//1024)])
            xyz=model.deform(i,local)
            T=model.closed_transforms[p['group']['id']]
            xyz=xyz@tensor(T[:3,:3]).T+tensor(T[:3,3])
            xyz=model.log_global_scale.exp()*(xyz@model.R.T)+model.translation_basis@model.screen_translation
            camera=xyz@model.extrinsics[:3,:3].T+model.extrinsics[:3,3]
            uv=camera[:,:2]/camera[:,2:3]
            uv=uv*torch.stack([model.intrinsics[0,0],model.intrinsics[1,1]])+model.intrinsics[:2,2]
            coords.append(uv)
        if not coords:continue
        uv=torch.cat(coords)
        pred=torch.cat([torch.quantile(uv,.01,dim=0),torch.quantile(uv,.99,dim=0)])
        tgt=tensor([np.quantile(xx/256,.01),np.quantile(yy/256,.01),
                    np.quantile(xx/256,.99),np.quantile(yy/256,.99)])
        boxes.append((pred-tgt).abs().mean())
    return torch.stack(boxes).mean() if boxes else model.log_shape.sum()*0


def closed_geometry_loss(model):
    loss,detail=model.losses()
    box=auto_box_loss(model)
    semantic=((model.render(kind='semantic')-model.semantic).square()*model.weight[None]).mean()/model.weight.mean()
    loss=loss+3*box+2*semantic
    return loss,dict(**detail,box=float(box.detach()),semantic=float(semantic.detach()))


def fit(batch,case,closed_drawers=False,geometry_closed_pose=False,unconstrained_pivots=False):
    root=batch/case
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cuda.matmul.allow_tf32=False
    start=time.monotonic()
    model=load(root)
    model.unconstrained_pivots=bool(unconstrained_pivots)
    owners=automatic_owners(model.groups)
    obs=observations(batch/'closed_references'/case,owners)[0]
    pose,trace0=pose_and_axis(model,obs,sweeps=2,closed_drawers=closed_drawers,geometry_closed_pose=geometry_closed_pose)
    model=rebuild_with_selected_axes(model,root,pose)
    alignment=align(model,100)
    pose,trace1=pose_and_axis(model,obs,sweeps=2,closed_drawers=closed_drawers,geometry_closed_pose=geometry_closed_pose)
    model=rebuild_with_selected_axes(model,root,pose)
    alignment+=align(model,40)
    pose,trace2=pose_and_axis(model,obs,sweeps=1,closed_drawers=closed_drawers,geometry_closed_pose=geometry_closed_pose)
    model=rebuild_with_selected_axes(model,root,pose)
    before=scores(model,obs,pose)
    preview(model,root/'auto_closed_before.png')
    with torch.no_grad():
        model.log_shape.zero_()
        model.vertical_shift.zero_()
    model.requires_grad_(True)
    model.log_global_scale.requires_grad_(False)
    model.screen_translation.requires_grad_(False)
    optimizer=torch.optim.Adam([{'params':[model.log_shape],'lr':.012},
                                {'params':[model.vertical_shift],'lr':.004}])
    best=float('inf')
    best_state=None
    best_step=0
    history=[]
    for step in range(240):
        optimizer.zero_grad(set_to_none=True)
        loss,detail=closed_geometry_loss(model)
        value=float(loss.detach())
        if value<best:
            best=value
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            best_step=step
        loss.backward()
        optimizer.step()
        model.project()
        history.append(dict(step=step,loss=value,**detail))
        if (step+1)%80==0:print('GEOMETRY',case,step+1,round(value,5),flush=True)
    with torch.no_grad():
        last_loss=closed_geometry_loss(model)[0]
        if float(last_loss)<best:
            best=float(last_loss)
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            best_step=240
    model.load_state_dict(best_state)
    after=scores(model,obs,pose)
    preview(model,root/'auto_closed_after.png')
    # Persist inferred axes, so later video/evaluation loads the same model.
    write_json(root/'motion_groups.json',model.manifest)
    cfg=read(root/'config.json')
    cfg['prior_manifest_sha256']=sha256(root/'motion_groups.json')
    cfg['geometry_policy']='Automatic closed-frame pose, pivot candidate, camera alignment and mask-derived panel boxes; 240 shape updates'
    write_json(root/'config.json',cfg)
    torch.save(dict(model=best_state,pose={str(k):v for k,v in pose.items()},
                    completed_updates=best_step,selection='minimum closed-image training objective'),
               root/'auto_closed_geometry.pt')
    write_json(root/'auto_closed_history.json',history)
    write_json(root/'auto_closed_fit.json',dict(case=case,pose={str(k):v for k,v in pose.items()},
        before=before,after=after,initial_pose_axis_trace=trace0,registered_pose_axis_trace=trace1+trace2,
        alignment_history=alignment,recovery=read(root/'automatic_recovery.json'),
        selected_step=best_step,selected_loss=best,seconds=time.monotonic()-start,
        closed_drawer_prior=bool(closed_drawers),geometry_closed_pose=bool(geometry_closed_pose),
        pivot_policy=('all visually scored edges' if unconstrained_pivots else 'body-contact gate: alternate hinge only if median line distance improves by >1.5% object diagonal'),
        axis_manifest_sha256=sha256(root/'motion_groups.json'),checkpoint_sha256=sha256(root/'auto_closed_geometry.pt'),
        policy='No manual part assignment or boxes; closed RGB/instance mask is the fitting observation.'))
    print('FIT COMPLETE',case,round(before['image']['IoU'],3),'->',round(after['image']['IoU'],3),
          'seconds',round(time.monotonic()-start,1),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=pathlib.Path,required=True)
    p.add_argument('--case',required=True)
    p.add_argument('--stage',choices=['prepare','fit'],required=True)
    p.add_argument('--closed-drawers',action='store_true')
    p.add_argument('--geometry-closed-pose',action='store_true')
    p.add_argument('--unconstrained-pivot-search',action='store_true')
    a=p.parse_args()
    batch=a.root.resolve()
    if a.stage=='prepare':prepare(batch,a.case)
    else:fit(batch,a.case,closed_drawers=a.closed_drawers,
             geometry_closed_pose=a.geometry_closed_pose,unconstrained_pivots=a.unconstrained_pivot_search)


if __name__=='__main__':main()
