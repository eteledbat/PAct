"""Closed-image geometry fit with the historical manual protocol and automatic labels.

GT meshes/joints and the historical manual recipe are never fitting inputs.  The
only differences from the manual optimizer are its source motion groups and
automatically measured instance boxes/symmetric pairs.
"""
import argparse
import json
import pathlib

import numpy as np
import torch

from auto_part_semantics import sha256, write_json
from fit_auto_closed_geometry import auto_box_loss, read, side_door_closed_q
from fit_closed_extrema_geometry import prepare as prepare_base
from fit_observed_drawer_pose import load, preview
from fit_two_extrema import observations, estimate, scores
from fit_vlm_component_geometry import automatic_owners


def automatic_pairs(root, manifest):
    """Share shape parameters for adjacent, similarly sized door/drawer leaves."""
    owners = automatic_owners(manifest['groups'])
    semantic_path = pathlib.Path(manifest['semantic_input'])
    if not semantic_path.is_absolute():
        semantic_path = pathlib.Path(__file__).resolve().parents[2] / semantic_path
    classes = {int(p['instance_id']): p['semantic'] for p in read(semantic_path)['parts']}
    ids = np.load(read(root/'prepared.json')['closed_mask'])
    rows = []
    for instance, gid in owners.items():
        category = classes.get(instance)
        if category not in ('door','drawer'):
            # The automatic axis router may have recognized a leaf that the
            # VLM left as unknown; use its generated motion-group class.
            name=next(g['name'] for g in manifest['groups'] if g['id']==gid)
            category=next((kind for kind in ('door','drawer') if kind in name),category)
        if category not in ('door', 'drawer'):
            continue
        yy, xx = np.where(ids == instance)
        if len(xx) < 32:
            continue
        rows.append((gid, category, np.array([xx.min(), yy.min(), xx.max(), yy.max()], float)))
    proposals = []
    for i, (ga, ca, a) in enumerate(rows):
        for gb, cb, b in rows[i+1:]:
            if ca != cb:
                continue
            ah, bh = a[3]-a[1]+1, b[3]-b[1]+1
            aw, bw = a[2]-a[0]+1, b[2]-b[0]+1
            overlap = max(0., min(a[3], b[3])-max(a[1], b[1])+1)
            if overlap/min(ah, bh) < .7 or min(ah,bh)/max(ah,bh) < .7:
                continue
            if min(aw,bw)/max(aw,bw) < .6:
                continue
            if not (a[2] < b[0] or b[2] < a[0]):
                continue
            proposals.append((ga,gb))
    used = set()
    pairs = []
    for a,b in sorted(proposals):
        if a not in used and b not in used:
            pairs.append([a,b]);used.update((a,b))
    return pairs


def prepare(batch,case):
    root=batch/case
    prepare_base(batch,case)
    manifest=read(root/'motion_groups.json')
    pairs=automatic_pairs(root,manifest)
    manifest['pairs']=pairs
    manifest['automatic_shape_pairs']=pairs
    write_json(root/'motion_groups.json',manifest)
    cfg=read(root/'config.json')
    cfg['prior_manifest_sha256']=sha256(root/'motion_groups.json')
    cfg['geometry_policy']='Historical manual 100 alignment + 240 shape updates, automatic instance boxes and paired shapes, frozen published axes'
    write_json(root/'config.json',cfg)
    write_json(root/'automatic_manual_style_input.json',dict(case=case,pairs=pairs,
        published_model=read(root/'prepared.json')['published_model'],
        closed_mask=read(root/'prepared.json')['closed_mask'],
        policy='No manual recipes or boxes; original automatic VLM semantics and projected closed instance IDs only'))
    print('PREPARED',case,'pairs',pairs,flush=True)


def manual_style_loss(model,align=False):
    geom=model.geometry()
    rgb=model.render(geom)
    alpha=model.render(geom,'silhouette')
    photo=((rgb-model.rgb).abs()*model.weight[None]).mean()/model.weight.mean()
    sil=(alpha-model.alpha).abs().mean()
    if align:
        return photo+2*sil,dict(photo=float(photo.detach()),silhouette=float(sil.detach()))
    semantic=((model.render(geom,'semantic')-model.semantic).square()*model.weight[None]).mean()/model.weight.mean()
    # The historical hand-drawn boxes supervise movable leaves only.  Apply
    # the same scope to boxes measured from the automatic instance masks.
    visible=model.visible_owners
    model.visible_owners=[gid for gid in visible if model.byid[gid]['joint']['type']!='fixed']
    try:
        box=auto_box_loss(model)
    finally:
        model.visible_owners=visible
    prior=.002*model.log_shape.square().mean()+.01*model.vertical_shift.square().mean()
    loss=photo+2*sil+2*semantic+3*box+prior
    return loss,dict(photo=float(photo.detach()),silhouette=float(sil.detach()),
                     semantic=float(semantic.detach()),box=float(box.detach()),prior=float(prior.detach()))


def fit(batch,case):
    root=batch/case
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cuda.matmul.allow_tf32=False
    model=load(root)
    model.requires_grad_(False)
    obs=observations(batch/'closed_references'/case,automatic_owners(model.groups))[0]
    pose,trace=estimate(model,[obs])
    pose=pose[0]
    # A closed reference has a retracted drawer; side-opening panels lie in the
    # cabinet front plane.  These are global geometric rules, not case labels.
    for group in model.groups:
        gid=group['id']
        if str(gid) not in pose:
            continue
        joint=group['joint']
        if joint['type']=='prismatic':
            pose[str(gid)]=0.
        elif (joint['type']=='revolute' and 'door' in group['name']
              and abs(joint['axis']['direction'][1])>.9):
            pose[str(gid)]=side_door_closed_q(model,group)
    model.set_pose(pose)
    optimizer=torch.optim.Adam([{'params':[model.log_global_scale],'lr':.008},
                                {'params':[model.screen_translation],'lr':.006}])
    model.log_global_scale.requires_grad_(True)
    model.screen_translation.requires_grad_(True)
    history=[]
    for step in range(100):
        optimizer.zero_grad(set_to_none=True)
        loss,detail=manual_style_loss(model,align=True)
        if not torch.isfinite(loss):raise FloatingPointError((case,'alignment',step))
        loss.backward();optimizer.step();model.project()
        history.append(dict(stage='alignment',step=step+1,loss=float(loss.detach()),**detail))
    with torch.no_grad():
        model.log_shape.zero_();model.vertical_shift.zero_()
    before=scores(model,obs,pose)
    preview(model,root/'manual_style_before.png')
    model.log_global_scale.requires_grad_(False)
    model.screen_translation.requires_grad_(False)
    model.log_shape.requires_grad_(True)
    model.vertical_shift.requires_grad_(True)
    optimizer=torch.optim.Adam([{'params':[model.log_shape],'lr':.012},
                                {'params':[model.vertical_shift],'lr':.004}])
    for step in range(240):
        optimizer.zero_grad(set_to_none=True)
        loss,detail=manual_style_loss(model)
        if not torch.isfinite(loss):raise FloatingPointError((case,'geometry',step))
        loss.backward();optimizer.step();model.project()
        history.append(dict(stage='geometry',step=step+1,loss=float(loss.detach()),**detail))
        if (step+1)%80==0:print('GEOMETRY',case,step+1,round(float(loss.detach()),5),flush=True)
    # Match the historical manual protocol: final iteration, not GT/CD selection.
    after=scores(model,obs,pose)
    preview(model,root/'manual_style_after.png')
    torch.save(dict(model=model.state_dict(),pose=pose,completed_updates=240,
                    selection='historical manual final iteration'),root/'manual_style_geometry.pt')
    write_json(root/'manual_style_history.json',history)
    write_json(root/'manual_style_fit.json',dict(case=case,pose=pose,pose_trace=trace,before=before,after=after,
        checkpoint_sha256=sha256(root/'manual_style_geometry.pt'),
        axes_sha256=sha256(root/'motion_groups.json'),
        input=read(root/'automatic_manual_style_input.json'),
        training='100 alignment steps, 240 geometry steps, exact manual loss weights with automatic instance boxes and shape pairs; geometric side-door closure and retracted closed drawers; no GT 3D/CD during fitting'))
    print('FIT COMPLETE',case,round(before['image']['IoU'],3),'->',round(after['image']['IoU'],3),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=pathlib.Path,required=True)
    p.add_argument('--case',required=True)
    p.add_argument('--stage',choices=('prepare','fit'),required=True)
    a=p.parse_args()
    batch=a.root.resolve()
    if a.stage=='prepare':prepare(batch,a.case)
    else:fit(batch,a.case)


if __name__=='__main__':main()
