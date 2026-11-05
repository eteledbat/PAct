"""Post-fit metrics for the two-extrema, latest-published-rule experiment."""
import argparse
import copy
import json
import pathlib
import sys

import numpy as np
import torch
import trimesh
from PIL import Image

REPO=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'repro/tools'))
from auto_part_semantics import sha256,write_json
from collision_refinement_common import surface
from evaluate_multicase_closed import joint_scores,distance_scores
from evaluate_observed_drawer_pose import build_mesh
from fit_observed_drawer_pose import load
from fit_two_extrema import load_observations,scores,set_observation
from multicase_closed_common import tensor,image_metrics
from evaluate_freeart_partnet_case import gt_parts
from extract_motion_extrema import video_meshes


def read(path):return json.loads(pathlib.Path(path).read_text())


def gt_mesh(scene,case,frame):
    if case=='20411':
        states,_=gt_parts(scene)
        return trimesh.util.concatenate(states[frame])
    infos=read(scene/'gt/part_info.json')
    return trimesh.util.concatenate([trimesh.Trimesh(v,f,process=False) for v,f in video_meshes(scene,frame,infos)])


def heldout(model,case,optimized=False):
    if case=='20411':return [],None
    scene=pathlib.Path(model.config['source_scene']);camera=read(scene/'transforms.json')
    frames=[25,50,75,100,125,149];rows=[]
    model.set_pose({});geom=model.geometry(original=not optimized)
    for index in frames:
        frame=camera['frames'][index]
        assert frame['state']==0 and frame['time']==0
        rgba=np.asarray(Image.open(scene/frame['file_path']).convert('RGBA').resize((256,256),Image.Resampling.LANCZOS)).astype(np.float32)/255
        alpha=rgba[:,:,3];rgb=rgba[:,:,:3]*alpha[:,:,None]+1-alpha[:,:,None]
        c2w=np.array(frame['transform_matrix']);c2w[:3,1:3]*=-1
        ext=tensor(np.linalg.inv(c2w));intr=tensor([[camera['focal_x']/camera['w'],0,camera['cx']/camera['w']],
                                                    [0,camera['focal_y']/camera['h'],camera['cy']/camera['h']],[0,0,1]])
        with torch.no_grad():
            predicted=model.render(geom,method='original' if not optimized else 'prior',extrinsics=ext,intrinsics=intr,reference=True)
            silhouette=model.render(geom,'silhouette',method='original' if not optimized else 'prior',extrinsics=ext,intrinsics=intr,reference=True)
            row=image_metrics(predicted,silhouette,tensor(rgb).permute(2,0,1),tensor(alpha))
        rows.append(dict(frame=index,**row))
    mean={k:float(np.mean([r[k] for r in rows])) for k in ['IoU','PSNR','foreground_PSNR','foreground_L1']}
    return rows,mean


def evaluate(batch,case):
    cases=read(batch/'protocol.json')['cases']
    assert all((batch/c/'two_endpoint_fit.json').exists() for c in cases),'All fitting must finish before GT metrics'
    root=batch/case;fit=read(root/'two_endpoint_fit.json')
    assert sha256(root/'two_endpoint_geometry.pt')==fit['geometry_checkpoint_sha256']
    torch.set_num_threads(4)
    model=load(root);model.requires_grad_(False)
    observations_data=load_observations(root);poses=fit['poses']
    methods={}
    for name in ['origin','axis','axis_geometry']:
        if name=='axis_geometry':
            model.load_state_dict(torch.load(root/'two_endpoint_geometry.pt',map_location='cuda',weights_only=False)['model'])
        endpoint=[]
        for i,(obs,pose) in enumerate(zip(observations_data,poses)):
            endpoint.append(scores(model,obs,pose,original=name=='origin',progress=float(i)))
        rows,mean=heldout(model,case,optimized=name=='axis_geometry')
        methods[name]=dict(endpoints=endpoint,heldout_views=rows,heldout_mean=mean)
    model.set_pose({})
    # The cached GT target list is dataset metadata from the earlier run.
    # Recompute fixed-body alignment and component correspondence on THIS
    # newly generated Origin; never reuse the old body's similarity.
    new_run=model.manifest['source_run']
    old_inventory=read(REPO/'repro/outputs/original_mask_multicase_20261029_v1/input_inventory.json')
    model.manifest['source_run']=next(x['run'] for x in old_inventory if x['case']==case)
    try:axis=joint_scores(model)
    finally:model.manifest['source_run']=new_run
    methods['origin']['axis']=axis['joint_methods']['original']
    methods['axis']['axis']=axis['joint_methods']['prior']
    methods['axis_geometry']['axis']=axis['joint_methods']['optimized']
    for method,source in [('origin','original'),('axis','prior'),('axis_geometry','optimized')]:
        methods[method]['surface_reconstruction']=axis['surface_reconstruction'][source]
    # Same six-state bidirectional surface-distance family as the old protocol,
    # with frames evenly spaced between this experiment's selected endpoints.
    alignment=axis['alignment']
    R=np.array(alignment['R_export_to_GT']);scale=alignment['scale'];shift=np.array(alignment['translation']);D=alignment['GT_static_diagonal']
    scene=pathlib.Path(model.config['source_scene'])
    first,last=fit['endpoint_frames'];frames=np.rint(np.linspace(first,last,6)).astype(int)
    fitted_tree=read(root/'two_endpoint_motion_groups.json')['groups']
    original_state={k:v.detach().clone() for k,v in model.state_dict().items()}
    trajectory={}
    for method in ['origin','axis','axis_geometry']:
        model.load_state_dict(original_state)
        if method!='axis_geometry':
            with torch.no_grad():model.log_shape.zero_();model.vertical_shift.zero_()
        tree=model.asset['nodes'] if method=='origin' else fitted_tree
        sequence=[]
        for i,(t,frame) in enumerate(zip(np.linspace(0,1,6),frames)):
            truth=surface(gt_mesh(scene,case,int(frame)),12000,177+i)
            prediction=build_mesh(model,optimized=method=='axis_geometry',tree=tree,progress=float(t))
            predicted=scale*(surface(prediction,12000,178+i)@R.T)+shift
            sequence.append(dict(progress=float(t),frame=int(frame),**distance_scores(predicted,truth,D)))
        trajectory[method]=sequence
    result=dict(case=case,endpoint_frames=fit['endpoint_frames'],methods=methods,alignment=alignment,
                axis_correspondence=axis['correspondence'],six_state_surface=trajectory,
                evaluation_reference=str(REPO/'repro/outputs/closed_multicase_20261028_v1'/case/'evaluation.json'),
                protocol='Endpoint IoU/part mIoU are training diagnostics. Held-out state0 frames25/50/75/100/125/149 are untouched camera views; q0 canonical rest. GT axis/type/pivot and surface metrics computed only after all fits. Six-state trajectory uses six recorded frames between selected extremes; prediction interpolates its fitted endpoint joints.',
                surface_metric='16000 or 12000 seeded mesh samples, nonsquared symmetric Chamfer normalized by frozen GT body diagonal; F-score at 1% and 5% D',
                old_protocol_comparison_caveat='Old closed-image experiment optimizes a different conditioning image and manual semantic recipe; compare per-case metrics with this input change explicitly disclosed.')
    write_json(root/'evaluation.json',result)
    print('EVALUATED',case,{k:round(v['axis']['axis_correct_rate'],3) for k,v in methods.items()},flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--case');a=p.parse_args();batch=a.root.resolve()
    for case in ([a.case] if a.case else read(batch/'protocol.json')['cases']):evaluate(batch,case)


if __name__=='__main__':main()
