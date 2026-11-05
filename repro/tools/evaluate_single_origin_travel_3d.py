"""Ten-case post-fit 3D surface-voxel IoU and axis/type/pivot comparison.

The published and GT meshes are not watertight. Consequently this reports
surface-shell voxel overlap, not a fictitious solid-volume IoU. All methods
share the original fixed-body-only alignment and the selected endpoint frames.
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import torch
from scipy.ndimage import binary_dilation,generate_binary_structure

REPO=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'repro/tools'))
from auto_part_semantics import sha256,write_json
from collision_refinement_common import surface
from evaluate_multicase_closed import joint_scores
from evaluate_observed_drawer_pose import build_mesh
from evaluate_freeart_partnet_case import gt_parts
from evaluate_two_extrema import gt_mesh
from fit_observed_drawer_pose import load,read
from fit_single_origin_extrema_travel import source

SAMPLES=80000
GRID_DIVISOR=96
RADII=(0,1,2)


def shell(points,origin,pitch,shape):
    indices=np.floor((points-origin)/pitch).astype(np.int64)
    keep=np.all((indices>=0)&(indices<np.array(shape)),axis=1)
    indices=indices[keep]
    grid=np.zeros(shape,dtype=bool)
    grid[indices[:,0],indices[:,1],indices[:,2]]=True
    return grid


def overlap(a,b):
    union=np.count_nonzero(a|b)
    return float(np.count_nonzero(a&b)/union) if union else None


def evaluate(batch,case):
    out=batch/case;fit=read(out/'travel_fit.json');travel=read(out/'travel_motion_groups.json')['groups']
    root,modelroot=source(case)
    assert sha256(modelroot/'observed_pose_geometry.pt')==fit['source_checkpoint_sha256']
    assert sha256(modelroot/'motion_groups.json')==read(out/'travel_motion_groups.json')['source_motion_groups_sha256']
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    model=load(modelroot,checkpoint='observed_pose_geometry.pt');model.requires_grad_(False)
    assert next(g for g in model.groups if g['id']==10)['joint']['type']=='fixed'
    with torch.no_grad():axis=joint_scores(model)
    A=axis['alignment'];R=np.asarray(A['R_export_to_GT']);scale=A['scale'];shift=np.asarray(A['translation']);D=A['GT_static_diagonal']
    scene=pathlib.Path(model.config['source_scene']);frames=fit['frames']
    if case=='20411':
        import trimesh
        states,_=gt_parts(scene)
        truth_meshes=[trimesh.util.concatenate(states[frame]) for frame in frames]
    else:truth_meshes=[gt_mesh(scene,case,frame) for frame in frames]
    methods=[('origin',model.asset['nodes'],False),('single_image_geometry',model.groups,True),
             ('extrema_travel',travel,True)]
    clouds={};mesh_checks={}
    for i,(frame,truth) in enumerate(zip(frames,truth_meshes)):
        clouds[('GT',i)]=surface(truth,SAMPLES,500+i)
        mesh_checks[('GT',i)]=dict(watertight=bool(truth.is_watertight),faces=len(truth.faces))
        for name,tree,optimized in methods:
            pred=build_mesh(model,optimized=optimized,tree=tree,progress=float(i))
            clouds[(name,i)]=scale*(surface(pred,SAMPLES,500+i)@R.T)+shift
            mesh_checks[(name,i)]=dict(watertight=bool(pred.is_watertight),faces=len(pred.faces))
            del pred
        print('SAMPLED',case,frame,flush=True)
    pitch=D/GRID_DIVISOR
    allpoints=np.concatenate(list(clouds.values()),axis=0)
    lower=allpoints.min(0)-3*pitch;upper=allpoints.max(0)+3*pitch
    shape=tuple((np.ceil((upper-lower)/pitch).astype(int)+1).tolist())
    del allpoints
    assert np.prod(shape)<25_000_000,(case,shape)
    occupancy={key:shell(points,lower,pitch,shape) for key,points in clouds.items()}
    del clouds
    structure=generate_binary_structure(3,1)
    rows={name:[] for name,_,_ in methods}
    for i,frame in enumerate(frames):
        gt=occupancy[('GT',i)]
        for name,_,_ in methods:
            pred=occupancy[(name,i)]
            scores={}
            for radius in RADII:
                if radius:
                    a=binary_dilation(pred,structure=structure,iterations=radius)
                    b=binary_dilation(gt,structure=structure,iterations=radius)
                else:a,b=pred,gt
                scores[f'surface_voxel_IoU_r{radius}']=overlap(a,b)
            rows[name].append(dict(frame=frame,**scores))
        print('OVERLAP',case,frame,{name:round(rows[name][-1]['surface_voxel_IoU_r1'],3) for name in rows},flush=True)
    summary={name:{f'surface_voxel_IoU_r{radius}_mean':float(np.mean([v[f'surface_voxel_IoU_r{radius}'] for v in values]))
                   for radius in RADII} for name,values in rows.items()}
    axes={'origin':axis['joint_methods']['original'],
          'single_image_geometry':axis['joint_methods']['prior'],
          'extrema_travel':axis['joint_methods']['prior']}
    assert all(a['joint']['axis']==b['joint']['axis'] for a,b in zip(model.groups,travel))
    result=dict(case=case,frames=frames,source_model=str(modelroot),
        checkpoint_sha256=fit['source_checkpoint_sha256'],travel_fit_sha256=sha256(out/'travel_fit.json'),
        travel_motion_sha256=sha256(out/'travel_motion_groups.json'),
        alignment=A,axis_correspondence=axis['correspondence'],axis=axes,
        per_endpoint=rows,summary=summary,
        grid=dict(pitch=pitch,pitch_percent_GT_static_diagonal=100/GRID_DIVISOR,
                  shape=shape,origin=lower.tolist(),samples_per_mesh=SAMPLES,
                  dilation_voxels=list(RADII),connectivity=6,seed_base=500),
        mesh_checks={f'{name}:{i}':value for (name,i),value in mesh_checks.items()},
        definition='3D IoU of shared-grid voxelized area-uniform surface samples, with optional 6-neighbor shell dilation of 0,1,2 voxels. The primary r1 score represents about 1% GT static-body diagonal thickness. This is not solid occupancy IoU: source and GT meshes are open/non-watertight. Same grid and frozen original fixed-body similarity for all methods and both frames.',
        pose_policy='Origin uses original source geometry and source joint ranges; single_image_geometry uses the published single-image checkpoint with original VLM ranges; extrema_travel uses the identical checkpoint and axes with only fitted numeric range endpoints.',
        axis_policy='GT one-to-one geometry correspondence and fixed-body similarity recomputed from this original single-image model, then frozen across all three variants. Correct means type match, unoriented direction error <=5 degrees, and revolute pivot-line distance <=5% GT static-body diagonal. Prismatic axis position is unidentifiable and omitted.')
    write_json(out/'three_way_3d_axis.json',result)
    print('EVALUATED',case,{name:round(item['surface_voxel_IoU_r1_mean'],3) for name,item in summary.items()},
          {name:v['axis_correct_count'] for name,v in axes.items()},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--case',required=True);a=p.parse_args()
    evaluate(a.root.resolve(),a.case)
