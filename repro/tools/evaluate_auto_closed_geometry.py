"""Post-fit comparison on the same closed image and GT mesh (never training)."""
import argparse
import copy
import pathlib
import sys

import numpy as np
import torch
from scipy.ndimage import binary_dilation,generate_binary_structure

sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from auto_part_semantics import write_json
from collision_refinement_common import surface
from evaluate_closed_extrema_geometry import shell,overlap,SAMPLES,GRID_DIVISOR
from evaluate_freeart_partnet_case import gt_parts
from evaluate_multicase_closed import joint_scores
from evaluate_observed_drawer_pose import build_mesh
from evaluate_two_extrema import gt_mesh
from fit_observed_drawer_pose import load
from fit_single_origin_extrema_travel import source
from multicase_closed_common import Model
from fit_auto_closed_geometry import read


def fixed_pose(groups,pose):
    tree=copy.deepcopy(groups)
    for group in tree:
        q=pose.get(str(group['id']),pose.get(group['id'],0.))
        group['joint']['range']=[float(q),float(q)]
    return tree


def evaluate(batch,case):
    root=batch/case
    fit=read(root/'auto_closed_fit.json')
    _,published=source(case)
    manual=batch.parent/'closed_multicase_20261028_v1'/case
    baseline=batch.parent/'closed_travel_reference_geometry_v011_20261031_v2'/case
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    native=load(published)
    new=load(root,checkpoint='auto_closed_geometry.pt')
    older=load(baseline,checkpoint='observed_pose_geometry.pt')
    human=Model(read(manual/'config.json'),read(manual/'motion_groups.json'),manual)
    human.load_state_dict(torch.load(manual/'checkpoint.pt',map_location='cuda',weights_only=False)['model'])
    for model in [native,new,older,human]:model.requires_grad_(False)
    old_axis=joint_scores(native)
    new_axis=joint_scores(new)
    A=old_axis['alignment'];R=np.asarray(A['R_export_to_GT']);scale=A['scale'];shift=np.asarray(A['translation']);D=A['GT_static_diagonal']
    frame=int(read(root/'prepared.json')['closed_frame'])
    scene=pathlib.Path(native.config['source_scene'])
    if case=='20411':
        import trimesh
        truth=trimesh.util.concatenate(gt_parts(scene)[0][frame])
    else:truth=gt_mesh(scene,case,frame)
    manual_pose={str(g['id']):g['observed_q'] for g in human.groups}
    older_pose=read(baseline/'fit_seal.json')['observed_pose']
    models={
        'origin':(native,native.asset['nodes'],False,0. if case=='20411' else 1.),
        'previous_auto':(older,fixed_pose(older.groups,older_pose),True,0.),
        'new_auto':(new,fixed_pose(new.groups,fit['pose']),True,0.),
        'manual':(human,fixed_pose(human.groups,manual_pose),True,0.),
    }
    pitch=D/GRID_DIVISOR
    clouds={'GT':surface(truth,SAMPLES,800)}
    for name,(model,tree,opt,progress) in models.items():
        mesh=build_mesh(model,optimized=opt,tree=tree,progress=progress)
        clouds[name]=scale*(surface(mesh,SAMPLES,800)@R.T)+shift
        print('SAMPLED',case,name,flush=True)
    all_points=np.concatenate(list(clouds.values()))
    lower=all_points.min(0)-3*pitch
    upper=all_points.max(0)+3*pitch
    shape=tuple((np.ceil((upper-lower)/pitch).astype(int)+1).tolist())
    assert np.prod(shape)<25_000_000,(case,shape)
    occupancies={name:shell(points,lower,pitch,shape) for name,points in clouds.items()}
    structure=generate_binary_structure(3,1)
    result={}
    for name in models:
        result[name]={}
        for radius in [0,1,2]:
            a=binary_dilation(occupancies[name],structure=structure,iterations=radius) if radius else occupancies[name]
            b=binary_dilation(occupancies['GT'],structure=structure,iterations=radius) if radius else occupancies['GT']
            result[name][f'surface_voxel_IoU_r{radius}']=overlap(a,b)
    axes={'origin':old_axis['joint_methods']['original'],
          'previous_auto':old_axis['joint_methods']['prior'],
          'new_auto':new_axis['joint_methods']['optimized'],
          'manual':read(manual/'evaluation.json')['joint_methods']['optimized']}
    write_json(root/'auto_closed_evaluation.json',dict(case=case,closed_frame=frame,
        surface_voxel_3d_IoU=result,axis=axes,alignment=A,
        same_image_2d=dict(previous_auto=read(baseline/'fit_seal.json')['geometry_fit']['image'],
                           new_auto=fit['after']['image'],
                           manual=read(manual/'fit_results.json')['image_metrics']['optimized']),
        definition='80k area-uniform surface points per mesh, shared 3D grid at GT static diagonal/96, r1 one-voxel surface dilation. Open meshes: not solid volume IoU. GT geometry and joints read only after fitting.'))
    print('EVALUATED',case,{k:round(v['surface_voxel_IoU_r1'],3) for k,v in result.items()},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=pathlib.Path,required=True)
    p.add_argument('--case',required=True)
    a=p.parse_args()
    evaluate(a.root.resolve(),a.case)
