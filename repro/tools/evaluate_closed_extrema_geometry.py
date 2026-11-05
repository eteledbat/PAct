"""Ten-case 3D surface voxel IoU, closed-image 2D IoU and frozen-axis audit."""
import argparse,copy,pathlib,sys
import numpy as np,torch
from scipy.ndimage import binary_dilation,generate_binary_structure
REPO=pathlib.Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO/'repro/tools'))
from auto_part_semantics import sha256,write_json
from fit_single_origin_extrema_travel import source,EXTREMES
from fit_observed_drawer_pose import load,read
from evaluate_multicase_closed import joint_scores
from evaluate_observed_drawer_pose import build_mesh
from evaluate_freeart_partnet_case import gt_parts
from evaluate_two_extrema import gt_mesh
from collision_refinement_common import surface

SAMPLES=80000;GRID_DIVISOR=96;RADII=(0,1,2)
def shell(points,origin,pitch,shape):
    idx=np.floor((points-origin)/pitch).astype(np.int64);idx=idx[np.all((idx>=0)&(idx<np.array(shape)),axis=1)]
    grid=np.zeros(shape,dtype=bool);grid[idx[:,0],idx[:,1],idx[:,2]]=True;return grid
def overlap(a,b):
    n=np.count_nonzero(a|b);return float(np.count_nonzero(a&b)/n) if n else None

def evaluate(batch,case):
    root=batch/case;seal=read(root/'fit_seal.json');_,oldroot=source(case)
    assert sha256(root/'motion_groups.json')==sha256(oldroot/'motion_groups.json')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    old=load(oldroot);new=load(root,checkpoint='observed_pose_geometry.pt');old.requires_grad_(False);new.requires_grad_(False)
    with torch.no_grad():old_axis=joint_scores(old);new_axis=joint_scores(new)
    A=old_axis['alignment'];R=np.asarray(A['R_export_to_GT']);scale=A['scale'];shift=np.asarray(A['translation']);D=A['GT_static_diagonal']
    manifest=read(EXTREMES/case/'manifest.json');frames=[int(x['frame']) for x in manifest['endpoints']]
    scene=pathlib.Path(old.config['source_scene'])
    if case=='20411':
        import trimesh
        states,_=gt_parts(scene);truths=[trimesh.util.concatenate(states[f]) for f in frames]
    else:truths=[gt_mesh(scene,case,f) for f in frames]
    methods={'origin':(old,old.asset['nodes'],False),'axis_only':(old,old.groups,False),
             'closed_geometry':(new,new.groups,True)}
    clouds={};checks={}
    for i,(frame,truth) in enumerate(zip(frames,truths)):
        clouds['GT',i]=surface(truth,SAMPLES,500+i);checks['GT',i]=bool(truth.is_watertight)
        for name,(model,tree,opt) in methods.items():
            mesh=build_mesh(model,optimized=opt,tree=tree,progress=float(i))
            clouds[name,i]=scale*(surface(mesh,SAMPLES,500+i)@R.T)+shift
            checks[name,i]=bool(mesh.is_watertight)
        print('SAMPLED',case,frame,flush=True)
    pitch=D/GRID_DIVISOR;allp=np.concatenate(list(clouds.values()));lower=allp.min(0)-3*pitch;upper=allp.max(0)+3*pitch
    shape=tuple((np.ceil((upper-lower)/pitch).astype(int)+1).tolist());del allp
    assert np.prod(shape)<25_000_000,(case,shape)
    occ={key:shell(pts,lower,pitch,shape) for key,pts in clouds.items()};del clouds
    structure=generate_binary_structure(3,1);rows={k:[] for k in methods}
    for i,frame in enumerate(frames):
        gt=occ['GT',i]
        for name in methods:
            pred=occ[name,i];scores={}
            for radius in RADII:
                a=binary_dilation(pred,structure=structure,iterations=radius) if radius else pred
                b=binary_dilation(gt,structure=structure,iterations=radius) if radius else gt
                scores[f'surface_voxel_IoU_r{radius}']=overlap(a,b)
            rows[name].append(dict(frame=frame,**scores))
        print('OVERLAP',case,frame,{n:round(rows[n][-1]['surface_voxel_IoU_r1'],3) for n in methods},flush=True)
    summary={name:{f'surface_voxel_IoU_r{radius}_mean':float(np.mean([v[f'surface_voxel_IoU_r{radius}'] for v in vals])) for radius in RADII}
             for name,vals in rows.items()}
    # Also evaluate the closed observation at its independently image-estimated
    # pose; native-range playback above can miss this pose entirely.
    closed_frame=int(seal['closed_frame']);native_closed_progress=0. if case=='20411' else 1.
    closed_truth=(__import__('trimesh').util.concatenate(gt_parts(scene)[0][closed_frame]) if case=='20411' else gt_mesh(scene,case,closed_frame))
    pose=seal['observed_pose']
    fixed_tree=copy.deepcopy(old.groups)
    for group in fixed_tree:
        if str(group['id']) in pose:group['joint']['range']=[float(pose[str(group['id'])])]*2
    closed_clouds={'GT':surface(closed_truth,SAMPLES,800),
                   'origin':scale*(surface(build_mesh(old,optimized=False,tree=old.asset['nodes'],progress=native_closed_progress),SAMPLES,800)@R.T)+shift,
                   'axis_only':scale*(surface(build_mesh(old,optimized=False,tree=fixed_tree,progress=0.),SAMPLES,800)@R.T)+shift,
                   'closed_geometry':scale*(surface(build_mesh(new,optimized=True,tree=fixed_tree,progress=0.),SAMPLES,800)@R.T)+shift}
    cp=np.concatenate(list(closed_clouds.values()));clower=cp.min(0)-3*pitch;cupper=cp.max(0)+3*pitch
    cshape=tuple((np.ceil((cupper-clower)/pitch).astype(int)+1).tolist());del cp
    assert np.prod(cshape)<25_000_000,(case,cshape)
    co={key:shell(pts,clower,pitch,cshape) for key,pts in closed_clouds.items()};del closed_clouds
    calibrated={}
    for name in methods:
        calibrated[name]={}
        for radius in RADII:
            x=binary_dilation(co[name],structure=structure,iterations=radius) if radius else co[name]
            y=binary_dilation(co['GT'],structure=structure,iterations=radius) if radius else co['GT']
            calibrated[name][f'surface_voxel_IoU_r{radius}']=overlap(x,y)
    print('CLOSED POSE',case,{n:round(v['surface_voxel_IoU_r1'],3) for n,v in calibrated.items()},flush=True)
    axes={'origin':old_axis['joint_methods']['original'],'axis_only':old_axis['joint_methods']['prior'],
          'closed_geometry':new_axis['joint_methods']['optimized']}
    assert axes['axis_only']['axis_correct_count']==axes['closed_geometry']['axis_correct_count']
    write_json(root/'metrics.json',dict(case=case,frames=frames,closed_reference_frame=closed_frame,
        surface_voxel_3d_IoU=summary,per_endpoint=rows,calibrated_closed_3d_IoU=calibrated,
        closed_2d_silhouette_IoU={'origin_at_native_endpoint':seal['origin_at_native_endpoint']['image']['IoU'],
                                 'axis_at_checked_pose':seal['position_check']['image']['IoU'],
                                 'closed_geometry_at_checked_pose':seal['geometry_fit']['image']['IoU']},
        closed_part_mean_IoU={'axis_at_checked_pose':seal['position_check']['part_mean_IoU'],
                              'closed_geometry_at_checked_pose':seal['geometry_fit']['part_mean_IoU']},
        axis=axes,axis_correspondence=old_axis['correspondence'],alignment=A,
        grid=dict(pitch=pitch,shape=shape,samples_per_mesh=SAMPLES,dilation_voxels=RADII),mesh_watertight={f'{name}:{i}':value for (name,i),value in checks.items()},
        definition='Area-uniform sampled surface voxel IoU, shared grid, r1 about 1% GT fixed-body diagonal. Meshes are open, so this is not solid volume IoU. Same original fixed-body alignment and reference endpoint frames across methods. Axis correct: GT type match, unsigned direction <=5 degrees, revolute pivot line <=5% GT static-body diagonal. Prismatic position not identifiable.',
        geometry_fit_note='Only closed extreme RGB, projected instance mask and known camera fit observed pose and geometry; GT meshes and joints are evaluated only after fitting. The published axis tree and native ranges are unchanged.'))
    print('EVALUATED',case,{n:round(v['surface_voxel_IoU_r1_mean'],3) for n,v in summary.items()},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--case',required=True);a=p.parse_args();evaluate(a.root.resolve(),a.case)
