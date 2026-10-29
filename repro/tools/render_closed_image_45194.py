"""Export fitted articulated parts and render original/prior/optimized triplets."""
import argparse
import copy
import json
import pathlib
import time

import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from plyfile import PlyData,PlyElement
from scipy.spatial.transform import Rotation
import torch
import trimesh

from collision_refinement_common import moved,sha256,transforms
from fit_closed_image_45194 import CovGaussians,Model,REPO,tensor,write_json
from semantic_motion_prior_45194 import UNDO
from modules.pact.renderers.gaussian_render import GaussianRenderer
import utils3d


def export_parts(model,root):
    out=root/'optimized_asset';(out/'ply').mkdir(parents=True,exist_ok=True);(out/'gs').mkdir(exist_ok=True)
    nodes=copy.deepcopy(model.groups)
    source_mesh={p['node']['id']:p['mesh'] for p in model.asset['parts']}
    source_ply={}
    checks=[]
    with torch.no_grad():
        for i,(part,node) in enumerate(zip(model.parts,nodes)):
            prefix=node['membership_prefix'];nid=node['source_node'];gid=node['id']
            source=source_mesh[nid]
            vi=model.membership[prefix+'_vertices'];fi=model.membership[prefix+'_faces']
            vertices=model.deform(i,tensor(part['vertex_local'])).cpu().numpy()
            lookup=np.full(len(source.vertices),-1,np.int64);lookup[vi]=np.arange(len(vi))
            faces=lookup[source.faces[fi]]
            mesh=trimesh.Trimesh(vertices=vertices,faces=faces,process=False)
            if hasattr(source.visual,'vertex_colors') and len(source.visual.vertex_colors)==len(source.vertices):
                mesh.visual.vertex_colors=np.asarray(source.visual.vertex_colors)[vi]
            meshpath=out/'ply'/f'part_{gid}.ply';mesh.export(meshpath)
            if nid not in source_ply:source_ply[nid]=PlyData.read(model.manifest['sources'][str(nid)]['gs'])['vertex'].data
            records=source_ply[nid][model.membership[prefix+'_gaussians']].copy()
            xyz,cov=model.deform(i);xyz=xyz.cpu().numpy();cov=cov.cpu().numpy()
            serialized=xyz@UNDO
            for axis,key in enumerate(['x','y','z']):records[key]=serialized[:,axis]
            serialized_cov=UNDO.T[None]@cov@UNDO[None]
            # Automatic connected-component partitions can contain mesh-only
            # fragments with no Gaussian center. Preserve the mesh and a valid
            # empty Gaussian PLY rather than dropping the fragment.
            eig,axes=np.linalg.eigh(serialized_cov.astype(np.float64))
            if len(records):
                assert eig.min()>0,'Non-positive covariance'
                flip=np.linalg.det(axes)<0;axes[flip,:,0]*=-1
                quat=Rotation.from_matrix(axes).as_quat()[:,[3,0,1,2]]
            else:
                quat=np.empty((0,4),dtype=np.float64)
            for j in range(3):records[f'scale_{j}']=.5*np.log(eig[:,j])
            for j in range(4):records[f'rot_{j}']=quat[:,j]
            gspath=out/'gs'/f'part_{gid}.ply'
            PlyData([PlyElement.describe(records,'vertex')],text=False).write(gspath)
            roundtrip=(axes*eig[:,None,:])@axes.transpose(0,2,1)
            error=float(np.max(np.linalg.norm(roundtrip-serialized_cov,axis=(1,2))/np.linalg.norm(serialized_cov,axis=(1,2)).clip(1e-20))) if len(records) else 0.
            checks.append({'group_id':gid,'vertices':len(vertices),'faces':len(faces),'gaussians':len(records),
                           'minimum_covariance_eigenvalue':float(eig.min()) if len(records) else None,'covariance_factorization_relative_error':error,
                           'mesh_sha256':sha256(meshpath),'gaussian_sha256':sha256(gspath)})
            node['plys']=[f'ply/part_{gid}.ply'];node['gs']=[f'gs/part_{gid}.ply']
            node['children']=[g['id'] for g in nodes if g['parent']==gid]
            node['aabb']={'min':mesh.bounds[0].tolist(),'max':mesh.bounds[1].tolist(),
                          'center':mesh.bounds.mean(0).tolist(),'size':np.ptp(vertices,axis=0).tolist()}
    write_json(out/'object.json',{'diffuse_tree':nodes,'source_prior':model.config['prior_root'],
               'optimization':'Single closed RGB frame, structure-preserving part deformation',
               'rest_pose':'Original generation reference pose; observed closed fitting pose stored separately'})
    write_json(root/'asset_export_checks.json',{'parts':checks,'new_asset':str(out),
               'all_original_source_files_unchanged':all(sha256(x['path'])==x['sha256'] for x in model.manifest['geometry_files'])})
    return checks


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
    root=pathlib.Path(a.root).resolve()
    config=json.loads((root/'config.json').read_text());fit=json.loads((root/'fit_results.json').read_text())
    prior=pathlib.Path(config['prior_root']);manifest=json.loads((prior/'motion_groups.json').read_text())
    assert sha256(prior/'motion_groups.json')==config['prior_manifest_sha256']
    assert sha256(root/'checkpoint.pt')==fit['checkpoint_sha256']
    assert sha256(root/'optimized_parts.npz')==fit['optimized_parts_sha256']
    torch.set_num_threads(4)
    model=Model(config,manifest,root)
    model.load_state_dict(torch.load(root/'checkpoint.pt',map_location='cuda',weights_only=False)['model'])
    model.eval();model.requires_grad_(False)
    export_checks=export_parts(model,root)
    original=model.geometry(original=True);optimized=model.geometry()
    originals=model.asset['nodes'];groups=model.groups
    methods=['original','motion_prior','image_optimized']
    titles={'original':'Original PAct','motion_prior':'Independent doors / drawers','image_optimized':'Closed-image geometry fit'}
    renderer=GaussianRenderer({'resolution':512,'near':.05,'far':20.,'bg_color':(1.,1.,1.)})
    renderer.pipe.compute_cov3D_python=True
    clouds=[]
    for method in methods:
        tree=originals if method=='original' else groups
        geom=optimized if method=='image_optimized' else original
        for t in np.linspace(0,1,11):
            world=transforms(tree,t)
            for part,(xyz,_) in zip(model.parts,geom):
                nid=part['group']['source_node'] if method=='original' else part['group']['id']
                clouds.append(moved(xyz[::197].cpu().numpy(),world[nid]))
    cloud=np.concatenate(clouds);center=(cloud.min(0)+cloud.max(0))/2
    radius=1.45*np.linalg.norm(np.ptp(cloud,axis=0))
    intr=utils3d.torch.intrinsics_from_fov_xy(tensor(np.deg2rad(42.)),tensor(np.deg2rad(42.)))
    cameras=[]
    for yaw in [.38,-.72]:
        eye=center+radius*np.array([np.sin(yaw)*np.cos(.22),np.sin(.22),np.cos(yaw)*np.cos(.22)])
        cameras.append(utils3d.torch.extrinsics_look_at(tensor(eye),tensor(center),tensor([0,1,0])))
    colors=torch.cat([p['colors'] for p in model.parts]).contiguous()
    alpha=torch.cat([p['alpha'] for p in model.parts]).contiguous()
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)

    @torch.no_grad()
    def frame(method,t,camera=0,closed_sweep=False,observation=False):
        tree=copy.deepcopy(originals if method=='original' else groups)
        if closed_sweep:
            for n in tree:
                start=fit['closed_observation_pose'].get(str(n['id']),0.)
                n['joint']['range'][0]=start
        world=transforms(tree,float(t))
        geometry=optimized if method=='image_optimized' else original
        xyzs,covs=[],[]
        for part,(xyz,cov) in zip(model.parts,geometry):
            nid=part['group']['source_node'] if method=='original' else part['group']['id']
            T=world[nid];R=tensor(T[:3,:3]);tr=tensor(T[:3,3])
            xyzs.append(xyz@R.T+tr);covs.append(R[None]@cov@R.T[None])
        xyz=torch.cat(xyzs);cov=torch.cat(covs)
        if observation:
            # Original's one hinge is assigned the left closed angle. It cannot
            # independently satisfy both leaves' opposite closing angles.
            closedtree=copy.deepcopy(tree)
            for n in closedtree:
                angle=fit['closed_observation_pose'].get(str(n['id']),0.)
                if method=='original' and n['id']==0:angle=fit['closed_observation_pose']['11']
                n['joint']['range']=[angle,angle]
            tr=transforms(closedtree,1.)
            xyzs,covs=[],[]
            for part,(x,c) in zip(model.parts,geometry):
                nid=part['group']['source_node'] if method=='original' else part['group']['id']
                T=tr[nid];A=model.R@tensor(T[:3,:3]);scale=model.log_global_scale.exp()
                translation=torch.stack([model.screen_translation[0]*0,model.screen_translation[0],model.screen_translation[1]])
                xyzs.append(scale*(x@A.T+tensor(T[:3,3])@model.R.T)+translation)
                covs.append(scale.square()*(A[None]@c@A.T[None]))
            xyz,cov=torch.cat(xyzs),torch.cat(covs)
        g=CovGaussians(xyz.contiguous(),cov.contiguous(),alpha)
        rgb=renderer.render(g,model.extrinsics if observation else cameras[camera],model.intrinsics if observation else intr,
                            colors_overwrite=colors)['color']
        pic=Image.fromarray((rgb.clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype('uint8'))
        d=ImageDraw.Draw(pic);d.rectangle((0,0,511,53),fill='white')
        d.text((10,7),titles[method],fill='black',font=font)
        d.text((10,29),'Closed reference pose' if observation else f'45194 | motion fraction {t:.2f}',fill='#555555',font=font)
        return np.asarray(pic)
    reference=Image.open(root/'closed_reference.png').convert('RGBA').resize((512,512),Image.Resampling.LANCZOS)
    bg=Image.new('RGBA',reference.size,'white');bg.alpha_composite(reference)
    reference=bg.convert('RGB');draw=ImageDraw.Draw(reference);draw.rectangle((0,0,511,53),fill='white')
    draw.text((10,7),'Closed RGB target (frame 250)',fill='black',font=font)
    still=np.concatenate([np.asarray(reference)]+[frame(m,0.,observation=True) for m in methods],axis=1)
    Image.fromarray(still).save(root/'closed_target_and_three_versions.png')
    started=time.monotonic()
    filenames={m:root/f'45194_{m}.mp4' for m in methods}
    comparison=root/'45194_three_versions_comparison.mp4'
    writers={m:imageio.get_writer(path,fps=20,codec='libx264',quality=8) for m,path in filenames.items()}
    cmp=imageio.get_writer(comparison,fps=20,codec='libx264',quality=8)
    sequence=np.r_[np.zeros(10),np.linspace(0,1,41),np.ones(10),np.linspace(1,0,41),np.zeros(10)]
    try:
        for i,t in enumerate(sequence):
            panels={m:[frame(m,float(t),c) for c in [0,1]] for m in methods}
            for m in methods:writers[m].append_data(np.concatenate(panels[m],axis=1))
            panel=np.concatenate([np.concatenate([panels[m][c] for m in methods],axis=1) for c in [0,1]],axis=0)
            cmp.append_data(panel)
            if i in [10,30,50]:Image.fromarray(panel).save(root/f'three_versions_t{t:.2f}.png')
            if i%30==0:print('RENDER triplet',i,flush=True)
    finally:
        cmp.close()
        for writer in writers.values():writer.close()
    closed_video=root/'45194_optimized_closed_to_open.mp4'
    with imageio.get_writer(closed_video,fps=20,codec='libx264',quality=8) as writer:
        for t in sequence:
            writer.append_data(np.concatenate([frame('image_optimized',float(t),c,closed_sweep=True) for c in [0,1]],axis=1))
    video_checks=[]
    for path in [*filenames.values(),comparison,closed_video]:
        with imageio.get_reader(path) as reader:
            count=reader.count_frames();meta=reader.get_meta_data();reader.get_data(0);reader.get_data(count-1)
        video_checks.append({'file':path.name,'frames':count,'fps':meta['fps'],'size':list(meta['size']),'sha256':sha256(path)})
    # Verify deformation is baked once, while animation remains a rigid map.
    max_rigid_error=0.
    for part,(xyz,_) in zip(model.parts,optimized):
        sample=xyz[::199].cpu().numpy().astype(np.float64)
        d0=np.linalg.norm(sample-sample[0],axis=1)
        for t in np.linspace(0,1,41):
            posed=moved(sample,transforms(groups,t)[part['group']['id']])
            max_rigid_error=max(max_rigid_error,float(np.max(abs(np.linalg.norm(posed-posed[0],axis=1)-d0))))
    assert max_rigid_error<1e-12
    assert all(sha256(x['path'])==x['sha256'] for x in manifest['geometry_files'])
    assert sha256(prior/'motion_groups.json')==config['prior_manifest_sha256']
    write_json(root/'render_verification.json',{'videos':video_checks,'shared_camera_center':center.tolist(),
               'shared_camera_radius':float(radius),'same_motion_intervals_prior_vs_optimized':True,
               'comparison_motion':'Original reference pose to original predicted endpoint, then return. These are not claimed fully closed/open physical limits.',
               'extra_closed_sweep':'Separate optimized closed-to-open video begins at the image-fitting closed pose and ends at the same original predicted endpoint.',
               'motion_during_video':'Fixed optimized geometry; only rigid joint transforms vary',
               'max_rigid_pair_distance_error':max_rigid_error,'original_asset_files_unchanged':True,
               'prior_motion_metadata_unchanged':True,'exported_parts':export_checks,
               'elapsed_render_seconds':time.monotonic()-started})
    print('VIDEOS COMPLETE',root,flush=True)


if __name__=='__main__':main()
