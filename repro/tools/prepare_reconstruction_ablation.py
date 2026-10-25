"""Package calibrated RGB/oracle 2D labels without supplying 3D targets to fitting."""
import argparse
import datetime
import hashlib
import json
import pathlib
import subprocess
import sys

import nvdiffrast.torch as dr
import numpy as np
import torch
import trimesh
from PIL import Image

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from modules.pact.renderers.mesh_renderer import intrinsics_to_projection

PALETTE = np.array([[.20,.52,.77],[.94,.53,.18],[.25,.70,.42],[.65,.26,.67],[.72,.65,.20],[.20,.70,.70],[.8,.3,.3],[.4,.4,.4]], np.float32)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True); a = p.parse_args()
    root = pathlib.Path(a.root).resolve(); root.mkdir(exist_ok=False)
    config = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'git_sha': subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
              'cases': {}, 'resolution': 192, 'prealign_steps': 200, 'steps': 200,
              'collision_weight': .02, 'collision_sources_per_part': 1024,
              'geometry_grid': 4, 'geometry_bound_fraction_D': .03,
              'pivot_bound_fraction_D': .10, 'direction_bound_degrees': 5.,
              'train_frames': [0,25,50,75,100,125], 'validation_frames': [12,37,62,87,112,137],
              'experiments': [{'name':'8961_native','case':'8961','range':'native'},
                              {'name':'8961_sweep90','case':'8961','range':'signed90_assumption'},
                              {'name':'45194_native','case':'45194','range':'native'}],
              'protocol': 'Multi-view RGB + oracle 2D part-mask post-reconstruction refinement, not original single-view PAct training. Six train and six held-out cameras at the same observed state. No GT axis/mesh point loss; fixed final step; no seed/lambda search. Gaussian centers and mesh vertices share a bounded trilinear deformation field; opacity, scales and color frozen in A/B.'}
    for scene in ['8961','45194']:
        run = pathlib.Path((REPO/f'repro/current_gt_{scene}.txt').read_text().strip())
        provenance = json.loads((run/'input_provenance.json').read_text())
        src = pathlib.Path(provenance['source_data_root']); camera = json.loads((src/'transforms.json').read_text())
        obj = next(run.rglob('object.json')); nodes = json.loads(obj.read_text())['diffuse_tree']
        by_mesh = {v['gt_mesh_index']: v['pact_node_id'] for v in provenance['part_correspondence']}
        index_by_id = {n['id']: i for i,n in enumerate(nodes)}
        vertices, faces, face_labels, offset = [], [], [], 0
        paths = sorted((src/'gt').glob('part_*.ply'),key=lambda x:int(x.stem.split('_')[-1]))
        for mesh_index, path in enumerate(paths):
            mesh = trimesh.load(path,process=False,force='mesh');vertices.append(mesh.vertices);faces.append(mesh.faces+offset);offset+=len(mesh.vertices)
            face_labels.extend([index_by_id[by_mesh[mesh_index]]+1]*len(mesh.faces))
        v = torch.tensor(np.concatenate(vertices),device='cuda',dtype=torch.float32)
        f = torch.tensor(np.concatenate(faces),device='cuda',dtype=torch.int32)
        v = torch.cat([v,torch.ones(len(v),1,device='cuda')],1)
        ctx = dr.RasterizeCudaContext();resolution=config['resolution'];rgb,labels,exts,intrs,ious=[],[],[],[],[]
        for frame_index in config['train_frames']+config['validation_frames']:
            frame=camera['frames'][frame_index];assert frame['state']==0 and frame['time']==0.
            image=Image.open(src/frame['file_path']).convert('RGBA').resize((resolution,resolution),Image.Resampling.LANCZOS)
            rgba=np.asarray(image).astype(np.float32)/255.;alpha=rgba[:,:,3:4];rgb.append(rgba[:,:,:3]*alpha+1-alpha)
            c2w=np.array(frame['transform_matrix']);c2w[:3,1:3]*=-1;ext=np.linalg.inv(c2w).astype(np.float32)
            intr=np.array([[camera['focal_x']/camera['w'],0,camera['cx']/camera['w']],[0,camera['focal_y']/camera['h'],camera['cy']/camera['h']],[0,0,1]],np.float32)
            projection=intrinsics_to_projection(torch.tensor(intr,device='cuda'),.1,20.)
            clip=(v@(projection@torch.tensor(ext,device='cuda')).T)[None]
            with torch.no_grad():raster,_=dr.rasterize(ctx,clip,f,(resolution,resolution))
            tri=raster[0,:,:,3].cpu().numpy().astype(int);lab=np.zeros(tri.shape,np.uint8);valid=tri>0;lab[valid]=np.array(face_labels,np.uint8)[tri[valid]-1]
            foreground=alpha[:,:,0]>=.5;iou=float((valid&foreground).sum()/(valid|foreground).sum());assert iou>.95,(scene,frame_index,iou)
            labels.append(lab);exts.append(ext);intrs.append(intr);ious.append(iou)
        directory=root/'data'/scene;directory.mkdir(parents=True)
        np.savez_compressed(directory/'observations.npz',rgb=np.array(rgb),labels=np.array(labels),extrinsics=np.array(exts),intrinsics=np.array(intrs),palette=PALETTE[:len(nodes)])
        meta={'case':scene,'run':str(run),'train_indices':list(range(6)),'validation_indices':list(range(6,12)),
              'train_frames':config['train_frames'],'validation_frames':config['validation_frames'],
              'silhouette_IoU':ious,'part_correspondence':provenance['part_correspondence'],
              'source_data_root':str(src),'oracle_2D_labels':True,
              'GT_training_usage':'GT meshes projected into 2D IDs only; no GT 3D vertices/axis/type/range saved as fitting targets.',
              'observations_sha256':hashlib.sha256((directory/'observations.npz').read_bytes()).hexdigest()}
        (directory/'manifest.json').write_text(json.dumps(meta,indent=2));config['cases'][scene]=str(run)
        print('PACKAGED',scene,'IoU min',min(ious),flush=True)
    (root/'config.json').write_text(json.dumps(config,indent=2));(REPO/'repro/current_reconstruction_ablation.txt').write_text(str(root)+'\n')
    print(root,flush=True)


if __name__=='__main__':main()
