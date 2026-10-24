"""Prepare an oracle 2D part mask at the published FreeArt3D rendering camera.

GT endpoint motion partitions disconnected triangles; no joint labels are read.
Only the selected RGB and visible 2D IDs are passed to PAct. GT mesh copies are
kept outside the input folder for later evaluation, never for reconstruction.
"""
import argparse, hashlib, json, os, pathlib, sys
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import cv2
import numpy as np
import torch
import trimesh
from PIL import Image
import nvdiffrast.torch as dr

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from modules.pact.datasets import ImageConditioned_dataset
from modules.pact.renderers.mesh_renderer import intrinsics_to_projection

p = argparse.ArgumentParser()
p.add_argument('--scene', required=True)
p.add_argument('--frame', type=int, default=5)
a = p.parse_args()
source = pathlib.Path('/root/autodl-tmp/FreeArt3D/refinement/data/inputs') / a.scene
out = ROOT / 'repro/inputs' / f'freeart_partnet_{a.scene}_frame{a.frame:02d}'
out.mkdir(exist_ok=False)
(out / 'Sample').mkdir()
paths = [source / 'gt_mesh' / f'{i:02d}.glb' for i in range(6)]
full = [trimesh.load(x, process=False, force='mesh') for x in paths]
base, final = full[0], full[-1]
assert all(np.array_equal(m.faces, base.faces) and m.vertices.shape == base.vertices.shape for m in full)
displacement = final.vertices - base.vertices
mag = np.linalg.norm(displacement, axis=1)
assert mag.max() > 1e-5
moving = mag > .1 * mag.max()
face_counts = moving[base.faces].sum(1)
assert not ((face_counts > 0) & (face_counts < 3)).any()
assert np.max(np.linalg.norm(displacement[moving] - np.median(displacement[moving], axis=0), axis=1)) < 1e-4
# Blender imports glTF Y-up into Z-up. Reproduce pipelines/render.py camera,
# including its bbox based on the LAST state, without fitting any camera.
R_glb_to_blender = np.array([[1,0,0],[0,0,-1],[0,1,0]], float)
v_last = final.vertices @ R_glb_to_blender.T
center = (v_last.min(0) + v_last.max(0)) / 2
distance = 2.1 * np.ptp(v_last, axis=0).max()
eye = center + distance * np.array([np.sin(np.pi/8), -np.cos(np.pi/8), np.sin(np.pi/4)])
forward = center-eye
forward /= np.linalg.norm(forward)
right = np.cross(forward, [0.,0.,1.]); right /= np.linalg.norm(right)
down = np.cross(forward, right)
rotation = np.stack([right, down, forward])
extrinsic = np.eye(4); extrinsic[:3,:3] = rotation; extrinsic[:3,3] = -rotation @ eye
focal = 1/(2*np.tan(np.deg2rad(45)/2))
intr = np.array([[focal,0,.5],[0,focal,.5],[0,0,1]], float)
rgb_path = source / f'{a.frame:02d}_pure.png'
rgb = Image.open(rgb_path).convert('RGBA'); w,h = rgb.size
vertices = torch.tensor(full[a.frame].vertices @ R_glb_to_blender.T, dtype=torch.float32, device='cuda')
faces = torch.tensor(full[a.frame].faces, dtype=torch.int32, device='cuda')
proj = intrinsics_to_projection(torch.tensor(intr,device='cuda',dtype=torch.float32), .1, 20.)
ext = torch.tensor(extrinsic,device='cuda',dtype=torch.float32)
clip = (torch.cat([vertices,torch.ones(len(vertices),1,device='cuda')],1) @ (proj @ ext).T)[None]
with torch.no_grad(): raster,_ = dr.rasterize(dr.RasterizeCudaContext(),clip,faces,(h,w))
tri = raster[0,:,:,3].cpu().numpy().astype(int)
visible = tri > 0
labels = np.zeros((h,w), np.int32)
face_labels = np.where(face_counts == 3, 2, 1)
labels[visible] = face_labels[tri[visible]-1]
foreground = np.asarray(rgb)[:,:,3] >= 128
iou = float((visible & foreground).sum() / (visible | foreground).sum())
(out / 'camera_check.json').write_text(json.dumps({'silhouette_iou':iou,'R_glb_to_blender':R_glb_to_blender.tolist(),'extrinsic':extrinsic.tolist(),'intrinsic_normalized':intr.tolist()},indent=2))
print('CAMERA SILHOUETTE IOU', iou, flush=True)
assert iou > .98, 'Published camera does not agree with actual RGB; do not infer from a mismatched mask'
name = f'{a.scene}_{a.frame:02d}'
rgb.save(out / 'Sample' / (name+'_processed.png'))
assert cv2.imwrite(str(out/'Sample'/(name+'_mask.exr')),np.repeat(labels[:,:,None],3,axis=2).astype(np.float32))
preview = np.full((h,w,3),255,np.uint8)
preview[labels==1] = [140,140,150]; preview[labels==2] = [240,130,40]
Image.fromarray(preview).save(out/'gt_part_mask_preview.png')
pack = ImageConditioned_dataset(str(out))[0]
unordered,ordered = pack['unordered_masks'].numpy(),pack['masks'].numpy()
mapping = []
for idx in [0,1]:
    ids = np.unique(ordered[unordered==idx+1]); ids = ids[ids>0]
    assert len(ids)==1, f'Part {idx} is not visible in the original 37x37 reader'
    mapping.append({'gt_mesh_index':idx,'input_label':idx+1,'pact_node_id':int(ids[0]-1),'visible_pixels':int((labels==idx+1).sum())})
files = [rgb_path,*paths,pathlib.Path('/root/autodl-tmp/FreeArt3D/pipelines/render.py')]
manifest = {'scene_id':a.scene,'frame_index':a.frame,'source_image':str(rgb_path),'source_data_root':str(source),
    'protocol':'Oracle visible 2D part segmentation from actual GT mesh, with published FreeArt3D camera; RGB is the existing 05_pure image. GT type, axis, origin and range are excluded from PAct input.',
    'selection':'Fixed last existing frame (05) chosen before inference to expose the drawer; single seed42, no GT metric sample selection.',
    'silhouette_iou':iou,'part_count':int(pack['num_parts']),'part_correspondence':mapping,'input_dir':str(out),
    'GT_partition':'Same-topology endpoint motion separates fixed/moving triangles; this oracle information enters only the 2D mask.',
    'files':[{'path':str(x),'sha256':hashlib.sha256(x.read_bytes()).hexdigest()} for x in files]}
(out/'input_provenance.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2))
