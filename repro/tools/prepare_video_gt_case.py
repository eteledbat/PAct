"""Prepare an explicit GT-mask PAct input from existing calibrated VideoArtGS data.
No joint axis is read for rendering. Only stored part meshes and camera are used.
"""
import argparse,hashlib,json,os,pathlib,sys
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
import numpy as np,torch,trimesh,cv2
from PIL import Image
import nvdiffrast.torch as dr
ROOT=pathlib.Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from modules.pact.pipelines import PActPipeline
from modules.pact.renderers.mesh_renderer import intrinsics_to_projection
from modules.pact.datasets import ImageConditioned_dataset
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--frame',type=int,default=0);a=p.parse_args();scene=pathlib.Path('/root/autodl-tmp/VideoArtGS/data/videoartgs/sapien')/a.scene;out=ROOT/'repro/inputs'/f'video_gt_{a.scene}_frame{a.frame:06d}';out.mkdir(exist_ok=False);sample=out/'Sample';sample.mkdir()
config=json.loads((scene/'transforms.json').read_text());frame=config['frames'][a.frame];src=scene/frame['file_path'];image=Image.open(src).convert('RGBA');rgb=np.asarray(image);h,w=rgb.shape[:2]
parts=sorted((scene/'gt').glob('part_*.ply'),key=lambda p:int(p.stem.split('_')[-1]));vertices=[];faces=[];face_part=[];offset=0
for idx,path in enumerate(parts):
 mesh=trimesh.load(path,process=False,force='mesh');vertices.append(np.asarray(mesh.vertices));faces.append(np.asarray(mesh.faces)+offset);face_part.extend([idx+1]*len(mesh.faces));offset+=len(mesh.vertices)
vertices=torch.tensor(np.concatenate(vertices),device='cuda',dtype=torch.float32);faces=torch.tensor(np.concatenate(faces),device='cuda',dtype=torch.int32)
c2w=np.asarray(frame['transform_matrix']);c2w[:3,1:3]*=-1;ext=torch.tensor(np.linalg.inv(c2w),device='cuda',dtype=torch.float32);intr=torch.tensor([[config['focal_x']/w,0,config['cx']/w],[0,config['focal_y']/h,config['cy']/h],[0,0,1]],device='cuda',dtype=torch.float32);proj=intrinsics_to_projection(intr,.1,20.);hom=torch.cat([vertices,torch.ones(len(vertices),1,device='cuda')],1);clip=(hom@(proj@ext).T).unsqueeze(0)
with torch.no_grad():rast,_=dr.rasterize(dr.RasterizeCudaContext(),clip,faces,(h,w))
tri=rast[0,:,:,3].cpu().numpy().astype(np.int64);labels=np.zeros((h,w),np.int32);valid=tri>0;labels[valid]=np.asarray(face_part)[tri[valid]-1]
foreground=rgb[:,:,3]>=128;intersection=(foreground&valid).sum();union=(foreground|valid).sum();iou=float(intersection/union);assert iou>.98,f'GT/camera silhouette alignment failed: IoU={iou}'
name=f'{a.scene}_{a.frame:06d}';image.save(sample/(name+'_processed.png'));exr=sample/(name+'_mask.exr');assert cv2.imwrite(str(exr),np.repeat(labels[:,:,None],3,axis=2).astype(np.float32));preview=np.full((h,w,3),255,np.uint8);palette=np.array([[60,140,200],[240,130,40],[70,180,100],[200,80,160],[150,120,220],[130,170,40],[70,190,190],[150,90,60]],np.uint8)
for idx in range(1,len(parts)+1):preview[labels==idx]=palette[(idx-1)%len(palette)]
Image.fromarray(preview).save(out/'gt_part_mask_preview.png');pack=ImageConditioned_dataset(str(out))[0];unordered=pack['unordered_masks'].numpy();ordered=pack['masks'].numpy();mapping=[]
for mesh_idx in range(len(parts)):
 seen=unordered==mesh_idx+1;ids=np.unique(ordered[seen]);ids=ids[ids>0];assert len(ids)==1,f'Part {mesh_idx} not visible after37x37 sampling: {ids}'
 mapping.append({'gt_mesh_index':mesh_idx,'input_label':mesh_idx+1,'pact_node_id':int(ids[0]-1),'visible_pixels':int((labels==mesh_idx+1).sum())})
manifest={'scene_id':a.scene,'frame_index':a.frame,'source_image':str(src),'source_data_root':str(scene),'protocol':'Explicit GT-mask / oracle part segmentation. GT mesh only projected to2D mask; no GT axis/type/origin passed to PAct.','silhouette_iou':iou,'mask_labels':'0 background; mesh_index+1 foreground; original PAct bottom-up reorder recorded','part_count':int(pack['num_parts']),'part_correspondence':mapping,'files':[{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for path in [src,scene/'transforms.json',*parts]],'input_dir':str(out)};(out/'input_provenance.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
