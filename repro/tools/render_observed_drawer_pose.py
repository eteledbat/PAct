"""GT | Origin | VLM axes | VLM pose+geometry | human pose+geometry.

Canonical full-range playback preserves every model's prior numeric joint range.
Observed q is used in single-input fitting, and shown separately in input stills.
"""
import argparse,pathlib,copy
import imageio.v2 as imageio,numpy as np,torch
from PIL import Image,ImageDraw,ImageFont
from fit_observed_drawer_pose import load,read,REPO
from collision_refinement_common import moved,transforms,sha256
from multicase_closed_common import CovGaussians,tensor,write_json
from add_gt_multicase_videos import GT,MeshExtractResult,MeshRenderer,SEQ
from modules.pact.renderers.gaussian_render import GaussianRenderer
import utils3d

FONT=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
TITLES=['GT | reference mesh','Origin | original PAct','VLM | axes only','VLM | pose + geometry','Human | pose + geometry']

def render(batch,case):
 root=batch/case;torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;v=load(root/'vlm',checkpoint='observed_pose_geometry.pt');h=load(root/'human',checkpoint='observed_pose_geometry.pt');v.requires_grad_(False);h.requires_grad_(False);gt=GT(root);evaluation=read(root/'evaluation.json')
 # For20411 actual GT closed00->open05; all predicted full native intervals0->1.
 if case!='20411':
  from render_auto_semantic_four_versions import gt_at_source
  gt,_=gt_at_source(root)
 methods=[dict(title=TITLES[1],model=v,tree=v.asset['nodes'],geometry=v.geometry(original=True),raw=True),dict(title=TITLES[2],model=v,tree=v.groups,geometry=v.geometry(original=True),raw=False),dict(title=TITLES[3],model=v,tree=v.groups,geometry=v.geometry(),raw=False),dict(title=TITLES[4],model=h,tree=h.groups,geometry=h.geometry(),raw=False)]
 clouds=[]
 for method in methods:
  for t in [0.,.5,1.]:
   tr=transforms(method['tree'],t)
   for p,(x,_) in zip(method['model'].parts,method['geometry']):clouds.append(moved(x[::max(1,len(x)//1200)].cpu().numpy(),tr[p['group']['source_node'] if method['raw'] else p['group']['id']]))
 for t in [0.,.5,1.]:clouds.append(gt.vertices(t)[::100])
 cloud=np.concatenate(clouds);center=(cloud.min(0)+cloud.max(0))/2;radius=1.6*np.linalg.norm(np.ptp(cloud,axis=0));intr=utils3d.torch.intrinsics_from_fov_xy(tensor(np.deg2rad(43.)),tensor(np.deg2rad(43.)));cameras=[]
 for yaw in [.35,-.7]:
  eye=center+radius*np.array([np.sin(yaw)*np.cos(.22),np.sin(.22),np.cos(yaw)*np.cos(.22)]);cameras.append(utils3d.torch.extrinsics_look_at(tensor(eye),tensor(center),tensor([0,1,0])))
 gr=GaussianRenderer(dict(resolution=384,near=.05,far=20.,bg_color=(1.,1.,1.)));gr.pipe.compute_cov3D_python=True;mr=MeshRenderer(dict(resolution=384,near=.05,far=20.,ssaa=2));cache={}
 def header(rgb,title,t,detail):
  out=Image.fromarray(rgb);d=ImageDraw.Draw(out);d.rectangle((0,0,383,47),fill='white');d.text((7,4),title,font=FONT,fill='black');d.text((7,26),detail+f' | {t:.2f}',font=FONT,fill='#555555');return np.array(out)
 @torch.inference_mode()
 def gt_frame(t,view):
  key=(float(t),view)
  if key not in cache:
   verts=tensor(gt.vertices(t));mesh=MeshExtractResult(verts,gt.faces,gt.attrs);o=mr.render(mesh,cameras[view],intr,return_types=['color','mask','normal']);rgb=(o['color']*(.65+.35*abs(2*o['normal'][2:3]-1))+1-o['mask'].unsqueeze(0)).clamp(0,1);cache[key]=header((rgb.permute(1,2,0).cpu().numpy()*255).astype('uint8'),TITLES[0],t,'Actual closed00 -> open05' if case=='20411' else 'Actual source -> GT endpoint')
  return cache[key]
 @torch.inference_mode()
 def prediction(method,t,view):
  m=method['model'];tr=transforms(method['tree'],t);xs=[];cs=[]
  for p,(x,c) in zip(m.parts,method['geometry']):
   T=tr[p['group']['source_node'] if method['raw'] else p['group']['id']];R=tensor(T[:3,:3]);xs.append(x@R.T+tensor(T[:3,3]));cs.append(R[None]@c@R.T[None])
  gs=CovGaussians(torch.cat(xs).contiguous(),torch.cat(cs).contiguous(),torch.cat([p['alpha'] for p in m.parts]).contiguous());rgb=gr.render(gs,cameras[view],intr,colors_overwrite=torch.cat([p['colors'] for p in m.parts]).contiguous())['color'].clamp(0,1);return header((rgb.permute(1,2,0).cpu().numpy()*255).astype('uint8'),method['title'],t,'Original numeric stroke retained')
 path=root/f'{case}_GT_Origin_VLMAxis_VLMPoseGeo_HumanPoseGeo.mp4'
 with imageio.get_writer(path,fps=16,codec='libx264',quality=8) as writer:
  for i,t in enumerate(SEQ):
   rows=[np.concatenate([gt_frame(float(t),view),*[prediction(method,float(t),view) for method in methods]],axis=1) for view in range(2)];out=np.concatenate(rows,axis=0);writer.append_data(out)
   if i in [6,22,38]:Image.fromarray(out).save(root/f'five_versions_t{float(t):.2f}.png')
   if i%28==0:print('RENDER',case,i,'/',len(SEQ),flush=True)
 with imageio.get_reader(path) as reader:
  count=reader.count_frames();meta=reader.get_meta_data()
  for frame in reader:assert frame.shape[:2]==(768,1920)
 assert count==84
 write_json(root/'video_results.json',dict(video=path.name,frames=count,fps=meta['fps'],size=list(meta['size']),sha256=sha256(path),all_frames_decoded=True,versions=TITLES,cameras=dict(center=center.tolist(),radius=float(radius)),pose_note='Full native predicted intervals retained; observed pose is for single-image fitting, not a learned travel limit. GT00->05 for20411; other cases source->farther endpoint. Progress shared, mechanical strokes not matched to GT.',GT_files_sha256={str(p):sha256(p) for p in gt.source_paths}));print('VIDEO COMPLETE',path,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);a=p.parse_args();batch=a.root.resolve()
 for case in read(batch/'protocol.json')['cases']:render(batch,case)
