"""Visibility-aware instance-mask metrics and semantic ownership videos."""
import argparse,copy,json,pathlib,time
import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from semantic_parts_ablation import SemanticModel,BASE,read,registration
from multicase_closed_common import CovGaussians,tensor,write_json
from collision_refinement_common import transforms,sha256
from modules.pact.renderers.gaussian_render import GaussianRenderer
import utils3d

p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--case',required=True);a=p.parse_args();root=pathlib.Path(a.root).resolve()/a.case;case=a.case
cfg=read(root/'config.json');model=SemanticModel(cfg,read(root/'motion_groups.json'),root);model.load_state_dict(torch.load(root/'checkpoint.pt',map_location='cuda',weights_only=False)['model']);model.requires_grad_(False)
oldroot=BASE/case;old=SemanticModel(read(oldroot/'config.json'),read(oldroot/'motion_groups.json'),oldroot);old.load_state_dict(torch.load(oldroot/'checkpoint.pt',map_location='cuda',weights_only=False)['model']);old.requires_grad_(False);torch.set_num_threads(4)
labels=np.load(root/'observation.npz')['labels'];owners=sorted(int(x) for x in np.unique(labels) if x)
raw_owner={}
for g in old.groups:
 raw_owner.setdefault(g['source_node'],g['shape_owner'] if g['shape_owner'] is not None else g['id'])

@torch.inference_mode()
def part_scores(m,method):
 oldowners=m.visible_owners;m.visible_owners=owners
 oldmap=m.owner_map
 if method=='original':m.owner_map={g['id']:raw_owner[g['source_node']] for g in m.groups}
 geom=m.geometry(original=method!='optimized');probs=m.probabilities(geom,method='original' if method=='original' else 'prior')
 m.owner_map=oldmap;m.visible_owners=oldowners
 probs=probs.cpu().numpy();background=(1-probs.sum(0)).clip(0,1);indices=np.concatenate([background[None],probs]).argmax(0);pred=np.array([0,*owners])[indices]
 rows=[]
 for gid in owners:
  truth=labels==gid;guess=pred==gid;inter=int((truth&guess).sum());union=int((truth|guess).sum());dice=2*inter/max(int(truth.sum()+guess.sum()),1)
  rows.append(dict(id=gid,IoU=inter/max(union,1),Dice=dice,target_pixels=int(truth.sum()),predicted_pixels=int(guess.sum())))
 rgb=np.full((*pred.shape,3),255,np.uint8)
 for gid in owners:rgb[pred==gid]=(np.array(cfg['palette'][str(gid)])*255).astype(np.uint8)
 return dict(rows=rows,mean_IoU=float(np.mean([r['IoU'] for r in rows])),mean_Dice=float(np.mean([r['Dice'] for r in rows]))),Image.fromarray(rgb)

metrics={};mask_images=[]
for name,m in [('previous',old),('semantic',model)]:
 metrics[name]={}
 for method in ['original','prior','optimized']:
  score,pic=part_scores(m,method);metrics[name][method]=score;mask_images.append((name+' '+method,pic))
canvas=Image.new('RGB',(1024,600),'white');draw=ImageDraw.Draw(canvas);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
target=Image.open(root/'front_panel_annotations.png').convert('RGB')
for row in range(2):
 for col,(name,pic) in enumerate([('RGB polygon target',target),*mask_images[row*3:(row+1)*3]]):
  x=col*256;y=row*300;draw.text((x+5,y+5),name,fill='black',font=font);canvas.paste(pic,(x,y+32))
canvas.save(root/'part_mask_fit_comparison.png')
write_json(root/'part_mask_metrics.json',dict(protocol='Visible functional-instance hard argmax IoU/Dice against manually traced CLOSED RGB polygons. These masks are training annotations, not independent GT segmentation. All instances rendered jointly with Gaussian alpha visibility. Origin colors use original source ownership.',**metrics))

render=read(root/'render_results.json');center=np.array(render['shared_camera_center']);radius=render['shared_camera_radius'];renderer=GaussianRenderer({'resolution':384,'near':.05,'far':20.,'bg_color':(1.,1.,1.)});renderer.pipe.compute_cov3D_python=True
intr=utils3d.torch.intrinsics_from_fov_xy(tensor(np.deg2rad(43.)),tensor(np.deg2rad(43.)));cameras=[]
for yaw in [.35,-.7]:
 eye=center+radius*np.array([np.sin(yaw)*np.cos(.22),np.sin(.22),np.cos(yaw)*np.cos(.22)]);cameras.append(utils3d.torch.extrinsics_look_at(tensor(eye),tensor(center),tensor([0,1,0])))
geoms={('semantic',method):model.geometry(original=method!='optimized') for method in ['original','prior','optimized']};geoms[('previous','optimized')]=old.geometry()
@torch.inference_mode()
def frame(m,version,method,t,view,semantic):
 tr=transforms(m.asset['nodes'] if method=='original' else m.groups,t);xyz=[];cov=[];colors=[];alphas=[]
 for part,(x,c) in zip(m.parts,geoms[(version,method)]):
  g=part['group'];T=tr[g['source_node'] if method=='original' else g['id']];R=tensor(T[:3,:3]);xyz.append(x@R.T+tensor(T[:3,3]));cov.append(R[None]@c@R.T[None]);alphas.append(part['alpha'])
  if semantic:
   owner=raw_owner[g['source_node']] if method=='original' else m.owner_map[g['id']];colors.append(tensor(cfg['palette'][str(owner)]).expand(len(x),3))
  else:colors.append(part['colors'])
 gs=CovGaussians(torch.cat(xyz).contiguous(),torch.cat(cov).contiguous(),torch.cat(alphas).contiguous());rgb=renderer.render(gs,cameras[view],intr,colors_overwrite=torch.cat(colors).contiguous())['color'];return (rgb.clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype('uint8')

def title(array,text,subtitle):
 pic=Image.fromarray(array);d=ImageDraw.Draw(pic);d.rectangle((0,0,383,47),fill='white');d.text((8,5),text,font=font,fill='black');d.text((8,27),subtitle,font=font,fill='#555555');return np.array(pic)
seq=np.r_[np.zeros(6),np.linspace(0,1,33),np.ones(6),np.linspace(1,0,33),np.zeros(6)]
mask_path=root/f'{case}_semantic_masks_three_stages.mp4';compare_path=root/f'{case}_previous_vs_semantic_with_GT.mp4'
source=imageio.get_reader(root/f'{case}_four_versions_with_GT.mp4');outputs=[mask_path,compare_path]
with imageio.get_writer(mask_path,fps=16,codec='libx264',quality=8) as mw,imageio.get_writer(compare_path,fps=16,codec='libx264',quality=8) as cw:
 for i,t in enumerate(seq):
  comparison=source.get_data(i);rows=[];comparison_rows=[]
  for view in [0,1]:
   semantic_panels=[title(frame(model,'semantic',method,float(t),view,True),caption,f'{case} | motion fraction {t:.2f}') for method,caption in [('original','Origin: source ownership'),('prior','Semantic motion prior'),('optimized','Closed-image fit')]]
   rows.append(np.concatenate(semantic_panels,1));oldrgb=title(frame(old,'previous','optimized',float(t),view,False),'Previous closed-image fit',f'{case} | motion fraction {t:.2f}')
   comparison_rows.append(np.concatenate([comparison[view*384:(view+1)*384,:384],oldrgb,comparison[view*384:(view+1)*384,3*384:4*384]],1))
  masks=np.concatenate(rows,0);panel=np.concatenate(comparison_rows,0);mw.append_data(masks);cw.append_data(panel)
  if i in [6,22,38]:
   Image.fromarray(masks).save(root/f'semantic_masks_motion_t{t:.2f}.png');Image.fromarray(panel).save(root/f'previous_vs_semantic_t{t:.2f}.png')
  if i%28==0:print('SEMANTIC MASK RENDER',case,i,flush=True)
source.close();videos=[]
for path in outputs:
 with imageio.get_reader(path) as r:
  n=r.count_frames();meta=r.get_meta_data()
  for i in range(n):r.get_data(i)
 assert n==84
 videos.append(dict(file=path.name,frames=n,size=list(meta['size']),fps=meta['fps'],sha256=sha256(path),all_frames_decoded=True))
write_json(root/'semantic_video_results.json',dict(videos=videos,shared_camera=render,mask_legend=read(root/'semantic_annotations.json')['names'],geometry_preserved_before_image_fit=True))
print('DETAILS COMPLETE',case,metrics,flush=True)
