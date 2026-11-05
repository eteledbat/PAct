"""Drawer-only comparison: GT | frozen VLM geometry/native range | new range."""
import argparse,pathlib,copy
import imageio.v2 as imageio,numpy as np,torch
from PIL import Image,ImageDraw,ImageFont
from plyfile import PlyData
from scipy.spatial.transform import Rotation
from fit_vlm_drawer_ranges import read
from collision_refinement_common import transforms,moved,sha256
from multicase_closed_common import tensor,CovGaussians,write_json
from add_gt_multicase_videos import GT,MeshExtractResult,MeshRenderer,SEQ
from modules.pact.renderers.gaussian_render import GaussianRenderer
import utils3d

FONT=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)

def render(batch,case):
 root=batch/case;assert read(batch/'ALL_FITS_SEAL.json')['state']=='complete';evaluation=read(root/'evaluation.json');fit=read(root/'range_fit.json');targets={r['group'] for r in fit['rows']};trees={name:read(root/name/'object.json')['diffuse_tree'] for name in ['native','full']};gt=GT(root);front_world=gt.R@np.array([0.,0.,1.])
 if case!='20411':
  for r in gt.records:
   if r['type']=='prismatic':
    a,b=r['reference_endpoint_a'],r['open_endpoint_b']
    if (b-a)*np.dot(np.array(r['axis_world']),front_world)<0:r['reference_endpoint_a'],r['open_endpoint_b']=b,a
   else:r['reference_endpoint_a']=r['mesh_frame0_value'];r['open_endpoint_b']=r['mesh_frame0_value']
 # This is a drawer-only diagnostic playback. Source assets retain all other
 # joint intervals untouched; only their displayed pose is held at canonical q0.
 for name,tree in trees.items():
  trees[name]=copy.deepcopy(tree)
  for g in trees[name]:
   if g['id'] not in targets:g['joint']['range']=[0.,0.]
 undo=np.array([[1,0,0],[0,0,1],[0,-1,0]],float);parts=[]
 for node in trees['native']:
  data=PlyData.read(node['gs'][0])['vertex'].data;x=np.column_stack([data[k] for k in ['x','y','z']]).astype(np.float32)@undo.T;scale=np.exp(np.column_stack([data[k] for k in ['scale_0','scale_1','scale_2']]).astype(np.float32));q=np.column_stack([data[k] for k in ['rot_0','rot_1','rot_2','rot_3']]);rot=undo[None]@Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix() if len(data) else np.empty((0,3,3));cov=(rot*scale[:,None,:]**2)@rot.transpose(0,2,1);alpha=(1/(1+np.exp(-data['opacity'].astype(np.float32))))[:,None];color=(.28209479177387814*np.column_stack([data[k] for k in ['f_dc_0','f_dc_1','f_dc_2']])+.5).clip(0,1);parts.append(dict(id=node['id'],xyz=tensor(x),cov=tensor(cov),alpha=tensor(alpha),color=tensor(color)))
 clouds=[]
 for name,tree in trees.items():
  for t in [0.,.5,1.]:
   tr=transforms(tree,t)
   for p in parts:clouds.append(moved(p['xyz'][::max(1,len(p['xyz'])//1200)].cpu().numpy(),tr[p['id']]))
 for t in [0.,.5,1.]:clouds.append(gt.vertices(t)[::100])
 cloud=np.concatenate(clouds);center=(cloud.min(0)+cloud.max(0))/2;radius=1.55*np.linalg.norm(np.ptp(cloud,axis=0));intr=utils3d.torch.intrinsics_from_fov_xy(tensor(np.deg2rad(43.)),tensor(np.deg2rad(43.)));cameras=[]
 for yaw in [.35,-.7]:
  eye=center+radius*np.array([np.sin(yaw)*np.cos(.22),np.sin(.22),np.cos(yaw)*np.cos(.22)]);cameras.append(utils3d.torch.extrinsics_look_at(tensor(eye),tensor(center),tensor([0,1,0])))
 gr=GaussianRenderer(dict(resolution=384,near=.05,far=20.,bg_color=(1.,1.,1.)));gr.pipe.compute_cov3D_python=True;mr=MeshRenderer(dict(resolution=384,near=.05,far=20.,ssaa=2));alpha=torch.cat([p['alpha'] for p in parts]).contiguous();colors=torch.cat([p['color'] for p in parts]).contiguous();titles=['GT | drawer closed -> open','VLM geometry | native range','Same geometry | new range'];cache={}
 def label(rgb,index,t):
  pic=Image.fromarray(rgb);d=ImageDraw.Draw(pic);d.rectangle((0,0,383,47),fill='white');d.text((6,4),titles[index],font=FONT,fill='black');d.text((6,26),f'{case}: drawer-only | {t:.2f}',font=FONT,fill='#555555');return np.array(pic)
 @torch.inference_mode()
 def gtframe(t,view):
  key=(float(t),view)
  if key not in cache:
   mesh=MeshExtractResult(tensor(gt.vertices(t)),gt.faces,gt.attrs);o=mr.render(mesh,cameras[view],intr,return_types=['color','mask','normal']);rgb=(o['color']*(.65+.35*abs(2*o['normal'][2:3]-1))+1-o['mask'].unsqueeze(0)).clamp(0,1);cache[key]=label((rgb.permute(1,2,0).cpu().numpy()*255).astype('uint8'),0,t)
  return cache[key]
 @torch.inference_mode()
 def pred(name,t,view):
  tr=transforms(trees[name],t);xs=[];cs=[]
  for p in parts:
   T=tr[p['id']];R=tensor(T[:3,:3]);xs.append(p['xyz']@R.T+tensor(T[:3,3]));cs.append(R[None]@p['cov']@R.T[None])
  gs=CovGaussians(torch.cat(xs).contiguous(),torch.cat(cs).contiguous(),alpha);rgb=gr.render(gs,cameras[view],intr,colors_overwrite=colors)['color'].clamp(0,1);return label((rgb.permute(1,2,0).cpu().numpy()*255).astype('uint8'),1 if name=='native' else 2,t)
 path=root/f'{case}_GT_VLMNativeRange_OptimizedRange.mp4';torch.set_num_threads(4)
 with imageio.get_writer(path,fps=16,codec='libx264',quality=8) as writer:
  for i,t in enumerate(SEQ):
   out=np.concatenate([np.concatenate([gtframe(float(t),view),pred('native',float(t),view),pred('full',float(t),view)],axis=1) for view in range(2)],axis=0);writer.append_data(out)
   if i in [6,22,38]:Image.fromarray(out).save(root/f'range_comparison_t{float(t):.2f}.png')
   if i%28==0:print('RENDER',case,i,flush=True)
 with imageio.get_reader(path) as reader:
  count=reader.count_frames();meta=reader.get_meta_data()
  for frame in reader:assert frame.shape[:2]==(768,1152)
 assert count==84;write_json(root/'video_results.json',dict(video=path.name,sha256=sha256(path),frames=count,fps=meta['fps'],size=list(meta['size']),all_frames_decoded=True,GT_files_sha256={str(p):sha256(p) for p in gt.source_paths},playback='Only primary semantic drawer joints animate; all other displayed joints q0. GT closed->open from annotated endpoints, prediction original/new respective numeric intervals; common progress, same geometry and axes before/after. Original exported non-target intervals unchanged.'));print('VIDEO COMPLETE',case,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);a=p.parse_args();batch=a.root.resolve()
 for case in read(batch/'protocol.json')['cases']:render(batch,case)
