"""Render actual exported triangles under original PAct joint transforms; no repairs."""
import argparse,json,os,pathlib,sys,time
import numpy as np,torch,trimesh,imageio
from PIL import Image,ImageDraw
ROOT=pathlib.Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from modules.pact.representations.mesh import MeshExtractResult
from modules.pact.renderers.mesh_renderer import MeshRenderer
from modules.utils.articulation_utils import _local_transform,_build_hierarchy
import utils3d
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();run=pathlib.Path(a.run);obj=next(run.rglob('object.json'));nodes=json.loads(obj.read_text())['diffuse_tree'];out=run/'mesh_previews';out.mkdir(exist_ok=False);meshes=[trimesh.load(obj.parent/n['plys'][0],force='mesh',process=False) for n in nodes];renderer=MeshRenderer({'resolution':512,'near':.1,'far':10.});lookup={n['id']:n for n in nodes};order=_build_hierarchy(nodes);started=time.monotonic()
fov=torch.deg2rad(torch.tensor(50.,device='cuda'));intr=utils3d.torch.intrinsics_from_fov_xy(fov,fov)
@torch.inference_mode()
def frame(t,yaw=.55):
 transforms={-1:np.eye(4)}
 for nid in order:transforms[nid]=transforms.get(lookup[nid].get('parent',-1),np.eye(4))@_local_transform(lookup[nid],float(t))
 vertices=[];faces=[];attrs=[];offset=0
 for i,(node,m) in enumerate(zip(nodes,meshes)):
  tr=transforms[node['id']];vertices.append(np.asarray(m.vertices)@tr[:3,:3].T+tr[:3,3]);faces.append(m.faces+offset);offset+=len(m.vertices);rgb=[[.20,.52,.77],[.94,.53,.18],[.25,.7,.42]][i%3];attrs.append(np.tile(rgb+[0.,0.,0.],(len(m.vertices),1)))
 representation=MeshExtractResult(torch.tensor(np.concatenate(vertices),device='cuda',dtype=torch.float32),torch.tensor(np.concatenate(faces),device='cuda',dtype=torch.int64),torch.tensor(np.concatenate(attrs),device='cuda',dtype=torch.float32))
 origin=torch.tensor([np.sin(yaw)*3.0,.7,np.cos(yaw)*3.0],device='cuda',dtype=torch.float32);ext=utils3d.torch.extrinsics_look_at(origin,torch.tensor([0.,0.,.35],device='cuda'),torch.tensor([0.,1.,0.],device='cuda'))
 image=renderer.render(representation,ext,intr,return_types=['color','mask','normal']);shade=.6+.4*torch.abs(2*image['normal'][2:3]-1);rgb=image['color']*shade+1-image['mask'].unsqueeze(0);arr=(rgb.clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8);pic=Image.fromarray(arr);ImageDraw.Draw(pic).text((12,12),f'Original exported mesh | motion t={t:.2f}',fill='black');return np.array(pic)
frames=[]
for i,t in enumerate(np.linspace(0,1,21)):
 image=frame(float(t));frames.append(image)
 if i in [0,10,20]:
  views=[image,frame(float(t),2.4),frame(float(t),4.3)];Image.fromarray(np.concatenate(views,axis=1)).save(out/f'mesh_state{i:02d}_3views.png')
 print('RENDER',i,flush=True)
imageio.mimsave(out/'raw_mesh_motion_21.mp4',frames,fps=10);(out/'render_info.json').write_text(json.dumps({'states':21,'actual_triangles':True,'mesh_repair':False,'motion':'Original _local_transform using exported object.json; no GT fitting','elapsed_seconds':time.monotonic()-started,'gpu_peak_allocated_mb':torch.cuda.max_memory_allocated()/2**20},indent=2));print('MESH PREVIEWS COMPLETE',out,flush=True)
