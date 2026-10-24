"""Shared-camera GT / original / refined actual-triangle motion previews."""
import argparse,json,pathlib,sys
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image,ImageDraw
from collision_refinement_common import load_asset,candidate_nodes,moved,transforms
from evaluate_freeart_partnet_case import gt_parts
REPO=pathlib.Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO))
from modules.pact.representations.mesh import MeshExtractResult
from modules.pact.renderers.mesh_renderer import MeshRenderer
import utils3d
p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
root=pathlib.Path(a.root);config=json.loads((root/'experiment.json').read_text());name,run=next(iter(config['cases'].items()))
asset=load_asset(run);provenance=json.loads((asset['run']/'input_provenance.json').read_text());gt,delta=gt_parts(pathlib.Path(provenance['source_data_root']))
metrics=json.loads((root/'GT_geometry_and_motion.json').read_text());base=metrics['alignment'];R,s,t=np.array(base['R_export_to_gtworld']),base['scale'],np.array(base['translation'])
methods=['GT','original',*config['variants']]
nodes={m:candidate_nodes(asset,None if m=='original' else root/f'{name}_{m}/candidate.json') for m in methods if m!='GT'}
out=root/'GT_mesh_previews';out.mkdir(exist_ok=False);torch.set_num_threads(2)

def geometry(method,progress):
    if method=='GT':return [(m.vertices+(progress*delta if i else 0),m.faces,np.array([.94,.53,.18]) if i else np.array([.55,.55,.60])) for i,m in enumerate(gt[0])]
    world=transforms(nodes[method],progress);rows=[]
    for part in asset['parts']:
        moving=part['node']['joint']['type']!='fixed';vertices=s*(moved(part['mesh'].vertices,world[part['node']['id']])@R.T)+t
        rows.append((vertices,part['mesh'].faces,np.array([.94,.53,.18]) if moving else np.array([.55,.55,.60])))
    return rows
cloud=np.concatenate([vertices for method in methods for progress in [0.,1.] for vertices,_,_ in geometry(method,progress)])
center=(cloud.min(0)+cloud.max(0))/2;radius=1.15*np.linalg.norm(np.ptp(cloud,axis=0));renderer=MeshRenderer({'resolution':448,'near':.05,'far':20.})
fov=torch.deg2rad(torch.tensor(50.,device='cuda'));intr=utils3d.torch.intrinsics_from_fov_xy(fov,fov)

@torch.inference_mode()
def render(method,progress,yaw):
    vertices=[];faces=[];attrs=[];offset=0
    for v,f,c in geometry(method,progress):
        vertices.append(v);faces.append(f+offset);offset+=len(v);attrs.append(np.tile(np.r_[c,0.,0.,0.],(len(v),1)))
    rep=MeshExtractResult(torch.tensor(np.concatenate(vertices),device='cuda',dtype=torch.float32),torch.tensor(np.concatenate(faces),device='cuda',dtype=torch.int64),torch.tensor(np.concatenate(attrs),device='cuda',dtype=torch.float32))
    eye=center+radius*np.array([np.sin(yaw),.45,np.cos(yaw)])
    ext=utils3d.torch.extrinsics_look_at(torch.tensor(eye,device='cuda',dtype=torch.float32),torch.tensor(center,device='cuda',dtype=torch.float32),torch.tensor([0.,1.,0.],device='cuda'))
    output=renderer.render(rep,ext,intr,return_types=['color','mask','normal']);shade=.65+.35*torch.abs(2*output['normal'][2:3]-1)
    rgb=(output['color']*shade+1-output['mask'].unsqueeze(0)).clamp(0,1)
    pic=Image.fromarray((255*rgb.permute(1,2,0).cpu().numpy()).astype('uint8'));ImageDraw.Draw(pic).text((8,8),f'{method} | t={progress:.2f}',fill='black');return np.asarray(pic)
frames=[]
for state,progress in enumerate(np.linspace(0,1,21)):
    panel=np.concatenate([np.concatenate([render(m,float(progress),yaw) for m in methods],axis=1) for yaw in [.55,2.0]],axis=0)
    frames.append(panel)
    if state in [0,10,20]:Image.fromarray(panel).save(out/f'GT_original_refined_state{state:02d}.png')
    print('GT COMPARISON',state,flush=True)
imageio.mimsave(out/'GT_original_refined.mp4',frames,fps=10)
(out/'render_info.json').write_text(json.dumps({'methods':methods,'frames':21,'actual_triangles':True,'geometry_repair':False,'candidate_range_override':False,'GT_motion':'Original actual GT state00 to05, uniform translation verified from all six meshes.','prediction_alignment':str(asset['run']/'joint_evaluation/joint_errors.json'),'camera_center':center.tolist(),'camera_radius':radius},indent=2))
