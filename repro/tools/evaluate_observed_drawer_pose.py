"""Independent post-seal evaluation of observed-pose and geometry ablations."""
import argparse,copy,pathlib,json
import numpy as np,torch,trimesh
from PIL import Image,ImageDraw,ImageFont
from fit_observed_drawer_pose import load,read,metrics,REPO
from collision_refinement_common import surface,sha256,transforms
from multicase_closed_common import tensor,write_json
from evaluate_multicase_closed import gt_and_alignment,distance_scores

FONT=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
NAMES={'origin':'Origin / raw q0','vlm_axis':'VLM axes / q0','vlm_pose':'VLM axes + observed pose','vlm_frozen_pose_geometry':'VLM geometry / fixed q0','vlm_pose_geometry':'VLM pose + geometry','human_axis':'Human axes / q0','human_pose':'Human axes + observed pose','human_frozen_pose_geometry':'Human geometry / fixed q0','human_pose_geometry':'Human pose + geometry'}

def build_mesh(m,optimized=False,pose=False,tree=None,progress=0.):
 motion=m.closed_transforms if pose else transforms(tree,progress) if tree is not None else {g['id']:np.eye(4) for g in m.groups};byid={p['node']['id']:p['mesh'] for p in m.asset['parts']};result=[]
 for nid,source in byid.items():
  v=np.array(source.vertices).copy()
  for i,p in enumerate(m.parts):
   g=p['group']
   if g['source_node']!=nid:continue
   vi=m.membership[g['membership_prefix']+'_vertices'];x=m.deform(i,tensor(p['vertex_local'])).detach().cpu().numpy() if optimized else np.array(source.vertices)[vi];T=motion[g['source_node'] if tree is m.asset['nodes'] else g['id']];v[vi]=x@T[:3,:3].T+T[:3,3]
  result.append(trimesh.Trimesh(v,np.array(source.faces),process=False))
 return trimesh.util.concatenate(result)

def image(m):
 with torch.no_grad():a=m.render();return Image.fromarray((a.clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype('uint8'))
def panel(titles,images,path):
 out=Image.new('RGB',(256*len(images),304),'white');d=ImageDraw.Draw(out)
 for j,(title,im) in enumerate(zip(titles,images)):out.paste(im,(j*256,48));d.text((j*256+4,4),title,font=FONT,fill='black')
 out.save(path)


def evaluate(batch,case):
 root=batch/case;assert all((batch/c/b/'fit_seal.json').exists() for c in read(batch/'protocol.json')['cases'] for b in ['vlm','human']);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;models={b:load(root/b) for b in ['vlm','human']};m=models['vlm'];_,body,moving,_,_,_,_,_=gt_and_alignment(m);alignment=read(REPO/f'repro/outputs/general_semantic_priors_20261029_v2/{case}/evaluation.json')['alignment'];R=np.array(alignment['R_export_to_GT']);s=alignment['scale'];shift=np.array(alignment['translation']);D=alignment['GT_static_diagonal'];truth_source=surface(trimesh.util.concatenate([body,*moving]),16000,177)
 if case=='20411':
  from evaluate_freeart_partnet_case import gt_parts
  states,delta=gt_parts(pathlib.Path(m.config['source_scene']));truth_closed=surface(trimesh.util.concatenate(states[0]),16000,177)
 else:states=None;delta=None;truth_closed=None
 variants=[('origin','vlm',None,False),('vlm_axis','vlm',None,False),('vlm_pose','vlm',None,True),('vlm_frozen_pose_geometry','vlm','frozen_pose_geometry.pt',False),('vlm_pose_geometry','vlm','observed_pose_geometry.pt',True),('human_axis','human',None,False),('human_pose','human',None,True),('human_frozen_pose_geometry','human','frozen_pose_geometry.pt',False),('human_pose_geometry','human','observed_pose_geometry.pt',True)];rows={};pictures={};native_states={};pretraining={b:{k:v.detach().clone() for k,v in mm.state_dict().items()} for b,mm in models.items()}
 for name,branch,checkpoint,posed in variants:
  mm=models[branch];mm.load_state_dict(pretraining[branch]);mm.requires_grad_(False)
  if checkpoint:mm.load_state_dict(torch.load(root/branch/checkpoint,map_location='cuda',weights_only=False)['model'])
  q=read(root/branch/'pose_estimation.json')['observed_pose'] if posed else {};mm.set_pose(q);score=metrics(mm);pictures[name]=image(mm);mesh=build_mesh(mm,optimized=checkpoint is not None,pose=True);points=s*(surface(mesh,16000,178)@R.T)+shift;score['surface_at_fitted_input_pose']=distance_scores(points,truth_source,D);canonical=build_mesh(mm,optimized=checkpoint is not None,pose=False);canonical_points=s*(surface(canonical,16000,178)@R.T)+shift;score['canonical_surface_vs_GT_closed']=distance_scores(canonical_points,truth_closed,D) if truth_closed is not None else None;score['observed_pose']=q
  # Read old, frozen axis evaluations only AFTER all new fits sealed.
  reference=REPO/f'repro/outputs/{"vlm_component_geometry_20261029_v1" if branch=="vlm" else "general_semantic_priors_20261029_v2"}/{case}/evaluation.json';joint=read(reference)['joint_methods']['original' if name=='origin' else 'prior'];score['axis']=dict(GT_joint_count=joint['GT_joint_count'],axis_correct_count=joint['axis_correct_count'],mean_axis_angle=joint['conditional_mean_angle_degrees'],type_correct_GT_coverage=joint['type_correct_GT_coverage']);rows[name]=score
  if states is not None and name in ['origin','vlm_axis','vlm_pose_geometry','human_pose_geometry']:
   tree=mm.asset['nodes'] if name=='origin' else mm.groups;sequence=[]
   for i in range(6):
    geom=build_mesh(mm,optimized=checkpoint is not None,tree=tree,progress=i/5);truth=surface(trimesh.util.concatenate(states[i]),16000,177);pred=s*(surface(geom,16000,178)@R.T)+shift;sequence.append(dict(progress=i/5,**distance_scores(pred,truth,D)))
   native_states[name]=sequence
 inputpic=Image.fromarray((m.rgb.clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype('uint8'));panel(['Original input RGB','Origin: canonical q0','VLM axes: q0','VLM: pose + geometry','Human: pose + geometry'],[inputpic,*[pictures[k] for k in ['origin','vlm_axis','vlm_pose_geometry','human_pose_geometry']]],root/'five_versions_input_pose.png');panel(['Original input RGB','VLM geo / fixed q0','VLM pose only','VLM pose + geometry','Human pose + geometry'],[inputpic,*[pictures[k] for k in ['vlm_frozen_pose_geometry','vlm_pose','vlm_pose_geometry','human_pose_geometry']]],root/'pose_ablation.png');shutil_config=read(root/'vlm/config.json');write_json(root/'config.json',shutil_config)
 result=dict(case=case,alignment=alignment,methods=rows,native_range_six_state_surface_CD=native_states,GT_only_after_all_fit_seals=True,original_input_image=m.config['source_image'],metrics_note='Input image IoU and instance mIoU are training diagnostics.3D surfaces use one frozen raw-body GT similarity and source face ordering. Input-pose CD includes pose+shape. Native-range trajectory separately retains full original stroke; no range fitting.',GT_displacement_world=delta.tolist() if delta is not None else None);write_json(root/'evaluation.json',result);print('EVALUATED',case,{k:round(v['image']['IoU']*100,2) for k,v in rows.items()},flush=True)


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);a=p.parse_args();batch=a.root.resolve()
 for case in read(batch/'protocol.json')['cases']:evaluate(batch,case)
if __name__=='__main__':main()
