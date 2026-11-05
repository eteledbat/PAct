"""Post-seal evaluation for the ten-case published-v0.1.0 replay."""
import argparse,copy,pathlib
import numpy as np,torch,trimesh
from PIL import Image
from fit_observed_drawer_pose import read,load,metrics,REPO
from evaluate_observed_drawer_pose import build_mesh,image,panel
from evaluate_multicase_closed import gt_and_alignment,distance_scores
from collision_refinement_common import surface,sha256
from multicase_closed_common import tensor,image_metrics,write_json

VARIANTS=[('origin',None,False),('vlm_axis',None,False),('vlm_pose',None,True),('vlm_frozen_pose_geometry','frozen_pose_geometry.pt',False),('vlm_pose_geometry','observed_pose_geometry.pt',True)]

def evaluate(batch,case):
 assert (batch/'ALL_FITS_SEAL.json').exists()
 for c,h in read(batch/'ALL_FITS_SEAL.json')['seals'].items():assert sha256(batch/c/'vlm/fit_seal.json')==h
 root=batch/case;r=root/'vlm';torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;m=load(r);m.requires_grad_(False);initial={k:v.detach().clone() for k,v in m.state_dict().items()};write_json(root/'config.json',read(r/'config.json'))
 _,body,moving,*_=gt_and_alignment(m);truth=surface(trimesh.util.concatenate([body,*moving]),16000,177)
 alignment=read(REPO/f'repro/outputs/general_semantic_priors_20261029_v2/{case}/evaluation.json')['alignment'];R=np.array(alignment['R_export_to_GT']);s=alignment['scale'];shift=np.array(alignment['translation']);D=alignment['GT_static_diagonal']
 reference=read(REPO/f'repro/outputs/vlm_component_geometry_20261029_v1/{case}/evaluation.json');rows={};pictures={};novel={};poses=read(r/'pose_estimation.json')['observed_pose'];camera=None
 if case!='20411':camera=read(pathlib.Path(m.config['source_scene'])/'transforms.json')
 for name,checkpoint,posed in VARIANTS:
  m.load_state_dict(initial)
  if checkpoint:m.load_state_dict(torch.load(r/checkpoint,map_location='cuda',weights_only=False)['model'])
  q=poses if posed else {};m.set_pose(q);score=metrics(m);pictures[name]=image(m);mesh=build_mesh(m,optimized=checkpoint is not None,pose=True);pred=s*(surface(mesh,16000,178)@R.T)+shift;score['surface_at_fitted_input_pose']=distance_scores(pred,truth,D);score['observed_pose']=q
  key='original' if name=='origin' else 'prior';score['axis']=copy.deepcopy(reference['joint_methods'][key]);score['checkpoint_sha256']=sha256(r/checkpoint) if checkpoint else None
  if camera:
   views=[];panels=[]
   for index in [25,50,75,100,125,149]:
    frame=camera['frames'][index];assert frame['state']==0 and frame['time']==0.;im=Image.open(pathlib.Path(m.config['source_scene'])/frame['file_path']).convert('RGBA').resize((256,256),Image.Resampling.LANCZOS);rgba=np.asarray(im).astype(np.float32)/255;target=tensor(rgba[:,:,:3]*rgba[:,:,3:4]+1-rgba[:,:,3:4]).permute(2,0,1);mask=tensor(rgba[:,:,3]);c2w=np.array(frame['transform_matrix']);c2w[:3,1:3]*=-1;ext=tensor(np.linalg.inv(c2w));intr=tensor([[camera['focal_x']/camera['w'],0,camera['cx']/camera['w']],[0,camera['focal_y']/camera['h'],camera['cy']/camera['h']],[0,0,1]])
    with torch.no_grad():
     g=m.geometry();rgb=m.render(g,extrinsics=ext,intrinsics=intr,reference=False);alpha=m.render(g,'silhouette',extrinsics=ext,intrinsics=intr,reference=False)
    views.append(dict(frame=index,**image_metrics(rgb,alpha,target,mask)))
   score['heldout_views']=views;score['heldout_mean']={k:float(np.mean([v[k] for v in views])) for k in ['IoU','PSNR','foreground_PSNR','foreground_L1']}
  else:score['heldout_views']=[];score['heldout_mean']=None
  rows[name]=score
 inputpic=Image.fromarray((m.rgb.clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype('uint8'))
 panel(['Original input RGB','Origin / q0','VLM axes / q0','VLM pose + geometry'],[inputpic,pictures['origin'],pictures['vlm_axis'],pictures['vlm_pose_geometry']],root/'input_pose_comparison.png')
 panel(['Original input RGB','Geometry / fixed q0','Observed pose only','Pose + geometry'],[inputpic,pictures['vlm_frozen_pose_geometry'],pictures['vlm_pose'],pictures['vlm_pose_geometry']],root/'pose_ablation.png')
 result=dict(case=case,alignment=alignment,methods=rows,GT_only_after_all_fit_seals=True,original_input_image=m.config['source_image'],axis_correspondence='Frozen original/prior axis/type-blind geometric correspondences from pre-fit v7 evaluation; geometry does not change kinematics.',heldout_protocol='Frames25/50/75/100/125/149 at same source state0, fixed camera registration, no GT pose/camera fit. Apply the INPUT-estimated observed q unchanged at all novel cameras.20411 has no novel cameras.',surface_protocol='16000 samples, nonsquared bidirectional mean CD / fixedGTbodydiagonal. Frozen original-body alignment. Includes pose+shape errors. No per-part alignment.')
 write_json(root/'evaluation.json',result)
 # Whole-object motion diagnostic; independent GT only after fitting seal.
 from add_gt_multicase_videos import GT
 if case=='20411':gt=GT(root)
 else:
  from render_auto_semantic_four_versions import gt_at_source
  gt,_=gt_at_source(root)
 trajectory={}
 for name,checkpoint in [('origin',None),('vlm_axis',None),('vlm_pose_geometry','observed_pose_geometry.pt')]:
  m.load_state_dict(initial)
  if checkpoint:m.load_state_dict(torch.load(r/checkpoint,map_location='cuda',weights_only=False)['model'])
  values=[]
  for t in np.linspace(0,1,6):
   tree=m.asset['nodes'] if name=='origin' else m.groups;mesh=build_mesh(m,optimized=checkpoint is not None,tree=tree,progress=float(t));target_mesh=trimesh.Trimesh(gt.vertices(float(t)),gt.faces.cpu().numpy(),process=False);truthp=s*(surface(target_mesh,16000,177)@R.T)+shift;pred=s*(surface(mesh,16000,178)@R.T)+shift;values.append(dict(progress=float(t),**distance_scores(pred,truthp,D)))
  trajectory[name]=values
 result['native_range_six_state_surface']=trajectory;write_json(root/'evaluation.json',result)
 print('EVALUATED',case,{k:round(v['image']['IoU']*100,2) for k,v in rows.items()},flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--case');a=p.parse_args()
 for c in ([a.case] if a.case else read(a.root/'protocol.json')['cases']):evaluate(a.root.resolve(),c)
