"""Single-input observed prismatic pose, then common geometry fit: human/VLM.

No extra RGB, GT geometry, GT joints, depth, or prior fitted geometry is used.
Pose search is bounded by each existing predicted interval. The input pose is
separate from the frozen mechanical interval and from canonical asset geometry.
"""
import argparse,copy,datetime,json,os,pathlib,shutil,subprocess,sys,time
import numpy as np,torch
from PIL import Image
from scipy.optimize import minimize_scalar
from collision_refinement_common import sha256,transforms
from multicase_closed_common import REPO,Model,tensor,image_metrics,write_json
from fit_vlm_component_geometry import VLMComponentFitModel,semantic_target,mask_scores,automatic_owners
from fit_auto_semantic_initial_frame import basis_for

AUTO=REPO/'repro/outputs/vlm_component_geometry_20261029_v1'
HUMAN=REPO/'repro/outputs/general_semantic_priors_20261029_v2'
LABELS={'100481':{1:10,2:11,3:12},'101808':{1:10,2:11,3:12},'10489':{1:10,2:12,3:11},'10655':{1:10,2:12,3:11},'20411':{1:10,2:11},'31249':{1:10,2:14,3:13,4:11,5:12},'45194':{1:10,2:12,3:11,4:14,5:13},'45503':{1:10,2:13,3:11,4:12},'8961':{1:10,2:11,3:12},'9016':{1:10,2:12,3:11},'103015':{1:10,2:11,3:12,4:13}}

def read(p):return json.loads(pathlib.Path(p).read_text())
def guard(root):
 denied=[];reads=[]
 def hook(event,args):
  if event=='open' and isinstance(args[0],(str,bytes)):
   p=os.fsdecode(args[0]);bad=any(x in p for x in ['/gt/','/gt_mesh/','joint_info','joint_evaluation','mobility_v2','/depth/','/closed_multicase_','/semantic_parts_'])
   if '/FreeArt3D/' in p and p.endswith('_pure.png') and not p.endswith('/05_pure.png'):bad=True
   if '/images/' in p and p.endswith('.png') and not p.endswith('/000000.png'):bad=True
   if bad:denied.append(p);raise PermissionError('Forbidden fitting observation: '+p)
   if p.endswith(('.json','.npz','.npy','.png','.pt','.ply')):reads.append(p)
 sys.addaudithook(hook);return denied,reads

class PoseModel(VLMComponentFitModel):
 def set_pose(self,pose):
  self.input_pose={int(k):float(v) for k,v in pose.items()};nodes=copy.deepcopy(self.groups)
  for g in nodes:g['joint']['range']=[self.input_pose.get(g['id'],0.)]*2
  self.closed_transforms=transforms(nodes,0.)

 def project(self):
  # The same generic bounds in both branches. No case-specific basket rule.
  super().project()


def load(root,registration=True,checkpoint=None):
 m=PoseModel(read(root/'config.json'),read(root/'motion_groups.json'),root)
 if registration:
  s=torch.load(root/'registration.pt',map_location='cuda',weights_only=False)
  with torch.no_grad():
   for name in ['log_global_scale','screen_translation']:getattr(m,name).copy_(s[name])
 if checkpoint:m.load_state_dict(torch.load(root/checkpoint,map_location='cuda',weights_only=False)['model'])
 m.set_pose({});return m

@torch.no_grad()
def metrics(m,original=False):
 g=m.geometry(original=original);return dict(image=image_metrics(m.render(g),m.render(g,'silhouette'),m.rgb,m.alpha),parts=mask_scores(m,g))

@torch.no_grad()
def preview(m,path):
 g=m.geometry();rgb=m.render(g);p=m.probabilities(g).cpu().numpy();bg=np.clip(1-p.sum(0),0,1);ids=np.array([0,*m.visible_owners])[np.concatenate([bg[None],p]).argmax(0)];out=np.ones((*ids.shape,3),np.float32)
 for gid in m.visible_owners:out[ids==gid]=m.config['palette'][str(gid)]
 pictures=[m.rgb,rgb,tensor(np.load(m.root/'observation.npz')['semantic']).permute(2,0,1),tensor(out).permute(2,0,1)]
 panel=torch.cat(pictures,2).clamp(0,1);Image.fromarray((255*panel.permute(1,2,0).cpu().numpy()).astype(np.uint8)).save(path)


def prepare(batch,case,branch):
 root=batch/case/branch;root.mkdir(parents=True,exist_ok=False);denied,reads=guard(root);source=(AUTO if branch=='vlm' else HUMAN)/case;man=copy.deepcopy(read(source/'motion_groups.json'));cfg=copy.deepcopy(read(source/'config.json'));groups=man['groups'];shutil.copyfile(source/'membership.npz',root/'membership.npz');membership=np.load(root/'membership.npz')
 # Human mapping is an already declared annotation, not a new region or GT axis.
 if branch=='human':
  mapping=LABELS[case];inverse={v:k for k,v in mapping.items()};byid={g['id']:g for g in groups}
  def owner(g):return owner(byid[g['shape_owner']]) if g['shape_owner'] is not None else g['id']
  for g in groups:g['semantic_owner']=inverse.get(owner(g))
 else:mapping=automatic_owners(groups)
 owners=automatic_owners(groups);ids=np.load(AUTO/case/'original_instance_ids.npy');shutil.copyfile(AUTO/case/'original_instance_ids.npy',root/'original_instance_ids.npy');raw_kinematics={str(g['id']):dict(parent=g['parent'],joint=copy.deepcopy(g['joint'])) for g in groups};man.update(automatic_instance_to_primary_group={str(k):v for k,v in owners.items()},pairs=[],unresolved_geometry_frozen_groups=[g['id'] for g in groups if g.get('semantic_owner') is None])
 # Common deformation family / owner union basis in BOTH branches; no old boxes,
 # paired parameters, or old optimized weights. Only axes/ownership may differ.
 from collision_refinement_common import load_asset
 asset=load_asset(man['source_run']);meshes={p['node']['id']:p['mesh'] for p in asset['parts']}
 for owner_id,gid in owners.items():
  primary=next(g for g in groups if g['id']==gid);owned=[g for g in groups if g.get('semantic_owner')==owner_id];v=np.concatenate([np.array(meshes[g['source_node']].vertices)[membership[g['membership_prefix']+'_vertices']] for g in owned]);B,a,length,kind=basis_for(primary,v);primary.update(shape_owner=None,deformation_basis=B.tolist(),deformation_anchor=a.tolist(),length=length,kind=kind)
  for g in owned:
   if g['id']!=gid:g['shape_owner']=gid
 for g in groups:g.update(box=None,observed_q=0.)
 assert raw_kinematics=={str(g['id']):dict(parent=g['parent'],joint=g['joint']) for g in groups}
 # Identical per-instance palette; RGB/masks/cameras from the actual source input.
 palette={str(g['id']):np.random.default_rng(g.get('semantic_owner') or 0).uniform(.2,.9,3).tolist() for g in groups};labels=semantic_target(ids,owners);data=np.load(AUTO/case/'observation.npz');semantic=np.ones((*labels.shape,3),np.float32)
 for instance,gid in owners.items():semantic[labels==gid]=palette[str(gid)]
 np.savez_compressed(root/'observation.npz',rgb=data['rgb'],alpha=data['alpha'],extrinsics=data['extrinsics'],intrinsics=data['intrinsics'],labels=labels,semantic=semantic);shutil.copyfile(AUTO/case/'registration.pt',root/'registration.pt');cfg.update(prior_root=str(root),palette=palette,annotation_boxes_original_pixels={},annotation_polygons_original_pixels={},alignment_steps=0,geometry_steps=240,pose_policy='Search only prismatic observed pose in existing predicted interval; axes and mechanical intervals frozen',geometry_policy='Same automatic-instance radial deformation, bounds, RGB/silhouette/Dice/CE loss, Adam settings in human and VLM branches; best TRAINING objective checkpoint, no GT selection')
 man.update(schema='observed_drawer_pose_v1',branch=branch,source_prior=str(source),source_kinematics=raw_kinematics,selection='Existing human annotations or sealed VLM labels/axes; common single-input pose and geometry protocol')
 write_json(root/'motion_groups.json',man);cfg['prior_manifest_sha256']=sha256(root/'motion_groups.json');cfg['observations_sha256']=sha256(root/'observation.npz');write_json(root/'config.json',cfg);Image.open(cfg['source_image']).save(root/'input_reference.png');write_json(root/'prepared.json',dict(case=case,branch=branch,source_prior=str(source),source_native_object=man['source_object'],mask_sha256=sha256(root/'original_instance_ids.npy'),mask_changed_pixels=0,common_registration_sha256=sha256(root/'registration.pt'),instance_to_primary=owners,kinematics_unchanged=True,forbidden_reads=denied,training_reads=sorted(set(reads))))
 print('PREPARED',case,branch,flush=True)


def estimate(m):
 pose={};trace=[];m.requires_grad_(False);drawers=[g for g in m.groups if g['joint']['type']=='prismatic'];initial=metrics(m,True)
 @torch.no_grad()
 def objective(gid,q):
  pose[gid]=float(q);m.set_pose(pose);value=float(m.losses()[0]);return value
 for sweep in range(2):
  for g in drawers:
   lo,hi=sorted(g['joint']['range']);grid=np.linspace(lo,hi,49);values=[objective(g['id'],q) for q in grid];j=int(np.argmin(values));lower=grid[max(0,j-1)];upper=grid[min(len(grid)-1,j+1)];sol=minimize_scalar(lambda q:objective(g['id'],q),bounds=(lower,upper),method='bounded',options={'xatol':1e-5,'maxiter':32}) if upper>lower else None;candidates=[(values[j],grid[j])]
   if sol is not None:candidates.append((float(sol.fun),float(sol.x)))
   value,q=min(candidates);pose[g['id']]=float(q);m.set_pose(pose);trace.append(dict(sweep=sweep,group=g['id'],semantic_instance=g.get('semantic_owner'),unchanged_native_interval=g['joint']['range'],grid_q=grid.tolist(),grid_loss=values,selected_q=float(q),selected_training_loss=float(value)))
 m.set_pose(pose);return dict(observed_pose={str(k):v for k,v in pose.items()},initial=initial,after_pose=metrics(m,True),trace=trace,mechanical_ranges_optimized=False,observation='Only original conditioning RGB / exact original instance mask / frozen camera registration')


def geometry(m,root,name,pose):
 m.set_pose(pose);m.requires_grad_(True);m.log_global_scale.requires_grad_(False);m.screen_translation.requires_grad_(False)
 with torch.no_grad():m.log_shape.zero_();m.vertical_shift.zero_()
 preview(m,root/(name+'_before.png'));initial=metrics(m);opt=torch.optim.Adam([{'params':[m.log_shape],'lr':.012},{'params':[m.vertical_shift],'lr':.004}]);history=[];bestloss=float('inf');best=None;beststep=0
 for step in range(240):
  opt.zero_grad(set_to_none=True);loss,values=m.losses();value=float(loss.detach());assert torch.isfinite(loss)
  if value<bestloss:bestloss=value;best={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};beststep=step
  loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters());opt.step();m.project();history.append(dict(completed_updates=step,loss=value,**values))
  if (step+1)%80==0:print(root.parent.name,root.name,name,step+1,value,flush=True)
 with torch.no_grad():lastloss=float(m.losses()[0])
 if lastloss<bestloss:bestloss=lastloss;best={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};beststep=240
 final=metrics(m);m.load_state_dict(best);chosen=metrics(m);preview(m,root/(name+'_after.png'));torch.save(dict(model=m.state_dict(),pose={str(k):v for k,v in pose.items()},completed_updates=beststep,selection='Minimum training objective among fixed240 updates plus initial/final, no GT or held-out data'),root/(name+'.pt'));result=dict(initial=initial,selected=chosen,last_iterate=final,selected_completed_updates=beststep,selected_training_loss=bestloss,last_training_loss=lastloss,pose={str(k):v for k,v in pose.items()},checkpoint_sha256=sha256(root/(name+'.pt')),optimizer='Adam .012 shape / .004 shift,240 updates, existing bounds',selection='Best training objective; all branches and ablations use identical policy');write_json(root/(name+'_history.json'),history);write_json(root/(name+'_results.json'),result);return result


def fit(root):
 denied,reads=guard(root);torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42);torch.backends.cuda.matmul.allow_tf32=False;start=time.monotonic();m=load(root);kin=sha256(root/'motion_groups.json');obs=sha256(root/'observation.npz');pose=estimate(m);write_json(root/'pose_estimation.json',pose);preview(m,root/'pose_only.png');m.requires_grad_(True);ablation=geometry(m,root,'frozen_pose_geometry',{});full=geometry(m,root,'observed_pose_geometry',pose['observed_pose']);m.requires_grad_(False)
 # Export canonical deformed geometry, with observed pose kept separately.
 from render_closed_image_45194 import export_parts
 export_parts(m,root);obj=read(root/'optimized_asset/object.json');obj.update(optimization='Original single input + observed drawer pose, then canonical geometry fit; no closed RGB',rest_pose='Native generated canonical reference; observed input pose stored separately',observed_input_pose=pose['observed_pose']);write_json(root/'optimized_asset/object.json',obj)
 assert kin==sha256(root/'motion_groups.json') and obs==sha256(root/'observation.npz') and not denied;assert all(sha256(p['path'])==p['sha256'] for p in m.manifest['geometry_files'])
 write_json(root/'fit_seal.json',dict(state='complete',case=root.parent.name,branch=root.name,pose=pose['observed_pose'],frozen_pose_geometry=ablation,observed_pose_geometry=full,source_manifest_sha256=kin,source_observation_sha256=obs,GT_forbidden_reads=denied,training_reads=sorted(set(reads)),seconds=time.monotonic()-start));print('FIT COMPLETE',root.parent.name,root.name,full['selected'],flush=True)


def initialize(batch,cases):
 batch.mkdir(exist_ok=False);protocol=dict(cases=cases,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),single_image=True,branches=['vlm','human'],vlm_prior_root=str(AUTO),human_prior_root=str(HUMAN),training='Only native source conditioning RGB, existing unchanged oracle instance mask, known camera and original native mesh/Gaussians. No extra closed RGB, GT geometry/joints/depth, held-out views.',pose='Independent derivative-free1D coordinate search for prismatic observed q,49grid points+bounded local refinement,2 sweeps. Existing predicted intervals are search bounds and remain unchanged.',geometry='Same physical-instance parameter basis and RGB+2silhouette+.5Dice+.1CE+.002shape+.01shift objective; Adam .012/.004 for240 steps, minimum TRAINING objective; no GT checkpoint choice.',registration='Both branches reuse same frozen original-input RGB/silhouette-only registration.',ablation='Each branch also runs identical240 geometry updates with observed q frozen at0; this isolates adding pose estimation.',versions=['GT','Origin','VLM axis correction','VLM axes + observed pose + geometry','Human axes + observed pose + geometry'],evaluation='Only after every fit sealed. Same original fixed-body GT similarity; native ranges unchanged.20411 GT states used only for evaluation/video.',origin_conditioning='Original RGBA+oracle instance EXR, part IDs only. No manual/VLM text category or axis rule fed to original PAct inference.',code_sha256=sha256(__file__))
 write_json(batch/'protocol.json',protocol);snapshot=batch/'source_snapshot';snapshot.mkdir();shutil.copyfile(__file__,snapshot/pathlib.Path(__file__).name);protected={}
 for case in cases:
  for base in [AUTO,HUMAN]:
   for p in (base/case).rglob('*'):
    if p.is_file():protected[str(p)]=sha256(p)
 write_json(batch/'protected_source_files.json',protected)


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--cases',nargs='+',default=['20411']);p.add_argument('--case');p.add_argument('--branch',choices=['vlm','human']);p.add_argument('--stage',choices=['all','prepare','fit'],default='all');a=p.parse_args();batch=a.root.resolve()
 if a.case:
  if a.stage=='prepare':prepare(batch,a.case,a.branch)
  elif a.stage=='fit':fit(batch/a.case/a.branch)
  return
 if not batch.exists():initialize(batch,a.cases)
 states={}
 for stage,marker in [('prepare','prepared.json'),('fit','fit_seal.json')]:
  for case in read(batch/'protocol.json')['cases']:
   for branch in ['vlm','human']:
    root=batch/case/branch;key=case+':'+branch+':'+stage
    if (root/marker).exists():states[key]='complete';continue
    with (batch/(case+'_'+branch+'_'+stage+'.log')).open('w') as log:r=subprocess.run([sys.executable,__file__,'--root',str(batch),'--case',case,'--branch',branch,'--stage',stage],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4'})
    states[key]='complete' if r.returncode==0 else 'failed';write_json(batch/'controller_status.json',states)
    if r.returncode:raise RuntimeError(key+' failed')
 states['state']='fit_complete';write_json(batch/'controller_status.json',states);print('ALL FITS SEALED',flush=True)
if __name__=='__main__':main()
