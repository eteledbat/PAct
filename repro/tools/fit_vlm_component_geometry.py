"""VLM component ownership -> frozen v7 joints -> original-mask geometry fit.

No manual case recipes, semantic labels/boxes, hand-selected components, GT
geometry/joints, depth, or held-out views enter prepare/fit. All native geometry
is partitioned exactly as the sealed automatic axis export. One shape parameter
set per automatic physical instance; its auxiliary fragments share deformation.
"""
import argparse,copy,datetime,json,os,pathlib,shutil,subprocess,sys,time
import numpy as np,torch
from PIL import Image,ImageDraw,ImageFont
from scipy.ndimage import binary_erosion
from collision_refinement_common import load_asset,sha256,transforms
from multicase_closed_common import Model,REPO,tensor,write_json,image_metrics
from original_mask_multicase import OriginalMaskModel
from fit_auto_semantic_initial_frame import basis_for
from render_closed_image_45194 import export_parts

FONT=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
def read(p):return json.loads(pathlib.Path(p).read_text())

def training_guard(root):
 denied=[];reads=[]
 def hook(event,args):
  if event=='open' and isinstance(args[0],(str,bytes)):
   name=os.fsdecode(args[0]);blocked=any(s in name for s in ['/gt/','/gt_mesh/','mobility_v2','part_info.json','joint_infos','joint_info.json','joint_evaluation','/depth/','/closed_multicase_','/original_mask_multicase_','/semantic_parts_','/auto_semantic_initial_frame_'])
   if '/images/' in name and name.endswith('.png') and not name.endswith('/000000.png'):blocked=True
   if '/FreeArt3D/' in name and name.endswith('_pure.png') and not name.endswith('/05_pure.png'):blocked=True
   if blocked:denied.append(name);raise PermissionError('Unapproved geometry-fitting input: '+name)
   if name.endswith(('.json','.npy','.npz','.png','.pt')):reads.append(name)
 sys.addaudithook(hook);return denied,reads

def automatic_owners(groups):
 result={}
 for g in groups:
  owner=g.get('semantic_owner')
  if owner is None:continue
  own_ids={v['id'] for v in groups if v.get('semantic_owner')==owner}
  primary=[v for v in groups if v['id'] in own_ids and v['parent'] not in own_ids]
  if len(primary)!=1:raise ValueError('Automatic physical instance needs exactly one primary group')
  result[int(owner)]=primary[0]['id']
 return result

def semantic_target(ids,owners,resolution=256):
 small=np.array(Image.fromarray(ids.astype(np.int32),mode='I').resize((resolution,resolution),Image.Resampling.NEAREST));labels=np.full(small.shape,-1,np.int32);labels[small==0]=0
 for owner,gid in owners.items():labels[small==owner]=gid
 return labels

class VLMComponentFitModel(OriginalMaskModel):
 def __init__(self,cfg,manifest,root):
  Model.__init__(self,cfg,manifest,root);labels=np.load(root/'observation.npz')['labels'];self.targets=tensor(labels);self.supervised=tensor(labels>=0)>.5
  self.visible_owners=sorted(int(v) for v in manifest['automatic_instance_to_primary_group'].values());self.owner_map={g['id']:manifest['automatic_instance_to_primary_group'].get(str(g.get('semantic_owner'))) for g in self.groups};self.interior={gid:tensor(binary_erosion(labels==gid,iterations=1))>.5 for gid in self.visible_owners}
 def losses(self,align=False):
  geom=self.geometry();rgb=self.render(geom);alpha=self.render(geom,'silhouette');photo=((rgb-self.rgb).abs()*self.weight[None]).mean()/self.weight.mean();sil=(alpha-self.alpha).abs().mean()
  if align:return photo+2*sil,dict(photo=float(photo.detach()),silhouette=float(sil.detach()))
  probs=self.probabilities(geom);dices=[];ces=[]
  for j,gid in enumerate(self.visible_owners):
   target=(self.targets==gid).float();p=probs[j]*self.supervised;dices.append(1-(2*(p*target).sum()+1)/(p.sum()+target.sum()+1))
   if self.interior[gid].any():ces.append(-probs[j][self.interior[gid]].clamp_min(1e-5).log().mean())
  dice=torch.stack(dices).mean() if dices else photo*0;ce=torch.stack(ces).mean() if ces else photo*0;prior=.002*self.log_shape.square().mean()+.01*self.vertical_shift.square().mean();loss=photo+2*sil+.5*dice+.1*ce+prior
  return loss,dict(photo=float(photo.detach()),silhouette=float(sil.detach()),part_dice=float(dice.detach()),balanced_CE=float(ce.detach()))
 @torch.no_grad()
 def project(self):
  Model.project(self)
  for gid in self.manifest.get('unresolved_geometry_frozen_groups',[]):
   i=self.indices[gid];self.log_shape[i]=0;self.vertical_shift[i]=0

def mask_scores(m,geometry):
 probs=m.probabilities(geometry).detach().cpu().numpy();bg=np.clip(1-probs.sum(0),0,1);pred=np.array([0,*m.visible_owners])[np.concatenate([bg[None],probs]).argmax(0)];target=m.targets.cpu().numpy();valid=target>=0;rows=[]
 for gid in m.visible_owners:
  x=(pred==gid)&valid;y=target==gid;rows.append(dict(group=gid,IoU=float((x&y).sum()/max((x|y).sum(),1))))
 return dict(rows=rows,mean_IoU=float(np.mean([r['IoU'] for r in rows])) if rows else None,ignored_pixels=int((~valid).sum()),note='Automatic functional grouping at input q0; raw Origin source semantic accuracy is not evaluated')

def prepare_case(batch,case):
 axis_root=pathlib.Path(read(batch/'protocol.json')['axis_root']);root=batch/case;root.mkdir(exist_ok=False);denied,reads=training_guard(root);src=axis_root/case;manifest=copy.deepcopy(read(src/'motion_groups.json'));cfg=copy.deepcopy(read(src/'config.json'));membership=np.load(src/'membership.npz');shutil.copyfile(src/'membership.npz',root/'membership.npz');asset=load_asset(manifest['source_run']);mesh={p['node']['id']:p['mesh'] for p in asset['parts']};groups=manifest['groups'];owners=automatic_owners(groups);ass=read(src/'component_assignment.json');sem_path=pathlib.Path(manifest['semantic_input']);sem=read(sem_path);auto_root=sem_path.parent;original_ids=np.load(auto_root/'instance_ids.npy');shutil.copyfile(auto_root/'instance_ids.npy',root/'original_instance_ids.npy');shutil.copyfile(sem_path,root/'automatic_semantics.json');shutil.copyfile(src/'component_assignment.json',root/'automatic_component_assignment.json');kinematics_before={str(g['id']):dict(parent=g['parent'],joint=copy.deepcopy(g['joint']),semantic_owner=g['semantic_owner'],source_node=g['source_node'],component=g['component']) for g in groups};unresolved=[]
 for owner,gid in owners.items():
  group=next(g for g in groups if g['id']==gid);owned=[g for g in groups if g.get('semantic_owner')==owner];verts=np.concatenate([np.asarray(mesh[g['source_node']].vertices)[membership[g['membership_prefix']+'_vertices']] for g in owned]);B,anchor,length,kind=basis_for(group,verts);group.update(kind=kind,shape_owner=None,deformation_basis=B.tolist(),deformation_anchor=anchor.tolist(),length=length)
  for g in owned:
   if g['id']!=gid:g['shape_owner']=gid
 # Fixed pull grips follow their automatically recognized mounting part in both
 # FK and deformation, while their semantic supervision retains their own ID.
 for g in groups:
  if g.get('semantic_owner') is None:
   unresolved.append(g['id']);v=np.asarray(mesh[g['source_node']].vertices)[membership[g['membership_prefix']+'_vertices']];B,anchor,length,kind=basis_for(g,v);g.update(kind=kind,deformation_basis=B.tolist(),deformation_anchor=anchor.tolist(),length=length)
 for assignment in ass['assignments']:
  owner=assignment['instance_id']
  if assignment.get('family')=='fixed_grip' and owner in owners:
   child=next(g for g in groups if g['id']==owners[owner]);parent=next((g for g in groups if g['id']==child['parent']),None)
   if parent is not None:child['shape_owner']=parent['id']
 for g in groups:g['box']=None;g['observed_q']=0.
 assert kinematics_before=={str(g['id']):dict(parent=g['parent'],joint=g['joint'],semantic_owner=g['semantic_owner'],source_node=g['source_node'],component=g['component']) for g in groups}
 labels=semantic_target(original_ids,owners);data=np.load(src/'observation.npz');palette={str(g['id']):np.random.default_rng(owners.get(g.get('semantic_owner'),g['id'])).uniform(.2,.9,3).tolist() for g in groups};semantic=np.ones((*labels.shape,3),np.float32)
 for owner,gid in owners.items():semantic[labels==gid]=palette[str(gid)]
 np.savez_compressed(root/'observation.npz',rgb=data['rgb'],alpha=data['alpha'],labels=labels,semantic=semantic,extrinsics=data['extrinsics'],intrinsics=data['intrinsics']);reg_root=pathlib.Path(read(axis_root/'protocol.json')['registration'])/case;shutil.copyfile(reg_root/'registration.pt',root/'registration.pt');source_image=pathlib.Path(cfg['source_image']);shutil.copyfile(source_image,root/'input_reference.png');manifest.update(schema='vlm_component_geometry_v1',groups=groups,automatic_instance_to_primary_group={str(k):v for k,v in owners.items()},source_axis_root=str(src),source_axis_manifest_sha256=sha256(src/'motion_groups.json'),source_axis_membership_sha256=sha256(src/'membership.npz'),source_kinematics=kinematics_before,unresolved_geometry_frozen_groups=unresolved,pairs=[],selection='Sealed VLM semantics + automatic visible-projection component ownership. Same physical-instance fragments share deformation; no manual recipes or instance names.')
 write_json(root/'motion_groups.json',manifest);cfg.update(prior_root=str(root),prior_manifest_sha256=sha256(root/'motion_groups.json'),observations_sha256=sha256(root/'observation.npz'),geometry_steps=240,alignment_steps=0,palette=palette,annotation_boxes_original_pixels={},annotation_polygons_original_pixels={},axis_policy='All joints/types/parents/axes/ranges frozen exactly from v7 automatic component axes',geometry_policy='Automatic physical-instance groups replace manual shape groups; frozen q0 RGB/silhouette registration; same bounded deformation family and240 steps',GT_access='No GT geometry/joints/depth/held-out RGB during prepare/fit; original existing oracle instance mask remains input.');cfg.pop('recipe_sha256',None);write_json(root/'config.json',cfg)
 unsupported=sorted(int(x) for x in np.unique(original_ids) if x and int(x) not in owners);write_json(root/'prepared.json',dict(case=case,source_axis_manifest_sha256=sha256(src/'motion_groups.json'),source_semantics_sha256=sha256(sem_path),source_mask_sha256=sha256(auto_root/'instance_ids.npy'),copied_mask_sha256=sha256(root/'original_instance_ids.npy'),mask_changed_pixels=0,supported_instance_to_shape_group=owners,unsupported_instances=unsupported,ignored_semantic_pixels=int((labels<0).sum()),registration_source=str(reg_root/'registration.pt'),registration_sha256=sha256(root/'registration.pt'),manual_case_recipe_used=False,manual_semantic_labels_used=False,GT_forbidden_reads=denied,training_file_access=sorted(set(reads))))
 Image.fromarray((semantic*255).astype(np.uint8)).save(root/'automatic_semantic_target.png');write_json(root/'status.json',dict(state='prepared'));print('PREPARED',case,'instances',len(owners),'ignored',unsupported,flush=True)

def load_model(root,registered=False,checkpoint=False):
 m=VLMComponentFitModel(read(root/'config.json'),read(root/'motion_groups.json'),root)
 if registered:
  state=torch.load(root/'registration.pt',map_location='cuda',weights_only=False)
  with torch.no_grad():
   for name in ['log_global_scale','screen_translation']:getattr(m,name).copy_(state[name])
 if checkpoint:m.load_state_dict(torch.load(root/'checkpoint.pt',map_location='cuda',weights_only=False)['model'])
 return m

def fit_case(root):
 import original_mask_semantic_100481 as mod
 mod.training_guard=training_guard;mod.load_model=load_model;mod.mask_scores=mask_scores;mod.fit(root);data=read(root/'fit_results.json');data.update(case=root.name,source_tool_sha256=sha256(__file__),training_frame=read(root/'config.json')['reference_frame'],manual_semantics_used=False,shape_ownership='automatic physical instances from VLM and projection');write_json(root/'fit_results.json',data);write_json(root/'status.json',dict(state='complete',**data))

def evaluate_case(root):
 batch=root.parent;assert all((batch/c/'fit_results.json').exists() for c in read(batch/'protocol.json')['cases'])
 import evaluate_multicase_closed as mod
 mod.Model=VLMComponentFitModel;original=mod.gt_and_alignment;reference=pathlib.Path(read(batch/'protocol.json')['evaluation_reference_root'])/root.name
 def frozen_alignment(model):
  targets,body,moving,_,_,_,_,_=original(model);a=read(reference/'evaluation.json')['alignment'];return targets,body,moving,np.array(a['R_export_to_GT']),a['scale'],np.array(a['translation']),a['GT_static_diagonal'],a
 mod.gt_and_alignment=frozen_alignment;mod.evaluate_case(root)

def render_case(root):
 import render_auto_semantic_four_versions as mod
 mod.AutoFitModel=VLMComponentFitModel;mod.TITLES=['GT | source to endpoint','Origin | original range','VLM axes | frozen v7','VLM-guided geometry fit'];m=load_model(root,checkpoint=True);m.requires_grad_(False);export_parts(m,root);del m;torch.cuda.empty_cache();mod.render_case(root)
 data=read(root/'auto_four_video_results.json');data['columns']=mod.TITLES;data['manual_geometry_guidance_replaced_by']='VLM categories, automatic component membership and semantic shape-parameter ownership';write_json(root/'auto_four_video_results.json',data)


def initialize(batch,axis_root,reference):
 batch.mkdir(exist_ok=False);axis_root=axis_root.resolve();seal=read(axis_root/'FINAL_SEAL.json');assert seal['state']=='complete';protocol=dict(cases=read(axis_root/'protocol.json')['cases'],created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),axis_root=str(axis_root),source_axis_final_seal_sha256=sha256(axis_root/'FINAL_SEAL.json'),evaluation_reference_root=str(reference.resolve()),versions=['GT','Origin','VLM axes (v7)','VLM-guided geometry fit'],manual_semantic_annotations_used=False,manual_component_groups_used=False,mask='Exact existing original instance masks. Only semantic IDs remapped; unsupported instances ignored, not hallucinated.',shape_ownership='One automatically matched physical instance per shape parameter set; same-instance fragments share deformation; independent doors/handles remain independent.',kinematics='Types, axes, parents and native ranges frozen from sealed v7, including handle-location-first flip-door rule.',fit='Original q0 conditioning RGB, silhouette and supported per-instance Dice/CE. Frozen previous RGB-only camera registration; fixed final240 geometry steps; no GT checkpoint selection.',loss='RGB weightedL1 +2 silhouetteL1 +.5 instanceDice +.1 balanced interiorCE +.002 shape prior +.01 shift prior',deformation='Existing anchored radial hinge deformation, in-plane scales, axis-direction shifts, unchanged opacity/color/count/topology. Existing bounds and optimizer retained.',evaluation='All11 fits sealed before GT/held-out views; frozen original raw-body GT similarity reused only in evaluation/render.',code_sha256=sha256(__file__))
 write_json(batch/'protocol.json',protocol);snap=batch/'source_snapshot';snap.mkdir()
 for name in ['fit_vlm_component_geometry.py','multicase_closed_common.py','fit_auto_semantic_initial_frame.py','original_mask_multicase.py','original_mask_semantic_100481.py','evaluate_multicase_closed.py','render_auto_semantic_four_versions.py','render_closed_image_45194.py']:shutil.copyfile(REPO/'repro/tools'/name,snap/name)
 write_json(batch/'source_axis_files_sha256.json',{str(p):sha256(p) for p in axis_root.rglob('*') if p.is_file()});print('PROTOCOL FROZEN',flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--axis-root',type=pathlib.Path);p.add_argument('--reference-root',type=pathlib.Path,default=REPO/'repro/outputs/general_semantic_priors_20261029_v2');p.add_argument('--case');p.add_argument('--stage',choices=['all','prepare','fit','evaluate','render'],default='all');a=p.parse_args();batch=a.root.resolve()
 if a.case:
  {'prepare':lambda:prepare_case(batch,a.case),'fit':lambda:fit_case(batch/a.case),'evaluate':lambda:evaluate_case(batch/a.case),'render':lambda:render_case(batch/a.case)}[a.stage]();return
 if not batch.exists():initialize(batch,a.axis_root,a.reference_root)
 states=read(batch/'controller_status.json') if (batch/'controller_status.json').exists() else {};markers={'prepare':'prepared.json','fit':'fit_results.json','evaluate':'evaluation.json','render':'auto_four_video_results.json'}
 for stage,marker in markers.items():
  for case in read(batch/'protocol.json')['cases']:
   root=batch/case;key=case+':'+stage
   if (root/marker).exists():states[key]='complete';continue
   if stage=='prepare':root.parent.mkdir(exist_ok=True)
   log=batch/(case+'_'+stage+'.log')
   with log.open('w') as f:proc=subprocess.run([sys.executable,__file__,'--root',str(batch),'--case',case,'--stage',stage],cwd=REPO,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4'})
   states[key]='complete' if proc.returncode==0 else 'failed';write_json(batch/'controller_status.json',states);print(key,states[key],flush=True)
   if proc.returncode:raise RuntimeError(key+' failed; '+str(log))
 states['state']='complete';write_json(batch/'controller_status.json',states);print('ALL COMPLETE',batch,flush=True)

if __name__=='__main__':main()
