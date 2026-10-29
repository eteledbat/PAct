"""Original-mask reconstruction with reusable handle and lifting-door priors."""
import argparse,copy,datetime,json,os,pathlib,shutil,subprocess,sys,time,types
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
import cv2,numpy as np,torch
from PIL import Image,ImageDraw
from plyfile import PlyData
from scipy.spatial import cKDTree
import original_mask_multicase as legacy
from multicase_closed_common import Model,REPO,tensor,write_json,image_metrics,component_labels,recipes,spec
from collision_refinement_common import load_asset,sha256
from semantic_motion_rules import discover_handles,top_hinged_door,radial_basis,outward_sign,project,resolve_parented_grips
from original_mask_semantic_100481 import source_group
from fit_closed_image_45194 import R_WORLD

PREVIOUS=REPO/'repro/outputs/original_mask_multicase_20261029_v1'
CASES=[*legacy.CASES,'103015']


def read(p):return json.loads(pathlib.Path(p).read_text())


def guard(root):
 cfg=read(root/'config.json') if (root/'config.json').exists() else next(x for x in read(root.parent/'input_inventory.json') if x['case']==root.name);allowed=pathlib.Path(cfg.get('source_image',cfg.get('rgb'))).resolve();denied=[];reads=[]
 def hook(event,args):
  if event=='open' and isinstance(args[0],(str,bytes)):
   name=os.fsdecode(args[0]);bad=any(s in name for s in ['/gt/','/gt_mesh/','mobility_v2','joint_infos','joint_info.json','joint_evaluation','/depth/','/original_mask_multicase_20261029_v1/','/closed_multicase_20261028_v1/','/semantic_parts_100481_45194_'])
   if '/images/' in name and name.endswith('.png') and pathlib.Path(name).resolve()!=allowed:bad=True
   if bad:denied.append(name);raise PermissionError('Training observation not allowed: '+name)
   if name.endswith(('.json','.png','.exr','.npz','.pt')):reads.append(name)
 sys.addaudithook(hook);return denied,reads


class GeneralModel(legacy.OriginalMaskModel):
 @torch.no_grad()
 def project(self):
  Model.project(self)
  for p in self.parts:
   if self.byid[p['owner']].get('motion_role')=='articulated_handle':
    self.log_shape[p['parameter_index'],1]=0;self.vertical_shift[p['parameter_index']]=0


def load_model(root,registered=False,checkpoint=False):
 m=GeneralModel(read(root/'config.json'),read(root/'motion_groups.json'),root)
 if registered:
  s=torch.load(root/'registration.pt',map_location='cuda',weights_only=False)
  with torch.no_grad():
   for name in ['log_global_scale','screen_translation']:getattr(m,name).copy_(s[name])
 if checkpoint:m.load_state_dict(torch.load(root/'checkpoint.pt',map_location='cuda',weights_only=False)['model'])
 return m


def initialize(batch):
 batch.mkdir(parents=True,exist_ok=False);inventory=read(PREVIOUS/'input_inventory.json');run=pathlib.Path((REPO/'repro/current_gt_103015.txt').read_text().strip());prov=read(run/'input_provenance.json');source=pathlib.Path(prov['input_dir']);inventory.append(dict(case='103015',run=str(run),provenance=str(run/'input_provenance.json'),mask=str(next((source/'Sample').glob('*_mask.exr'))),rgb=prov['source_image'],scene=prov['source_data_root'],frame=0))
 write_json(batch/'input_inventory.json',inventory)
 mappings=copy.deepcopy(legacy.LABEL_OWNERS);mappings['103015']={1:10,2:11,3:12,4:13}
 # Semantic task annotation: only family/attachment side, no numeric GT axes.
 roles={'103015':{'11':'upward_door','12':'upward_door','13':'upward_door'}}
 write_json(batch/'protocol.json',dict(cases=CASES,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),previous=str(PREVIOUS),label_to_owner=mappings,semantic_role_annotations=roles,
  training='Original conditioning RGB + EXACT existing oracle instance mask + camera; original pose q0; fresh100 registration and240 geometry steps. No GT joints/geometry/depth or heldout RGB in fitting.',
  motion_rules='Evidence-gated carrying-handle component/attachment rule runs on EVERY case, derives rotation axis from separated supports, optional canonical snap within12deg. Fixed door/drawer grips follow their parent. Upward doors use upper attachment edge and outward lifting sign. No case-ID branches in reusable rules.',
  new_case_selection='103015 upper three outward lifting windows selected from source RGB motion; only frame0 enters optimization. Selection frames175/299 were not passed to fitting. No GT axis used to set priors.',
  semantics='Native PAct semantic names retained as evidence; explicit instance semantic names/family annotations permitted as previous experiment. Declared handle semantic + paired supports can correct a mistaken native door class; shape alone cannot override a declared door/drawer. No claim of learned semantic recognition.',
  geometry='Original geometry/point count/topology conserved before image fit. Same bounded deformation and original-mask objective as previous batch; all joints frozen during geometry fit.',
  source_sha256=sha256(__file__),frozen_before_fit_source_sha256={name:sha256(REPO/'repro/tools'/name) for name in ['general_semantic_multicase.py','semantic_motion_rules.py','test_semantic_motion_rules.py']}))
 snapshot=batch/'source_snapshot';snapshot.mkdir()
 for name in ['general_semantic_multicase.py','semantic_motion_rules.py','test_semantic_motion_rules.py','original_mask_multicase.py','original_mask_semantic_100481.py','multicase_closed_common.py','evaluate_multicase_closed.py','render_multicase_closed.py','add_gt_multicase_videos.py']:
  shutil.copyfile(REPO/'repro/tools'/name,snapshot/name)
 protected={str(PREVIOUS/name):sha256(PREVIOUS/name) for name in ['FINAL_SEAL.json','protocol.json','REPORT.md','aggregate.json']}
 for case in legacy.CASES:
  for name in ['checkpoint.pt','motion_groups.json','fit_results.json','evaluation.json']:
   protected[str(PREVIOUS/case/name)]=sha256(PREVIOUS/case/name)
 write_json(batch/'previous_preservation.json',protected)


def prepare(root):
 batch=root.parent;case=root.name;inventory=next(i for i in read(batch/'input_inventory.json') if i['case']==case);root.mkdir();forbidden,training_reads=guard(root);raw=cv2.imread(inventory['mask'],cv2.IMREAD_UNCHANGED);raw=raw if raw.ndim==2 else raw[:,:,0];labels=raw.astype(np.uint8);Image.fromarray(labels).save(root/'original_instance_labels_800.png');asset=load_asset(inventory['run']);prov=read(inventory['provenance']);mapping={int(k):v for k,v in read(batch/'protocol.json')['label_to_owner'][case].items()}
 recipe=copy.deepcopy(recipes().get(case))
 if recipe is None:
  corr={p['input_label']:p['pact_node_id'] for p in prov['part_correspondence']};recipe=dict(body=corr[1],groups=[spec(owner,'upward_door_'+str(label),corr[label],'hinge_x',edge='upper') for label,owner in mapping.items() if owner!=10],pairs=[])
 groups,sources,files=legacy.build_prior_groups(root,case,asset,recipe)
 role_annot=read(batch/'protocol.json')['semantic_role_annotations'].get(case,{})
 for g in groups:g['motion_role']=role_annot.get(str(g['id']),'static_body' if g['id']==10 else 'existing_family_prior')
 manifest=dict(schema='general_semantic_priors_v1',case=case,source_run=inventory['run'],source_object=str(asset['object_path']),source_object_sha256=sha256(asset['object_path']),sources=sources,groups=groups,pairs=recipe['pairs'],membership_sha256=sha256(root/'membership.npz'),geometry_files=files,selection='Native geometry; generic handle evidence and declared lifting-door role applied after fresh original-frame registration')
 write_json(root/'motion_groups.json',manifest)
 scene=pathlib.Path(inventory['scene']);source=pathlib.Path(inventory['rgb'])
 if case=='20411':
  cp=REPO/'repro/inputs/freeart_partnet_20411_frame05/camera_check.json';camera=read(cp);ext=np.array(camera['extrinsic']);intr=np.array(camera['intrinsic_normalized']);world_R=camera['R_glb_to_blender']
 else:
  cp=scene/'transforms.json';camera=read(cp);c2w=np.array(camera['frames'][0]['transform_matrix']);c2w[:3,1:3]*=-1;ext=np.linalg.inv(c2w);intr=np.array([[camera['focal_x']/camera['w'],0,camera['cx']/camera['w']],[0,camera['focal_y']/camera['h'],camera['cy']/camera['h']],[0,0,1]]);world_R=R_WORLD.tolist()
 im=Image.open(source).convert('RGBA');im.save(root/'input_reference.png');im.save(root/'closed_reference.png');rgba=np.array(im.resize((256,256),Image.Resampling.LANCZOS)).astype(float)/255;alpha=rgba[:,:,3];rgb=rgba[:,:,:3]*alpha[:,:,None]+1-alpha[:,:,None];semantic800=np.zeros_like(labels)
 for label,owner in mapping.items():semantic800[labels==label]=owner
 Image.fromarray(semantic800).save(root/'semantic_ids_800.png');ids=np.array(Image.fromarray(semantic800).resize((256,256),Image.Resampling.NEAREST));semantic=np.ones((256,256,3))
 for owner in mapping.values():semantic[ids==owner]=legacy.COLORS[owner]
 Image.fromarray((semantic*255).astype(np.uint8)).save(root/'approved_semantic_target.png');np.savez_compressed(root/'observation.npz',rgb=rgb,alpha=alpha,labels=ids,semantic=semantic,extrinsics=ext,intrinsics=intr)
 palette={str(g['id']):legacy.COLORS[g['shape_owner'] if g['shape_owner'] is not None else g['id']] for g in groups}
 cfg=dict(case=case,prior_root=str(root),prior_manifest_sha256=sha256(root/'motion_groups.json'),source_image=str(source),source_image_sha256=sha256(source),reference_frame=inventory['frame'],source_camera=str(cp),camera_sha256=sha256(cp),source_scene=str(scene),world_R=world_R,resolution=256,alignment_steps=100,geometry_steps=240,palette=palette,observations_sha256=sha256(root/'observation.npz'),seed=42,annotation_boxes_original_pixels={},annotation_polygons_original_pixels={},axis_policy='Reusable semantic attachment rules; frozen before geometry fitting',training_pose='Original conditioning pose q0')
 write_json(root/'config.json',cfg)
 names={str(g['id']):g['name'] for g in groups if g['shape_owner'] is None};write_json(root/'semantic_labels.json',dict(owner_names=names,native_names={str(n['id']):n['name'] for n in asset['nodes']},source_original_mask=inventory['mask'],original_mask_changed_pixels=int((labels!=raw).sum())))
 assert np.array_equal(labels,raw)


def repartition(root,model,rows,cache,axes,role):
 manifest=read(root/'motion_groups.json');cfg=read(root/'config.json');meshes={p['node']['id']:p['mesh'] for p in model.asset['parts']};buckets={}
 for row in rows:buckets.setdefault((row['owner'],row['source']),[]).append(row['component'])
 owners=sorted({row['owner'] for row in rows});body=next(g['source_node'] for g in model.groups if g['id']==10);primary={10:(10,body)}
 for owner in owners:
  if owner==10:continue
  matches=[key for key in buckets if key[0]==owner];primary[owner]=max(matches,key=lambda key:sum(int((cache[key[1]][0]==cid).sum()) for cid in buckets[key]))
 specs=[(owner,*primary[owner]) for owner in owners];extra=[key for key in sorted(buckets) if key not in primary.values()];next_id=max(owners)+1;specs.extend((next_id+i,*key) for i,key in enumerate(extra));groups=[];arrays={}
 for gid,owner,nid in specs:
  mesh=meshes[nid];lab,glab=cache[nid];cs=buckets[(owner,nid)];sel=np.isin(lab,cs);fs=sel[mesh.faces];assert ((fs.sum(1)%3)==0).all();vi=np.flatnonzero(sel);fi=np.flatnonzero(fs[:,0]);gi=np.flatnonzero(np.isin(glab,cs));g=source_group(gid,nid,mesh,dict(vertices=len(vi),faces=len(fi),gaussians=len(gi)),-1 if gid==10 else 10 if gid in owners else owner,None if gid in owners else owner);g.update(name='fixed body' if owner==10 else role+' '+str(owner),components=cs,semantic_owner=owner,motion_role='static_body' if owner==10 else role if gid in owners else 'attached_fragment')
  for key,ids in [('vertices',vi),('faces',fi),('gaussians',gi)]:arrays[f'g{gid}_{key}']=ids.astype(np.int64)
  if gid in axes:
   info=axes[gid];v=mesh.vertices[vi]
   if role=='articulated_handle':
    B,length=radial_basis(v,info['origin'],info['direction']);native=next(n for n in model.asset['nodes'] if n['id']==info['front_source']);span=abs(float(np.diff(native['joint']['range'])[0]));assumed=native['joint']['type'] not in ['revolute','continuous'] or span<1e-6;span=90. if assumed else span;bodycenter=meshes[body].bounds.mean(0);sign=outward_sign(info['direction'],v.mean(0)-info['origin'],info['origin']-bodycenter);joint=dict(type='revolute',axis=dict(origin=info['origin'].tolist(),direction=info['direction'].tolist()),range=[0.,sign*span]);info['audit'].update(span_assumed=assumed,inherited_span=span,opening_sign=sign);g.update(kind='hinge_handle',joint=joint,deformation_basis=B.tolist(),deformation_anchor=info['origin'].tolist(),length=length)
   else:
    native=next(n for n in model.asset['nodes'] if n['id']==nid);joint,B,length,audit=top_hinged_door(v,meshes[body].vertices,native['joint']);info.update(audit=audit);g.update(kind='hinge_up',joint=joint,deformation_basis=B.tolist(),deformation_anchor=joint['axis']['origin'],length=length,edge='upper')
  groups.append(g)
 for nid,mesh in meshes.items():
  subset=[g for g in groups if g['source_node']==nid]
  for key,count in [('vertices',len(mesh.vertices)),('faces',len(mesh.faces)),('gaussians',len(cache[nid][1]))]:assert np.array_equal(np.sort(np.concatenate([arrays[g['membership_prefix']+'_'+key] for g in subset])),np.arange(count))
 np.savez_compressed(root/'membership.npz',**arrays);manifest.update(groups=groups,pairs=[],membership_sha256=sha256(root/'membership.npz'));write_json(root/'motion_groups.json',manifest);cfg.update(palette={str(g['id']):legacy.COLORS[g['semantic_owner']] for g in groups},prior_manifest_sha256=sha256(root/'motion_groups.json'));write_json(root/'config.json',cfg)
 return manifest


def upward_partition(model,semantic_ids,owners):
 meshes={p['node']['id']:p['mesh'] for p in model.asset['parts']};source={}
 for p,(xyz,_) in zip(model.parts,model.geometry(original=True)):
  nid=p['group']['source_node'];ids=model.membership[p['group']['membership_prefix']+'_gaussians'];source.setdefault(nid,[]).append((ids,xyz.cpu().numpy()))
 rows=[];cache={};centers={o:np.median(np.c_[np.where(semantic_ids==o)[1],np.where(semantic_ids==o)[0]],0) for o in owners}
 for nid,mesh in meshes.items():
  lab=component_labels(mesh);xyz=np.empty((sum(len(ids) for ids,_ in source[nid]),3))
  for ids,x in source[nid]:xyz[ids]=x
  glab=lab[cKDTree(mesh.vertices).query(xyz,workers=4)[1]];cache[nid]=(lab,glab)
  for cid in np.unique(lab):
   v=mesh.vertices[lab==cid];uv,depth=project(model,v[::max(1,len(v)//6000)]);pix=np.rint(uv).astype(int);ok=(depth>0)&(pix[:,0]>=0)&(pix[:,0]<800)&(pix[:,1]>=0)&(pix[:,1]<800);seen=semantic_ids[pix[ok,1],pix[ok,0]];votes={o:int((seen==o).sum()) for o in [10,*owners]};best=max(votes,key=votes.get);owner=best if best in owners and votes[best]>.5*len(seen) else 10
   rows.append(dict(source=nid,component=int(cid),owner=owner,mask_votes=votes))
 supported={r['owner'] for r in rows};missing=set(owners)-supported
 if missing:raise ValueError('Missing independent upper door geometry; preserve failure: '+str(missing))
 return rows,cache,{o:{} for o in owners}


def prior(root):
 forbidden,reads=guard(root);manifest=read(root/'motion_groups.json');resolved,grip_audit=resolve_parented_grips(manifest['groups']);manifest['groups']=resolved;write_json(root/'motion_groups.json',manifest);cfg=read(root/'config.json');cfg['prior_manifest_sha256']=sha256(root/'motion_groups.json');write_json(root/'config.json',cfg);torch.set_num_threads(4);m=load_model(root,registered=True);m.requires_grad_(False);semantic=np.array(Image.open(root/'semantic_ids_800.png'));names=read(root/'semantic_labels.json')['owner_names'];up=[g['id'] for g in m.groups if g.get('motion_role')=='upward_door'];audit={}
 if up:
  rows,cache,axes=upward_partition(m,semantic,up);manifest=repartition(root,m,rows,cache,axes,'upward_door');audit=dict(dispatched='upward_door',accepted=True,component_evidence=rows,axes={str(gid):info['audit'] for gid,info in axes.items()})
 else:
  result,evidence=discover_handles(m,semantic,names);audit=dict(dispatched='handle_evidence_check',**evidence)
  if result is not None:
   manifest=repartition(root,m,result['rows'],result['cache'],result['axes'],'articulated_handle');audit['axes']={str(k):dict(origin=v['origin'].tolist(),direction=v['direction'].tolist(),attachments=np.asarray(v['attachments']).tolist(),automatic_mask_endpoint=v['mask_pixel'].tolist(),**v['audit']) for k,v in result['axes'].items()}
  else:manifest=m.manifest
 write_json(root/'axis_prior_results.json',dict(**audit,fixed_grips=grip_audit,geometry_changed=False,GT_forbidden_reads=forbidden,training_file_access=sorted(set(reads))))
 print('PRIOR',root.name,audit['dispatched'],audit['accepted'],flush=True)


def registration(root):
 legacy.load_model=load_model;legacy.training_guard=guard;legacy.registration(root)


def fit(root):
 legacy.load_model=load_model;legacy.training_guard=guard;legacy.basket.load_model=load_model;legacy.basket.training_guard=guard;legacy.basket.mask_scores=legacy.mask_scores;legacy.basket.fit(root)
 data=read(root/'fit_results.json');data.update(case=root.name,training_frame=read(root/'config.json')['reference_frame'],source_tool_sha256=sha256(__file__));write_json(root/'fit_results.json',data)


def evaluate(root):
 import evaluate_multicase_closed as mod
 mod.Model=GeneralModel;mod.evaluate_case(root)


def render(root):
 import render_multicase_closed as mod
 source=pathlib.Path(mod.__file__).read_text().replace("'optimized':'Closed-image shape fit'","'optimized':'Original-mask geometry fit'");custom=types.ModuleType('general_rule_renderer');custom.__file__=mod.__file__;exec(compile(source,mod.__file__,'exec'),custom.__dict__);custom.Model=GeneralModel;custom.render_case(root)


def gt(root):
 import add_gt_multicase_videos as mod
 if read(root/'axis_prior_results.json').get('accepted') and read(root/'axis_prior_results.json')['dispatched']=='handle_evidence_check':
  Base=mod.GT
  class ReferenceGT(Base):
   def __init__(self,path):
    super().__init__(path)
    for row in self.records:
     initial=row['mesh_frame0_value'];a=row['reference_endpoint_a'];row['reference_endpoint_a']=initial;row['open_endpoint_b']=initial+(a-initial+np.pi)%(2*np.pi)-np.pi
  mod.GT=ReferenceGT
 mod.TITLES=['GT (dataset mesh)','Origin / Original PAct','General motion prior','Original-mask geometry fit'];mod.render_case(root)


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--stage',default='all');p.add_argument('--case');a=p.parse_args();batch=pathlib.Path(a.root).resolve()
 if a.stage=='initialize':initialize(batch);return
 if a.case:globals()[a.stage](batch/a.case);return
 if not batch.exists():initialize(batch)
 states=read(batch/'controller_status.json') if (batch/'controller_status.json').exists() else {};markers=dict(prepare='config.json',registration='registration_results.json',prior='axis_prior_results.json',fit='fit_results.json',evaluate='evaluation.json',render='render_results.json',gt='gt_video_results.json')
 for stage,marker in markers.items():
  if stage=='evaluate':
   assert all((batch/c/'fit_results.json').exists() for c in CASES)
   run=pathlib.Path((REPO/'repro/current_gt_103015.txt').read_text().strip())
   if not (run/'joint_evaluation/joint_errors.json').exists():
    with (batch/'new_case_GT_evaluation.log').open('w') as log:
     result=subprocess.run([sys.executable,str(REPO/'repro/tools/evaluate_gt_joints.py'),'--run',str(run)],cwd=REPO,stdout=log,stderr=subprocess.STDOUT);assert result.returncode==0
  for case in CASES:
   out=batch/case;key=case+':'+stage
   if (out/marker).exists():states[key]='complete';continue
   env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENCV_IO_ENABLE_OPENEXR='1')
   if stage=='prepare':logpath=batch/f'prepare_{case}.log'
   else:logpath=out/f'{stage}.log'
   with logpath.open('w') as log:proc=subprocess.run([sys.executable,__file__,'--root',str(batch),'--case',case,'--stage',stage],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
   states[key]='complete' if proc.returncode==0 else 'failed';write_json(batch/'controller_status.json',states);print(key,states[key],flush=True)
   if proc.returncode:raise RuntimeError(key+' failed')
 states['state']='complete';write_json(batch/'controller_status.json',states)


if __name__=='__main__':main()
