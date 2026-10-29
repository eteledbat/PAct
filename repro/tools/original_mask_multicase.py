"""Exact original instance masks + explicit semantic articulation priors, ten cases.
Training reads only original conditioning RGB/mask/camera and native PAct geometry.
Old closed-image experiments are read ONLY by independent post-seal evaluation.
"""
import argparse,copy,datetime,json,os,pathlib,shutil,subprocess,sys,time
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
import cv2,numpy as np,torch
from PIL import Image,ImageDraw,ImageFont
from plyfile import PlyData
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree
from collision_refinement_common import load_asset,sha256,transforms
from multicase_closed_common import Model,REPO,CovGaussians,tensor,write_json,image_metrics,component_labels,recipes,spec
from semantic_motion_prior_45194 import UNDO
from fit_closed_image_45194 import R_WORLD
import original_mask_semantic_100481 as basket

CASES=['100481','101808','10489','10655','20411','31249','45194','45503','8961','9016']
BASE=REPO/'repro/outputs/closed_multicase_20261028_v1'
LABEL_OWNERS={
 '100481':{1:10,2:11,3:12},'101808':{1:10,2:11,3:12},
 '10489':{1:10,2:12,3:11},'10655':{1:10,2:12,3:11},'20411':{1:10,2:11},
 '31249':{1:10,2:14,3:13,4:11,5:12},'45194':{1:10,2:12,3:11,4:14,5:13},
 '45503':{1:10,2:13,3:11,4:12},'8961':{1:10,2:11,3:12},'9016':{1:10,2:12,3:11}}
COLORS={10:[.52,.55,.58],11:[.91,.20,.18],12:[.15,.35,.94],13:[.15,.75,.30],14:[.75,.20,.80]}
FONT=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
def read(p):return json.loads(pathlib.Path(p).read_text())

def training_guard(root):
 forbidden=[];reads=[];cfg=read(root/'config.json') if (root/'config.json').exists() else None
 allowed_rgb=cfg['source_image'] if cfg else None
 def hook(event,args):
  if event=='open' and isinstance(args[0],(str,bytes)):
   name=os.fsdecode(args[0]);denied=any(x in name for x in ['/gt/','/gt_mesh/','mobility_v2','joint_infos','joint_info.json','joint_evaluation','/depth/','/semantic_parts_100481_45194_','/closed_multicase_20261028_v1/'])
   if '/images/' in name and name.endswith('.png') and not name.endswith('/000000.png'):denied=True
   if '/FreeArt3D/' in name and name.endswith('_pure.png') and not name.endswith('/05_pure.png'):denied=True
   if denied:forbidden.append(name);raise PermissionError('Unapproved training data: '+name)
   if name.endswith(('.png','.exr','.pt','.npz','.json')):reads.append(name)
 sys.addaudithook(hook);return forbidden,reads

class OriginalMaskModel(Model):
 def __init__(self,cfg,manifest,root):
  super().__init__(cfg,manifest,root);labels=np.load(root/'observation.npz')['labels'];self.targets=tensor(labels);self.visible_owners=sorted(int(k) for k in np.unique(labels) if k)
  self.owner_map={g['id']:(g['shape_owner'] if g['shape_owner'] is not None else g['id']) for g in self.groups}
  self.interior={gid:tensor(binary_erosion(labels==gid,iterations=1))>.5 for gid in self.visible_owners}
 def probabilities(self,geometry=None,method='prior',reference=False,extrinsics=None,intrinsics=None):
  geometry=self.geometry() if geometry is None else geometry;xyz=[];cov=[];alpha=[];scale=self.log_global_scale.exp();shift=self.translation_basis@self.screen_translation;tr=self.original_closed_transforms if method=='original' else self.closed_transforms
  for part,(x,c) in zip(self.parts,geometry):
   g=part['group'];T=np.eye(4) if reference else tr[g['source_node'] if method=='original' else g['id']];A=self.R@tensor(T[:3,:3]);xyz.append(scale*(x@A.T+tensor(T[:3,3])@self.R.T)+shift);cov.append(scale.square()*(A[None]@c@A.T[None]));alpha.append(part['alpha'])
  gs=CovGaussians(torch.cat(xyz).contiguous(),torch.cat(cov).contiguous(),torch.cat(alpha).contiguous());result=[]
  for start in range(0,len(self.visible_owners),3):
   owners=self.visible_owners[start:start+3];colors=[]
   for p,(x,_) in zip(self.parts,geometry):
    color=torch.ones(3,device='cuda');owner=self.owner_map[p['group']['id']]
    if owner in owners:color[owners.index(owner)]=0
    colors.append(color.expand(len(x),3))
   rgb=self.renderer.render(gs,self.extrinsics if extrinsics is None else extrinsics,self.intrinsics if intrinsics is None else intrinsics,colors_overwrite=torch.cat(colors).contiguous())['color'];result.extend((1-rgb[:len(owners)]).unbind(0))
  return torch.stack(result).clamp(0,1)
 def losses(self,align=False):
  base,values=super().losses(align=align)
  if align:return base,values
  probs=self.probabilities();dices=[];ces=[]
  for j,gid in enumerate(self.visible_owners):
   target=(self.targets==gid).float();p=probs[j];dices.append(1-(2*(p*target).sum()+1)/(p.sum()+target.sum()+1))
   if self.interior[gid].any():ces.append(-p[self.interior[gid]].clamp_min(1e-5).log().mean())
  dice=torch.stack(dices).mean();ce=torch.stack(ces).mean();return base+.5*dice+.1*ce,dict(**values,part_dice=float(dice.detach()),balanced_CE=float(ce.detach()))
 @torch.no_grad()
 def project(self):
  super().project()
  if self.config['case']=='100481':
   for p in self.parts:
    if p['kind']=='hinge_z':self.log_shape[p['parameter_index'],1]=0;self.vertical_shift[p['parameter_index']]=0

def build_prior_groups(root,case,asset,recipe):
    byid={p['node']['id']:p for p in asset['parts']}
    groups=[spec(10,'fixed_body',recipe['body'],'fixed',parent=-1)]+copy.deepcopy(recipe['groups'])
    arrays={};source_cache={};body=byid[recipe['body']]['mesh'];body_tree=cKDTree(body.vertices);body_center=body.bounds.mean(0)
    for g in groups:
        nid=g['source_node'];part=byid[nid];mesh=part['mesh']
        if nid not in source_cache:
            lab=component_labels(mesh);plypath=asset['object_path'].parent/part['node']['gs'][0]
            v=PlyData.read(plypath)['vertex'].data
            xyz=np.column_stack([v[k] for k in ['x','y','z']]).astype(np.float64)@UNDO.T
            glab=lab[cKDTree(mesh.vertices).query(xyz,workers=4)[1]]
            source_cache[nid]=dict(mesh=mesh,lab=lab,glab=glab,plypath=plypath,gs_count=len(v))
        cache=source_cache[nid];lab=cache['lab']
        select=np.ones(len(mesh.vertices),bool) if g['components'] is None else np.isin(lab,g['components'])
        faces=select[mesh.faces];assert (faces.sum(1)%3==0).all(),'Component split crosses a face'
        vi=np.flatnonzero(select);fi=np.flatnonzero(faces[:,0]);gi=np.arange(cache['gs_count']) if g['components'] is None else np.flatnonzero(np.isin(cache['glab'],g['components']))
        prefix=f'g{g["id"]}';g['membership_prefix']=prefix
        for key,ids in [('vertices',vi),('faces',fi),('gaussians',gi)]:arrays[prefix+'_'+key]=ids.astype(np.int64)
        g['counts']={'vertices':len(vi),'faces':len(fi),'gaussians':len(gi)}
        vertices=np.asarray(mesh.vertices)[vi];center=vertices.mean(0)
        kind=g['kind'];origin=np.zeros(3);axis=np.zeros(3);B=np.eye(3);closed_q=0.;span=0.;assumed=False
        if kind=='hinge_y':
            horizontal=vertices[:,[0,2]];hc=horizontal.mean(0)
            _,ev=np.linalg.eigh(np.cov(horizontal.T));w=ev[:,-1];w*=np.sign(w[0])
            u=(horizontal-hc)@w;candidates=[]
            for q in [.01,.99]:
                xz=hc+np.quantile(u,q)*w;y=np.linspace(*np.quantile(vertices[:,1],[.1,.9]),31)
                line=np.c_[np.full_like(y,xz[0]),y,np.full_like(y,xz[1])]
                candidates.append((float(np.median(body_tree.query(line)[0])),xz))
            # Contact identifies opened leaves extending beyond their hinge.
            # Near ties use the outer-edge prior, rather than axis/GT errors.
            best=min(candidates,key=lambda x:x[0]);side=-1 if center[0]<body_center[0] else 1
            near=[x for x in candidates if x[0]<=best[0]+.01*asset['diagonal']]
            chosen=max(near,key=lambda x:side*x[1][0]);xz=chosen[1]
            origin=np.array([xz[0],np.median(vertices[:,1]),xz[1]]);axis=np.array([0.,1.,0.])
            toward=hc-xz
            if w@toward<0:w=-w
            width=np.array([w[0],0.,w[1]])
            B=np.column_stack([width,axis,np.cross(width,axis)])
            hinge_side=-1 if origin[0]<body_center[0] else 1
            target=0. if hinge_side<0 else np.pi
            closed_q=float(np.degrees(np.arctan2(w[1],w[0])-target))
            closed_q=(closed_q+180)%360-180
            old=part['node']['joint'];span=abs(float(np.diff(old['range'])[0])) if old['type'] in ['revolute','continuous'] else 90.
            assumed=old['type'] not in ['revolute','continuous'];span=max(span,1e-6)
            limits=[closed_q,closed_q+hinge_side*span]
            g['hinge_candidates']=[{'distance_to_body':d,'xz':z.tolist()} for d,z in candidates]
        elif kind=='hinge_x':
            yz=vertices[:,[1,2]];c=yz.mean(0);_,ev=np.linalg.eigh(np.cov(yz.T));w=ev[:,-1]
            edge=g.get('edge') or 'lower';ybound=np.quantile(vertices[:,1],.01 if edge=='lower' else .99)
            near=vertices[np.abs(vertices[:,1]-ybound)<=.03*max(np.ptp(vertices[:,1]),1e-3)]
            if not len(near):near=vertices
            origin=np.array([np.median(vertices[:,0]),ybound,np.median(near[:,2])]);axis=np.array([1.,0.,0.])
            toward=c-origin[[1,2]]
            if w@toward<0:w=-w
            width=np.array([0.,w[0],w[1]])
            B=np.column_stack([width,axis,np.cross(width,axis)])
            target=np.pi if edge=='upper' else 0.
            closed_q=float(np.degrees(target-np.arctan2(w[1],w[0])));closed_q=(closed_q+180)%360-180
            old=part['node']['joint'];span=abs(float(np.diff(old['range'])[0])) if old['type'] in ['revolute','continuous'] else 90.
            assumed=old['type'] not in ['revolute','continuous'];limits=[closed_q,closed_q+span]
        elif kind=='slider_z':
            origin=center.copy();axis=np.array([0.,0.,1.]);old=part['node']['joint']
            span=abs(float(np.diff(old['range'])[0])) if old['type']=='prismatic' else .25*asset['diagonal']
            assumed=old['type']!='prismatic';limits=[0.,span]
        else:limits=[0.,0.]
        typ='revolute' if kind.startswith('hinge') else ('prismatic' if kind=='slider_z' else 'fixed')
        g['joint']={'type':typ,'range':limits,'axis':{'direction':axis.tolist(),'origin':origin.tolist()}}
        g['observed_q']=closed_q;g['span_assumed']=assumed
        if not kind.startswith('hinge'):origin=vertices.min(0)/2+vertices.max(0)/2
        g['deformation_anchor']=origin.tolist();g['deformation_basis']=B.tolist()
        local=(vertices-origin)@B;g['length']=max(float(np.quantile(local[:,0],.99)),1e-3) if kind.startswith('hinge') else 1.
    for nid,cache in source_cache.items():
        sg=[g for g in groups if g['source_node']==nid]
        for kind,count in [('vertices',len(cache['mesh'].vertices)),('faces',len(cache['mesh'].faces)),('gaussians',cache['gs_count'])]:
            idx=np.concatenate([arrays[g['membership_prefix']+'_'+kind] for g in sg]);assert np.array_equal(np.sort(idx),np.arange(count)),(case,nid,kind)
    assert set(source_cache)==set(byid),'Every original part must be retained'
    np.savez_compressed(root/'membership.npz',**arrays)
    sources={str(nid):{'mesh':str(byid[nid]['mesh_path']),'gs':str(cache['plypath'])} for nid,cache in source_cache.items()}
    frozen=[{'path':p,'sha256':sha256(p)} for item in sources.values() for p in item.values()]
    # Geometry-based family/contact priors are unchanged from baseline for 9 cases.
    # Fitting is at actual generation pose q0, independent of display intervals.
    for g in groups:
        g['geometric_closed_q']=g['observed_q'];g['observed_q']=0.;g['box']=None
    return groups,sources,frozen

def prepare(root):
 case=root.name;forbidden,reads=training_guard(root);inventory=next(x for x in read(root.parent/'input_inventory.json') if x['case']==case);run=pathlib.Path(inventory['run']);asset=load_asset(run);recipe=copy.deepcopy(recipes()[case])
 labels=np.array(Image.open(root/'original_instance_labels_800.png'));raw=cv2.imread(inventory['mask'],cv2.IMREAD_UNCHANGED);raw=raw if raw.ndim==2 else raw[:,:,0];assert np.array_equal(labels,raw)
 if case=='100481':
  byid={p['node']['id']:p for p in asset['parts']};body=next(n['id'] for n in asset['nodes'] if n['joint']['type']=='fixed');groups=[];arrays={};sources={};frozen=[]
  for gid,nid in zip([10,11,12],[body,*sorted(n for n in byid if n!=body)]):
   p=byid[nid];mesh=p['mesh'];gs=asset['object_path'].parent/p['node']['gs'][0];counts=dict(vertices=len(mesh.vertices),faces=len(mesh.faces),gaussians=len(PlyData.read(gs)['vertex'].data));groups.append(basket.source_group(gid,nid,mesh,counts,-1 if gid==10 else 10));sources[str(nid)]=dict(mesh=str(p['mesh_path']),gs=str(gs))
   for key,n in counts.items():arrays[f'g{gid}_{key}']=np.arange(n,dtype=np.int64)
   frozen.extend(dict(path=str(p),sha256=sha256(p)) for p in [p['mesh_path'],gs])
  np.savez_compressed(root/'membership.npz',**arrays);pairs=[]
 else:groups,sources,frozen=build_prior_groups(root,case,asset,recipe);pairs=recipe['pairs']
 manifest=dict(schema='original_mask_multicase_v1',case=case,source_run=str(run),source_object=str(asset['object_path']),source_object_sha256=sha256(asset['object_path']),sources=sources,groups=groups,pairs=pairs,membership_sha256=sha256(root/'membership.npz'),geometry_files=frozen,selection='Existing explicit physical component/family priors, no 2D boxes; 100481 uses fresh mask-voted leg grouping after registration')
 write_json(root/'motion_groups.json',manifest)
 source=pathlib.Path(inventory['rgb']);scene=pathlib.Path(inventory['scene'])
 if case=='20411':
  camera_path=REPO/'repro/inputs/freeart_partnet_20411_frame05/camera_check.json';cam=read(camera_path);ext=np.array(cam['extrinsic']);intr=np.array(cam['intrinsic_normalized']);world_R=np.array(cam['R_glb_to_blender'])
 else:
  camera_path=scene/'transforms.json';cam=read(camera_path);frame=cam['frames'][0];assert (scene/frame['file_path']).resolve()==source.resolve();c2w=np.array(frame['transform_matrix']);c2w[:3,1:3]*=-1;ext=np.linalg.inv(c2w);intr=np.array([[cam['focal_x']/cam['w'],0,cam['cx']/cam['w']],[0,cam['focal_y']/cam['h'],cam['cy']/cam['h']],[0,0,1.]]);world_R=R_WORLD
 image=Image.open(source).convert('RGBA');image.save(root/'input_reference.png');image.save(root/'closed_reference.png')
 mapping=LABEL_OWNERS[case];semantic_ids=np.zeros_like(labels)
 for k,owner in mapping.items():semantic_ids[labels==k]=owner
 restored=np.zeros_like(labels)
 for k,owner in mapping.items():restored[semantic_ids==owner]=k
 assert np.array_equal(labels,restored);Image.fromarray(semantic_ids).save(root/'semantic_ids_800.png')
 rgba=np.array(image.resize((256,256),Image.Resampling.LANCZOS)).astype(np.float32)/255.;alpha=rgba[:,:,3];rgb=rgba[:,:,:3]*alpha[:,:,None]+1-alpha[:,:,None];ids=np.array(Image.fromarray(semantic_ids).resize((256,256),Image.Resampling.NEAREST));semantic=np.ones((256,256,3),np.float32)
 for owner in mapping.values():semantic[ids==owner]=COLORS[owner]
 Image.fromarray((semantic*255).astype(np.uint8)).save(root/'approved_semantic_target.png');np.savez_compressed(root/'observation.npz',rgb=rgb,alpha=alpha,labels=ids,semantic=semantic,extrinsics=ext,intrinsics=intr)
 palette={str(g['id']):COLORS[g['shape_owner'] if g['shape_owner'] is not None else g['id']] for g in groups}
 cfg=dict(case=case,prior_root=str(root),prior_manifest_sha256=sha256(root/'motion_groups.json'),source_image=str(source),source_image_sha256=sha256(source),reference_frame=inventory['frame'],source_camera=str(camera_path),camera_sha256=sha256(camera_path),source_scene=str(scene),world_R=world_R.tolist(),resolution=256,alignment_steps=100,geometry_steps=240,palette=palette,observations_sha256=sha256(root/'observation.npz'),seed=42,annotation_boxes_original_pixels={},annotation_polygons_original_pixels={},axis_policy='Explicit semantic family direction and generated-geometry attachment; 100481 original-mask endpoint fit. Freeze before geometry optimization.',GT_access='Existing original oracle 2D instance mask is authorized input. No GT mesh/joint/depth/validation RGB or old checkpoint in training.',training_pose='Actual original conditioning pose q0; frame05 for20411, frame0 for other9')
 write_json(root/'config.json',cfg)
 names={10:'fixed basket and lining' if case=='100481' else 'fixed body'};names.update({g['id']:g['name'] for g in groups if g['id']!=10})
 if case=='100481':names.update({11:'left handle',12:'right handle'})
 sem={str(k):dict(owner=owner,name=names[owner],pixels=int((labels==k).sum())) for k,owner in mapping.items()};write_json(root/'semantic_labels.json',dict(source_original_mask=inventory['mask'],source_mask_sha256=sha256(inventory['mask']),label_to_semantics=sem,changed_pixels=0,annotation='Only existing instance IDs named by original RGB; no selected regions'))
 maskpic=np.full((800,800,3),255,np.uint8)
 for owner in mapping.values():maskpic[semantic_ids==owner]=(np.array(COLORS[owner])*255).astype(np.uint8)
 rgbpic=Image.fromarray((rgb*255).astype(np.uint8)).resize((400,400));maskpic=Image.fromarray(maskpic).resize((400,400));overlay=Image.blend(rgbpic,maskpic,.4);panel=Image.new('RGB',(1200,470),'white');d=ImageDraw.Draw(panel)
 for j,(title,pic) in enumerate(zip(['Original conditioning RGB','Exact original mask + semantics','Overlay'],[rgbpic,maskpic,overlay])):panel.paste(pic,(j*400,28));d.text((j*400+5,5),title,font=FONT,fill='black')
 for j,(k,value) in enumerate(sem.items()):d.text((5+(j%3)*400,435+(j//3)*17),f'{k}: {value["name"]}',font=FONT,fill='black')
 panel.save(root/'original_mask_semantic_comparison.png');write_json(root/'mask_integrity.json',dict(original_mask_changed_pixels=0,semantic_id_remap=mapping,source_mask_sha256=sha256(inventory['mask']),native_label_copy_sha256=sha256(root/'original_instance_labels_800.png'),GT_forbidden_reads=forbidden))
 if case!='100481':write_json(root/'axis_prior_results.json',dict(method='Same explicit physical grouping/family/contact priors as closed_multicase for9 nonbasket cases',geometry_changed=False,GT_forbidden_reads=forbidden,all_observed_q_zero=True,manual_2D_regions_used=False))
 print('PREPARED',case,flush=True)

def load_model(root,registered=False,checkpoint=False):
 m=OriginalMaskModel(read(root/'config.json'),read(root/'motion_groups.json'),root)
 if registered:
  state=torch.load(root/'registration.pt',map_location='cuda',weights_only=False)
  with torch.no_grad():
   for name in ['log_global_scale','screen_translation']:getattr(m,name).copy_(state[name])
 if checkpoint:m.load_state_dict(torch.load(root/'checkpoint.pt',map_location='cuda',weights_only=False)['model'])
 return m

def registration(root):
 forbidden,reads=training_guard(root);torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42);torch.backends.cuda.matmul.allow_tf32=False;m=load_model(root);m.log_shape.requires_grad_(False);m.vertical_shift.requires_grad_(False);opt=torch.optim.Adam([{'params':[m.log_global_scale],'lr':.008},{'params':[m.screen_translation],'lr':.006}]);history=[]
 for step in range(100):
  opt.zero_grad(set_to_none=True);loss,values=m.losses(align=True);assert torch.isfinite(loss);loss.backward();opt.step();m.project();history.append(dict(step=step+1,loss=float(loss.detach()),**values))
 torch.save({name:getattr(m,name).detach() for name in ['log_global_scale','screen_translation']},root/'registration.pt');m.preview('registered_native_source');write_json(root/'registration_results.json',dict(history=history,scale=float(m.log_global_scale.exp()),translation=m.screen_translation.detach().cpu().tolist(),GT_forbidden_reads=forbidden,training_file_access=sorted(set(reads)),checkpoint_sha256=sha256(root/'registration.pt')));print('REGISTERED',root.name,flush=True)

def prior(root):
 if root.name!='100481':return
 # The exact approved single-case algorithm, rerun from fresh registration.
 basket.load_model=load_model;basket.training_guard=training_guard;basket.prior(root)

def mask_scores(m,geometry):
 probs=m.probabilities(geometry).detach().cpu().numpy();bg=np.clip(1-probs.sum(0),0,1);pred=np.array([0,*m.visible_owners])[np.concatenate([bg[None],probs]).argmax(0)];target=m.targets.cpu().numpy();rows=[]
 for gid in m.visible_owners:
  a=pred==gid;b=target==gid;rows.append(dict(id=gid,IoU=float((a&b).sum()/max((a|b).sum(),1))))
 return dict(rows=rows,mean_IoU=float(np.mean([r['IoU'] for r in rows])))

def fit(root):
 # Same final-step240 optimizer as approved100481, generalized to visible IDs.
 basket.training_guard=training_guard;basket.load_model=load_model;basket.mask_scores=mask_scores;basket.fit(root)
 data=read(root/'fit_results.json');data.update(case=root.name,source_tool_sha256=sha256(__file__),training_frame=read(root/'config.json')['reference_frame']);write_json(root/'fit_results.json',data);write_json(root/'status.json',dict(state='complete',**data))

def render(root):
 import render_multicase_closed as mod
 # Change title only; source renderer behavior preserved in an isolated module.
 import types
 code=pathlib.Path(mod.__file__).read_text().replace("'optimized':'Closed-image shape fit'","'optimized':'Original-mask geometry fit'")
 custom=types.ModuleType('original_mask_renderer');custom.__file__=mod.__file__;exec(compile(code,mod.__file__,'exec'),custom.__dict__);custom.Model=OriginalMaskModel;custom.render_case(root)
 data=read(root/'render_results.json');data['training_pose']='Original conditioning q0, NOT video closed start';data['range_note']='Origin raw intervals. For9 nonbasket cases, same geometric closed -> inherited-span prior intervals as baseline.100481 uses approved input->outward native span. Fixed unlock assumptions explicitly preserved.';write_json(root/'render_results.json',data);shutil.copyfile(root/'closed_target_and_three_versions.png',root/'input_target_and_three_versions.png')
 asset=read(root/'optimized_asset/object.json');asset['optimization']='Original conditioning RGB and unchanged original instance mask; axes and topology frozen';write_json(root/'optimized_asset/object.json',asset)

def gt_video(root):
 if root.name=='100481':basket.gt_video(root);return
 import add_gt_multicase_videos as mod
 if root.name=='20411':
  OriginalGT=mod.GT
  class InputPoseGT(OriginalGT):
   def __init__(self,path):
    super().__init__(path)
    # Base class samples actual00->05 meshes; native fitted rest is05, but
    # predicted animation0 starts05 and adds positive native travel as baseline.
    # GT stays the true00->05 trajectory; deliberately no prediction range fix.
  mod.GT=InputPoseGT
 mod.TITLES=['GT (dataset mesh)','Origin / Original PAct','Motion prior','Original-mask geometry fit']
 mod.render_case(root)
 data=read(root/'gt_video_results.json');data['fourth_method']='Original-mask geometry fit';data['training_pose']='Actual original conditioning pose q0; display intervals unchanged from baseline for9 nonbasket cases';write_json(root/'gt_video_results.json',data)

def initialize(batch):
 protocol=dict(cases=CASES,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),stages=['GT','Origin','Motion prior','Original-mask geometry fit'],changed_native_mask_pixels=0,label_to_owner=LABEL_OWNERS,manual_2D_regions_used=False,previous_fit_checkpoints_used=False,training_observation='Only original generation input RGB + exact existing instance mask + its camera; frame05 for20411, frame0 for other9',semantics='Explicit human names on existing instance IDs. Original masks are existing oracle/GT-projected segmentation, not automatic masks.',motion_prior='9 nonbasket cases reuse existing physical-component/family/attachment rules for controlled comparison;100481 reruns approved geometry leg + original-mask-vote grouping and automatic mask-endpoint pivot fit. No GT axis fitting.',registration='Fresh100 original RGB/silhouette steps; geometry fixed; no copied old camera registration.',geometry='Final240 bounded geometry steps. RGB+2 silhouette+2 palette+.5 mean-instance Dice+.1 balanced-interior CE+shape regularizer. No hand boxes/polygons. Axes/types/tree/ranges frozen.',evaluation='All10 fits sealed before GT evaluation. Same6 held-out native reference-pose cameras as baseline;20411 unavailable. Compare both original published metrics and frozen common-new-registration old/new metrics; no evaluation-driven checkpoint choice.',baseline=str(BASE),code_sha256=sha256(__file__),GT_policy='Existing2D oracle input mask allowed. GT meshes/joints read only by post-seal evaluator/render. All cases included regardless of results.')
 write_json(batch/'protocol.json',protocol);snapshot=batch/'source_snapshot';snapshot.mkdir(exist_ok=True)
 for name in ['original_mask_multicase.py','original_mask_semantic_100481.py','multicase_closed_common.py','evaluate_multicase_closed.py','render_multicase_closed.py','add_gt_multicase_videos.py','render_closed_image_45194.py']:shutil.copyfile(REPO/'repro/tools'/name,snapshot/name)
 # Freeze all baseline outputs for later read-only comparison integrity audit.
 files=[p for p in BASE.rglob('*') if p.is_file()];write_json(batch/'baseline_hashes.json',{str(p):sha256(p) for p in files})
 print('PROTOCOL FROZEN',flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--case',choices=CASES);p.add_argument('--stage',default='all',choices=['all','initialize','prepare','registration','prior','fit','evaluate','render','gt','compare']);a=p.parse_args();batch=pathlib.Path(a.root).resolve()
 if a.stage=='initialize':initialize(batch);return
 if a.case:
  root=batch/a.case
  if a.stage=='evaluate':
   assert all((batch/c/'fit_results.json').exists() for c in CASES),'Seal every case before GT evaluation'
   import evaluate_multicase_closed as mod
   mod.Model=OriginalMaskModel;mod.evaluate_case(root);return
  if a.stage=='compare':
   from compare_original_mask_multicase import compare_case
   compare_case(root);return
  globals()[a.stage if a.stage!='gt' else 'gt_video'](root);return
 if not (batch/'protocol.json').exists():initialize(batch)
 states=read(batch/'controller_status.json') if (batch/'controller_status.json').exists() else {};markers={'prepare':'config.json','registration':'registration_results.json','prior':'axis_prior_results.json','fit':'fit_results.json','evaluate':'evaluation.json','render':'render_results.json','gt':'gt_video_results.json','compare':'baseline_comparison.json'}
 for stage in markers:
  for case in CASES:
   if stage=='prior' and case!='100481':continue
   root=batch/case;key=case+':'+stage
   if (root/markers[stage]).exists():states[key]='complete';continue
   env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENCV_IO_ENABLE_OPENEXR='1')
   with (root/f'{stage}.log').open('w') as f:proc=subprocess.run([sys.executable,__file__,'--root',str(batch),'--case',case,'--stage',stage],cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT)
   states[key]='complete' if proc.returncode==0 else 'failed';write_json(batch/'controller_status.json',states);print(key,states[key],flush=True)
   if proc.returncode:raise RuntimeError(key+' failed; inspect log')
 states['state']='complete';write_json(batch/'controller_status.json',states);print('BATCH COMPLETE',batch,flush=True)

if __name__=='__main__':main()
