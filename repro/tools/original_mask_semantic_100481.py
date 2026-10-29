"""100481: unchanged original instance mask -> semantic axes -> geometry fit.

Only the approved frame-0 mask/RGB and native generated geometry enter fitting.
No hand polygons, hand rivets, prior closed-image checkpoints, GT joints/meshes,
or held-out images enter prepare/registration/prior/geometry workers.
"""
import argparse,copy,datetime,json,os,pathlib,shutil,subprocess,sys,time
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from plyfile import PlyData
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree
import torch
from collision_refinement_common import load_asset,sha256,transforms
from multicase_closed_common import Model,REPO,CovGaussians,tensor,write_json,image_metrics,component_labels
from fit_closed_image_45194 import R_WORLD

APPROVED=REPO/'repro/outputs/mask_semantics_100481_20261028'
OWNERS=[10,11,12]
COLORS={10:[.52,.55,.58],11:[.91,.20,.18],12:[.15,.35,.94]}
NAMES={10:'fixed basket including lining',11:'left handle',12:'right handle'}

def read(p):return json.loads(pathlib.Path(p).read_text())

def training_guard(root):
    forbidden=[];reads=[]
    def hook(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)):
            name=os.fsdecode(args[0])
            denied=any(x in name for x in ['/gt/','/gt_mesh/','mobility_v2','joint_infos','joint_info.json','joint_evaluation','/depth/','/semantic_parts_100481_45194_','/closed_multicase_20261028_v1/'])
            if '/images/' in name and name.endswith('.png') and not name.endswith('/000000.png'):denied=True
            if denied:forbidden.append(name);raise PermissionError('Unapproved observation or GT read: '+name)
            if name.endswith(('.png','.exr','.npz','.pt','.json')):reads.append(name)
    sys.addaudithook(hook)
    return forbidden,reads

class OriginalMaskModel(Model):
    def __init__(self,cfg,manifest,root):
        super().__init__(cfg,manifest,root)
        labels=np.load(root/'observation.npz')['labels'];self.targets=tensor(labels)
        self.visible_owners=OWNERS;self.owner_map={g['id']:g['shape_owner'] if g['shape_owner'] is not None else g['id'] for g in self.groups}
        self.interior={gid:tensor(binary_erosion(labels==gid,iterations=1))>.5 for gid in OWNERS}

    def probabilities(self,geometry=None,method='prior'):
        geometry=self.geometry() if geometry is None else geometry;xyz=[];cov=[];alpha=[];colors=[];s=self.log_global_scale.exp();shift=self.translation_basis@self.screen_translation
        tr=self.original_closed_transforms if method=='original' else self.closed_transforms
        for part,(x,c) in zip(self.parts,geometry):
            g=part['group'];T=tr[g['source_node'] if method=='original' else g['id']];A=self.R@tensor(T[:3,:3]);xyz.append(s*(x@A.T+tensor(T[:3,3])@self.R.T)+shift);cov.append(s.square()*(A[None]@c@A.T[None]));alpha.append(part['alpha'])
            color=torch.ones(3,device='cuda');color[OWNERS.index(self.owner_map[g['id']])]=0;colors.append(color.expand(len(x),3))
        gs=CovGaussians(torch.cat(xyz).contiguous(),torch.cat(cov).contiguous(),torch.cat(alpha).contiguous());rgb=self.renderer.render(gs,self.extrinsics,self.intrinsics,colors_overwrite=torch.cat(colors).contiguous())['color']
        return (1-rgb).clamp(0,1)

    def losses(self,align=False):
        base,values=super().losses(align=align)
        if align:return base,values
        probs=self.probabilities();dices=[];ces=[]
        for j,gid in enumerate(OWNERS):
            target=(self.targets==gid).float();p=probs[j];dices.append(1-(2*(p*target).sum()+1)/(p.sum()+target.sum()+1));ces.append(-p[self.interior[gid]].clamp_min(1e-5).log().mean())
        dice=torch.stack(dices).mean();ce=torch.stack(ces).mean();return base+.5*dice+.1*ce,dict(**values,part_dice=float(dice.detach()),balanced_CE=float(ce.detach()))

    @torch.no_grad()
    def project(self):
        super().project()
        for p in self.parts:
            if p['kind']=='hinge_z':
                i=p['parameter_index'];self.log_shape[i,1]=0;self.vertical_shift[i]=0


def source_group(gid,nid,mesh,counts,parent=-1,owner=None):
    return dict(id=gid,name=f'native source {nid}',source_node=nid,kind='fixed',parent=parent,components=None,shape_owner=owner,box=None,edge=None,
       membership_prefix=f'g{gid}',counts=counts,joint={'type':'fixed','range':[0.,0.],'axis':{'direction':[0.,0.,0.],'origin':[0.,0.,0.]}},observed_q=0.,span_assumed=False,
       deformation_anchor=mesh.bounds.mean(0).tolist(),deformation_basis=np.eye(3).tolist(),length=1.)


def prepare(batch):
    batch.mkdir(parents=True,exist_ok=False);root=batch/'100481';root.mkdir();forbidden,reads=training_guard(root)
    approved=read(APPROVED/'semantic_labels.json');source_rgb=pathlib.Path(approved['source_rgb']);source_labels=APPROVED/'100481_original_instance_labels.png';labels=np.array(Image.open(source_labels));assert labels.shape==(800,800) and set(np.unique(labels))=={0,1,2,3}
    # Verify this indexed PNG was the exact original EXR, already approved.
    assert approved['changed_label_pixels']==0 and approved['mask_exactly_equal_to_original']
    run=pathlib.Path((REPO/'repro/current_gt_100481.txt').read_text().strip());asset=load_asset(run);byid={p['node']['id']:p for p in asset['parts']};body=next(n['id'] for n in asset['nodes'] if n['joint']['type']=='fixed')
    groups=[];arrays={};sources={};files=[]
    order=[body,*sorted(nid for nid in byid if nid!=body)]
    for gid,nid in zip([10,11,12],order):
        p=byid[nid];mesh=p['mesh'];gs=asset['object_path'].parent/next(n['gs'][0] for n in asset['nodes'] if n['id']==nid);n=len(PlyData.read(gs)['vertex'].data);counts=dict(vertices=len(mesh.vertices),faces=len(mesh.faces),gaussians=n)
        groups.append(source_group(gid,nid,mesh,counts,-1 if gid==10 else 10));sources[str(nid)]=dict(mesh=str(p['mesh_path']),gs=str(gs))
        for key,count in counts.items():arrays[f'g{gid}_{key}']=np.arange(count,dtype=np.int64)
        files.extend([dict(path=str(path),sha256=sha256(path)) for path in [p['mesh_path'],gs]])
    np.savez_compressed(root/'membership.npz',**arrays)
    scene=source_rgb.parents[1];camera_path=scene/'transforms.json';cam=read(camera_path);frame=cam['frames'][0];assert (scene/frame['file_path']).resolve()==source_rgb.resolve();c2w=np.array(frame['transform_matrix']);c2w[:3,1:3]*=-1;extrinsics=np.linalg.inv(c2w)
    intr=np.array([[cam['focal_x']/cam['w'],0,cam['cx']/cam['w']],[0,cam['focal_y']/cam['h'],cam['cy']/cam['h']],[0,0,1.]])
    source=Image.open(source_rgb).convert('RGBA');source.save(root/'input_reference.png');source.save(root/'closed_reference.png');source.save(root/'approved_input_RGB.png')
    shutil.copyfile(source_labels,root/'original_instance_labels_800.png');semantic_ids=np.where(labels==1,10,np.where(labels==2,11,np.where(labels==3,12,0))).astype(np.uint8);Image.fromarray(semantic_ids).save(root/'semantic_ids_800.png')
    restored=np.where(semantic_ids==10,1,np.where(semantic_ids==11,2,np.where(semantic_ids==12,3,0))).astype(np.uint8);assert np.array_equal(restored,labels)
    rgba=np.array(source.resize((256,256),Image.Resampling.LANCZOS)).astype(np.float32)/255.;alpha=rgba[:,:,3];rgb=rgba[:,:,:3]*alpha[:,:,None]+1-alpha[:,:,None];ids=np.array(Image.fromarray(semantic_ids).resize((256,256),Image.Resampling.NEAREST));semantic=np.ones((256,256,3),np.float32)
    for gid,color in COLORS.items():semantic[ids==gid]=color
    Image.fromarray((semantic*255).astype(np.uint8)).save(root/'approved_semantic_target.png')
    np.savez_compressed(root/'observation.npz',rgb=rgb,alpha=alpha,labels=ids,semantic=semantic,extrinsics=extrinsics,intrinsics=intr)
    manifest=dict(schema='original_mask_semantic_100481_v1',case='100481',source_run=str(run),source_object=str(asset['object_path']),source_object_sha256=sha256(asset['object_path']),sources=sources,groups=groups,pairs=[],geometry_files=files,membership_sha256=sha256(root/'membership.npz'),selection='Native source grouping for RGB-only registration, before semantic ownership and axes')
    write_json(root/'motion_groups.json',manifest)
    cfg=dict(case='100481',prior_root=str(root),prior_manifest_sha256=sha256(root/'motion_groups.json'),source_image=str(source_rgb),source_image_sha256=sha256(source_rgb),reference_frame=0,source_camera=str(camera_path),camera_sha256=sha256(camera_path),source_scene=str(scene),world_R=R_WORLD.tolist(),resolution=256,alignment_steps=100,geometry_steps=240,palette=COLORS,observations_sha256=sha256(root/'observation.npz'),seed=42,training_pose='Actual original input frame0 with handles upright, NOT a closed-frame mask',annotation_polygons_original_pixels={},axis_policy='Semantic family normal-to-front axes; numeric locations from generated leg attachments and endpoints extracted from approved mask; freeze after prior stage',GT_access='Only the existing approved oracle INSTANCE mask is an input; no GT geometry, GT joints, depth, or held-out images in fitting.')
    write_json(root/'config.json',cfg);shutil.copyfile(APPROVED/'100481_original_mask_semantic_comparison.png',root/'approved_mask_comparison.png')
    write_json(batch/'protocol.json',dict(cases=['100481'],created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),stages=['Origin','Motion prior','Original-mask geometry fit'],approved_mask=str(source_labels),approved_mask_sha256=sha256(source_labels),changed_native_mask_pixels=0,training_observations=[str(source_rgb),str(source_labels),str(camera_path)],manual_polygons_used=False,manual_rivet_coordinates_used=False,previous_fit_checkpoints_used=False,
       preparation='Fresh native PAct export, no previous curated component recipe. Thin upright leg components classified by geometry; original mask votes assign left/right. Same-side front/back legs share a hinge.',
       axis_prior='Hard family direction +Z in canonical Y-up/front+Z frame. Pivot solves weighted attachment fit using generated lower leg endpoints (weight1 each) and automatic bottom3% original mask endpoint backprojection at generated front-leg depth (weight2). No GT axis fitting.',
       registration='Fresh100 RGB/silhouette-only steps at original frame0 with geometry fixed. Same camera registration for all variants.',geometry='Final240 steps with axes/types/tree/ranges fixed. Same bounded Gaussian/mesh deformation family; no new points, faces, opacity or textures. Axis thickness and axis shifts frozen for handles.',
       objective='RGB weighted L1 + 2 silhouette L1 + 2 semantic palette MSE + .5 mean visible-instance Dice + .1 balanced interior CE + small shape regularizer; no hand boxes.',
       animation='All methods start at the actual input pose. Origin keeps native raw intervals. Both corrected handles inherit the front-leg source absolute span, left outward +Z/right outward -Z. GT starts at frame0 and moves to closest equivalent annotated endpoint a, only for post-fit visualization. Progress synchronized, angular amplitudes may differ; no inferred mechanical closed limit for predictions.',GT_policy=cfg['GT_access'],code_sha256=sha256(__file__)))
    write_json(root/'mask_integrity.json',dict(approved_mask_sha256=sha256(source_labels),native_label_copy_sha256=sha256(root/'original_instance_labels_800.png'),changed_pixels=0,class_pixel_counts={str(k):int((labels==k).sum()) for k in [1,2,3]},semantic_only_id_relabel={1:10,2:11,3:12},source_RGB_sha256=sha256(source_rgb),GT_forbidden_reads=forbidden))
    print('PREPARED',root,flush=True)


def load_model(root,registered=False,checkpoint=False):
    cfg=read(root/'config.json');manifest=read(root/'motion_groups.json');m=OriginalMaskModel(cfg,manifest,root)
    if registered:
        state=torch.load(root/'registration.pt',map_location='cuda',weights_only=False)
        with torch.no_grad():m.log_global_scale.copy_(state['log_global_scale']);m.screen_translation.copy_(state['screen_translation'])
    if checkpoint:m.load_state_dict(torch.load(root/'checkpoint.pt',map_location='cuda',weights_only=False)['model'])
    return m


def registration(root):
    forbidden,reads=training_guard(root);torch.set_num_threads(4);torch.manual_seed(42);m=load_model(root);m.log_shape.requires_grad_(False);m.vertical_shift.requires_grad_(False);opt=torch.optim.Adam([{'params':[m.log_global_scale],'lr':.008},{'params':[m.screen_translation],'lr':.006}]);history=[]
    for step in range(100):
        opt.zero_grad(set_to_none=True);loss,values=m.losses(align=True);assert torch.isfinite(loss);loss.backward();opt.step();m.project();history.append(dict(step=step+1,loss=float(loss.detach()),**values))
    torch.save({name:getattr(m,name).detach() for name in ['log_global_scale','screen_translation']},root/'registration.pt');m.preview('registered_native_source');write_json(root/'registration_results.json',dict(history=history,scale=float(m.log_global_scale.exp()),translation=m.screen_translation.detach().cpu().tolist(),GT_forbidden_reads=forbidden,checkpoint_sha256=sha256(root/'registration.pt')));print('REGISTERED',history[-1],flush=True)


def project_native(m,points):
    R=m.R.detach().cpu().numpy();s=float(m.log_global_scale.exp());shift=(m.translation_basis@m.screen_translation).detach().cpu().numpy();ext=m.extrinsics.cpu().numpy();intr=m.intrinsics.cpu().numpy();world=s*(points@R.T)+shift;cam=world@ext[:3,:3].T+ext[:3,3];uv=cam[:,:2]/cam[:,2:3];uv=uv*np.array([intr[0,0],intr[1,1]])+intr[:2,2];return uv*800,cam[:,2]


def unproject(m,pixel,z):
    ext=m.extrinsics.cpu().numpy();intr=m.intrinsics.cpu().numpy();R=m.R.cpu().numpy();s=float(m.log_global_scale.exp());shift=(m.translation_basis@m.screen_translation).detach().cpu().numpy();ray=np.linalg.solve(intr,np.r_[pixel/800,1.]);wo=-ext[:3,:3].T@ext[:3,3];wd=ext[:3,:3].T@ray;o=(wo-shift)@R/s;d=wd@R/s;return o+((z-o[2])/d[2])*d


def prior(root):
    forbidden,reads=training_guard(root);torch.set_num_threads(4);m=load_model(root,registered=True);m.requires_grad_(False);manifest=read(root/'motion_groups.json');cfg=read(root/'config.json');asset=m.asset;meshes={p['node']['id']:p['mesh'] for p in asset['parts']};parts={p['group']['source_node']:p for p in m.parts};labels=np.array(Image.open(root/'original_instance_labels_800.png'));bounds=np.array([v.bounds for v in meshes.values()]);height=float(bounds[:,1,1].max()-bounds[:,0,1].min());rows=[];cache={}
    mask_x={label:float(np.median(np.where(labels==label)[1])) for label in [2,3]}
    for nid,mesh in meshes.items():
        lab=component_labels(mesh);p=parts[nid];xyz=(p['local']@p['B'].T+p['anchor']).cpu().numpy();glab=lab[cKDTree(mesh.vertices).query(xyz,workers=4)[1]];cache[nid]=(lab,glab)
        for cid in np.unique(lab):
            v=np.asarray(mesh.vertices)[lab==cid];span=np.ptp(v,axis=0);leg=span[1]>2*max(span[0],span[2]) and span[1]>.15*height;owner=10;votes={};why='Non-leg component remains fixed basket'
            if leg:
                points=v[::max(1,len(v)//4000)];uv,depth=project_native(m,points);pixel=np.rint(uv).astype(int);valid=(depth>0)&(pixel[:,0]>=0)&(pixel[:,0]<800)&(pixel[:,1]>=0)&(pixel[:,1]<800);seen=labels[pixel[valid,1],pixel[valid,0]];votes={label:int((seen==label).sum()) for label in [1,2,3]};best=max([2,3],key=lambda k:votes[k]);other=5-best
                if votes[best]>max(.1*len(seen),2*votes[other]):owner=11 if best==2 else 12;why='Upright thin leg, original mask vote'
                else:
                    x=float(np.median(uv[:,0]));label=min([2,3],key=lambda k:abs(x-mask_x[k]));owner=11 if label==2 else 12;why='Upright thin leg, projected side matched to original semantic mask'
            rows.append(dict(source=nid,component=int(cid),owner=owner,is_leg=bool(leg),span=span.tolist(),center=v.mean(0).tolist(),mask_votes=votes,rule=why))
    assert all(sum(r['owner']==gid for r in rows)>=2 for gid in [11,12]),'Need front and back leg evidence for each handle'
    buckets={}
    for row in rows:buckets.setdefault((row['owner'],row['source']),[]).append(row['component'])
    body_source=next(n['id'] for n in asset['nodes'] if n['joint']['type']=='fixed');primary={10:(10,body_source)}
    for owner in [11,12]:
        front=max([r for r in rows if r['owner']==owner],key=lambda r:r['center'][2]);primary[owner]=(owner,front['source'])
    specs=[(owner,*primary[owner]) for owner in OWNERS];extras=[key for key in sorted(buckets) if key not in primary.values()];specs.extend((13+i,*key) for i,key in enumerate(extras));groups=[];arrays={};axis_audit={}
    for gid,owner,nid in specs:
        mesh=meshes[nid];lab,glab=cache[nid];components=buckets[(owner,nid)];select=np.isin(lab,components);fs=select[mesh.faces];assert (fs.sum(1)%3==0).all();vi=np.flatnonzero(select);fi=np.flatnonzero(fs[:,0]);gi=np.flatnonzero(np.isin(glab,components));counts=dict(vertices=len(vi),faces=len(fi),gaussians=len(gi));g=source_group(gid,nid,mesh,counts,-1 if gid==10 else 10 if gid in [11,12] else owner,None if gid in OWNERS else owner);g.update(components=components,name=NAMES[owner]+(' auxiliary fragment' if gid not in OWNERS else ''),semantic_owner=owner)
        for suffix,ids in [('vertices',vi),('faces',fi),('gaussians',gi)]:arrays[f'g{gid}_{suffix}']=ids.astype(np.int64)
        groups.append(g)
    for owner in [11,12]:
        selected=[r for r in rows if r['owner']==owner];attachments=[]
        for r in selected:
            mesh=meshes[r['source']];lab=cache[r['source']][0];v=np.asarray(mesh.vertices)[lab==r['component']];low=v[v[:,1]<=np.quantile(v[:,1],.03)];attachments.append(np.median(low,axis=0))
        front=max(range(len(selected)),key=lambda j:selected[j]['center'][2]);front_depth=float(attachments[front][2]);label=2 if owner==11 else 3;yy,xx=np.where(labels==label);low=yy>=np.quantile(yy,.97);pixel=np.array([np.median(xx[low]),np.median(yy[low])]);observed=unproject(m,pixel,front_depth)
        attachments=np.array(attachments);origin=np.r_[(attachments[:,:2].sum(0)+2*observed[:2])/(len(attachments)+2),attachments[:,2].mean()];axis=np.array([0.,0.,1.]);g=next(g for g in groups if g['id']==owner);v=np.asarray(meshes[g['source_node']].vertices)[arrays[g['membership_prefix']+'_vertices']];xy=v[:,:2];_,evec=np.linalg.eigh(np.cov(xy.T));radial=np.r_[evec[:,-1],0.]
        if radial[1]<0:radial=-radial
        B=np.column_stack([radial,axis,np.cross(radial,axis)]);length=max(float(np.quantile((v-origin)@radial,.99)),.05);raw=next(n for n in asset['nodes'] if n['id']==g['source_node']);span=abs(float(np.diff(raw['joint']['range'])[0]));assert raw['joint']['type']=='revolute' and span>0
        g.update(kind='hinge_z',deformation_anchor=origin.tolist(),deformation_basis=B.tolist(),length=length,joint={'type':'revolute','range':[0.,span if owner==11 else -span],'axis':{'direction':axis.tolist(),'origin':origin.tolist()}},observed_q=0.,semantic_constraint=dict(class_name='handle',axis_prior='normal to basket front',attachment_fit='all generated leg lower endpoints + automatically extracted original-mask lower endpoint'))
        axis_audit[str(owner)]=dict(generated_attachments=attachments.tolist(),automatically_extracted_mask_endpoint=pixel.tolist(),mask_endpoint_unprojection=observed.tolist(),axis_origin=origin.tolist(),axis_direction=axis.tolist(),geometry_attachment_weight=1.,mask_endpoint_weight=2.,geometry_perpendicular_residual=float(np.sqrt(np.mean(np.sum((attachments[:,:2]-origin[:2])**2,axis=1)))),inherited_source=raw['id'],native_absolute_span=span)
    partitions=[]
    for nid,mesh in meshes.items():
        sg=[g for g in groups if g['source_node']==nid]
        for suffix,n in [('vertices',len(mesh.vertices)),('faces',len(mesh.faces)),('gaussians',len(cache[nid][1]))]:
            ids=np.concatenate([arrays[g['membership_prefix']+'_'+suffix] for g in sg]);assert np.array_equal(np.sort(ids),np.arange(n))
        partitions.append(dict(source=nid,vertices=len(mesh.vertices),faces=len(mesh.faces),gaussians=len(cache[nid][1])))
    np.savez_compressed(root/'membership.npz',**arrays);manifest.update(groups=groups,membership_sha256=sha256(root/'membership.npz'),selection='Geometry leg-shape prior and original-mask semantic votes; zero hand polygons or hand coordinate annotations',axis_reference_pose='All q=0 at observed frame0; animation is native-span outward sweep, not a calibrated closed stop');write_json(root/'motion_groups.json',manifest)
    cfg['palette']={str(g['id']):COLORS[g.get('semantic_owner',g['id'])] for g in groups};cfg['prior_manifest_sha256']=sha256(root/'motion_groups.json');write_json(root/'config.json',cfg)
    write_json(root/'axis_prior_results.json',dict(component_classification=rows,axes=axis_audit,source_partitions=partitions,geometry_changed=False,GT_forbidden_reads=forbidden,approved_mask_sha256=sha256(root/'original_instance_labels_800.png'),registration_sha256=sha256(root/'registration.pt')));print('AXIS PRIOR COMPLETE',axis_audit,flush=True)


def mask_scores(m,geometry):
    probs=m.probabilities(geometry).cpu().numpy();bg=np.clip(1-probs.sum(0),0,1);pred=np.array([0,*OWNERS])[np.concatenate([bg[None],probs]).argmax(0)];target=m.targets.cpu().numpy();rows=[]
    for gid in OWNERS:
        a=pred==gid;b=target==gid;rows.append(dict(id=gid,IoU=float((a&b).sum()/max((a|b).sum(),1))))
    return dict(rows=rows,mean_IoU=float(np.mean([r['IoU'] for r in rows])))


def fit(root):
    forbidden,reads=training_guard(root);torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42);torch.backends.cuda.matmul.allow_tf32=False;m=load_model(root,registered=True);m.log_global_scale.requires_grad_(False);m.screen_translation.requires_grad_(False);m.preview('before_geometry_fit');start=time.monotonic();history=[];opt=torch.optim.Adam([{'params':[m.log_shape],'lr':.012},{'params':[m.vertical_shift],'lr':.004}]);manifest_before=sha256(root/'motion_groups.json');obs_before=sha256(root/'observation.npz')
    for step in range(240):
        opt.zero_grad(set_to_none=True);loss,values=m.losses();assert torch.isfinite(loss);loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters());opt.step();m.project();history.append(dict(step=step+1,loss=float(loss.detach()),**values))
        if (step+1)%40==0:print('GEOMETRY FIT',history[-1],flush=True)
    m.preview('after_geometry_fit');torch.save({'model':m.state_dict(),'optimizer':opt.state_dict(),'stage':'complete','step':240},root/'checkpoint.pt');results={};part_metrics={};export={}
    with torch.no_grad():
        for method in ['original','prior','optimized']:
            geom=m.geometry(original=method!='optimized');which='original' if method=='original' else 'prior';results[method]=image_metrics(m.render(geom,method=which),m.render(geom,'silhouette',method=which),m.rgb,m.alpha)
            part_metrics[method]=mask_scores(m,geom)
        for i,p in enumerate(m.parts):
            prefix=p['group']['membership_prefix'];xyz,cov=m.deform(i);export[prefix+'_xyz']=xyz.cpu().numpy();export[prefix+'_covariance']=cov.cpu().numpy();export[prefix+'_mesh_vertices']=m.deform(i,tensor(p['vertex_local'])).cpu().numpy()
    np.savez_compressed(root/'optimized_parts.npz',**export);write_json(root/'history.json',history);assert sha256(root/'motion_groups.json')==manifest_before and sha256(root/'observation.npz')==obs_before and not forbidden;assert all(sha256(f['path'])==f['sha256'] for f in m.manifest['geometry_files'])
    seal=dict(case='100481',image_metrics=results,part_mask_metrics=part_metrics,checkpoint_sha256=sha256(root/'checkpoint.pt'),optimized_parts_sha256=sha256(root/'optimized_parts.npz'),config_sha256=sha256(root/'config.json'),manifest_sha256=manifest_before,observation_sha256=obs_before,GT_forbidden_reads=forbidden,training_file_access=sorted(set(reads)),fit_seconds=time.monotonic()-start,source_tool_sha256=sha256(__file__),registered_scale=float(m.log_global_scale.exp()),translation_parameters=m.screen_translation.detach().cpu().tolist(),pose={str(g['id']):0. for g in m.groups},training_frame=0)
    write_json(root/'fit_results.json',seal);write_json(root/'status.json',dict(state='complete',**seal));print('FIT COMPLETE',results,part_metrics,flush=True)


def render(root):
    import render_multicase_closed as mod
    mod.Model=OriginalMaskModel;mod.render_case(root)
    data=read(root/'render_results.json');data['range_note']='All variants start at observed input q0; original native intervals retained; corrected handles use each native front-leg absolute span with opposite outward signs. No closed pose or closed limit inferred.';data['training_frame']=0;data['fourth_method']='Original-mask geometry fit';write_json(root/'render_results.json',data)
    shutil.copyfile(root/'closed_target_and_three_versions.png',root/'input_target_and_three_versions.png')
    asset=read(root/'optimized_asset/object.json');asset['optimization']='Original frame0 RGB and exact original instance mask, bounded part shape fit; axes and topology frozen';asset['rest_pose']='Observed original input pose q0';write_json(root/'optimized_asset/object.json',asset)


def gt_video(root):
    import add_gt_multicase_videos as mod
    class ReferenceGT(mod.GT):
        def __init__(self,path):
            super().__init__(path)
            for row in self.records:
                initial=row['mesh_frame0_value'];a=row['reference_endpoint_a'];delta=(a-initial+np.pi)%(2*np.pi)-np.pi
                row['source_original_annotated_a']=a;row['source_original_annotated_b']=row['open_endpoint_b'];row['reference_endpoint_a']=initial;row['open_endpoint_b']=initial+delta;row['range_source']='Actual observed frame0 -> closest equivalent native annotated endpoint a; GT visualization only, not model fitting'
    mod.GT=ReferenceGT;mod.TITLES=['GT (dataset mesh)','Origin / Original PAct','Motion prior','Original-mask geometry fit']
    def label(array,index,t,diagnostic=False):
        pic=Image.fromarray(array.copy());d=ImageDraw.Draw(pic);d.rectangle((0,0,383,47),fill='white');d.text((8,5),mod.TITLES[index],font=mod.FONT,fill='black');detail=['Input pose -> GT folded limit','Native original interval','Semantic axis / native span','Same axis / geometry fit'][index];d.text((8,27),f'{detail} | {t:.2f}',font=mod.FONT,fill='#555555');return np.array(pic)
    mod.label=label;mod.render_case(root);data=read(root/'gt_video_results.json');data['phase']='All start at observed frame0. GT frame0->native annotated endpoint a, using shortest equivalent angle. Origin keeps raw native interval. Prior/fit use inherited absolute front-leg span and semantic outward signs. Common progress, different true/predicted angle amplitudes.';data['training_observation']='Approved original input mask/RGB frame0; fourth column is original-mask geometry fit';write_json(root/'gt_video_results.json',data)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--stage',choices=['all','prepare','registration','prior','fit','evaluate','render','gt'],default='all');a=p.parse_args();batch=pathlib.Path(a.root).resolve();root=batch/'100481'
    if a.stage=='prepare':prepare(batch);return
    if a.stage=='registration':registration(root);return
    if a.stage=='prior':prior(root);return
    if a.stage=='fit':fit(root);return
    if a.stage=='evaluate':
        import evaluate_multicase_closed as mod
        mod.Model=OriginalMaskModel;mod.evaluate_case(root);return
    if a.stage=='render':render(root);return
    if a.stage=='gt':gt_video(root);return
    if not batch.exists():prepare(batch)
    snapshot=batch/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for name in ['original_mask_semantic_100481.py','multicase_closed_common.py','evaluate_multicase_closed.py','render_multicase_closed.py','add_gt_multicase_videos.py','render_closed_image_45194.py']:
        shutil.copyfile(REPO/'repro/tools'/name,snapshot/name)
    states={}
    for stage,marker in [('registration','registration_results.json'),('prior','axis_prior_results.json'),('fit','fit_results.json'),('evaluate','evaluation.json'),('render','render_results.json'),('gt','gt_video_results.json')]:
        if (root/marker).exists():states[stage]='complete';continue
        env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4')
        with (root/f'{stage}.log').open('w') as log:proc=subprocess.run([sys.executable,__file__,'--root',str(batch),'--stage',stage],env=env,cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
        states[stage]='complete' if proc.returncode==0 else 'failed';write_json(batch/'controller_status.json',states);print(stage,states[stage],flush=True)
        if proc.returncode:raise RuntimeError(f'{stage} failed')
    write_json(batch/'controller_status.json',dict(state='complete',stages=states));print('ALL COMPLETE',batch,flush=True)

if __name__=='__main__':main()
