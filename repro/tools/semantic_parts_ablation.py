"""RGB-only semantic mask / kinematic ablation on 100481 and 45194.

No GT geometry, joint, depth, or held-out RGB reads in prepare/prior/fit.
Origin geometry and all source files are preserved. Explicit reviewed visible
polygons are annotations, not a learned semantic detector.
"""
import argparse,copy,datetime,json,os,pathlib,shutil,subprocess,sys,time
import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.ndimage import binary_erosion
import torch
from collision_refinement_common import sha256,transforms
from multicase_closed_common import Model,REPO,CovGaussians,tensor,write_json,image_metrics

BASE=REPO/'repro/outputs/closed_multicase_20261028_v1'
CASES=['100481','45194']
POLYGONS={
 '100481':{11:[[141,646],[275,488],[304,512],[165,667]],12:[[487,512],[512,487],[660,645],[635,669]]},
 '45194':{11:[[177,280],[395,280],[395,536],[177,536]],12:[[400,280],[619,280],[619,536],[400,536]],
          13:[[145,173],[394,173],[395,262],[145,262]],14:[[400,173],[650,173],[650,262],[399,262]]}}
NAMES={'100481':{10:'fixed basket + lining',11:'left visible handle',12:'right visible handle'},
       '45194':{10:'fixed cabinet / frame',11:'left hinged door + grip',12:'right hinged door + grip',13:'left sliding drawer + insert',14:'right sliding drawer'}}


def read(p):return json.loads(p.read_text())


def guard_gt():
    reads=[]
    def hook(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)):
            name=os.fsdecode(args[0])
            if any(s in name for s in ['/gt/','/gt_mesh/','mobility_v2','joint_infos','joint_info.json','joint_evaluation','/depth/','evaluation.json']):
                reads.append(name);raise PermissionError('GT forbidden in semantic preparation/fitting: '+name)
    sys.addaudithook(hook)
    return reads


def registration(root,model):
    checkpoint=torch.load(BASE/model.config['case']/'checkpoint.pt',map_location='cuda',weights_only=False)
    with torch.no_grad():
        for name in ['log_global_scale','screen_translation']:
            getattr(model,name).copy_(checkpoint['model'][name])
    model.log_global_scale.requires_grad_(False);model.screen_translation.requires_grad_(False)


def set_closed(model,values):
    nodes=copy.deepcopy(model.groups)
    for node in nodes:
        q=float(values.get(node['id'],node['observed_q']));node['joint']['range']=[q,q]
    model.closed_transforms=transforms(nodes,1.)


class SemanticModel(Model):
    def __init__(self,config,manifest,root):
        super().__init__(config,manifest,root)
        labels=np.load(root/'observation.npz')['labels']
        self.targets=tensor(labels)
        self.visible_owners=sorted(int(x) for x in np.unique(labels) if x)
        self.owner_map={g['id']:(g['shape_owner'] if g['shape_owner'] is not None else g['id']) for g in self.groups}
        # Keep Origin's closed-state convention exactly the baseline convention.
        baseline=read(BASE/config['case']/'motion_groups.json');original=copy.deepcopy(self.asset['nodes']);by_source={}
        for g in baseline['groups']:
            if g['kind']!='fixed':by_source.setdefault(g['source_node'],g['observed_q'])
        for g in original:g['joint']['range']=[by_source.get(g['id'],0.)]*2
        self.original_closed_transforms=transforms(original,1.)
        self.interior={gid:tensor(binary_erosion(labels==gid,iterations=1))>.5 for gid in self.visible_owners}

    def probabilities(self,geometry=None,method='prior',reference=False,extrinsics=None,intrinsics=None):
        geometry=self.geometry() if geometry is None else geometry
        xyz,cov,alpha=[],[],[];s=self.log_global_scale.exp();translation=self.translation_basis@self.screen_translation
        tr=self.original_closed_transforms if method=='original' else self.closed_transforms
        for p,(x,c) in zip(self.parts,geometry):
            nid=p['group']['source_node'] if method=='original' else p['group']['id'];T=np.eye(4) if reference else tr[nid]
            A=self.R@tensor(T[:3,:3]);xyz.append(s*(x@A.T+tensor(T[:3,3])@self.R.T)+translation)
            cov.append(s.square()*(A[None]@c@A.T[None]));alpha.append(p['alpha'])
        gs=CovGaussians(torch.cat(xyz).contiguous(),torch.cat(cov).contiguous(),torch.cat(alpha).contiguous())
        result=[]
        for start in range(0,len(self.visible_owners),3):
            owners=self.visible_owners[start:start+3];colors=[]
            for p,(x,_) in zip(self.parts,geometry):
                color=torch.ones(3,device='cuda');owner=self.owner_map[p['group']['id']]
                if owner in owners:color[owners.index(owner)]=0
                colors.append(color.expand(len(x),3))
            rgb=self.renderer.render(gs,self.extrinsics if extrinsics is None else extrinsics,self.intrinsics if intrinsics is None else intrinsics,colors_overwrite=torch.cat(colors).contiguous())['color']
            result.extend((1-rgb[:len(owners)]).unbind(0))
        return torch.stack(result).clamp(0,1)

    def losses(self,align=False):
        geometry=self.geometry();rgb=self.render(geometry);probs=self.probabilities(geometry);alpha=probs.sum(0).clamp(0,1)
        photo=((rgb-self.rgb).abs()*self.weight[None]).mean()/self.weight.mean();sil=(alpha-self.alpha).abs().mean()
        dices=[];ces=[]
        for j,gid in enumerate(self.visible_owners):
            target=(self.targets==gid).float();p=probs[j]
            dices.append(1-(2*(p*target).sum()+1)/(p.sum()+target.sum()+1))
            interior=self.interior[gid]
            if interior.any():ces.append(-p[interior].clamp_min(1e-5).log().mean())
        dice=torch.stack(dices).mean();ce=torch.stack(ces).mean()
        prior=.002*self.log_shape.square().mean()+.01*self.vertical_shift.square().mean()
        base_loss,base_values=super().losses(align=False)
        loss=base_loss+.5*dice+.1*ce
        return loss,dict(**base_values,part_dice=float(dice.detach()),balanced_CE=float(ce.detach()))

    @torch.no_grad()
    def project(self):
        super().project()
        if self.config['case']=='100481':
            for i,p in enumerate(self.parts):
                if p['kind']=='hinge_z':
                    # Semantic handle: keep hinge-axis thickness and attachment;
                    # preserve radial deformation capacity of the old model.
                    self.log_shape[p['parameter_index'],1]=0
                    self.vertical_shift[p['parameter_index']]=0


def unproject_to_front(model,pixel,z):
    intr=model.intrinsics.cpu().numpy();ext=model.extrinsics.cpu().numpy();R=model.R.cpu().numpy();s=float(model.log_global_scale.exp());shift=(model.translation_basis@model.screen_translation).cpu().numpy()
    uv=np.array([pixel[0]/800,pixel[1]/800,1.]);ray=np.linalg.solve(intr,uv);world_o=-ext[:3,:3].T@ext[:3,3];world_d=ext[:3,:3].T@ray
    o=(world_o-shift)@R/s;d=world_d@R/s
    return o+((z-o[2])/d[2])*d


def correct_basket_ownership(out,cfg,manifest):
    """Move whole connected components, preserving every vertex/face/Gaussian.

    The native source0 is mostly a basket PANEL, source2 contains both front
    handle legs, and source1 contains both back legs. Semantic instances are
    left/right handles across depth, rather than source-file instances.
    """
    from multicase_closed_common import component_labels
    from scipy.spatial import cKDTree
    model=SemanticModel(cfg,manifest,out)
    bysource={p['group']['source_node']:p for p in model.parts}
    meshes={p['node']['id']:p['mesh'] for p in model.asset['parts']}
    specs=[(10,'fixed basket and lining',1,[0,1],-1,None),
           (11,'left handle front leg',2,[0],10,None),
           (12,'right handle front leg',2,[1],10,None),
           (13,'left handle back leg',1,[2],11,11),
           (14,'right handle back leg',1,[3],12,12),
           (15,'basket front panel',0,[0],10,10),
           (16,'left rivet fragment',0,[1],11,11),
           (17,'right rivet fragment',0,[2],12,12)]
    cache={};arrays={};groups=[]
    original={g['source_node']:g for g in manifest['groups']}
    for nid,mesh in meshes.items():
        lab=component_labels(mesh);p=bysource[nid];xyz=(p['local']@p['B'].T+p['anchor']).cpu().numpy();glab=lab[cKDTree(mesh.vertices).query(xyz,workers=4)[1]]
        cache[nid]=(lab,glab)
    for gid,name,nid,components,parent,owner in specs:
        mesh=meshes[nid];lab,glab=cache[nid];select=np.isin(lab,components);fs=select[mesh.faces];assert (fs.sum(1)%3==0).all()
        vi=np.flatnonzero(select);fi=np.flatnonzero(fs[:,0]);gi=np.flatnonzero(np.isin(glab,components));prefix=f'g{gid}'
        for key,values in [('vertices',vi),('faces',fi),('gaussians',gi)]:arrays[prefix+'_'+key]=values.astype(np.int64)
        g=copy.deepcopy(original[nid]);g.update(id=gid,name=name,source_node=nid,components=components,parent=parent,shape_owner=owner,box=None,edge=None,
          membership_prefix=prefix,counts=dict(vertices=len(vi),faces=len(fi),gaussians=len(gi)),kind='hinge_z' if gid in [11,12] else 'fixed',observed_q=0.)
        if gid not in [11,12]:g['joint']={'type':'fixed','range':[0.,0.],'axis':{'direction':[0.,0.,0.],'origin':[0.,0.,0.]}}
        groups.append(g)
    checks=[]
    for nid,mesh in meshes.items():
        srcgroups=[g for g in groups if g['source_node']==nid];row={'source':nid}
        for key,n in [('vertices',len(mesh.vertices)),('faces',len(mesh.faces)),('gaussians',len(cache[nid][1]))]:
            assigned=np.concatenate([arrays[g['membership_prefix']+'_'+key] for g in srcgroups]);assert np.array_equal(np.sort(assigned),np.arange(n));row[key]=n
        checks.append(row)
    np.savez_compressed(out/'membership.npz',**arrays);manifest['groups']=groups;manifest['membership_sha256']=sha256(out/'membership.npz')
    manifest['recipe']={'body':1,'groups':specs,'pairs':[],'note':'Whole-component semantic ownership correction; left/right front and back legs share one physical handle joint. No points/faces/Gaussians added or removed.'}
    cfg['palette'].update({str(g['id']):cfg['palette'][str(g['shape_owner'] or g['id'])] for g in groups if g['id'] not in [10,11,12]})
    write_json(out/'semantic_ownership_audit.json',dict(protocol='Manual connected-component identities reviewed against existing oracle frame0 instance mask and the closed RGB; no new GT projection/axis/depth read',
      groups=specs,source_partition_bijections=checks,geometry_changed=False,raw_source0='basket panel, not a handle',raw_source1='basket body plus two back handle legs',raw_source2='both front handle legs',
      physical_handles='Each front leg and its back leg now follow the SAME independent left/right hinge; rivet fragments follow their handle',
      unavailable_geometry='U-shaped top bridges absent in generated geometry, not synthesized'))
    del model
    return cfg,manifest


def prepare(root):
    forbidden=guard_gt();root.mkdir(exist_ok=False,parents=True)
    protocol=dict(cases=CASES,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),baseline=str(BASE),
      stages=['Origin (unchanged source)','Semantic motion prior (no geometry change)','Closed-image fit (semantic masks)'],
      training='Only the same one closed RGB/alpha and its camera; manually traced visible polygons. Baseline RGB-only registration reused and frozen. 240 final geometry steps; no checkpoint selection.',
      loss='Retain original RGB + 2 silhouette + 2 part-palette MSE + 3 front-box L1 + shape prior; add .5 mean per-instance soft Dice + .1 class-balanced interior CE. Handle boxes replaced by precise polygons because they contain basket body.',
      semantic_motion='100481 first reassigns complete connected components into fixed panel/body and independent left/right front+back handle legs. Then handle: revolute axis normal to basket front, pivot from annotated rivet projected onto generated handle depth, closed q from RGB mask-only discrete search. Native absolute spans retained. 45194: existing independent side-hinged doors and sliding drawers retained; insert follows left drawer.',
      data_note='Origin was already generated from oracle GT INSTANCE masks at frame0 (existing pipeline). New training masks are traced from CLOSED RGB, not GT projections. Semantic names are explicit human priors, not an automatic semantic model.',
      geometry='Same generated points, covariance, opacity, color, meshes and topology. 100481 membership is corrected by whole-component semantic ownership, preserving bijections of all points/faces/Gaussians. 45194 membership unchanged. Same bounded part deformation as baseline; handle thickness/axial shift frozen. No new handle/door geometry synthesized.',
      GT_access='No GT mesh, joint, depth or held-out RGB during preparation/prior/fit; audit hook fails such reads. Evaluation/render GT allowed only after fit seals.',
      weights_shared_between_cases=True,source_sha256=sha256(__file__))
    write_json(root/'protocol.json',protocol)
    for case in CASES:
        src=BASE/case;out=root/case;out.mkdir()
        for name in ['membership.npz','closed_reference.png']:
            shutil.copyfile(src/name,out/name)
        cfg=read(src/'config.json');manifest=read(src/'motion_groups.json')
        obs=dict(np.load(src/'observation.npz'));rgb=np.array(Image.open(src/'closed_reference.png').convert('RGBA'));labels=np.where(rgb[:,:,3]>127,10,0).astype(np.uint8)
        for gid,polygon in POLYGONS[case].items():
            tile=Image.new('L',(800,800));ImageDraw.Draw(tile).polygon([tuple(x) for x in polygon],fill=1);select=np.array(tile)>0;labels[select&(rgb[:,:,3]>127)]=gid
        Image.fromarray(labels).save(out/'semantic_labels_800.png')
        labels256=np.array(Image.fromarray(labels).resize((256,256),Image.Resampling.NEAREST))
        obs['labels']=labels256;semantic=np.ones((256,256,3),np.float32)
        for gid in np.unique(labels256):
            if gid:semantic[labels256==gid]=cfg['palette'][str(gid)]
        obs['semantic']=semantic;np.savez_compressed(out/'observation.npz',**obs)
        Image.fromarray((semantic*255).astype('uint8')).save(out/'front_panel_annotations.png')
        cfg.update(prior_root=str(out),baseline_root=str(src),observations_sha256=sha256(out/'observation.npz'),annotation_polygons_original_pixels=POLYGONS[case],
                   semantic_names=NAMES[case],objective=protocol['loss'],alignment_steps=0,geometry_steps=240,
                   selection='Fixed final240, shared weights, no GT selection',registration_policy='Reuse and freeze original baseline RGB-only registration',GT_access=protocol['GT_access'])
        write_json(out/'config.json',cfg);write_json(out/'motion_groups.json',manifest)
        if case=='100481':
            cfg,manifest=correct_basket_ownership(out,cfg,manifest)
            write_json(out/'config.json',cfg);write_json(out/'motion_groups.json',manifest)
            model=SemanticModel(cfg,manifest,out);registration(out,model)
            pivots={11:[277,510],12:[512,510]}
            for i,g in enumerate(manifest['groups']):
                if g['id'] not in [11,12]:continue
                part=model.parts[i];v=part['vertex_local']@part['B'].cpu().numpy().T+part['anchor'].cpu().numpy();origin=unproject_to_front(model,pivots[g['id']],float(np.median(v[:,2])))
                # Radial PCA in XY, with sign toward the generated part.
                h=v[:,:2];_,ev=np.linalg.eigh(np.cov(h.T));w=ev[:,-1]
                if w@(h.mean(0)-origin[:2])<0:w=-w
                radial=np.r_[w,0.];axis=np.array([0.,0.,1.]);B=np.column_stack([radial,axis,np.cross(radial,axis)])
                length=max(float(np.quantile((v-origin)@radial,.99)),.05)
                span=abs(float(np.diff(g['joint']['range'])[0]));g.update(name=NAMES[case][g['id']],kind='hinge_z',deformation_anchor=origin.tolist(),deformation_basis=B.tolist(),length=length,observed_q=0.,box=None,
                  semantic_constraint=dict(class_name='basket handle',attachment='visible rivet on generated handle front plane',annotated_pivot_pixel=pivots[g['id']],axis_prior='front normal +Z',hidden_geometry='not inferred'))
                g['joint']={'type':'revolute','axis':{'direction':axis.tolist(),'origin':origin.tolist()},'range':[0.,(-1 if g['id']==11 else 1)*span]}
            del model
        else:
            for g in manifest['groups']:
                g['semantic_constraint']=dict(class_name='fixed cabinet' if g['id']==10 else 'door, vertical outer hinge' if g['id'] in [11,12] else 'sliding drawer, rail normal to front' if g['id'] in [13,14] else 'drawer insert, fixed to parent 13',pixel_supervision='visible RGB polygon' if g['id']!=15 else 'occluded; rigid parent only')
        manifest.update(schema='semantic_parts_mask_ablation_v1',baseline_manifest_sha256=sha256(src/'motion_groups.json'),selection='Explicit part names and RGB-only visible polygons; no GT motion values',membership_sha256=sha256(out/'membership.npz'))
        write_json(out/'motion_groups.json',manifest);cfg['prior_manifest_sha256']=sha256(out/'motion_groups.json');write_json(out/'config.json',cfg)
        annotation=dict(case=case,source_image=cfg['source_image'],polygons=POLYGONS[case],names=NAMES[case],
          pixel_counts={int(gid):int((labels256==gid).sum()) for gid in np.unique(labels256)},
          occluded_part_policy='45194 group15 has no standalone closed-image pixels; attached rigidly to drawer13 and shares its semantic owner',
          mask_semantics='Functional rigid instances; door grips are part of their door, drawer grips/insert follow drawer, basket lining is fixed body',
          annotation_source='Manual trace on only this closed RGB; alpha for foreground; no GT geometry projection',GT_forbidden_reads=forbidden)
        write_json(out/'semantic_annotations.json',annotation)
        print('PREPARED',case,flush=True)


def prior_search(root):
    forbidden=guard_gt();cfg=read(root/'config.json');manifest=read(root/'motion_groups.json');model=SemanticModel(cfg,manifest,root);registration(root,model);model.requires_grad_(False);torch.set_num_threads(4)
    values={g['id']:g['observed_q'] for g in model.groups};trace=[]
    if cfg['case']=='100481':
        with torch.no_grad():
            geom=model.geometry(original=True)
            for step,half in [(15.,180.),(3.,15.),(.75,3.)]:
                for gid in [11,12]:
                    center=0. if half==180 else values[gid];best=(float('inf'),None)
                    for q in np.arange(center-half,center+half+step*.1,step):
                        test=dict(values);test[gid]=float(q);set_closed(model,test);probs=model.probabilities(geom);j=model.visible_owners.index(gid);target=(model.targets==gid).float();p=probs[j]
                        score=float(1-(2*(p*target).sum()+1)/(p.sum()+target.sum()+1));trace.append(dict(gid=gid,q=float(q),score=score,step=step))
                        if score<best[0]:best=(score,float(q))
                    values[gid]=best[1]
                    print('SEMANTIC PRIOR',gid,step,best,flush=True)
        for g in manifest['groups']:
            if g['id'] in [11,12]:
                span=abs(float(np.diff(g['joint']['range'])[0]));g['observed_q']=values[g['id']];g['joint']['range']=[values[g['id']],values[g['id']]+(-1 if g['id']==11 else 1)*span]
    write_json(root/'motion_groups.json',manifest);cfg['prior_manifest_sha256']=sha256(root/'motion_groups.json');write_json(root/'config.json',cfg)
    write_json(root/'motion_prior_search.json',dict(case=cfg['case'],closed_values=values,search=trace,geometry_unchanged=True,GT_forbidden_reads=forbidden))


def fit(root):
    forbidden=guard_gt();cfg=read(root/'config.json');manifest=read(root/'motion_groups.json');torch.manual_seed(42);np.random.seed(42);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    model=SemanticModel(cfg,manifest,root);registration(root,model);model.preview('aligned_before_geometry')
    optimizer=torch.optim.Adam([{'params':[model.log_shape],'lr':.012},{'params':[model.vertical_shift],'lr':.004}]);history=[];start=time.monotonic()
    for step in range(240):
        optimizer.zero_grad(set_to_none=True);loss,values=model.losses();assert torch.isfinite(loss);loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters());optimizer.step();model.project();history.append(dict(step=step+1,loss=float(loss.detach()),**values))
        if (step+1)%40==0:print(cfg['case'],history[-1],flush=True)
    model.preview('optimized_closed_fit');torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'stage':'complete','step':240},root/'checkpoint.pt')
    result={};export={}
    with torch.no_grad():
        for method in ['original','prior','optimized']:
            geom=model.geometry(original=method!='optimized');which='original' if method=='original' else 'prior'
            result[method]=image_metrics(model.render(geom,method=which),model.render(geom,'silhouette',method=which),model.rgb,model.alpha)
        for i,p in enumerate(model.parts):
            prefix=p['group']['membership_prefix'];xyz,cov=model.deform(i);export[prefix+'_xyz']=xyz.cpu().numpy();export[prefix+'_covariance']=cov.cpu().numpy();export[prefix+'_mesh_vertices']=model.deform(i,tensor(p['vertex_local'])).cpu().numpy()
    np.savez_compressed(root/'optimized_parts.npz',**export);write_json(root/'history.json',history)
    assert not forbidden and all(sha256(x['path'])==x['sha256'] for x in manifest['geometry_files'])
    seal=dict(case=cfg['case'],image_metrics=result,checkpoint_sha256=sha256(root/'checkpoint.pt'),optimized_parts_sha256=sha256(root/'optimized_parts.npz'),config_sha256=sha256(root/'config.json'),manifest_sha256=sha256(root/'motion_groups.json'),
      GT_forbidden_reads=forbidden,source_tool_sha256=sha256(__file__),fit_seconds=time.monotonic()-start,pose={str(g['id']):g['observed_q'] for g in model.groups},registered_scale=float(model.log_global_scale.exp()),translation_parameters=model.screen_translation.cpu().tolist())
    write_json(root/'fit_results.json',seal);write_json(root/'status.json',dict(state='complete',**seal));print('FIT COMPLETE',cfg['case'],result,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--stage',choices=['prepare','prior','fit','evaluate','render','gt']);p.add_argument('--case');a=p.parse_args();root=pathlib.Path(a.root).resolve()
    if a.stage=='prepare':prepare(root);return
    out=root/a.case
    if a.stage=='prior':prior_search(out)
    elif a.stage=='fit':fit(out)
    elif a.stage=='evaluate':
        import evaluate_multicase_closed as mod
        mod.Model=SemanticModel;mod.evaluate_case(out)
    elif a.stage=='render':
        import render_multicase_closed as mod
        mod.Model=SemanticModel;mod.render_case(out)
    elif a.stage=='gt':
        from add_gt_multicase_videos import render_case
        render_case(out)


if __name__=='__main__':main()
