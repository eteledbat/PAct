"""Case-configured semantic articulation + single closed-image part refinement.

All recipes and visible 2D annotations are fixed before fitting or GT evaluation.
They are explicit case priors, not a learned semantic instance classifier.
"""
import copy
import json
import os
import pathlib
import sys
import time

import numpy as np
from PIL import Image
from plyfile import PlyData
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import torch

from collision_refinement_common import load_asset,sha256,transforms
from fit_closed_image_45194 import CovGaussians,R_WORLD,tensor,write_json
from semantic_motion_prior_45194 import UNDO

REPO=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
from modules.pact.renderers.gaussian_render import GaussianRenderer


def spec(gid,name,source,kind,parent=10,components=None,owner=None,box=None,edge=None):
    return dict(id=gid,name=name,source_node=source,kind=kind,parent=parent,
                components=components,shape_owner=owner,box=box,edge=edge)


def recipes():
    # Pixel annotations are on the chosen 800x800 closed RGB only.
    return {
      '100481':dict(body=1,groups=[spec(11,'basket_lower_handle',0,'hinge_x',box=[145,480,350,650],edge='upper'),spec(12,'basket_upper_handle',2,'hinge_x',box=[448,480,650,650],edge='lower')],pairs=[],note='Basket stress case: horizontal handle prior; generated parts may not represent full handles.'),
      '101808':dict(body=0,groups=[spec(11,'lower_oven_door',1,'hinge_x',box=[272,455,520,650],edge='lower'),spec(12,'upper_oven_door_fragment',2,'hinge_x',box=[272,325,520,438],edge='lower')],pairs=[],note='Upper door is only a small generated fragment; retained as a failure case. Unlocking fixed fragment assumes a 90 degree visualization span.'),
      '10489':dict(body=1,groups=[spec(11,'lower_fridge_door',0,'hinge_y',box=[305,240,491,586]),spec(12,'upper_fridge_door',2,'hinge_y',box=[305,61,491,226])],pairs=[]),
      '10655':dict(body=0,groups=[spec(11,'left_door',1,'hinge_y',box=[273,176,400,634]),spec(12,'right_door',2,'hinge_y',box=[403,176,540,634])],pairs=[[11,12]]),
      '20411':dict(body=0,groups=[spec(11,'drawer',1,'slider_z',box=[286,274,484,359])],pairs=[],note='Published fixed FreeArt3D camera; closed frame00. No novel-camera set available.'),
      '31249':dict(body=2,groups=[spec(11,'left_door',0,'hinge_y',components=[0],box=[216,321,394,526]),spec(12,'right_door',0,'hinge_y',components=[1],box=[404,321,583,526]),spec(13,'left_drawer',3,'slider_z',box=[173,205,351,267]),spec(14,'right_drawer',4,'slider_z',box=[446,205,624,267]),spec(15,'right_door_duplicate_layer',1,'fixed',parent=12,components=[1],owner=12),spec(16,'center_vent',1,'fixed',components=[0],owner=10)],pairs=[[11,12],[13,14]],note='Two source door components separated; a drawer-labelled duplicate right door is attached to right hinge, central vent fixed.'),
      '45194':dict(body=1,groups=[spec(11,'left_door',0,'hinge_y',components=[0],box=[177,280,396,536]),spec(12,'right_door',0,'hinge_y',components=[1],box=[400,280,619,536]),spec(13,'left_drawer',2,'slider_z',box=[145,173,395,263]),spec(14,'right_drawer',3,'slider_z',box=[400,173,650,263]),spec(15,'left_drawer_insert',4,'fixed',parent=13,owner=13)],pairs=[[11,12],[13,14]]),
      '45503':dict(body=1,groups=[spec(11,'lower_drop_door',0,'hinge_x',box=[233,600,567,746],edge='lower'),spec(12,'right_upper_door',2,'hinge_y',box=[405,233,566,578]),spec(13,'left_upper_door',3,'hinge_y',components=[0],box=[234,233,395,578]),spec(14,'center_frame',3,'fixed',components=[1],owner=10)],pairs=[[12,13]]),
      '8961':dict(body=2,groups=[spec(11,'left_shutter',0,'hinge_y',box=[246,166,394,638]),spec(12,'right_shutter',1,'hinge_y',box=[405,166,554,638])],pairs=[[11,12]],note='Native predicted strokes below one degree are retained; a separately labelled 90-degree diagnostic is also rendered.'),
      '9016':dict(body=0,groups=[spec(11,'left_door',1,'hinge_y',box=[191,219,394,564]),spec(12,'right_door',2,'hinge_y',box=[405,219,607,564])],pairs=[[11,12]])}


def component_labels(mesh):
    e=mesh.edges_unique
    return connected_components(coo_matrix((np.ones(len(e)),(e[:,0],e[:,1])),shape=(len(mesh.vertices),)*2),directed=False)[1]


def prepare_case(batch,case,recipe):
    root=batch/case;root.mkdir()
    run=pathlib.Path((REPO/f'repro/current_gt_{case}.txt').read_text().strip())
    asset=load_asset(run);byid={p['node']['id']:p for p in asset['parts']}
    provenance=json.loads((run/'input_provenance.json').read_text());scene=pathlib.Path(provenance['source_data_root'])
    if case=='20411':
        source=scene/'00_pure.png';frame_index=0
        cm=REPO/'repro/inputs/freeart_partnet_20411_frame05/camera_check.json';cam=json.loads(cm.read_text())
        extrinsic=np.array(cam['extrinsic']);intr=np.array(cam['intrinsic_normalized'])
        world_R=np.array(cam['R_glb_to_blender']);camera_path=cm
    else:
        camera_path=scene/'transforms.json';cam=json.loads(camera_path.read_text())
        frame_index=250 if case=='45194' else len(cam['frames'])-1
        f=cam['frames'][frame_index];source=scene/f['file_path'];c2w=np.array(f['transform_matrix']);c2w[:3,1:3]*=-1
        extrinsic=np.linalg.inv(c2w);intr=np.array([[cam['focal_x']/cam['w'],0,cam['cx']/cam['w']],[0,cam['focal_y']/cam['h'],cam['cy']/cam['h']],[0,0,1]])
        world_R=R_WORLD
    original=Image.open(source).convert('RGBA');original.save(root/'closed_reference.png')
    res=256;rgba=np.asarray(original.resize((res,res),Image.Resampling.LANCZOS)).astype(np.float32)/255.
    alpha=rgba[:,:,3];rgb=rgba[:,:,:3]*alpha[:,:,None]+1-alpha[:,:,None]
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
    manifest={'schema':'multicase_semantic_motion_v1','case':case,'source_run':str(run),'source_object':str(asset['object_path']),
              'source_object_sha256':sha256(asset['object_path']),'sources':sources,'groups':groups,'pairs':recipe['pairs'],
              'membership_sha256':sha256(root/'membership.npz'),'geometry_files':frozen,'recipe':recipe,
              'selection':'Manual family and geometry-instance priors; no GT joints, no axis-error selection',
              'range_policy':'Closed pose from leaf geometry; inherited absolute native span, or explicitly assumed span if original fixed. Not calibrated mechanical limits.'}
    write_json(root/'motion_groups.json',manifest)
    labels=np.where(alpha>.5,10,0).astype(np.int32);palette={10:[.5,.5,.5]};colors=[[.9,.15,.15],[.15,.3,.95],[.15,.75,.3],[.75,.2,.8],[.8,.6,.1]]
    movable=[g for g in groups if g['kind']!='fixed']
    for j,g in enumerate(movable):
        gid=g['id'];palette[gid]=colors[j%len(colors)];x0,y0,x1,y1=np.rint(np.array(g['box'])*res/800).astype(int)
        labels[y0:y1,x0:x1]=np.where(alpha[y0:y1,x0:x1]>.5,gid,0)
    bygroup={g['id']:g for g in groups}
    def appearance_owner(g):
        if g['kind']!='fixed' or g['parent']==-1:return g['id']
        if g['shape_owner'] is not None:return appearance_owner(bygroup[g['shape_owner']])
        return appearance_owner(bygroup[g['parent']])
    for g in groups:
        if g['id'] not in palette:palette[g['id']]=palette.get(appearance_owner(g),[.5,.5,.5])
    semantic=np.ones((res,res,3),np.float32)
    for gid,color in palette.items():semantic[labels==gid]=color
    Image.fromarray((semantic*255).astype('uint8')).save(root/'front_panel_annotations.png')
    np.savez_compressed(root/'observation.npz',rgb=rgb,alpha=alpha,semantic=semantic,labels=labels,extrinsics=extrinsic,intrinsics=intr)
    config={'case':case,'prior_root':str(root),'prior_manifest_sha256':sha256(root/'motion_groups.json'),
            'source_image':str(source),'source_image_sha256':sha256(source),'reference_frame':frame_index,
            'source_camera':str(camera_path),'camera_sha256':sha256(camera_path),'source_scene':str(scene),
            'world_R':world_R.tolist(),'resolution':256,'alignment_steps':100,'geometry_steps':240,
            'annotation_boxes_original_pixels':{str(g['id']):g['box'] for g in movable},'palette':palette,
            'observations_sha256':sha256(root/'observation.npz'),'seed':42,'recipe_sha256':sha256(batch/'recipes.json'),
            'selection':'One fixed closed image, final 240 steps, no per-case tuning. Same losses/weights/bounds across cases.',
            'objective':'RGB weighted L1 + 2 silhouette L1 + 2 part-palette MSE + 3 front-box L1 + .002 shape prior',
            'axis_policy':'Semantic/attachment prior before fitting; image optimization freezes axes, types, tree and ranges',
            'GT_access':'Forbidden during fitting. Evaluation occurs in a separate process after all candidate seals.'}
    write_json(root/'config.json',config)


class Model(torch.nn.Module):
    def __init__(self,config,manifest,root):
        super().__init__();self.config=config;self.manifest=manifest;self.root=root
        self.asset=load_asset(manifest['source_run']);self.groups=manifest['groups'];self.membership=np.load(root/'membership.npz')
        self.byid={g['id']:g for g in self.groups};self.indices={g['id']:i for i,g in enumerate(self.groups)}
        self.log_global_scale=torch.nn.Parameter(tensor(np.log(1. if config['case']=='20411' else 1.7)))
        self.screen_translation=torch.nn.Parameter(tensor([0.,0.]))
        self.log_shape=torch.nn.Parameter(torch.zeros(len(self.groups),2,device='cuda'))
        self.vertical_shift=torch.nn.Parameter(torch.zeros(len(self.groups),device='cuda'))
        self.R=tensor(config['world_R']);self.parts=[];source_data={}
        mesh_byid={p['node']['id']:p['mesh'] for p in self.asset['parts']}
        def shape_owner(gid):
            g=self.byid[gid]
            return shape_owner(g['shape_owner']) if g['shape_owner'] is not None else gid
        shared={}
        for pair in manifest['pairs']:
            for gid in pair:shared[gid]=pair[0]
        for g in self.groups:
            gid=g['id'];owner=shape_owner(gid);og=self.byid[owner];param_owner=shared.get(owner,owner)
            nid=g['source_node']
            if nid not in source_data:
                v=PlyData.read(manifest['sources'][str(nid)]['gs'])['vertex'].data
                xyz=np.column_stack([v[k] for k in ['x','y','z']]).astype(np.float32)@UNDO.T
                scale=np.exp(np.column_stack([v[k] for k in ['scale_0','scale_1','scale_2']]).astype(np.float32))
                q=np.column_stack([v[k] for k in ['rot_0','rot_1','rot_2','rot_3']]);r=UNDO[None]@Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix()
                cov=(r*scale[:,None,:]**2)@r.transpose(0,2,1)
                alpha=(1/(1+np.exp(-v['opacity'].astype(np.float32))))[:,None]
                color=(.28209479177387814*np.column_stack([v[k] for k in ['f_dc_0','f_dc_1','f_dc_2']])+.5).clip(0,1)
                source_data[nid]=(xyz,cov,alpha,color)
            xyz,cov,alpha,color=[x[self.membership[g['membership_prefix']+'_gaussians']] for x in source_data[nid]]
            B=np.array(og['deformation_basis']);anchor=np.array(og['deformation_anchor'])
            verts=np.asarray(mesh_byid[nid].vertices)[self.membership[g['membership_prefix']+'_vertices']]
            p={'group':g,'B':tensor(B),'anchor':tensor(anchor),'length':og['length'],'kind':og['kind'],
               'parameter_index':self.indices[param_owner],'owner':owner,'local':tensor((xyz-anchor)@B),
               'cov_local':tensor(B.T[None]@cov@B[None]),'alpha':tensor(alpha),'colors':tensor(color),
               'vertex_local':(verts-anchor)@B}
            if g['box'] is not None:
                select=verts
                if g['kind']=='slider_z':select=select[select[:,2]>=np.quantile(select[:,2],.75)]
                select=select[::max(1,len(select)//2048)];p['front_local']=tensor((select-anchor)@B)
            self.parts.append(p)
        self.closed_nodes=copy.deepcopy(self.groups)
        for g in self.closed_nodes:g['joint']['range']=[g['observed_q']]*2
        self.closed_transforms=transforms(self.closed_nodes,1.)
        self.original_closed=copy.deepcopy(self.asset['nodes']);q_by_source={}
        for g in self.groups:
            if g['kind']!='fixed':q_by_source.setdefault(g['source_node'],g['observed_q'])
        for g in self.original_closed:g['joint']['range']=[q_by_source.get(g['id'],0.)]*2
        self.original_closed_transforms=transforms(self.original_closed,1.)
        data=np.load(root/'observation.npz');self.rgb=tensor(data['rgb']).permute(2,0,1);self.alpha=tensor(data['alpha'])
        self.semantic=tensor(data['semantic']).permute(2,0,1);self.extrinsics=tensor(data['extrinsics']);self.intrinsics=tensor(data['intrinsics'])
        self.translation_basis=self.extrinsics[:3,:3].T[:,:2]
        self.weight=1+3*self.alpha
        self.renderer=GaussianRenderer({'resolution':config['resolution'],'near':.05,'far':20.,'bg_color':(1.,1.,1.)})
        self.renderer.pipe.compute_cov3D_python=True

    def deform(self,index,local=None):
        p=self.parts[index];i=p['parameter_index'];shape=self.log_shape[i].exp();source=p['local'] if local is None else local
        if p['kind'].startswith('hinge'):
            u=(source[:,0]/p['length']).clamp(0,1);s=3*u.square()-2*u.pow(3)
            w=source[:,0]+p['length']*(shape[0]-1)*s;deriv=1+(shape[0]-1)*6*u*(1-u)
        else:w=source[:,0]*shape[0];deriv=shape[0].expand(len(source))
        pos=torch.stack([w,source[:,1]*shape[1],source[:,2]],1)
        direction=p['B'][:,1] if p['kind'].startswith('hinge') else tensor([0,1,0])
        shift=self.vertical_shift[i]*direction if p['owner']!=10 else direction*0
        xyz=pos@p['B'].T+p['anchor']+shift
        if local is not None:return xyz
        scale=torch.stack([deriv,shape[1].expand(len(source)),torch.ones_like(deriv)],1)
        cov=p['cov_local']*scale[:,:,None]*scale[:,None,:]
        return xyz,p['B'][None]@cov@p['B'].T[None]

    def geometry(self,original=False):
        result=[]
        for i,p in enumerate(self.parts):
            if original:result.append((p['local']@p['B'].T+p['anchor'],p['B'][None]@p['cov_local']@p['B'].T[None]))
            else:result.append(self.deform(i))
        return result

    def render(self,geometry=None,kind='rgb',method='prior',extrinsics=None,intrinsics=None,reference=False):
        geometry=self.geometry() if geometry is None else geometry
        xyz,cov,alpha,color=[],[],[],[];s=self.log_global_scale.exp();translation=self.translation_basis@self.screen_translation
        tr=self.original_closed_transforms if method=='original' else self.closed_transforms
        for p,(x,c) in zip(self.parts,geometry):
            nid=p['group']['source_node'] if method=='original' else p['group']['id'];T=np.eye(4) if reference else tr[nid]
            A=self.R@tensor(T[:3,:3]);xyz.append(s*(x@A.T+tensor(T[:3,3])@self.R.T)+translation)
            cov.append(s.square()*(A[None]@c@A.T[None]));alpha.append(p['alpha'])
            if kind=='rgb':color.append(p['colors'])
            elif kind=='semantic':color.append(tensor(self.config['palette'][str(p['group']['id'])]).expand(len(x),3))
            else:color.append(torch.zeros_like(p['colors']))
        gs=CovGaussians(torch.cat(xyz).contiguous(),torch.cat(cov).contiguous(),torch.cat(alpha).contiguous())
        rgb=self.renderer.render(gs,self.extrinsics if extrinsics is None else extrinsics,self.intrinsics if intrinsics is None else intrinsics,
                                 colors_overwrite=torch.cat(color).contiguous())['color']
        return 1-rgb[0] if kind=='silhouette' else rgb

    def losses(self,align=False):
        geom=self.geometry();rgb=self.render(geom);alpha=self.render(geom,'silhouette')
        photo=((rgb-self.rgb).abs()*self.weight[None]).mean()/self.weight.mean();sil=(alpha-self.alpha).abs().mean()
        semantic=photo*0 if align else ((self.render(geom,'semantic')-self.semantic).square()*self.weight[None]).mean()/self.weight.mean()
        boxes=[]
        if not align:
            for i,p in enumerate(self.parts):
                if 'front_local' not in p:continue
                x=self.deform(i,p['front_local']);T=self.closed_transforms[p['group']['id']]
                x=x@tensor(T[:3,:3]).T+tensor(T[:3,3]);x=self.log_global_scale.exp()*(x@self.R.T)+self.translation_basis@self.screen_translation
                camera=x@self.extrinsics[:3,:3].T+self.extrinsics[:3,3];uv=camera[:,:2]/camera[:,2:3]
                uv=uv*torch.stack([self.intrinsics[0,0],self.intrinsics[1,1]])+self.intrinsics[:2,2]
                pred=torch.cat([torch.quantile(uv,.01,dim=0),torch.quantile(uv,.99,dim=0)])
                target=tensor(p['group']['box'])/800.
                boxes.append((pred-target).abs().mean())
        box=torch.stack(boxes).mean() if boxes else photo*0
        prior=.002*self.log_shape.square().mean()+.01*self.vertical_shift.square().mean()
        loss=photo+2*sil+(0 if align else 2)*semantic+3*box+(0 if align else prior)
        return loss,dict(photo=float(photo.detach()),silhouette=float(sil.detach()),semantic=float(semantic.detach()),box=float(box.detach()))

    @torch.no_grad()
    def project(self):
        self.log_global_scale.clamp_(np.log(.2),np.log(4.));self.screen_translation.clamp_(-.8,.8)
        for p in self.parts:
            i=p['parameter_index'];self.log_shape[i,0].clamp_(np.log(.8),np.log(3.5 if p['kind'].startswith('hinge') else 1.2));self.log_shape[i,1].clamp_(np.log(.8),np.log(1.2))
        self.vertical_shift.clamp_(-.2,.2);self.vertical_shift[0]=0.

    @torch.no_grad()
    def preview(self,name):
        panel=torch.cat([self.rgb,self.render(),self.semantic,self.render(kind='semantic')],2).clamp(0,1)
        Image.fromarray((panel.permute(1,2,0).cpu().numpy()*255).astype('uint8')).save(self.root/(name+'.png'))


def image_metrics(rgb,alpha,target,mask):
    d=rgb-target;mse=float(d.square().mean());fg=mask>.5;fmse=float(d[:,fg].square().mean());a=alpha>.5
    return {'IoU':float((a&fg).sum()/(a|fg).sum()),'PSNR':float(-10*np.log10(max(mse,1e-12))),
            'foreground_PSNR':float(-10*np.log10(max(fmse,1e-12))),'foreground_L1':float(d[:,fg].abs().mean())}


def fit_case(root):
    config=json.loads((root/'config.json').read_text());manifest=json.loads((root/'motion_groups.json').read_text())
    forbidden=[]
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)):
            name=os.fsdecode(args[0])
            if any(x in name for x in ['/gt/','/gt_mesh/','mobility_v2','joint_infos','joint_info.json','joint_evaluation','/depth/']):
                forbidden.append(name);raise PermissionError('GT data forbidden in fitting: '+name)
    sys.addaudithook(guard)
    torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42);torch.backends.cuda.matmul.allow_tf32=False
    model=Model(config,manifest,root);model.log_shape.requires_grad_(False);model.vertical_shift.requires_grad_(False)
    optimizer=torch.optim.Adam([{'params':[model.log_global_scale],'lr':.008},{'params':[model.screen_translation],'lr':.006}])
    history=[];start=time.monotonic()
    def save(stage,step):
        torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'stage':stage,'step':step},root/'checkpoint.tmp');(root/'checkpoint.tmp').replace(root/'checkpoint.pt')
        write_json(root/'status.json',{'state':'running','stage':stage,'step':step,'seconds':time.monotonic()-start})
    model.preview('initial_closed_fit')
    for stage,steps in [('alignment',100),('geometry',240)]:
        if stage=='geometry':
            model.preview('aligned_before_geometry')
            model.log_global_scale.requires_grad_(False);model.screen_translation.requires_grad_(False)
            model.log_shape.requires_grad_(True);model.vertical_shift.requires_grad_(True)
            optimizer=torch.optim.Adam([{'params':[model.log_shape],'lr':.012},{'params':[model.vertical_shift],'lr':.004}])
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True);loss,values=model.losses(align=stage=='alignment');assert torch.isfinite(loss)
            loss.backward()
            for p in model.parameters():
                if p.grad is not None:assert torch.isfinite(p.grad).all()
            optimizer.step();model.project();history.append(dict(stage=stage,step=step+1,loss=float(loss.detach()),**values))
            if (step+1)%40==0 or step+1==steps:save(stage,step+1);print(config['case'],history[-1],flush=True)
    model.preview('optimized_closed_fit');save('complete',240)
    with torch.no_grad():
        result={}
        for method in ['original','prior','optimized']:
            geom=model.geometry(original=method!='optimized')
            result[method]=image_metrics(model.render(geom,method='original' if method=='original' else 'prior'),model.render(geom,'silhouette',method='original' if method=='original' else 'prior'),model.rgb,model.alpha)
    export={}
    with torch.no_grad():
        for i,p in enumerate(model.parts):
            prefix=p['group']['membership_prefix'];xyz,cov=model.deform(i)
            export[prefix+'_xyz']=xyz.cpu().numpy();export[prefix+'_covariance']=cov.cpu().numpy();export[prefix+'_mesh_vertices']=model.deform(i,tensor(p['vertex_local'])).cpu().numpy()
    np.savez_compressed(root/'optimized_parts.npz',**export)
    assert not forbidden and all(sha256(x['path'])==x['sha256'] for x in manifest['geometry_files'])
    write_json(root/'history.json',history)
    seal={'case':config['case'],'image_metrics':result,'checkpoint_sha256':sha256(root/'checkpoint.pt'),
          'optimized_parts_sha256':sha256(root/'optimized_parts.npz'),'config_sha256':sha256(root/'config.json'),
          'manifest_sha256':sha256(root/'motion_groups.json'),'GT_forbidden_reads':forbidden,
          'source_tool_sha256':sha256(__file__),'fit_seconds':time.monotonic()-start,
          'pose':{str(g['id']):g['observed_q'] for g in model.groups},
          'registered_scale':float(model.log_global_scale.exp()),'translation_parameters':model.screen_translation.detach().cpu().tolist()}
    write_json(root/'fit_results.json',seal);write_json(root/'status.json',{'state':'complete',**seal})
    print('FIT COMPLETE',config['case'],json.dumps(result),flush=True)
