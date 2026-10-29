"""Single closed RGB image fitting after rigid-instance/semantic motion correction.

Only image-space annotations, RGB/alpha, and its stored camera are observations.
No GT mesh, joint annotation, depth, or other RGB frame enters optimization.
"""
import argparse
import copy
import json
import os
import pathlib
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
from plyfile import PlyData
from scipy.spatial.transform import Rotation
import torch

from collision_refinement_common import load_asset, sha256, transforms
from semantic_motion_prior_45194 import UNDO, verify as verify_prior

REPO=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
from modules.pact.renderers.gaussian_render import GaussianRenderer

R_WORLD=np.array([[0,0,-1],[-1,0,0],[0,1,0]],np.float32)
PALETTE={10:[.5,.5,.5],11:[.9,.15,.15],12:[.15,.3,.95],13:[.15,.75,.3],14:[.75,.2,.8],15:[.15,.75,.3]}


def tensor(x):return torch.as_tensor(x,device='cuda',dtype=torch.float32)


def write_json(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


class CovGaussians:
    active_sh_degree=0
    def __init__(self,xyz,cov,alpha):self.get_xyz=xyz;self.cov=cov;self.get_opacity=alpha
    def get_covariance(self,scaling_modifier=1.):
        c=self.cov*scaling_modifier**2
        return torch.stack([c[:,0,0],c[:,0,1],c[:,0,2],c[:,1,1],c[:,1,2],c[:,2,2]],1).contiguous()


def prepare(prior,root,resolution):
    root.mkdir(parents=True,exist_ok=False)
    verify_prior(prior)
    manifest=json.loads((prior/'motion_groups.json').read_text())
    scene=pathlib.Path('/root/autodl-tmp/VideoArtGS/data/videoartgs/sapien/45194')
    camera=json.loads((scene/'transforms.json').read_text())
    frame_index=250
    frame=camera['frames'][frame_index]
    source=scene/frame['file_path']
    original=Image.open(source).convert('RGBA')
    original.save(root/'closed_reference.png')
    rgba=np.asarray(original.resize((resolution,resolution),Image.Resampling.LANCZOS)).astype(np.float32)/255.
    alpha=rgba[:,:,3]
    rgb=rgba[:,:,:3]*alpha[:,:,None]+1-alpha[:,:,None]
    # Explicit visual annotations on this one front image, in original pixels.
    # These are rectangles covering visible front panels, not GT 3D projections.
    boxes={11:[177,280,396,536],12:[400,280,619,536],
           13:[145,173,395,263],14:[400,173,650,263]}
    labels=np.zeros((resolution,resolution),np.int32)
    labels[alpha>.5]=10
    for gid,(x0,y0,x1,y1) in boxes.items():
        x0,x1=int(round(x0*resolution/800)),int(round(x1*resolution/800))
        y0,y1=int(round(y0*resolution/800)),int(round(y1*resolution/800))
        labels[y0:y1,x0:x1]=np.where(alpha[y0:y1,x0:x1]>.5,gid,0)
    semantic=np.ones((resolution,resolution,3),np.float32)
    for gid,color in PALETTE.items():semantic[labels==gid]=color
    Image.fromarray((semantic*255).astype('uint8')).save(root/'front_panel_annotations.png')
    c2w=np.asarray(frame['transform_matrix'],np.float32)
    c2w[:3,1:3]*=-1
    extrinsics=np.linalg.inv(c2w).astype(np.float32)
    intrinsics=np.array([[camera['focal_x']/camera['w'],0,camera['cx']/camera['w']],
                         [0,camera['focal_y']/camera['h'],camera['cy']/camera['h']],[0,0,1]],np.float32)
    np.savez_compressed(root/'observation.npz',rgb=rgb,alpha=alpha,semantic=semantic,labels=labels,
                        extrinsics=extrinsics,intrinsics=intrinsics)
    config={'prior_root':str(prior),'prior_manifest_sha256':sha256(prior/'motion_groups.json'),
            'reference_frame':frame_index,'source_image':str(source),'source_image_sha256':sha256(source),
            'camera_metadata_sha256':sha256(scene/'transforms.json'),
            'observation_sha256':sha256(root/'observation.npz'),'annotation_boxes_original_pixels':boxes,
            'annotation_provenance':'Manual visible front-panel rectangles on the same closed RGB image; no GT 3D projection',
            'resolution':resolution,'alignment_steps':100,'geometry_steps':240,
            'seed':42,'selection':'Fixed final step; no GT validation or frame/checkpoint search',
            'geometry':'Hinge-anchored smooth horizontal door expansion, bounded in-plane height/width scales and vertical assembly offsets; left/right pairs share parameters. Depth and topology retained.',
            'frozen':'Motion groups/parents, hinge lines/directions, rail directions, original numeric motion intervals, opacity, color, point count, topology, hidden depth',
            'loss':'RGB foreground-weighted L1 + 2*silhouette L1 + 2*visible-part palette MSE + 3*annotated front-panel projected bounding-box L1 + small shape prior; left/right pairs shared',
            'source_tool_sha256':sha256(__file__),
            'single_view_limit':'Fit to one observed front view; no claim of unseen geometry or motion accuracy improvement'}
    write_json(root/'config.json',config)
    return config,manifest


class Model(torch.nn.Module):
    def __init__(self,config,manifest,root):
        super().__init__()
        self.config=config;self.manifest=manifest;self.root=root
        self.asset=load_asset(manifest['source_run'])
        self.membership=np.load(pathlib.Path(config['prior_root'])/'membership.npz')
        self.groups=manifest['groups'];self.parts=[]
        self.log_global_scale=torch.nn.Parameter(tensor(np.log(1.7)))
        self.screen_translation=torch.nn.Parameter(tensor([0.,0.]))
        self.log_shape=torch.nn.Parameter(torch.zeros(len(self.groups),2,device='cuda'))
        self.vertical_shift=torch.nn.Parameter(torch.zeros(len(self.groups),device='cuda'))
        self.R=tensor(R_WORLD)
        source_data={}
        mesh_byid={p['node']['id']:p['mesh'] for p in self.asset['parts']}
        closed=copy.deepcopy(self.groups)
        for i,g in enumerate(self.groups):
            nid=g['source_node'];gid=g['id']
            if nid not in source_data:
                v=PlyData.read(manifest['sources'][str(nid)]['gs'])['vertex'].data
                xyz=np.column_stack([v[k] for k in ['x','y','z']]).astype(np.float32)@UNDO.T
                scales=np.exp(np.column_stack([v[k] for k in ['scale_0','scale_1','scale_2']]).astype(np.float32))
                q=np.column_stack([v[k] for k in ['rot_0','rot_1','rot_2','rot_3']])
                rot=UNDO[None]@Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix()
                cov=(rot*scales[:,None,:]**2)@rot.transpose(0,2,1)
                alpha=(1/(1+np.exp(-v['opacity'].astype(np.float32))))[:,None]
                color=(.28209479177387814*np.column_stack([v[k] for k in ['f_dc_0','f_dc_1','f_dc_2']])+.5).clip(0,1)
                source_data[nid]=(xyz,cov,alpha,color)
            ids=self.membership[g['membership_prefix']+'_gaussians']
            xyz,cov,alpha,color=[x[ids] for x in source_data[nid]]
            vertices=np.asarray(mesh_byid[nid].vertices)[self.membership[g['membership_prefix']+'_vertices']]
            B=np.eye(3)
            if gid in [11,12]:
                anchor=np.asarray(g['joint']['axis']['origin'])
                horizontal=vertices[:,[0,2]]
                _,ev=np.linalg.eigh(np.cov(horizontal.T));w=ev[:,-1]
                toward=horizontal.mean(0)-anchor[[0,2]]
                if w@toward<0:w=-w
                width=np.array([w[0],0,w[1]])
                B=np.column_stack([width,[0,1,0],np.cross(width,[0,1,0])])
                length=float(np.quantile((vertices-anchor)@width,.99))
                theta=float(np.arctan2(w[1],w[0]))
                if gid==12:theta-=np.pi
                closed[i]['joint']['range']=[np.degrees(theta)]*2
            else:
                anchor=(vertices.min(0)+vertices.max(0))/2
                length=1.
                closed[i]['joint']['range']=[0.,0.]
            local=(xyz-anchor)@B
            localcov=B.T[None]@cov@B[None]
            self.parts.append({'group':g,'B':tensor(B),'anchor':tensor(anchor),'length':length,
                               'local':tensor(local),'cov_local':tensor(localcov),
                               'alpha':tensor(alpha),'colors':tensor(color),'vertex_local':(vertices-anchor)@B})
            if gid in [11,12,13,14]:
                selected=vertices
                if gid in [13,14]:selected=selected[selected[:,2]>=np.quantile(selected[:,2],.75)]
                selected=selected[::max(1,len(selected)//2048)]
                self.parts[-1]['front_local']=tensor((selected-anchor)@B)
        self.closed_nodes=closed;self.closed_transforms=transforms(closed,1.)
        data=np.load(root/'observation.npz')
        self.rgb=tensor(data['rgb']).permute(2,0,1);self.alpha=tensor(data['alpha'])
        self.semantic=tensor(data['semantic']).permute(2,0,1)
        self.extrinsics=tensor(data['extrinsics']);self.intrinsics=tensor(data['intrinsics'])
        self.weight=1+3*self.alpha
        self.renderer=GaussianRenderer({'resolution':config['resolution'],'near':.05,'far':20.,'bg_color':(1.,1.,1.)})
        self.renderer.pipe.compute_cov3D_python=True

    def deform(self,index,local=None):
        part=self.parts[index]
        paired_index={2:1,4:3,5:3}.get(index,index)
        shape=self.log_shape[paired_index].exp()
        source=part['local'] if local is None else local
        if part['group']['id'] in [11,12]:
            u=(source[:,0]/part['length']).clamp(0,1)
            smooth=3*u.square()-2*u.pow(3)
            width=source[:,0]+part['length']*(shape[0]-1)*smooth
            deriv=1+(shape[0]-1)*6*u*(1-u)
        else:
            width=source[:,0]*shape[0]
            deriv=shape[0].expand(len(source))
        pos=torch.stack([width,source[:,1]*shape[1],source[:,2]],1)
        shift=self.vertical_shift[paired_index]
        vertical=torch.stack([shift*0,shift,shift*0])
        xyz=pos@part['B'].T+part['anchor']+vertical
        if local is not None:return xyz
        scale=torch.stack([deriv,shape[1].expand(len(source)),torch.ones_like(deriv)],1)
        cov=part['cov_local']*scale[:,:,None]*scale[:,None,:]
        cov=part['B'][None]@cov@part['B'].T[None]
        return xyz,cov

    def geometry(self,original=False):
        result=[]
        for i,part in enumerate(self.parts):
            if original:
                xyz=part['local']@part['B'].T+part['anchor']
                cov=part['B'][None]@part['cov_local']@part['B'].T[None]
            else:xyz,cov=self.deform(i)
            result.append((xyz,cov))
        return result

    def render(self,geometry=None,kind='rgb'):
        geometry=self.geometry() if geometry is None else geometry
        xyz,cov,alpha,colors=[],[],[],[]
        s=self.log_global_scale.exp()
        translation=torch.stack([self.screen_translation[0]*0,self.screen_translation[0],self.screen_translation[1]])
        for part,(x,c) in zip(self.parts,geometry):
            T=self.closed_transforms[part['group']['id']]
            A=self.R@tensor(T[:3,:3])
            x=s*(x@A.T+tensor(T[:3,3])@self.R.T)+translation
            c=s.square()*(A[None]@c@A.T[None])
            xyz.append(x);cov.append(c);alpha.append(part['alpha'])
            if kind=='rgb':colors.append(part['colors'])
            elif kind=='semantic':colors.append(tensor(PALETTE[part['group']['id']]).expand(len(x),3))
            elif kind=='silhouette':colors.append(torch.zeros_like(part['colors']))
        g=CovGaussians(torch.cat(xyz).contiguous(),torch.cat(cov).contiguous(),torch.cat(alpha).contiguous())
        image=self.renderer.render(g,self.extrinsics,self.intrinsics,colors_overwrite=torch.cat(colors).contiguous())['color']
        return 1-image[0] if kind=='silhouette' else image

    def losses(self,align=False):
        geometry=self.geometry()
        rgb=self.render(geometry,'rgb');alpha=self.render(geometry,'silhouette')
        photo=((rgb-self.rgb).abs()*self.weight[None]).mean()/self.weight.mean()
        silhouette=(alpha-self.alpha).abs().mean()
        if align:semantic=photo*0
        else:
            seg=self.render(geometry,'semantic')
            semantic=((seg-self.semantic).square()*self.weight[None]).mean()/self.weight.mean()
        boxes=[]
        if not align:
            scale=self.log_global_scale.exp()
            translation=torch.stack([self.screen_translation[0]*0,self.screen_translation[0],self.screen_translation[1]])
            for i,part in enumerate(self.parts):
                gid=part['group']['id']
                if 'front_local' not in part:continue
                x=self.deform(i,part['front_local'])
                T=self.closed_transforms[gid]
                x=x@tensor(T[:3,:3]).T+tensor(T[:3,3])
                x=scale*(x@self.R.T)+translation
                camera=x@self.extrinsics[:3,:3].T+self.extrinsics[:3,3]
                uv=camera[:,:2]/camera[:,2:3]
                uv=uv*torch.stack([self.intrinsics[0,0],self.intrinsics[1,1]])+self.intrinsics[:2,2]
                lo=torch.quantile(uv,.01,dim=0);hi=torch.quantile(uv,.99,dim=0)
                pred=torch.cat([lo,hi])
                annotations=self.config['annotation_boxes_original_pixels']
                target=tensor(annotations.get(str(gid),annotations.get(gid)))/800.
                boxes.append((pred-target).abs().mean())
        boxloss=torch.stack(boxes).mean() if boxes else photo*0
        # Low shape prior allows image-supported door widening while retaining
        # non-door geometry. A symmetric pair prior limits asymmetric overfit.
        prior=.002*self.log_shape.square().mean()+.01*(self.log_shape[1]-self.log_shape[2]).square().mean()+.01*self.vertical_shift.square().mean()
        return photo+2*silhouette+(0 if align else 2)*semantic+(0 if align else prior)+3*boxloss,photo,silhouette,semantic

    @torch.no_grad()
    def project(self):
        self.log_global_scale.clamp_(np.log(.8),np.log(3.))
        self.screen_translation.clamp_(-.6,.6)
        for i,p in enumerate(self.parts):
            gid=p['group']['id']
            self.log_shape[i,0].clamp_(np.log(.8),np.log(3.5 if gid in [11,12] else 1.2))
            self.log_shape[i,1].clamp_(np.log(.8),np.log(1.2))
        # Invisible internal insert follows the same shape change as its drawer.
        self.log_shape[5].copy_(self.log_shape[3])
        self.log_shape[2].copy_(self.log_shape[1])
        self.log_shape[4].copy_(self.log_shape[3])
        self.vertical_shift.clamp_(-.2,.2)
        self.vertical_shift[0]=0.
        self.vertical_shift[5]=self.vertical_shift[3]
        self.vertical_shift[2]=self.vertical_shift[1]
        self.vertical_shift[4]=self.vertical_shift[3]

    @torch.no_grad()
    def preview(self,name):
        rgb=self.render();seg=self.render(kind='semantic')
        panel=torch.cat([self.rgb,rgb,self.semantic,seg],2).clamp(0,1)
        Image.fromarray((panel.permute(1,2,0).cpu().numpy()*255).astype('uint8')).save(self.root/f'{name}.png')


def metrics(model,original=False):
    with torch.no_grad():
        geometry=model.geometry(original=original)
        rgb=model.render(geometry,'rgb');alpha=model.render(geometry,'silhouette')
        diff=(rgb-model.rgb)
        fg=model.alpha>.5
        mse=float(diff.square().mean())
        fgmse=float(diff[:,fg].square().mean())
        a,b=alpha>.5,fg
        return {'PSNR':float(-10*np.log10(max(mse,1e-12))),
                'foreground_PSNR':float(-10*np.log10(max(fgmse,1e-12))),
                'foreground_L1':float(diff[:,fg].abs().mean()),
                'silhouette_IoU':float((a&b).sum()/(a|b).sum()),
                'meaning':'Fitted training image only, same registered camera/similarity; not held-out evidence'}


def fit(prior,root,resolution):
    config,manifest=prepare(prior,root,resolution)
    blocked=[]
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)):
            name=os.fsdecode(args[0])
            if any(x in name for x in ['/gt/','mobility_v2','joint_infos','point_cloud.ply','/depth/','joint_evaluation']):
                blocked.append(name)
                raise PermissionError('GT/depth target forbidden: '+name)
    sys.addaudithook(guard)
    torch.manual_seed(42);np.random.seed(42);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    model=Model(config,manifest,root)
    model.log_shape.requires_grad_(False)
    model.vertical_shift.requires_grad_(False)
    optimizer=torch.optim.Adam([{'params':[model.log_global_scale],'lr':.008},
                                {'params':[model.screen_translation],'lr':.006}])
    history=[];started=time.monotonic()
    def save(stage,step):
        torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'history':history,
                    'stage':stage,'step':step},root/'checkpoint.tmp')
        (root/'checkpoint.tmp').replace(root/'checkpoint.pt')
        write_json(root/'status.json',{'state':'running','stage':stage,'step':step,'elapsed_seconds':time.monotonic()-started})
    model.preview('initial_closed_fit')
    for step in range(config['alignment_steps']):
        optimizer.zero_grad(set_to_none=True)
        loss,photo,sil,sem=model.losses(align=True)
        assert torch.isfinite(loss)
        loss.backward();optimizer.step();model.project()
        history.append({'stage':'alignment','step':step+1,'loss':float(loss),'photo':float(photo),'silhouette':float(sil)})
        if (step+1)%20==0:
            save('alignment',step+1);print('ALIGN',history[-1],flush=True)
    model.preview('aligned_before_geometry')
    baseline=metrics(model,original=True)
    model.log_global_scale.requires_grad_(False);model.screen_translation.requires_grad_(False)
    model.log_shape.requires_grad_(True)
    model.vertical_shift.requires_grad_(True)
    optimizer=torch.optim.Adam([{'params':[model.log_shape],'lr':.012},
                                {'params':[model.vertical_shift],'lr':.004}])
    for step in range(config['geometry_steps']):
        optimizer.zero_grad(set_to_none=True)
        loss,photo,sil,sem=model.losses()
        assert torch.isfinite(loss)
        loss.backward()
        assert model.log_shape.grad is not None and torch.isfinite(model.log_shape.grad).all()
        optimizer.step();model.project()
        history.append({'stage':'geometry','step':step+1,'loss':float(loss),'photo':float(photo),
                        'silhouette':float(sil),'semantic':float(sem),'shape_scales':model.log_shape.detach().exp().cpu().tolist(),
                        'vertical_offsets':model.vertical_shift.detach().cpu().tolist()})
        if (step+1)%20==0:
            save('geometry',step+1);print('GEOMETRY',history[-1],flush=True)
    model.preview('optimized_closed_fit')
    optimized=metrics(model)
    save('complete',config['geometry_steps'])
    write_json(root/'history.json',history)
    export={}
    mesh_byid={p['node']['id']:p['mesh'] for p in model.asset['parts']}
    with torch.no_grad():
        for i,p in enumerate(model.parts):
            gid=p['group']['id'];prefix=f'g{gid}'
            x,c=model.deform(i)
            export[prefix+'_xyz']=x.cpu().numpy()
            export[prefix+'_covariance']=c.cpu().numpy()
            export[prefix+'_mesh_vertices']=model.deform(i,tensor(p['vertex_local'])).cpu().numpy()
            source=p['group']['source_node']
            vertex_ids=model.membership[prefix+'_vertices']
            source_faces=mesh_byid[source].faces[model.membership[prefix+'_faces']]
            lut=np.full(len(mesh_byid[source].vertices),-1,np.int64);lut[vertex_ids]=np.arange(len(vertex_ids))
            export[prefix+'_mesh_faces']=lut[source_faces]
    np.savez_compressed(root/'optimized_parts.npz',**export)
    assert all(sha256(x['path'])==x['sha256'] for x in manifest['geometry_files']) and not blocked
    assert sha256(prior/'motion_groups.json')==config['prior_manifest_sha256']
    result={'before':baseline,'after':optimized,
            'registered_scale':float(model.log_global_scale.exp()),'screen_translation_world':model.screen_translation.detach().cpu().tolist(),
            'shape_scales':{str(p['group']['id']):model.log_shape[i].detach().exp().cpu().tolist() for i,p in enumerate(model.parts)},
            'vertical_offsets':{str(p['group']['id']):float(model.vertical_shift[i]) for i,p in enumerate(model.parts)},
            'closed_observation_pose':{str(g['id']):g['joint']['range'][0] for g in model.closed_nodes},
            'motion_metadata_frozen':True,'original_files_unchanged':True,'forbidden_reads':blocked,
            'optimized_parts_sha256':sha256(root/'optimized_parts.npz'),
            'checkpoint_sha256':sha256(root/'checkpoint.pt'),
            'elapsed_seconds':time.monotonic()-started,'peak_GPU_MiB':torch.cuda.max_memory_allocated()/2**20}
    write_json(root/'fit_results.json',result)
    write_json(root/'status.json',{'state':'complete',**result})
    print('FIT COMPLETE',json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prior',required=True);p.add_argument('--root',required=True)
    p.add_argument('--resolution',type=int,default=256);a=p.parse_args()
    fit(pathlib.Path(a.prior).resolve(),pathlib.Path(a.root).resolve(),a.resolution)
