"""Shared image-fitting and differentiable geometry for paired reconstruction trials."""
import json
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

REPO=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
from modules.pact.renderers.gaussian_render import GaussianRenderer
from collision_refinement_common import load_asset, sha256
from refine_collision_axes import AxisParameters

R_EXPORT_TO_WORLD=np.array([[0,0,-1],[-1,0,0],[0,1,0]],np.float32)
UNDO_GS=np.array([[1,0,0],[0,0,1],[0,-1,0]],np.float32)


def tensor(value):return torch.as_tensor(value,dtype=torch.float32,device='cuda')


def quaternion_product(a,b):
    w,x,y,z=a.unbind(-1);v,i,j,k=b.unbind(-1)
    return torch.stack([w*v-x*i-y*j-z*k,w*i+x*v+y*k-z*j,w*j-x*k+y*v+z*i,w*k+x*j-y*i+z*v],-1)


def interpolation(points,bounds,grid=4):
    low,high=bounds;u=((points-low)/(high-low).clip(1e-5)*(grid-1)).clip(0,grid-1-1e-6)
    base=np.floor(u).astype(np.int64);frac=u-base;indices=[];weights=[]
    for a in [0,1]:
        for b in [0,1]:
            for c in [0,1]:
                corner=base+np.array([a,b,c]);indices.append(corner[:,0]*grid*grid+corner[:,1]*grid+corner[:,2]);weights.append(np.prod(np.where(np.array([a,b,c]),frac,1-frac),axis=1))
    return np.stack(indices,1),np.stack(weights,1).astype(np.float32)


class Joints(AxisParameters):
    def world_q(self,qs):
        directions,origins=self.axes();B=len(qs);identity=torch.eye(4,device='cuda').expand(B,4,4);world={-1:identity};quats={-1:tensor([1,0,0,0]).expand(B,4)};lookup={n['id']:n for n in self.nodes}
        def visit(nid):
            if nid in world:return
            node=lookup[nid];parent=node.get('parent',-1);visit(parent);local=identity.clone();quat=tensor([1,0,0,0]).expand(B,4)
            if nid in self.indices:
                i=self.indices[nid];d,o=directions[i],origins[i];q=qs[:,i]
                if node['joint']['type'] in ['revolute','continuous']:
                    x,y,z=d.unbind();zero=x*0;K=torch.stack([zero,-z,y,z,zero,-x,-y,x,zero]).reshape(3,3);angle=torch.deg2rad(q)
                    R=torch.eye(3,device='cuda')+angle.sin()[:,None,None]*K+(1-angle.cos())[:,None,None]*(K@K)
                    local[:,:3,:3]=R;local[:,:3,3]=o-torch.einsum('bij,j->bi',R,o)
                    quat=torch.cat([torch.cos(angle[:,None]/2),torch.sin(angle[:,None]/2)*d[None]],1)
                else:local[:,:3,3]=q[:,None]*d
            world[nid]=world[parent]@local;quats[nid]=quaternion_product(quats[parent],quat)
        for node in self.nodes:visit(node['id'])
        return world,quats


class RenderGaussians:
    active_sh_degree=0
    def __init__(self,xyz,scales,rotations,alpha):self.get_xyz=xyz;self.get_scaling=scales;self.get_rotation=rotations;self.get_opacity=alpha


class Reconstruction(torch.nn.Module):
    def __init__(self,run,config,data):
        super().__init__();self.config=config;self.asset=load_asset(run);self.parts=[];self.geometry=torch.nn.ParameterList();self.D=self.asset['diagonal'];self.bound=config['geometry_bound_fraction_D']*self.D
        self.joints=Joints(self.asset,'both');self.joints.max_position=config['pivot_bound_fraction_D']*self.D
        self.log_scale=torch.nn.Parameter(tensor(np.log(1.5)));self.translation=torch.nn.Parameter(tensor([0.,0.,0.]));self.R=tensor(R_EXPORT_TO_WORLD)
        q=Rotation.from_matrix(R_EXPORT_TO_WORLD).as_quat();self.qglobal=tensor(q[[3,0,1,2]])
        for part in self.asset['parts']:
            path=self.asset['object_path'].parent/part['node']['gs'][0];v=PlyData.read(path)['vertex'].data
            xyz=np.column_stack([v[k] for k in ['x','y','z']]).astype(np.float32)@UNDO_GS.T
            scales=np.exp(np.column_stack([v[k] for k in ['scale_0','scale_1','scale_2']]).astype(np.float32))
            alpha=1/(1+np.exp(-v['opacity'].astype(np.float32)))
            rotations=np.column_stack([v[k] for k in ['rot_0','rot_1','rot_2','rot_3']]).astype(np.float32)
            rm=Rotation.from_quat(rotations[:,[1,2,3,0]]).as_matrix();rot=Rotation.from_matrix(UNDO_GS[None]@rm).as_quat()[:,[3,0,1,2]].copy()
            colors=(.28209479177387814*np.column_stack([v[k] for k in ['f_dc_0','f_dc_1','f_dc_2']])+.5).clip(0,1)
            bounds=np.array([np.minimum(xyz.min(0),part['mesh'].bounds[0]),np.maximum(xyz.max(0),part['mesh'].bounds[1])]);indices,weights=interpolation(xyz,bounds,config['geometry_grid'])
            kept=np.flatnonzero(alpha>=.05);rng=np.random.default_rng(0+int(part['node']['id']));source=np.sort(rng.choice(kept,min(config['collision_sources_per_part'],len(kept)),replace=False))
            part.update(gs_path=path,gs_sha256=sha256(path),gs_xyz=tensor(xyz),scales=tensor(scales),alpha=tensor(alpha[:,None]),rotations=tensor(rot),colors=tensor(colors),
                        indices=torch.as_tensor(indices,device='cuda'),weights=tensor(weights),bounds=bounds,retained=torch.as_tensor(kept,device='cuda'),source=torch.as_tensor(source,device='cuda'))
            self.parts.append(part);self.geometry.append(torch.nn.Parameter(torch.zeros(config['geometry_grid']**3,3,device='cuda')))
        initial=[]
        for node in self.joints.active:
            hi=node['joint']['range'][1];initial.append(np.sign(hi or 1)*30 if node['joint']['type'] in ['revolute','continuous'] else .5*hi)
        self.input_q=torch.nn.Parameter(tensor(initial));self.renderer=GaussianRenderer({'resolution':config['resolution'],'near':.1,'far':20.,'bg_color':(1.,1.,1.)})
        self.data={k:tensor(data[k]) for k in ['rgb','extrinsics','intrinsics','palette']};self.labels=torch.as_tensor(data['labels'],device='cuda')
        palette=torch.cat([torch.ones(1,3,device='cuda'),self.data['palette']],0);self.target_labels=palette[self.labels.long()].permute(0,3,1,2)
        self.data['rgb']=self.data['rgb'].permute(0,3,1,2)

    def deformed(self):return [p['gs_xyz']+(delta[p['indices']]*p['weights'][:,:,None]).sum(1) for p,delta in zip(self.parts,self.geometry)]

    def render(self,view,geometry=None,qs=None):
        points=self.deformed() if geometry is None else geometry;qs=self.input_q[None] if qs is None else qs
        world,quats=self.joints.world_q(qs);xyz,scales,rotations,alpha,colors,labels=[],[],[],[],[],[];scale=self.log_scale.exp()
        for i,(part,points_i) in enumerate(zip(self.parts,points)):
            nid=part['node']['id'];tr=world[nid][0];posed=points_i@tr[:3,:3].T+tr[:3,3];xyz.append(scale*(posed@self.R.T)+self.translation)
            scales.append(scale*part['scales']);rotations.append(quaternion_product(quaternion_product(self.qglobal,quats[nid][0]),part['rotations']));alpha.append(part['alpha']);colors.append(part['colors']);labels.append(self.data['palette'][i].expand(len(points_i),3))
        g=RenderGaussians(torch.cat(xyz),torch.cat(scales),torch.cat(rotations),torch.cat(alpha));args=(g,self.data['extrinsics'][view],self.data['intrinsics'][view])
        rgb=self.renderer.render(*args,colors_overwrite=torch.cat(colors))['color'];seg=self.renderer.render(*args,colors_overwrite=torch.cat(labels))['color']
        return rgb,seg

    def image_loss(self,view,geometry=None):
        rgb,seg=self.render(view,geometry);weight=1+2*(self.labels[view]>0).float()[None]
        photo=((rgb-self.data['rgb'][view]).abs()*weight).mean()/weight.mean();semantic=((seg-self.target_labels[view]).square()*weight).mean()/weight.mean()
        return photo+2*semantic,photo,semantic

    def regularizer(self):
        normalized=torch.stack(list(self.geometry))/self.bound;grid=self.config['geometry_grid'];field=normalized.reshape(len(self.geometry),grid,grid,grid,3)
        smooth=sum(field.diff(dim=d).square().mean() for d in [1,2,3])/3
        return .001*normalized.square().mean()+.001*smooth+.001*self.joints.regularizer()

    @torch.no_grad()
    def project(self):
        self.joints.project()
        for p in self.geometry:p.mul_((self.bound/p.norm(dim=-1,keepdim=True).clamp_min(1e-12)).clamp_max(1))
        self.log_scale.clamp_(np.log(.2),np.log(5.));self.translation.clamp_(-2,2)
        for i,node in enumerate(self.joints.active):self.input_q[i].clamp_(-150 if node['joint']['type'] in ['revolute','continuous'] else -self.D,150 if node['joint']['type'] in ['revolute','continuous'] else self.D)

    def configure(self,stage):
        prealign=stage=='prealign'
        for p in self.geometry:p.requires_grad_(not prealign)
        self.joints.theta.requires_grad_(not prealign);self.joints.phi.requires_grad_(not prealign)
        self.log_scale.requires_grad_(prealign);self.translation.requires_grad_(prealign);self.input_q.requires_grad_(prealign)

    def motion_q(self,mode,states=21):
        limits=tensor([n['joint']['range'] for n in self.joints.active]);
        if mode=='signed90_assumption':
            limits=limits.clone()
            for i,n in enumerate(self.joints.active):
                if n['joint']['type'] in ['revolute','continuous']:limits[i]=tensor([0.,np.sign(n['joint']['range'][1] or 1)*90.])
        paths=[]
        for i,node in enumerate(self.joints.active):
            q=limits[:,0].repeat(states,1);q[:,i]=torch.linspace(float(limits[i,0]),float(limits[i,1]),states,device='cuda');paths.append((f"only_node_{node['id']}",q))
        q=limits[:,0]+torch.linspace(0,1,states,device='cuda')[:,None]*(limits[:,1]-limits[:,0]);paths.append(('synchronized',q))
        return paths

    def collision(self,geometry,qs,source_indices=None):
        world,_=self.joints.world_q(qs);values=[];trees=[cKDTree(x[p['retained']].detach().cpu().numpy()) for x,p in zip(geometry,self.parts)]
        for i in range(len(self.parts)):
            for j in range(i+1,len(self.parts)):
                terms=[];mass=self.parts[i]['alpha'][self.parts[i]['retained']].sum()+self.parts[j]['alpha'][self.parts[j]['retained']].sum()
                for source,target in [(i,j),(j,i)]:
                    p,q=self.parts[source],self.parts[target];ids=p['source'] if source_indices is None else source_indices[source]
                    a,b=world[p['node']['id']],world[q['node']['id']];relative=torch.linalg.inv(b)@a
                    query=torch.einsum('nj,bij->bni',geometry[source][ids],relative[:,:3,:3])+relative[:,None,:3,3]
                    with torch.no_grad():nearest=trees[target].query(query.detach().cpu().numpy(),workers=4)[1];idx=q['retained'][torch.as_tensor(nearest,device='cuda')]
                    d2=(query-geometry[target][idx]).square().sum(-1);sigma_s=p['scales'][ids].mean(-1);sigma_t=q['scales'][idx].mean(-1)
                    value=p['alpha'][ids,0][None]*q['alpha'][idx,0]*torch.exp(-d2/(2*(sigma_s[None].square()+sigma_t.square())))
                    terms.append(len(p['retained'])*value.mean(-1)/mass)
                values.append(sum(terms))
        return torch.stack(values).mean(0)

    def deformed_mesh(self,index):
        p=self.parts[index];indices,weights=interpolation(p['mesh'].vertices,p['bounds'],self.config['geometry_grid']);delta=self.geometry[index].detach().cpu().numpy();mesh=p['mesh'].copy();mesh.vertices=np.asarray(mesh.vertices)+(delta[indices]*weights[:,:,None]).sum(1);return mesh
