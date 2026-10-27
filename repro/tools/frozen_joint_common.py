"""Rigid-only exports, differentiable FK and the existing Gaussian proxy.

All local geometry/appearance tensors are constants. No RGB fitting, deformation
field, densification, opacity update or independent part translation is present.
"""
import copy
import json
import pathlib

import numpy as np
import torch
import torch.nn.functional as F

from collision_refinement_common import load_asset, sha256
from refine_collision_axes import AxisParameters, GaussianObjective
from reconstruction_ablation_common import Joints

RAW_TO_EXPORT = np.array([[1,0,0],[0,0,1],[0,-1,0]], dtype=np.float32)


def tensor(x):
    return torch.as_tensor(x, dtype=torch.float32, device='cuda')


class FrozenJoints(AxisParameters):
    def __init__(self, asset):
        super().__init__(asset, 'both')
        self.register_buffer('limits0', tensor([n['joint']['range'] for n in self.active]))
        units = [15. if n['joint']['type'] in ['revolute','continuous'] else .1*self.diagonal
                 for n in self.active]
        self.register_buffer('range_units', tensor(units)[:, None])
        self.range_delta = torch.nn.Parameter(torch.zeros_like(self.limits0))
        self.override = None

    def axes(self):
        return super().axes() if self.override is None else self.override[:2]

    def limits(self):
        return self.limits0+self.range_units*self.range_delta if self.override is None else self.override[2]

    def world(self, qs):
        return Joints.world_q(self, qs)[0]

    def prior(self):
        return self.regularizer()+self.range_delta.square().sum(-1).mean()

    @torch.no_grad()
    def project(self):
        super().project()
        self.range_delta.clamp_(-1, 1)

    def export(self, asset):
        obj = copy.deepcopy(asset['object'])
        d, o = (v.detach().cpu().numpy() for v in self.axes())
        limits = self.limits().detach().cpu().numpy()
        for n in obj['diffuse_tree']:
            if n['id'] not in self.indices:
                continue
            i = self.indices[n['id']]
            n['joint']['axis']['direction'] = d[i].astype(float).tolist()
            if n['joint']['type'] in ['revolute', 'continuous']:
                n['joint']['axis']['origin'] = o[i].astype(float).tolist()
            n['joint']['range'] = limits[i].astype(float).tolist()
        obj['geometry_source_object'] = str(asset['object_path'])
        return obj


def paths(limits, states):
    t = torch.linspace(0, 1, states, device=limits.device)
    all_paths = []
    for i in range(len(limits)):
        basis = torch.eye(len(limits), device=limits.device)[i]
        qs = limits[:,0]+t[:,None]*(limits[:,1]-limits[:,0])*basis
        all_paths.append((f'only_index_{i}', qs))
    all_paths.append(('synchronized', limits[:,0]+t[:,None]*(limits[:,1]-limits[:,0])))
    return all_paths


class FrozenScene:
    def __init__(self, run, sources=512, seed=0):
        self.asset = load_asset(run, sources, seed)
        self.params = FrozenJoints(self.asset)
        self.objective = GaussianObjective(self.asset, self.params, tensor([[0.]*len(self.params.active)]))
        # Per-scene normalization on exactly the same full path grid in both arms.
        with torch.no_grad():
            qs = torch.cat([q for _,q in paths(self.params.limits0,21)])
            self.normalization = max(1e-8, float(self.proxy(qs).mean()))

    def proxy(self, qs):
        return self.proxy_states(qs)['proxy'].mean()

    def proxy_states(self, qs):
        world=self.params.world(qs);values=[];distances=[];gaps=[]
        data=self.objective.data
        for i in range(len(data)):
            for j in range(i+1,len(data)):
                terms=[]
                mass=data[i]['full_alpha'].sum()+data[j]['full_alpha'].sum()
                for source,target in [(i,j),(j,i)]:
                    a,b=data[source],data[target]
                    A=world[self.asset['parts'][source]['node']['id']]
                    B=world[self.asset['parts'][target]['node']['id']]
                    relative=torch.linalg.inv(B)@A
                    query=torch.einsum('nj,bij->bni',a['xyz'],relative[:,:3,:3])+relative[:,None,:3,3]
                    with torch.no_grad():
                        nearest=self.objective.trees[target].query(query.detach().cpu().numpy(),workers=4)[1]
                        idx=torch.as_tensor(nearest,device='cuda')
                    d2=(query-b['full_xyz'][idx]).square().sum(-1)
                    kernel=a['alpha'][None]*b['full_alpha'][idx]*torch.exp(-d2/(2*(a['sigma'][None].square()+b['full_sigma'][idx].square())))
                    terms.append(len(a['full_xyz'])*kernel.mean(-1)/mass)
                    distances.append(d2.sqrt().min(-1).values)
                    gaps.append((d2.sqrt()-a['sigma'][None]-b['full_sigma'][idx]).min(-1).values)
                values.append(sum(terms))
        return {'proxy':torch.stack(values).mean(0),
                'minimum_sampled_center_distance':torch.stack(distances).min(0).values,
                'minimum_sampled_one_sigma_gap':torch.stack(gaps).min(0).values}

    def minibatch(self, step, count=3):
        qs = torch.cat([q for _,q in paths(self.params.limits(),21)])
        ids = (torch.arange(count, device='cuda')+count*step)%len(qs)
        return self.proxy(qs[ids])

    def frozen_files(self):
        rows = [{'path':str(self.asset['object_path']), 'sha256':sha256(self.asset['object_path'])}]
        for p in self.asset['parts']:
            rows.extend({'path':str(p[k]), 'sha256':p[h]} for k,h in
                        [('mesh_path','mesh_sha256'),('gs_path','gs_sha256')])
        return rows


def continuous_from_raw(raw, scene):
    """Continuous pre-snap diagnostic, with the ORIGINAL interleaved range decoder.

    Official convert_data_range averages interleaved channels; preserve it even
    though the training encoder repeats each scalar in three consecutive slots.
    Axis snapping/bbox projection is separately evaluated using official code.
    """
    ids = [n['id'] for n in scene.params.active]
    values = raw[ids]
    R = tensor(RAW_TO_EXPORT)
    d = F.normalize(values[:,6:9], dim=-1)@R.T
    o = values[:,9:12]@R.T
    ranges = ((values[:,12:14]+values[:,14:16]+values[:,16:18])/3).clamp(-1,1)
    end = torch.stack([ranges[i,0]*360 if n['joint']['type'] in ['revolute','continuous']
                       else ranges[i,1] for i,n in enumerate(scene.params.active)])
    limits = torch.stack([end*0,end], -1)
    return d, o, limits


def original_postprocessed(raw, scene):
    """Use official decoding, then the original GS->Mesh bbox projection order.

    Type/semantics/tree remain the original prediction, as requested.
    """
    from modules.utils.articulation_utils import convert_data_2_info, post_process_arti_info
    info = convert_data_2_info(raw.detach().cpu().numpy())
    reps = []
    mesh_reps = []
    class Coordinates:
        pass
    for p in scene.asset['parts']:
        from plyfile import PlyData
        v = PlyData.read(p['gs_path'])['vertex'].data
        # save_ply rotates by RAW_TO_EXPORT: undo to native decoder frame.
        # Both save_ply's inverse-default rotation and the articulation exporter
        # map native coordinates to export via RAW_TO_EXPORT. These cancel here.
        xyz = np.column_stack([v[k] for k in ['x','y','z']])
        rep = Coordinates(); rep.get_xyz = tensor(xyz)
        reps.append(rep)
        mesh_rep=Coordinates();mesh_rep.get_xyz=tensor(np.asarray(p['mesh'].vertices)@RAW_TO_EXPORT)
        mesh_reps.append(mesh_rep)
    for i, n in enumerate(scene.asset['nodes']):
        info[i]['name'] = n['name']
        info[i]['joint']['type'] = n['joint']['type']
    info = post_process_arti_info(info,reps)
    info = post_process_arti_info(info,mesh_reps)
    obj = copy.deepcopy(scene.asset['object'])
    for n, new in zip(obj['diffuse_tree'], info):
        if n['id'] not in scene.params.indices:
            continue
        n['joint']['axis']['direction'] = (RAW_TO_EXPORT@np.array(new['joint']['axis']['direction'])).tolist()
        if n['joint']['type'] in ['revolute','continuous']:
            n['joint']['axis']['origin'] = (RAW_TO_EXPORT@np.array(new['joint']['axis']['origin'])).tolist()
        ranges = ((raw[n['id'],12:14]+raw[n['id'],14:16]+raw[n['id'],16:18])/3).clamp(-1,1).detach().cpu().numpy()
        n['joint']['range'] = [0.,float(ranges[0]*360 if n['joint']['type'] in ['revolute','continuous'] else ranges[1])]
    obj['geometry_source_object'] = str(scene.asset['object_path'])
    return obj


def state_from_nodes(scene, nodes):
    byid = {n['id']:n for n in nodes}
    active = [byid[n['id']]['joint'] for n in scene.params.active]
    return (tensor([j['axis']['direction'] for j in active]),
            tensor([j['axis']['origin'] for j in active]),
            tensor([j['range'] for j in active]))


def atomic_json(path, value):
    path = pathlib.Path(path); tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)); tmp.replace(path)
