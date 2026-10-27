"""Direction-only / pivot-only FK, outputs and strict physical field freezing."""
import copy

import torch

from axis_fixed_common import atomic_json,sha256,assert_ranges
from frozen_joint_common import (FrozenScene as BaseScene,FrozenJoints,paths,tensor,
    continuous_from_raw as decode,original_postprocessed as postprocess,
    state_from_nodes)
from collision_refinement_common import load_asset
from refine_collision_axes import GaussianObjective


class ModeJoints(FrozenJoints):
    def __init__(self,asset,mode,route):
        super().__init__(asset);self.mode=mode;self.route=route
        self.range_delta.requires_grad_(False)
        self.theta.requires_grad_(route=='direct' and mode=='direction')
        self.phi.requires_grad_(route=='direct' and mode=='position')

    def axes(self):
        d,o=super().axes()
        # These are physical FK constraints, independent of raw MLP channels.
        return (self.d0 if self.mode=='position' else d,
                self.o0 if self.mode=='direction' else o)


class ModeScene(BaseScene):
    def __init__(self,run,sources=512,mode='direction',route='direct'):
        self.asset=load_asset(run,sources,0);self.mode=mode;self.route=route
        self.params=ModeJoints(self.asset,mode,route)
        self.objective=GaussianObjective(self.asset,self.params,tensor([[0.]*len(self.params.active)]))
        with torch.no_grad():
            qs=torch.cat([q for _,q in paths(self.params.limits0,21)])
            self.normalization=max(1e-8,float(self.proxy(qs).mean()))
        self.assert_frozen()

    def assert_limits(self):
        p=self.params
        assert not p.range_delta.requires_grad and p.range_delta.grad is None
        assert not torch.count_nonzero(p.range_delta)
        assert torch.equal(p.limits(),p.limits0)

    def assert_frozen(self):
        self.assert_limits();p=self.params;d,o=p.axes()
        if self.mode=='direction':
            assert not p.phi.requires_grad and p.phi.grad is None
            assert not torch.count_nonzero(p.phi) and torch.equal(o,p.o0)
        else:
            assert not p.theta.requires_grad and p.theta.grad is None
            assert not torch.count_nonzero(p.theta) and torch.equal(d,p.d0)
        if self.route=='head':assert not p.theta.requires_grad and not p.phi.requires_grad


def output_mask(cache,scene):
    mask=torch.zeros_like(cache['raw'],device='cuda')
    for n in scene.params.active:
        if scene.mode=='direction':mask[n['id'],6:9]=1
        elif n['joint']['type'] in ['revolute','continuous']:mask[n['id'],9:12]=1
    return mask


def fixed_raw(pred,cache,scene):
    mask=output_mask(cache,scene);base=cache['raw'].to(pred.device)
    raw=mask*pred+(1-mask)*base
    assert torch.equal(raw[mask==0],base[mask==0])
    return raw


def continuous_from_raw(raw,scene):
    d,o,_=decode(raw,scene)
    return (scene.params.d0 if scene.mode=='position' else d,
            scene.params.o0 if scene.mode=='direction' else o,scene.params.limits0)


def preserve_fields(obj,scene):
    byid={n['id']:n for n in scene.asset['nodes']}
    for n in obj['diffuse_tree']:
        old=byid[n['id']];n['joint']['range']=copy.deepcopy(old['joint']['range'])
        if scene.mode=='direction':n['joint']['axis']['origin']=copy.deepcopy(old['joint']['axis']['origin'])
        else:n['joint']['axis']['direction']=copy.deepcopy(old['joint']['axis']['direction'])
    obj['geometry_source_object']=str(scene.asset['object_path']);obj['axis_single_variable_mode']=scene.mode
    assert_fields(obj,scene)
    return obj


def assert_fields(obj,scene):
    assert_ranges(obj,scene);byid={n['id']:n for n in scene.asset['nodes']}
    key='origin' if scene.mode=='direction' else 'direction'
    for n in obj['diffuse_tree']:
        old=byid[n['id']]
        assert n['joint']['axis'][key]==old['joint']['axis'][key],('Frozen field changed',scene.mode,n['id'])
        restored=copy.deepcopy(n)
        if n['id'] in scene.params.indices:
            if scene.mode=='direction':restored['joint']['axis']['direction']=old['joint']['axis']['direction']
            elif old['joint']['type'] in ['revolute','continuous']:restored['joint']['axis']['origin']=old['joint']['axis']['origin']
        assert restored==old,('Non-selected part field changed',scene.mode,n['id'])


def export_axis(scene):
    scene.assert_frozen()
    return preserve_fields(scene.params.export(scene.asset),scene)


def original_postprocessed(raw,scene):
    # Official bbox projection may change the frozen origin when direction changes.
    # Replay the requested frozen physical field AFTER that projection as well.
    return preserve_fields(postprocess(raw,scene),scene)
