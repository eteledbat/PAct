"""Axis-only continuation with exact original numeric endpoints and frozen parts."""
import copy

import torch
import torch.nn.functional as F

from frozen_joint_common import (FrozenScene as ParentScene, atomic_json,
    continuous_from_raw as parent_decode, original_postprocessed as parent_postprocess,
    paths, sha256, state_from_nodes, tensor)


class FrozenScene(ParentScene):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.params.range_delta.requires_grad_(False)
        self.assert_limits()

    def assert_limits(self):
        assert not self.params.range_delta.requires_grad
        assert self.params.range_delta.grad is None
        assert torch.count_nonzero(self.params.range_delta)==0
        assert torch.equal(self.params.limits(),self.params.limits0)


def output_mask(cache,scene):
    mask=torch.zeros_like(cache['raw'],device='cuda')
    for n in scene.params.active:
        mask[n['id'],6:9]=1
        if n['joint']['type'] in ['revolute','continuous']:
            mask[n['id'],9:12]=1
    return mask


def fixed_raw(pred,cache,scene):
    mask=output_mask(cache,scene);base=cache['raw'].to(pred.device)
    result=mask*pred+(1-mask)*base
    assert torch.equal(result[:,12:18],base[:,12:18])
    assert torch.equal(result[mask==0],base[mask==0])
    return result


def continuous_from_raw(raw,scene):
    d,o,_=parent_decode(raw,scene)
    # Native serialized limits are the physical training path, for every step.
    return d,o,scene.params.limits0


def preserve_ranges(obj,scene):
    byid={n['id']:n for n in scene.asset['nodes']}
    for n in obj['diffuse_tree']:
        n['joint']['range']=copy.deepcopy(byid[n['id']]['joint']['range'])
    obj['geometry_source_object']=str(scene.asset['object_path'])
    obj['axis_only_fixed_endpoints']=True
    assert_ranges(obj,scene)
    return obj


def assert_ranges(obj,scene):
    byid={n['id']:n for n in scene.asset['nodes']}
    for n in obj['diffuse_tree']:
        assert n['joint']['range']==byid[n['id']]['joint']['range'],('Range changed',n['id'])
        assert n['joint']['type']==byid[n['id']]['joint']['type']
        assert n.get('parent')==byid[n['id']].get('parent')


def original_postprocessed(raw,scene):
    return preserve_ranges(parent_postprocess(raw,scene),scene)


def export_axis(scene):
    scene.assert_limits()
    return preserve_ranges(scene.params.export(scene.asset),scene)
