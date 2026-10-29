"""Fit original conditioning RGB and verified instance masks on auto-axis assets.

One deformation group is kept for each original PAct source node.  Automatic
semantics only chose exported joints; geometric supervision uses only masks
whose source-node correspondence passed the preexisting projection gate.
GT geometry, joints and held-out views are read only in the separate evaluator.
"""
import argparse
import copy
import json
import pathlib
import shutil
import sys

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData

from collision_refinement_common import load_asset, sha256
from multicase_closed_common import Model, tensor, write_json
from original_mask_multicase import OriginalMaskModel, mask_scores
import original_mask_semantic_100481 as original_fit


def read(path):
    return json.loads(pathlib.Path(path).read_text())


def basis_for(node, vertices):
    joint = node['joint']
    if joint['type'] not in {'revolute', 'continuous'}:
        return np.eye(3), np.asarray(vertices).mean(0), 1.0, 'fixed' if joint['type'] == 'fixed' else 'slider_z'
    pivot = np.asarray(joint['axis']['origin'], float)
    axis = np.asarray(joint['axis']['direction'], float)
    norm = np.linalg.norm(axis)
    if norm < 1e-8:
        return np.eye(3), np.asarray(vertices).mean(0), 1.0, 'fixed'
    axis /= norm
    radial = np.asarray(vertices).mean(0) - pivot
    radial -= (radial @ axis) * axis
    if np.linalg.norm(radial) < 1e-8:
        radial = np.eye(3)[int(np.argmin(np.abs(axis)))]
        radial -= (radial @ axis) * axis
    radial /= np.linalg.norm(radial)
    normal = np.cross(radial, axis)
    basis = np.stack([radial, axis, normal], 1)
    length = max(float(np.quantile((np.asarray(vertices) - pivot) @ radial, .99)), 1e-3)
    return basis, pivot, length, 'hinge_auto'


class AutoFitModel(OriginalMaskModel):
    def losses(self, align=False):
        geometry = self.geometry()
        rgb = self.render(geometry)
        alpha = self.render(geometry, 'silhouette')
        photo = ((rgb-self.rgb).abs()*self.weight[None]).mean()/self.weight.mean()
        silhouette = (alpha-self.alpha).abs().mean()
        if align:
            return photo+2*silhouette, dict(photo=float(photo.detach()), silhouette=float(silhouette.detach()))
        probs = self.probabilities(geometry)
        dice = []
        ce = []
        for j, gid in enumerate(self.visible_owners):
            target = (self.targets == gid).float()
            p = probs[j]
            dice.append(1-(2*(p*target).sum()+1)/(p.sum()+target.sum()+1))
            if self.interior[gid].any():
                ce.append(-p[self.interior[gid]].clamp_min(1e-5).log().mean())
        part_dice = torch.stack(dice).mean() if dice else photo*0
        balanced_ce = torch.stack(ce).mean() if ce else photo*0
        prior = .002*self.log_shape.square().mean()+.01*self.vertical_shift.square().mean()
        loss = photo+2*silhouette+.5*part_dice+.1*balanced_ce+prior
        return loss, dict(photo=float(photo.detach()), silhouette=float(silhouette.detach()),
                          part_dice=float(part_dice.detach()), balanced_CE=float(balanced_ce.detach()))


def prepare_case(auto_root, registration_root, output, case):
    output.mkdir(parents=True, exist_ok=False)
    auto = auto_root/case
    reg = registration_root/case
    source_run = pathlib.Path(read(auto/'inputs.json')['source_run'])
    asset = load_asset(source_run)
    obj = read(auto/'axis_only_export/object.json')
    exported = {n['id']: n for n in obj['diffuse_tree']}
    source = {n['id']: n for n in asset['nodes']}
    assert set(exported) == set(source)
    static = next(n['id'] for n in asset['nodes'] if n['joint']['type'] == 'fixed' and n['parent'] == -1)
    order = [static] + sorted(n for n in source if n != static)
    gids = {nid: 10+i for i, nid in enumerate(order)}
    membership = {}
    groups = []
    files = []
    sources = {}
    for nid in order:
        p = next(p for p in asset['parts'] if p['node']['id'] == nid)
        node = exported[nid]
        vertices = np.asarray(p['mesh'].vertices)
        basis, anchor, length, kind = basis_for(node, vertices)
        gs = pathlib.Path(asset['object_path'].parent/source[nid]['gs'][0])
        count = len(PlyData.read(gs)['vertex'].data)
        gid = gids[nid]
        prefix = f'g{gid}'
        for key, n in [('vertices', len(vertices)), ('faces', len(p['mesh'].faces)), ('gaussians', count)]:
            membership[f'{prefix}_{key}'] = np.arange(n, dtype=np.int64)
        groups.append(dict(id=gid, name=node['name'], source_node=nid, kind=kind,
                           parent=-1 if node['parent'] == -1 else gids[node['parent']],
                           components=None, shape_owner=None, box=None, edge=None,
                           membership_prefix=prefix,
                           counts=dict(vertices=len(vertices), faces=len(p['mesh'].faces), gaussians=count),
                           joint=copy.deepcopy(node['joint']), observed_q=0.,
                           deformation_basis=basis.tolist(), deformation_anchor=anchor.tolist(),
                           length=length))
        sources[str(nid)] = dict(mesh=str(p['mesh_path']), gs=str(gs))
        files.extend(dict(path=str(path), sha256=sha256(path)) for path in [p['mesh_path'], gs])
    np.savez_compressed(output/'membership.npz', **membership)
    requests = read(auto/'axis_requests.json')['parts']
    verified = {r['instance_id']: gids[r['native_node_id']] for r in requests
                if r['correspondence']['verified'] and r['native_node_id'] in gids}
    # The Hungarian assignment is one-to-one; accepted masks therefore remain
    # independent even if two source nodes have the same predicted category.
    assert len(verified.values()) == len(set(verified.values()))
    ids = np.load(auto/'instance_ids.npy')
    ids256 = np.asarray(Image.fromarray(ids.astype('uint8')).resize((256,256), Image.Resampling.NEAREST))
    labels = np.zeros_like(ids256)
    for instance, gid in verified.items():
        labels[ids256 == instance] = gid
    old = np.load(reg/'observation.npz')
    palette = {str(g['id']): [float(x) for x in np.random.default_rng(g['id']).uniform(.2,.9,3)] for g in groups}
    semantic = np.ones((256,256,3), np.float32)
    for gid in np.unique(labels):
        if gid: semantic[labels == gid] = palette[str(int(gid))]
    np.savez_compressed(output/'observation.npz', rgb=old['rgb'], alpha=old['alpha'],
                        labels=labels, semantic=semantic, extrinsics=old['extrinsics'], intrinsics=old['intrinsics'])
    shutil.copyfile(reg/'registration.pt', output/'registration.pt')
    cfg = read(reg/'config.json')
    cfg.update(prior_root=str(output), palette=palette, observations_sha256=sha256(output/'observation.npz'),
               geometry_steps=240, alignment_steps=0, initial_frame_fit='Auto-axis source groups; only verified original masks supervise parts; RGB and full silhouette supervise all geometry')
    cfg.pop('recipe_sha256', None)
    manifest = dict(schema='auto_semantic_initial_frame_v1', case=case, source_run=str(source_run),
                    source_object=str(asset['object_path']), source_object_sha256=sha256(asset['object_path']),
                    auto_export=str(auto/'axis_only_export/object.json'),
                    auto_export_sha256=sha256(auto/'axis_only_export/object.json'), sources=sources,
                    groups=groups, pairs=[], membership_sha256=sha256(output/'membership.npz'),
                    geometry_files=files, verified_instance_to_group=verified,
                    selection='One original PAct source node per deformation group; joints from sealed automatic export')
    write_json(output/'motion_groups.json', manifest)
    cfg['prior_manifest_sha256'] = sha256(output/'motion_groups.json')
    write_json(output/'config.json', cfg)
    write_json(output/'prepared.json', dict(case=case, verified_instance_count=len(verified),
                                           source_node_count=len(groups), original_mask_changed=False,
                                           source_rgb=cfg['source_image'], source_registration=str(reg/'registration.pt')))


def load_model(root, registered=False, checkpoint=False):
    model = AutoFitModel(read(root/'config.json'), read(root/'motion_groups.json'), root)
    if registered:
        state = torch.load(root/'registration.pt', map_location='cuda', weights_only=False)
        with torch.no_grad():
            for name in ['log_global_scale', 'screen_translation']:
                getattr(model, name).copy_(state[name])
    if checkpoint:
        model.load_state_dict(torch.load(root/'checkpoint.pt', map_location='cuda', weights_only=False)['model'])
    return model


def guard(root):
    denied = []
    def audit(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes)):
            path = str(args[0])
            if any(s in path for s in ['/gt/', '/gt_mesh/', 'mobility_v2', 'joint_infos',
                                       'joint_info.json', 'joint_evaluation', '/depth/']):
                denied.append(path)
                raise PermissionError('GT observation forbidden during initial-frame fit: '+path)
    sys.addaudithook(audit)
    return denied, []


def fit_case(root):
    original_fit.training_guard = guard
    original_fit.load_model = load_model
    original_fit.mask_scores = mask_scores
    original_fit.fit(root)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--auto-root', type=pathlib.Path, required=True)
    p.add_argument('--registration-root', type=pathlib.Path, required=True)
    p.add_argument('--output', type=pathlib.Path, required=True)
    p.add_argument('--stage', choices=['prepare','fit'], required=True)
    p.add_argument('--case')
    a = p.parse_args()
    auto = a.auto_root.resolve()
    reg = a.registration_root.resolve()
    out = a.output.resolve()
    cases = [a.case] if a.case else read(auto/'protocol.json')['cases']
    if a.stage == 'prepare':
        out.mkdir(parents=True, exist_ok=False)
        write_json(out/'protocol.json', dict(cases=cases, auto_root=str(auto), registration_root=str(reg),
                                            input='Original conditioning RGB and existing exact instance masks',
                                            axis='Sealed automatic axis-only export',
                                            geometry='One original PAct node per deformation group; 240 steps; frozen axes and original RGB/silhouette registration',
                                            mask='Only automatically verified source-node correspondences enter instance-mask loss; no manual semantic labels or physical groups',
                                            GT='Excluded until all fits complete'))
        for case in cases:
            prepare_case(auto, reg, out/case, case)
    else:
        for case in cases:
            root = out/case
            if (root/'fit_results.json').exists():
                continue
            fit_case(root)


if __name__ == '__main__':
    main()
