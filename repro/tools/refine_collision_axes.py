"""Independent bounded axis refinement on frozen exported Gaussians.

This is a Gaussian-overlap feasibility experiment, NOT a solid collision solver.
No GT, pretrained model, geometry updates, range changes, or axis snapping.
"""
import argparse
import copy
import datetime
import json
import os
import pathlib
import sys
import time

import numpy as np
import torch
from scipy.spatial import cKDTree

from collision_refinement_common import load_asset, sha256, transforms


class AxisParameters(torch.nn.Module):
    def __init__(self, asset, variant, device='cuda'):
        super().__init__()
        self.nodes = asset['nodes']
        self.variant = variant
        self.diagonal = asset['diagonal']
        self.max_tangent = float(np.tan(np.deg2rad(5.)))
        self.max_position = .05 * self.diagonal
        self.active = [n for n in self.nodes if n['joint']['type'] in ['revolute', 'continuous', 'prismatic', 'screw']]
        self.indices = {n['id']: i for i, n in enumerate(self.active)}
        directions, origins, bases, pivot_mask = [], [], [], []
        for node in self.active:
            d = np.asarray(node['joint']['axis']['direction'], float)
            d /= np.linalg.norm(d)
            reference = np.eye(3)[np.argmin(abs(d))]
            u = np.cross(d, reference)
            u /= np.linalg.norm(u)
            basis = np.stack([u, np.cross(d, u)], axis=1)
            directions.append(d)
            origins.append(node['joint']['axis']['origin'])
            bases.append(basis)
            pivot_mask.append(node['joint']['type'] in ['revolute', 'continuous'])
        for name, value in [('d0', directions), ('o0', origins), ('basis', bases), ('pivot_mask', np.array(pivot_mask)[:, None])]:
            self.register_buffer(name, torch.tensor(np.array(value), device=device, dtype=torch.float32))
        self.theta = torch.nn.Parameter(torch.zeros(len(self.active), 2, device=device))
        self.phi = torch.nn.Parameter(torch.zeros(len(self.active), 2, device=device))
        self.theta.requires_grad_(variant in ['direction', 'both'])
        self.phi.requires_grad_(variant in ['position', 'both'])

    def axes(self):
        d = self.d0 + torch.einsum('nij,nj->ni', self.basis, self.theta)
        d = torch.nn.functional.normalize(d, dim=-1)
        o = self.o0 + torch.einsum('nij,nj->ni', self.basis, self.phi * self.pivot_mask)
        return d, o

    def regularizer(self):
        direction = (self.theta / self.max_tangent).square().sum(-1).mean()
        mask = self.pivot_mask[:, 0]
        position = ((self.phi / self.max_position).square().sum(-1) * mask).sum() / mask.sum().clamp_min(1)
        return direction + position

    @torch.no_grad()
    def project(self):
        for param, bound in [(self.theta, self.max_tangent), (self.phi, self.max_position)]:
            norm = param.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            param.mul_((bound / norm).clamp_max(1))
        self.phi.mul_(self.pivot_mask)

    def world(self, times):
        d, o = self.axes()
        B = len(times)
        identity = torch.eye(4, device=times.device).expand(B, 4, 4)
        lookup = {n['id']: n for n in self.nodes}
        world = {-1: identity}

        def visit(nid):
            if nid in world:
                return world[nid]
            node = lookup[nid]
            local = identity.clone()
            if nid in self.indices:
                idx = self.indices[nid]
                axis, origin = d[idx], o[idx]
                lo, hi = node['joint']['range']
                q = lo + (hi - lo) * times
                if node['joint']['type'] in ['revolute', 'continuous']:
                    x, y, z = axis.unbind()
                    zero = x * 0
                    K = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero]).reshape(3, 3)
                    angle = torch.deg2rad(q)
                    rot = torch.eye(3, device=times.device) + angle.sin()[:, None, None] * K + (1 - angle.cos())[:, None, None] * (K @ K)
                    local[:, :3, :3] = rot
                    local[:, :3, 3] = origin - torch.einsum('bij,j->bi', rot, origin)
                else:
                    local[:, :3, 3] = q[:, None] * axis
            world[nid] = visit(node.get('parent', -1)) @ local
            return world[nid]

        for node in self.nodes:
            visit(node['id'])
        return world

    def exported(self, asset):
        obj = copy.deepcopy(asset['object'])
        directions, origins = [x.detach().cpu().numpy() for x in self.axes()]
        for node in obj['diffuse_tree']:
            if node['id'] in self.indices:
                index = self.indices[node['id']]
                if self.variant in ['direction', 'both']:
                    node['joint']['axis']['direction'] = directions[index].astype(float).tolist()
                # Prismatic pivot is meaningless; preserve its original serialized value.
                if self.variant in ['position', 'both'] and node['joint']['type'] in ['revolute', 'continuous']:
                    node['joint']['axis']['origin'] = origins[index].astype(float).tolist()
        obj['geometry_source_object'] = str(asset['object_path'])
        return obj


class GaussianObjective:
    def __init__(self, asset, params, times, freeze_correspondences=False):
        self.asset, self.params, self.times = asset, params, times
        self.freeze_correspondences = freeze_correspondences
        self.correspondences = {}
        self.data = [{k: torch.tensor(p[k], device='cuda', dtype=torch.float32) for k in ['xyz', 'alpha', 'sigma', 'full_xyz', 'full_alpha', 'full_sigma']} for p in asset['parts']]
        self.trees = [cKDTree(p['full_xyz']) for p in asset['parts']]

    def __call__(self):
        world = self.params.world(self.times)
        points = []
        for part, data in zip(self.asset['parts'], self.data):
            tr = world[part['node']['id']]
            points.append(torch.einsum('nj,bij->bni', data['xyz'], tr[:, :3, :3]) + tr[:, None, :3, 3])
        values = []
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                x, y, a, b = points[i], points[j], self.data[i], self.data[j]
                ti = world[self.asset['parts'][i]['node']['id']]
                tj = world[self.asset['parts'][j]['node']['id']]
                # Rigid-invariant exact NN queries into ALL retained target centers.
                # Only source quadrature is sampled, avoiding a sparsified density field.
                if self.freeze_correspondences and (i, j) in self.correspondences:
                    ix, iy = self.correspondences[i, j]
                else:
                    with torch.no_grad():
                        xlocal = torch.einsum('bni,bij->bnj', x - tj[:, None, :3, 3], tj[:, :3, :3])
                        ylocal = torch.einsum('bni,bij->bnj', y - ti[:, None, :3, 3], ti[:, :3, :3])
                        ix = torch.tensor(self.trees[j].query(xlocal.cpu().numpy(), workers=4)[1], device='cuda')
                        iy = torch.tensor(self.trees[i].query(ylocal.cpu().numpy(), workers=4)[1], device='cuda')
                    if self.freeze_correspondences:
                        self.correspondences[i, j] = (ix, iy)
                ymatch = torch.einsum('bnj,bij->bni', b['full_xyz'][ix], tj[:, :3, :3]) + tj[:, None, :3, 3]
                xmatch = torch.einsum('bnj,bij->bni', a['full_xyz'][iy], ti[:, :3, :3]) + ti[:, None, :3, 3]
                dx = (x - ymatch).square().sum(-1)
                dy = (y - xmatch).square().sum(-1)
                kx = b['full_alpha'][ix] * torch.exp(-dx / (2 * (a['sigma'][None, :].square() + b['full_sigma'][ix].square())))
                ky = a['full_alpha'][iy] * torch.exp(-dy / (2 * (b['sigma'][None, :].square() + a['full_sigma'][iy].square())))
                # Uniform-source Horvitz-Thompson estimate of the full-center numerator.
                wa = len(a['full_xyz']) / len(a['xyz'])
                wb = len(b['full_xyz']) / len(b['xyz'])
                values.append((wa * (kx * a['alpha']).sum(-1) + wb * (ky * b['alpha']).sum(-1)) / (a['full_alpha'].sum() + b['full_alpha'].sum()))
        return torch.stack(values).mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--variant', choices=['direction', 'position', 'both'], default='both')
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--samples', type=int, default=2048)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    out = pathlib.Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=args.resume)
    blocked = []

    def guard(event, values):
        if event == 'open' and isinstance(values[0], (str, bytes)):
            path = os.fsdecode(values[0])
            if any(token in path for token in ['/VideoArtGS/', '/FreeArt3D/', '/Part2GS-data/', '/gt/', 'mobility', 'joint_infos', 'input_provenance', 'joint_evaluation', 'gt-joint-results']):
                blocked.append(path)
                raise PermissionError('GT/evaluation access is forbidden in refinement: ' + path)

    sys.addaudithook(guard)
    started = time.monotonic()
    torch.manual_seed(0)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    asset = load_asset(args.run, args.samples, seed=0)
    params = AxisParameters(asset, args.variant)
    # 21 equally spaced training states. t=0 has zero joint gradient by construction.
    objective = GaussianObjective(asset, params, torch.linspace(0, 1, 21, device='cuda'))
    with torch.no_grad():
        baseline = float(objective())
    assert baseline > 1e-12 and np.isfinite(baseline)
    optimizer = torch.optim.Adam([
        {'params': [params.theta], 'lr': .002}, {'params': [params.phi], 'lr': .0005 * asset['diagonal']}])
    protocol = {'run': str(asset['run']), 'variant': args.variant, 'steps': args.steps, 'samples_per_part': args.samples,
                'training_states': 21, 'sample_seed': 0, 'axis_bound_degrees': 5, 'pivot_bound_fraction_asset_diagonal': .05,
                'sampling': 'Uniform source quadrature, exact NN in all opacity-retained target centers, count reweighting for the full opacity-weighted kernel',
                'regularization_weight': .05, 'objective': 'mean nearest-center Gaussian overlap / original overlap + .05 * normalized quadratic axis/pivot displacement',
                'geometry_frozen': True, 'range_frozen': True, 'type_frozen': True, 'selection': 'Fixed final step only, no GT reads; all three ablations reported',
                'original_object_sha256': sha256(asset['object_path']),
                'source_sha256': {str(p): sha256(p) for p in [pathlib.Path(__file__), pathlib.Path(__file__).with_name('collision_refinement_common.py')]},
                'frozen_files': [{'mesh': str(p['mesh_path']), 'mesh_sha256': p['mesh_sha256'], 'gs': str(p['gs_path']), 'gs_sha256': p['gs_sha256']} for p in asset['parts']]}
    history, start = [], 0
    if args.resume:
        assert json.loads((out / 'protocol.json').read_text()) == protocol, 'Resume protocol/source mismatch'
        saved = torch.load(out / 'checkpoint.pt', map_location='cuda', weights_only=False)
        params.load_state_dict(saved['parameters'])
        optimizer.load_state_dict(saved['optimizer'])
        history, start = saved['history'], saved['step']
    else:
        (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))

    def save(state, step, **extra):
        status = {'state': state, 'step': step, 'total_steps': args.steps, 'variant': args.variant,
                  'elapsed_seconds': time.monotonic() - started, 'gpu_peak_allocated_MiB': torch.cuda.max_memory_allocated() / 2**20,
                  'updated_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra}
        tmp = out / 'status.tmp'
        tmp.write_text(json.dumps(status, indent=2, allow_nan=False))
        tmp.replace(out / 'status.json')

    try:
        for step in range(start, args.steps):
            optimizer.zero_grad(set_to_none=True)
            collision = objective()
            loss = collision / baseline + .05 * params.regularizer()
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            loss.backward()
            gradients = {name: float(p.grad.norm()) if p.grad is not None else None for name, p in [('direction', params.theta), ('pivot', params.phi)]}
            for p in params.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    raise RuntimeError('Nonfinite joint gradient')
            optimizer.step()
            params.project()
            row = {'step': step + 1, 'loss': float(loss.detach()), 'gaussian_proxy': float(collision.detach()), 'gradient_norms': gradients}
            history.append(row)
            if (step + 1) % 10 == 0 or step == 0 or step + 1 == args.steps:
                tmp = out / 'checkpoint.tmp'
                torch.save({'parameters': params.state_dict(), 'optimizer': optimizer.state_dict(), 'history': history, 'step': step + 1}, tmp)
                tmp.replace(out / 'checkpoint.pt')
                save('running', **row)
                print(json.dumps(row), flush=True)
        prediction = params.exported(asset)
        for before, after in zip(asset['nodes'], prediction['diffuse_tree']):
            assert before['joint']['range'] == after['joint']['range']
            assert before['joint']['type'] == after['joint']['type']
        (out / 'candidate.json').write_text(json.dumps(prediction, indent=2, allow_nan=False))
        (out / 'history.json').write_text(json.dumps(history, indent=2, allow_nan=False))
        with torch.no_grad():
            final = float(objective())
        for part in asset['parts']:
            assert sha256(part['mesh_path']) == part['mesh_sha256']
            assert sha256(part['gs_path']) == part['gs_sha256']
        seal = {'candidate_sha256': sha256(out / 'candidate.json'), 'GT_reads_blocked': blocked,
                'frozen_files_unchanged': True, 'final_training_proxy': final,
                'initial_training_proxy': baseline, 'selection': 'Fixed final step, not best collision or GT score'}
        (out / 'prediction_seal.json').write_text(json.dumps(seal, indent=2))
        save('complete', args.steps, **seal)
        print('COMPLETE', out, flush=True)
    except BaseException as exc:
        save('failed', len(history), error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    main()
