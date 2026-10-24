"""Frozen PAct exports and kinematics for an independent collision experiment.

No GT or previous GT evaluations are read here. Distances are model units.
"""
import copy
import hashlib
import json
import pathlib

import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree
import trimesh


def sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def load_asset(run, gaussian_count=0, seed=2026):
    run = pathlib.Path(run).resolve()
    paths = list(run.rglob('object.json'))
    assert len(paths) == 1
    path = paths[0]
    obj = json.loads(path.read_text())
    nodes = obj['diffuse_tree']
    parts = []
    undo = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)
    for node in nodes:
        mesh_path = path.parent / node['plys'][0]
        mesh = trimesh.load(mesh_path, process=False, force='mesh')
        part = {'node': node, 'mesh': mesh, 'mesh_path': mesh_path,
                'mesh_sha256': sha256(mesh_path)}
        if gaussian_count:
            gs_path = path.parent / node['gs'][0]
            v = PlyData.read(gs_path)['vertex'].data
            xyz = np.column_stack([v[k] for k in ['x', 'y', 'z']]).astype(float) @ undo.T
            alpha = 1 / (1 + np.exp(-v['opacity'].astype(float)))
            sigma = np.exp(np.column_stack([v[k] for k in ['scale_0', 'scale_1', 'scale_2']]).astype(float)).mean(1)
            kept = np.flatnonzero(alpha >= .05)
            ids = kept.copy()
            rng = np.random.default_rng(seed + int(node['id']))
            if len(ids) > gaussian_count:
                ids = np.sort(rng.choice(ids, gaussian_count, replace=False))
            part.update(xyz=xyz[ids], alpha=alpha[ids], sigma=sigma[ids],
                        full_xyz=xyz[kept], full_alpha=alpha[kept], full_sigma=sigma[kept],
                        retained_ids=kept,
                        gs_path=gs_path, gs_sha256=sha256(gs_path), sample_ids=ids)
        parts.append(part)
    bounds = np.array([p['mesh'].bounds for p in parts])
    diagonal = float(np.linalg.norm(bounds[:, 1].max(0) - bounds[:, 0].min(0)))
    return {'run': run, 'object_path': path, 'object': obj, 'nodes': nodes,
            'parts': parts, 'diagonal': diagonal}


def surface(mesh, count, seed):
    rng = np.random.default_rng(seed)
    ids = rng.choice(len(mesh.faces), count, p=mesh.area_faces / mesh.area)
    tri = mesh.vertices[mesh.faces[ids]]
    uv = rng.random((count, 2))
    flip = uv.sum(1) > 1
    uv[flip] = 1 - uv[flip]
    return tri[:, 0] + uv[:, :1] * (tri[:, 1] - tri[:, 0]) + uv[:, 1:] * (tri[:, 2] - tri[:, 0])


def local_transform(node, t):
    j = node['joint']
    tr = np.eye(4)
    d = np.asarray(j['axis']['direction'], float)
    if np.linalg.norm(d) > 1e-12:
        d = d / np.linalg.norm(d)
    else:
        d = np.array([0., 0., 1.])
    o = np.asarray(j['axis']['origin'], float)
    lo, hi = j['range']
    q = lo + (hi - lo) * t
    if j['type'] in ['revolute', 'continuous']:
        x, y, z = d
        K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        theta = np.deg2rad(q)
        R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
        tr[:3, :3] = R
        tr[:3, 3] = o - R @ o
    elif j['type'] in ['prismatic', 'screw']:
        tr[:3, 3] = d * q
    return tr


def transforms(nodes, t):
    lookup = {n['id']: n for n in nodes}
    world = {-1: np.eye(4)}

    def visit(nid):
        if nid in world:
            return world[nid]
        n = lookup[nid]
        world[nid] = visit(n.get('parent', -1)) @ local_transform(n, t)
        return world[nid]

    for nid in lookup:
        visit(nid)
    return world


def moved(points, transform):
    return points @ transform[:3, :3].T + transform[:3, 3]


def gaussian_pair(x, y, a, b, sa, sb):
    da, ia = cKDTree(y).query(x, workers=4)
    db, ib = cKDTree(x).query(y, workers=4)
    ka = b[ia] * np.exp(-da**2 / (2 * (sa**2 + sb[ia]**2)))
    kb = a[ib] * np.exp(-db**2 / (2 * (sb**2 + sa[ib]**2)))
    return {'gaussian_nearest_center_proxy': float((a @ ka + b @ kb) / (a.sum() + b.sum())),
            'minimum_sampled_center_distance': float(min(da.min(), db.min())),
            'negative_nearest_pair_1sigma_gap_fraction': float(np.r_[da - sa - sb[ia], db - sb - sa[ib]].__lt__(0).mean())}


def candidate_nodes(asset, candidate):
    return copy.deepcopy(asset['nodes']) if candidate is None else json.loads(pathlib.Path(candidate).read_text())['diffuse_tree']
