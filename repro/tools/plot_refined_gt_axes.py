"""After candidate sealing, overlay the original and all refined axes with GT."""
import argparse
import json
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import trimesh

from collision_refinement_common import candidate_nodes, load_asset

p = argparse.ArgumentParser()
p.add_argument('--root', required=True)
a = p.parse_args()
root = pathlib.Path(a.root)
config = json.loads((root / 'experiment.json').read_text())
asset = load_asset(config['cases']['31249'])
out = root / 'GT_axis_previews'
out.mkdir(exist_ok=False)
base = json.loads((asset['run'] / 'joint_evaluation/joint_errors.json').read_text())
provenance = json.loads((asset['run'] / 'input_provenance.json').read_text())
R, s, t = np.array(base['R_export_to_gtworld']), base['scale'], np.array(base['translation'])
D = base['GT_static_diagonal']
methods = ['original', 'position', 'direction', 'both']
nodes = {}
for m in methods:
    if m != 'original':
        assert (root / f'31249_{m}/prediction_seal.json').exists()
        assert json.loads((root / f'31249_{m}/status.json').read_text())['state'] == 'complete'
    nodes[m] = {n['id']: n for n in candidate_nodes(asset, None if m == 'original' else root / f'31249_{m}/candidate.json')}
mapping = provenance['part_correspondence']
root_id = next(x['pact_node_id'] for x in mapping if x['gt_mesh_index'] == 0)
gt_root = trimesh.load(pathlib.Path(provenance['source_data_root']) / 'gt/part_0.ply', process=False)
pred_root = next(p['mesh'] for p in asset['parts'] if p['node']['id'] == root_id)
center = gt_root.bounds.mean(0)
rng = np.random.default_rng(2026)

for joint in base['joints']:
    pid = joint['predicted_node_id']
    target = joint['GT_type']
    if nodes['original'][pid]['joint']['type'] != target:
        continue
    go, gd = np.array(joint['GT_origin_world']), np.array(joint['GT_direction_world'])
    gm = trimesh.load(pathlib.Path(provenance['source_data_root']) / f'gt/part_{joint["gt_mesh_index"]}.ply', process=False)
    pm = next(p['mesh'] for p in asset['parts'] if p['node']['id'] == pid)
    geometry = [(gt_root.vertices, 'grey', 'GT base'), (gm.vertices, '#77aa77', 'GT part'),
                (s * (pred_root.vertices @ R.T) + t, 'lightblue', 'PAct base'),
                (s * (pm.vertices @ R.T) + t, '#e7b080', 'PAct part')]
    subsets = [(v[rng.choice(len(v), min(4000, len(v)), replace=False)], c, label) for v, c, label in geometry]
    fig = plt.figure(figsize=(15, 5))
    for view, azim in enumerate([25, 125, 225]):
        ax = fig.add_subplot(1, 3, view + 1, projection='3d')
        for v, color, label in subsets:
            ax.scatter(*v.T, s=.3, c=color, alpha=.13, rasterized=True, label=label)
        axes = [(go, gd, 'red', 'GT')]
        for m, color in zip(methods, ['blue', '#00aaaa', '#88aa22', '#9b36b7']):
            j = nodes[m][pid]['joint']
            axes.append((s * (R @ np.array(j['axis']['origin'])) + t,
                         R @ np.array(j['axis']['direction']), color, m))
        for origin, direction, color, label in axes:
            direction = direction / np.linalg.norm(direction)
            projected = origin + np.dot(center - origin, direction) * direction
            line = projected + np.array([-.45, .45])[:, None] * D * direction
            ax.plot(*line.T, c=color, lw=1.6, label=label + ' axis')
        rad = D * .6
        ax.set(xlim=(center[0]-rad, center[0]+rad), ylim=(center[1]-rad, center[1]+rad),
               zlim=(center[2]-rad, center[2]+rad), xlabel='GT world x', ylabel='y', zlabel='z')
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=15, azim=azim)
        if view == 0:
            ax.legend(fontsize=6)
    fig.suptitle(f'31249 GT mobility {joint["gt_mobility_id"]} ({target}) | shared original alignment, frozen geometry')
    fig.tight_layout()
    fig.savefig(out / f'GT_mobility_{joint["gt_mobility_id"]}.png', dpi=160)
    plt.close(fig)
print(out)
