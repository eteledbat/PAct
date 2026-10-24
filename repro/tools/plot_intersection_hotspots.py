"""Show actual VTK transverse intersection segments through frozen mesh previews."""
import argparse
import json
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np

from collision_refinement_common import candidate_nodes, load_asset, moved, transforms

p = argparse.ArgumentParser()
p.add_argument('--root', required=True)
p.add_argument('--case', required=True)
a = p.parse_args()
root = pathlib.Path(a.root)
config = json.loads((root / 'experiment.json').read_text())
asset = load_asset(config['cases'][a.case])
out = root / f'hotspots_{a.case}'
out.mkdir(exist_ok=False)
methods = ['original', config.get('hotspot_variant', 'both')]
nodes = {m: candidate_nodes(asset, None if m == 'original' else root / f'{a.case}_{m}/candidate.json') for m in methods}
allpoints = []
states = {}
rng = np.random.default_rng(2026)
samples = [part['mesh'].vertices[rng.choice(len(part['mesh'].vertices), min(6000, len(part['mesh'].vertices)), replace=False)] for part in asset['parts']]
for method in nodes:
    for state in [0, 10, 20]:
        tr = transforms(nodes[method], state / 20)
        points = [moved(v, tr[p['node']['id']]) for v, p in zip(samples, asset['parts'])]
        states[method, state] = points
        allpoints.extend(points)
cloud = np.concatenate(allpoints)
center = (cloud.min(0) + cloud.max(0)) / 2
radius = (cloud.max(0) - cloud.min(0)).max() * .6
palette = ['#e88b31', '#43a875', '#888888', '#aa66cc', '#bc4057']
regions = []
for state in [0, 10, 20]:
    fig = plt.figure(figsize=(15, 9))
    for row, method in enumerate(methods):
        path = root / f'mesh_{a.case}_{method}/segments_{state:02d}.json'
        if path.exists():
            segments = json.loads(path.read_text())
        else:
            metrics = json.loads((path.parent / 'metrics.json').read_text())
            assert sum(p['line_length_over_diagonal'] for p in metrics['states'][state]['pairs']) == 0
            segments = []
        for group in segments:
            if group['segments']:
                endpoints = np.asarray(group['segments']).reshape(-1, 3)
                regions.append({'method': method, 'state': state, 'ids': group['ids'], 'shown_segment_count': len(group['segments']),
                                'bounds_world': [endpoints.min(0).tolist(), endpoints.max(0).tolist()]})
        for view, azim in enumerate([25, 125, 225]):
            ax = fig.add_subplot(2, 3, row * 3 + view + 1, projection='3d')
            for i, v in enumerate(states[method, state]):
                ax.scatter(*v.T, s=.18, color=palette[i % len(palette)], alpha=.12, rasterized=True)
            for group in segments:
                if group['segments']:
                    ax.add_collection3d(Line3DCollection(group['segments'], colors='red', linewidths=.65, alpha=.85))
            ax.set(xlim=(center[0]-radius, center[0]+radius), ylim=(center[1]-radius, center[1]+radius),
                   zlim=(center[2]-radius, center[2]+radius), xlabel='x', ylabel='y (up)', zlabel='z')
            ax.set_box_aspect([1, 1, 1])
            ax.view_init(elev=15, azim=azim, vertical_axis='y')
            ax.set_title(method)
    fig.suptitle(f'{a.case} t={state/20:.1f}: X-ray transverse surface intersections (red); NOT solid penetration volume')
    fig.tight_layout()
    fig.savefig(out / f'intersections_state{state:02d}.png', dpi=150)
    plt.close(fig)
(out / 'regions.json').write_text(json.dumps({'regions': regions, 'display_cap': 'At most5000 transverse segments per pair; see full numerical metrics separately',
                                            'interpretation': 'Actual VTK surface intersection segments, includes possible legal contact; not material penetration labels'}, indent=2))
print(out)
