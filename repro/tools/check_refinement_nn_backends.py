"""Compare independent CPU/GPU neighbors and kernels on real moving subsets."""
import json
import pathlib

import numpy as np
import torch
from kaolin.metrics.pointcloud import sided_distance
from scipy.spatial import cKDTree

from collision_refinement_common import candidate_nodes, load_asset, moved, transforms

repo = pathlib.Path(__file__).resolve().parents[2]
root = pathlib.Path((repo / 'repro/current_refinement.txt').read_text().strip())
config = json.loads((root / 'experiment.json').read_text())
torch.set_num_threads(2)
rows = []
for name, run in config['cases'].items():
    asset = load_asset(run, gaussian_count=128, seed=2026)
    parts = asset['parts']
    trees = [cKDTree(p['full_xyz']) for p in parts]
    targets = [torch.tensor(p['full_xyz'], device='cuda', dtype=torch.float32)[None] for p in parts]
    for method in ['original', 'position', 'direction', 'both']:
        nodes = candidate_nodes(asset, None if method == 'original' else root / f'{name}_{method}/candidate.json')
        for t in [0., .5, 1.]:
            tr = transforms(nodes, t)
            for i, source in enumerate(parts):
                for j, target in enumerate(parts):
                    if i == j:
                        continue
                    relative = np.linalg.inv(tr[target['node']['id']]) @ tr[source['node']['id']]
                    q = moved(source['xyz'], relative)
                    dc, ic = trees[j].query(q)
                    with torch.no_grad():
                        ig = sided_distance(torch.tensor(q, device='cuda', dtype=torch.float32)[None], targets[j])[1][0].cpu().numpy().astype(np.int64)
                    dg = np.linalg.norm(q - target['full_xyz'][ig], axis=1)
                    def kernel(d, idx):
                        return source['alpha'] * target['full_alpha'][idx] * np.exp(-d**2 / (2 * (source['sigma']**2 + target['full_sigma'][idx]**2)))
                    kc, kg = kernel(dc, ic), kernel(dg, ig)
                    rows.append({'case': name, 'method': method, 't': t, 'source': source['node']['id'], 'target': target['node']['id'],
                                 'max_distance_excess_GPU_over_CPU': float((dg - dc).max()),
                                 'neighbor_index_mismatches': int((ic != ig).sum()),
                                 'max_kernel_absolute_difference': float(abs(kc - kg).max()),
                                 'mean_kernel_absolute_difference': float(abs(kc - kg).mean())})
assert max(r['max_distance_excess_GPU_over_CPU'] for r in rows) < 2e-6
result = {'passed_distance_tolerance': True, 'source_points_per_pair': 128, 'states': [0., .5, 1.],
          'scope': 'Real-motion numerical backend comparison, not collision improvement statistics',
          'max_kernel_absolute_difference': max(r['max_kernel_absolute_difference'] for r in rows),
          'max_mean_kernel_absolute_difference': max(r['mean_kernel_absolute_difference'] for r in rows), 'rows': rows}
(root / 'nn-backend-check.json').write_text(json.dumps(result, indent=2))
print(json.dumps({k:v for k,v in result.items() if k != 'rows'}, indent=2))
