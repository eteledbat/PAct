"""Read-only motion screening. Gaussian overlap is not solid collision."""
import argparse
import json
import pathlib
import time

import numpy as np
from collision_refinement_common import load_asset, transforms, moved, gaussian_pair

parser = argparse.ArgumentParser()
parser.add_argument('--run', required=True)
parser.add_argument('--out', required=True)
parser.add_argument('--gaussians', type=int, default=12000)
args = parser.parse_args()
out = pathlib.Path(args.out)
out.mkdir(parents=True, exist_ok=False)
asset = load_asset(args.run, args.gaussians)
parts = asset['parts']
started = time.monotonic()
curves = []
for state, t in enumerate(np.linspace(0, 1, 21)):
    world = transforms(asset['nodes'], float(t))
    points = [moved(p['xyz'], world[p['node']['id']]) for p in parts]
    pairs = []
    for i, a in enumerate(parts):
        for j in range(i + 1, len(parts)):
            b = parts[j]
            result = gaussian_pair(points[i], points[j], a['alpha'], b['alpha'], a['sigma'], b['sigma'])
            pairs.append({'ids': [a['node']['id'], b['node']['id']], **result})
    curves.append({'state': state, 't': float(t), 'pairs': pairs})
    print('STATE', state, 'max_proxy', max(p['gaussian_nearest_center_proxy'] for p in pairs), flush=True)
summary = {'run': str(asset['run']), 'states': curves,
           'sampling': f'Uniform subset of at most {args.gaussians} opacity>=.05 GS per part; seed2026. Not comparable numerically with prior all-center proxy.',
           'mesh_audit': [{'id': p['node']['id'], 'watertight': bool(p['mesh'].is_watertight),
                           'winding_consistent': bool(p['mesh'].is_winding_consistent),
                           'self_intersection_verified': False, 'mesh_sha256': p['mesh_sha256']}
                          for p in parts],
           'predicted_joints': [{'id': n['id'], **n['joint']} for n in asset['nodes']],
           'solid_collision': {'status': 'unavailable until topology and self-intersection validation', 'volume': None},
           'elapsed_seconds': time.monotonic() - started}
(out / 'screen.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
print('COMPLETE', out, flush=True)
