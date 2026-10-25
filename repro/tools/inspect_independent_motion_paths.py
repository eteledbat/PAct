"""GT-free diagnosis of synchronized and one-joint-at-a-time native motion.

Exact full-target KD-tree nearest-center Gaussian proxies, not solid collision.
This script evaluates exports; it never changes geometry, joints, or ranges.
"""
import argparse
import itertools
import json
import pathlib
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from collision_refinement_common import load_asset, local_transform, moved, sha256, transforms


def independent_world(nodes, times):
    lookup = {node['id']: node for node in nodes}
    world = {-1: np.eye(4)}

    def visit(nid):
        if nid not in world:
            node = lookup[nid]
            world[nid] = visit(node.get('parent', -1)) @ local_transform(node, times.get(nid, 0.))
        return world[nid]

    for nid in lookup:
        visit(nid)
    return world


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--sources', type=int, default=4096)
    args = parser.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    asset = load_asset(args.run, gaussian_count=2048, seed=0)
    parts, nodes = asset['parts'], asset['nodes']
    active = [n['id'] for n in nodes if n['joint']['type'] != 'fixed' and abs(n['joint']['range'][1] - n['joint']['range'][0]) > 1e-8]
    samples, trees = [], []
    for part in parts:
        remaining = np.setdiff1d(part['retained_ids'], part['sample_ids'], assume_unique=True)
        rng = np.random.default_rng(2026 + int(part['node']['id']))
        selected = np.sort(rng.choice(remaining, min(args.sources, len(remaining)), replace=False))
        assert len(selected) and not np.intersect1d(selected, part['sample_ids']).size
        indices = np.searchsorted(part['retained_ids'], selected)
        samples.append({key: part['full_' + key][indices] for key in ['xyz', 'alpha', 'sigma']})
        trees.append(cKDTree(part['full_xyz']))
    # Verify hierarchy and range interpolation against the existing synchronized FK.
    max_fk_error = 0.
    for t in np.linspace(0, 1, 41):
        actual = independent_world(nodes, {n['id']: float(t) for n in nodes})
        reference = transforms(nodes, float(t))
        max_fk_error = max(max_fk_error, max(float(np.max(np.abs(actual[n['id']] - reference[n['id']]))) for n in nodes))
    assert max_fk_error < 1e-12
    pairs = list(itertools.combinations(range(len(parts)), 2))
    cache = {}

    def pair_metric(i, j, world):
        relative = np.linalg.inv(world[parts[j]['node']['id']]) @ world[parts[i]['node']['id']]
        key = (i, j, relative.round(13).tobytes())
        if key in cache:
            return cache[key]
        mass = parts[i]['full_alpha'].sum() + parts[j]['full_alpha'].sum()
        numerators, minima, gaps = [], [], []
        for source, target, matrix in [(i, j, relative), (j, i, np.linalg.inv(relative))]:
            query = moved(samples[source]['xyz'], matrix)
            distance, idx = trees[target].query(query, workers=4)
            values = samples[source]['alpha'] * parts[target]['full_alpha'][idx] * np.exp(-distance**2 / (2 * (samples[source]['sigma']**2 + parts[target]['full_sigma'][idx]**2)))
            numerators.append(float(len(parts[source]['full_xyz']) * values.mean() / mass))
            minima.append(float(distance.min()))
            gaps.append(float(np.mean(distance < samples[source]['sigma'] + parts[target]['full_sigma'][idx])))
        result = {'ids': [parts[i]['node']['id'], parts[j]['node']['id']], 'gaussian_overlap_proxy': sum(numerators),
                  'min_sampled_source_to_full_target_center_distance': min(minima),
                  'negative_nearest_pair_1sigma_gap_fraction': float(np.mean(gaps))}
        assert np.isfinite([result['gaussian_overlap_proxy'], min(minima), *gaps]).all()
        cache[key] = result
        return result

    paths = []
    schedules = [('synchronized', active)] + [(f'only_node_{nid}', [nid]) for nid in active]
    for label, moving_ids in schedules:
        states = []
        for state, t in enumerate(np.linspace(0, 1, 21)):
            world = independent_world(nodes, {nid: float(t) for nid in moving_ids})
            values = [pair_metric(i, j, world) for i, j in pairs]
            states.append({'state': state, 't': float(t), 'pairs': values,
                           'mean_pair_proxy': float(np.mean([v['gaussian_overlap_proxy'] for v in values])),
                           'worst_pair_proxy': float(max(v['gaussian_overlap_proxy'] for v in values))})
        initial = states[0]['mean_pair_proxy']
        paths.append({'name': label, 'moving_predicted_node_ids': moving_ids, 'states': states,
                      'initial_state_proxy': initial, 'mean_motion_only_proxy': float(np.mean([s['mean_pair_proxy'] for s in states[1:]])),
                      'max_state_proxy': max(s['mean_pair_proxy'] for s in states),
                      'max_state_minus_initial_proxy': max(s['mean_pair_proxy'] - initial for s in states)})
        print(label, 'COMPLETE', paths[-1]['mean_motion_only_proxy'], flush=True)
    assert all(p['states'][0]['pairs'] == paths[0]['states'][0]['pairs'] for p in paths)
    frozen = [{'path': str(p[key]), 'sha256': p[hashkey]} for p in parts for key, hashkey in [('mesh_path', 'mesh_sha256'), ('gs_path', 'gs_sha256')]]
    assert all(sha256(row['path']) == row['sha256'] for row in frozen)
    result = {'run': str(asset['run']), 'object_sha256': sha256(asset['object_path']),
              'protocol': 'Predicted native ranges only. 21 states per path; each active joint alone with all others at t=0, plus synchronized path. Hierarchy preserved. No GT files read; no optimization or range changes.',
              'nodes': [{'id': n['id'], 'parent': n.get('parent', -1), 'joint': n['joint']} for n in nodes],
              'sampling': 'Up to4096 sources/part, seed2026+node_id, excluding optimizer seed0 source set. All opacity>=.05 centers are exact KD-tree targets; population-count and opacity reweighting match the existing objective. Isotropic sigma is mean(exp(scale)).',
              'source_counts': {str(p['node']['id']): len(s['xyz']) for p, s in zip(parts, samples)},
              'full_target_counts': {str(p['node']['id']): len(p['full_xyz']) for p in parts},
              'paths': paths, 'checks': {'synchronized_FK_max_abs_error': max_fk_error, 'all_paths_start_identical': True, 'source_hashes_unchanged': True},
              'combination_coverage': 'Individual paths and one synchronized diagonal only; independent multi-joint combinations not exhaustively covered.',
              'solid_collision': {'volume': None, 'penetration_depth': None, 'status': 'unavailable', 'reason': 'Gaussian nearest-center overlap and sigma gaps are proxies; solid mesh validity not established.'},
              'frozen_files': frozen, 'elapsed_seconds': time.monotonic() - started}
    (out / 'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for path in paths:
        times = [s['t'] for s in path['states']]
        values = np.array([s['mean_pair_proxy'] for s in path['states']])
        axes[0].plot(times, values, label=path['name'])
        axes[1].plot(times, values - values[0], label=path['name'])
    axes[0].set_ylabel('Gaussian overlap proxy (not volume)')
    axes[1].set_ylabel('Proxy minus initial-state proxy (diagnostic)')
    for ax in axes:
        ax.set_xlabel('Fraction of predicted native range')
        ax.grid(alpha=.25)
        ax.legend(fontsize=7)
    fig.suptitle('Original PAct: individual-joint and synchronized motion, 21 states/path')
    fig.tight_layout()
    fig.savefig(out / 'motion_paths.png', dpi=180)
    fig.savefig(out / 'motion_paths.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
