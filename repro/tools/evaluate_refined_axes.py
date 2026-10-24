"""Post-seal GT and 41-state held-out-source collision evaluation.

Training uses CPU KD trees; this evaluation independently finds NN with Kaolin
CUDA then computes kernels in float64. Neither is a solid collision measure.
"""
import argparse
import fcntl
import json
import pathlib
import time

import numpy as np
import torch
from kaolin.metrics.pointcloud import sided_distance
from scipy.spatial import cKDTree

from collision_refinement_common import candidate_nodes, load_asset, moved, sha256, transforms


def angle(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if min(np.linalg.norm(a), np.linalg.norm(b)) < 1e-10:
        return None
    return float(np.degrees(np.arccos(np.clip(abs(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)), 0, 1))))


def gt_evaluation(asset, nodes):
    baseline = asset['run'] / 'joint_evaluation/joint_errors.json'
    if not baseline.exists():
        return {'status': 'unavailable', 'reason': 'No verified GT for official real-world sample'}
    base = json.loads(baseline.read_text())
    R = np.asarray(base['R_export_to_gtworld'])
    s, t, D = base['scale'], np.asarray(base['translation']), base['GT_static_diagonal']
    byid = {n['id']: n for n in nodes}
    rows = []
    for original in base['joints']:
        joint = byid[original['predicted_node_id']]['joint']
        direction = R @ np.array(joint['axis']['direction'])
        origin = s * (R @ np.array(joint['axis']['origin'])) + t
        gd, go = np.asarray(original['GT_direction_world']), np.asarray(original['GT_origin_world'])
        correct = original['GT_type'] == joint['type']
        pivot = float(np.linalg.norm(np.cross(origin - go, gd))) if correct and joint['type'] == 'revolute' else None
        row = {'gt_mobility_id': original['gt_mobility_id'], 'predicted_node_id': original['predicted_node_id'],
               'GT_type': original['GT_type'], 'predicted_type': joint['type'], 'type_correct': correct,
               'direction_vector_diagnostic_degrees': angle(direction, gd),
               'valid_axis_error_degrees': angle(direction, gd) if correct else None,
               'pivot_to_GT_axis_percent_static_diagonal': None if pivot is None else 100 * pivot / D}
        sensitivity = asset['run'] / 'joint_evaluation/geometry_registration_sensitivity.json'
        if sensitivity.exists():
            registration = json.loads(sensitivity.read_text())
            rr, tt = np.array(registration['rotation']), np.array(registration['translation'])
            row['same_baseline_registration_axis_error_degrees'] = angle(rr @ direction, gd) if correct else None
            row['same_baseline_registration_pivot_percent_diagonal'] = float(100 * np.linalg.norm(np.cross(rr @ origin + tt - go, gd)) / D) if pivot is not None else None
        rows.append(row)
    angles = [r['valid_axis_error_degrees'] for r in rows if r['valid_axis_error_degrees'] is not None]
    return {'status': 'measured', 'alignment': 'Exactly the same fixed rotation, scale and translation as the original GT evaluation; never re-fit per candidate',
            'joint_type_correct_count': sum(r['type_correct'] for r in rows), 'joint_count': len(rows),
            'valid_axis_count': len(angles), 'mean_valid_axis_error_degrees': float(np.mean(angles)) if angles else None,
            'joints': rows}


def evaluate(root, name, run, method):
    asset = load_asset(run, gaussian_count=2048, seed=0)
    candidate = None if method == 'original' else root / f'{name}_{method}/candidate.json'
    if candidate:
        directory = candidate.parent
        assert json.loads((directory / 'status.json').read_text())['state'] == 'complete'
        seal = json.loads((directory / 'prediction_seal.json').read_text())
        assert sha256(candidate) == seal['candidate_sha256']
        assert seal['frozen_files_unchanged'] and not seal['GT_reads_blocked']
    nodes = candidate_nodes(asset, candidate)
    candidate_hash = sha256(candidate or asset['object_path'])
    out = root / f'eval_{name}_{method}'
    out.mkdir(exist_ok=True)
    lock = (out / 'evaluation.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    result_path = out / 'metrics.json'
    if result_path.exists():
        assert json.loads(result_path.read_text())['candidate_sha256'] == candidate_hash
        print('ALREADY EVALUATED', name, method, flush=True)
        return
    started = time.monotonic()
    parts = asset['parts']
    samples = []
    for part in parts:
        remaining = np.setdiff1d(part['retained_ids'], part['sample_ids'], assume_unique=True)
        assert len(remaining) > 0
        rng = np.random.default_rng(2026 + int(part['node']['id']))
        selected = np.sort(rng.choice(remaining, min(12000, len(remaining)), replace=False))
        ids = np.searchsorted(part['retained_ids'], selected)
        samples.append({'xyz': part['full_xyz'][ids], 'alpha': part['full_alpha'][ids],
                        'sigma': part['full_sigma'][ids], 'original_ids': selected})
    targets = [torch.tensor(p['full_xyz'], dtype=torch.float32, device='cuda')[None] for p in parts]
    rows = []
    checks = []
    histories = [np.zeros(len(p['xyz'])) for p in samples]
    pair_count = len(parts) * (len(parts) - 1) // 2
    state_count = 41
    for state, t in enumerate(np.linspace(0, 1, state_count)):
        world = transforms(nodes, float(t))
        pairs = []
        for i in range(len(parts)):
            for j in range(i + 1, len(parts)):
                numerators, minima = [], []
                mass = parts[i]['full_alpha'].sum() + parts[j]['full_alpha'].sum()
                for source, target in [(i, j), (j, i)]:
                    source_tr, target_tr = world[parts[source]['node']['id']], world[parts[target]['node']['id']]
                    relative = np.linalg.inv(target_tr) @ source_tr
                    query = moved(samples[source]['xyz'], relative)
                    q = torch.tensor(query, dtype=torch.float32, device='cuda')[None]
                    with torch.no_grad():
                        idx = sided_distance(q, targets[target])[1][0].cpu().numpy().astype(np.int64)
                    d2 = np.sum((query - parts[target]['full_xyz'][idx])**2, axis=1)
                    values = samples[source]['alpha'] * parts[target]['full_alpha'][idx] * np.exp(-d2 / (2 * (samples[source]['sigma']**2 + parts[target]['full_sigma'][idx]**2)))
                    weighted = len(parts[source]['full_xyz']) * values / mass
                    numerators.append(float(weighted.mean()))
                    histories[source] += weighted / (pair_count * state_count)
                    minima.append(float(np.sqrt(d2.min())))
                    if state == 0:
                        ref, _ = cKDTree(parts[target]['full_xyz']).query(query[:31])
                        excess = np.sqrt(d2[:31]) - ref
                        assert np.max(excess) < 2e-6
                        checks.append({'source': parts[source]['node']['id'], 'target': parts[target]['node']['id'],
                                       'max_GPU_NN_distance_excess_over_CPU_on_31_real_points': float(np.max(excess))})
                pairs.append({'ids': [parts[i]['node']['id'], parts[j]['node']['id']],
                              'gaussian_overlap_proxy': float(sum(numerators)),
                              'min_heldout_source_to_full_target_center_distance': min(minima)})
        rows.append({'state': state, 't': float(t), 'interleaved_heldout_time': bool(state % 2), 'pairs': pairs,
                     'mean_pair_proxy': float(np.mean([p['gaussian_overlap_proxy'] for p in pairs]))})
        print(name, method, 'STATE', state, 'proxy', rows[-1]['mean_pair_proxy'], flush=True)
    for p, source in zip(parts, samples):
        assert not np.intersect1d(source['original_ids'], p['sample_ids']).size
    np.savez_compressed(out / 'source_contributions.npz', **{str(p['node']['id']): h for p, h in zip(parts, histories)})
    mean = float(np.mean([r['mean_pair_proxy'] for r in rows]))
    assert np.isclose(mean, sum(h.mean() for h in histories), rtol=1e-10)
    axis_changes = []
    original_byid = {n['id']: n for n in asset['nodes']}
    for n in nodes:
        old = original_byid[n['id']]
        assert n['joint']['type'] == old['joint']['type'] and n['joint']['range'] == old['joint']['range']
        if n['joint']['type'] == 'fixed':
            continue
        po = np.array(old['joint']['axis']['origin'])
        pn = np.array(n['joint']['axis']['origin'])
        anchor = []
        if n['joint']['type'] == 'revolute':
            for t in np.linspace(0, 1, 41):
                oldtr, newtr = transforms(asset['nodes'], t)[n['id']], transforms(nodes, t)[n['id']]
                anchor.append(float(np.linalg.norm(moved(po[None], newtr) - moved(po[None], oldtr)) / asset['diagonal']))
        axis_changes.append({'node_id': n['id'], 'type': n['joint']['type'],
                             'direction_change_degrees': angle(n['joint']['axis']['direction'], old['joint']['axis']['direction']),
                             'origin_shift_percent_asset_diagonal': float(100 * np.linalg.norm(pn - po) / asset['diagonal']) if n['joint']['type'] == 'revolute' else None,
                             'max_original_pivot_anchor_displacement_percent_asset_diagonal': 100 * max(anchor) if anchor else None})
    result = {'case': name, 'method': method, 'run': str(asset['run']), 'candidate_sha256': candidate_hash,
              'states': rows, 'mean_all_state_pair_proxy': mean,
              'mean_interleaved_heldout_time_proxy': float(np.mean([r['mean_pair_proxy'] for r in rows if r['interleaved_heldout_time']])),
              'max_state_mean_pair_proxy': float(max(r['mean_pair_proxy'] for r in rows)),
              'initial_state_mean_pair_proxy': rows[0]['mean_pair_proxy'],
              'sampling': 'Up to12000 source centers per part, seed2026, disjoint from training seed0 source centers. All retained centers remain NN targets; sources reweighted to full populations.',
              'sample_counts': {str(p['node']['id']): len(s['xyz']) for p, s in zip(parts, samples)},
              'full_counts': {str(p['node']['id']): len(p['full_xyz']) for p in parts},
              'GT': gt_evaluation(asset, nodes), 'axis_changes': axis_changes,
              'independent_GPU_CPU_NN_checks': checks, 'mesh_geometry_frozen': True,
              'solid_collision': {'status': 'unavailable', 'volume': None, 'penetration_depth': None,
                                  'reason': 'Open or self-intersecting/unverified mesh; Gaussian overlap is not solid overlap.'},
              'elapsed_seconds': time.monotonic() - started}
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False))
    print('EVALUATED', name, method, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--case')
    p.add_argument('--method', choices=['original', 'position', 'direction', 'both'])
    a = p.parse_args()
    root = pathlib.Path(a.root).resolve()
    config = json.loads((root / 'experiment.json').read_text())
    torch.set_num_threads(2)
    for name, run in config['cases'].items():
        if a.case and name != a.case:
            continue
        for method in ([a.method] if a.method else ['original', 'position', 'direction', 'both']):
            evaluate(root, name, run, method)


if __name__ == '__main__':
    main()
