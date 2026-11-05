"""Motion-only gravity feasibility experiment; never changes a geometry asset.

Static-under-gravity is a USER ASSUMPTION, not an observed release experiment.
Gravity alone cannot identify joint type or a vertical hinge's horizontal pivot.
"""
import argparse
import copy
import datetime
import json
import pathlib
import sys

import numpy as np

from collision_refinement_common import load_asset, local_transform, moved, sha256


UP = np.array([0., 1., 0.])
GRAVITY = -UP


def unit(x):
    x = np.asarray(x, float)
    if np.linalg.norm(x) < 1e-12:
        raise ValueError('Undefined axis direction')
    return x / np.linalg.norm(x)


def neutral_axis(joint):
    """Closest ideal all-state neutral axis within the original joint type."""
    d = unit(joint['axis']['direction'])
    if joint['type'] == 'prismatic':
        horizontal = d - UP * np.dot(d, UP)
        if np.linalg.norm(horizontal) < 1e-10:
            raise ValueError('Vertical slider: horizontal azimuth is unidentifiable')
        return unit(horizontal)
    if joint['type'] in ['revolute', 'continuous']:
        return UP * (-1 if d @ UP < 0 else 1)
    raise ValueError(joint['type'])


def state_metrics(node, com, diagonal, states=41):
    """r = g_hat dot dc/dt / D; t spans this candidate's motion interval.

    Physical generalized forces: prism mg*g_hat.d;
    hinge mg*g_hat.(d cross (c(q)-o)), with angle measured in radians.
    An ideal joint reacts to all force/torque components outside its free DOF.
    """
    joint = node['joint']
    delta = joint['range'][1] - joint['range'][0]
    if joint['type'] == 'fixed':
        d = np.zeros(3)
    else:
        d = unit(joint['axis']['direction'])
    origin = np.array(joint['axis']['origin'])
    rows = []
    max_fd_error = 0.
    for t in np.linspace(0, 1, states):
        center = moved(np.asarray(com)[None], local_transform(node, t))[0]
        if joint['type'] == 'prismatic':
            generalized = float(GRAVITY @ d)  # Q / (mg)
            velocity = delta * d
        elif joint['type'] in ['revolute', 'continuous']:
            tangent = np.cross(d, center-origin)
            generalized = float(GRAVITY @ tangent)  # torque / (mg), model units
            velocity = np.deg2rad(delta) * tangent
        else:
            generalized = 0.
            velocity = np.zeros(3)
        residual = float(GRAVITY @ velocity / diagonal)
        eps = 1e-5
        ca = moved(np.asarray(com)[None], local_transform(node, t-eps))[0]
        cb = moved(np.asarray(com)[None], local_transform(node, t+eps))[0]
        fd = float(GRAVITY @ (cb-ca)/(2*eps*diagonal))
        max_fd_error = max(max_fd_error, abs(fd-residual))
        rows.append({'t': float(t), 'com': center.tolist(),
                     'normalized_motion_gravity_residual': residual,
                     'generalized_force_div_mg': generalized})
    return {'node_id': node['id'], 'type': joint['type'], 'states': rows,
            'max_abs_residual': max(abs(r['normalized_motion_gravity_residual']) for r in rows),
            'mean_squared_residual': float(np.mean([r['normalized_motion_gravity_residual']**2 for r in rows])),
            'height_span_percent_D': float(100*np.ptp([r['com'][1] for r in rows])/diagonal),
            'max_finite_difference_error': max_fd_error}


def evaluate(run, out):
    asset = load_asset(run)
    obj = copy.deepcopy(asset['object'])
    baseline = asset['nodes']
    # This trial explicitly supports the original single fixed-parent star tree.
    byid = {n['id']: n for n in baseline}
    for n in baseline:
        if n['parent'] != -1:
            assert byid[n['parent']]['joint']['type'] == 'fixed'
    frozen = []
    for p in asset['parts']:
        n = p['node']
        for field in ['plys', 'gs']:
            for rel in n.get(field, []):
                path = asset['object_path'].parent/rel
                frozen.append({'path': str(path), 'sha256': sha256(path)})
    result = {'source_run': str(asset['run']), 'source_object': str(asset['object_path']),
              'source_object_sha256': sha256(asset['object_path']),
              'source_tool_sha256': sha256(__file__),
              'baseline_kind': 'Earliest sealed original official-weight 45194 inference; no later optimization',
              'world_up_export': UP.tolist(), 'gravity_export_unit': GRAVITY.tolist(),
              'gravity_definition': 'Canonical PAct Mesh/JSON Y-up; fixed independent of candidate joint; magnitude cancels',
              'physical_assumption': 'User assumes every interior configuration can remain released and static; ideal frictionless joints, fixed base, no actuation',
              'static_claim': 'Analytic zero driving force along allowed DOF; not an observed release or a contact/dynamics simulation',
              'mass_model': 'Surface-area centroid diagnostic. No calibrated mass/solid COM. Projected zero residual holds for EVERY COM.',
              'loss': 'Mean over non-root movable nodes and 41 states of (g_hat dot dc/dt / original asset diagonal)^2',
              'tie_break': 'Preserve original type and pivot when multiple zero-gravity candidates exist. No GT in selection.',
              'range_policy': 'Keep original numeric ranges and original types for selected candidate; no motion collapse',
              'baseline': [], 'gravity_only': [], 'type_probes': [], 'changes': []}
    for part, candidate in zip(asset['parts'], obj['diffuse_tree']):
        n, mesh = part['node'], part['mesh']
        com = np.asarray(mesh.centroid)
        if n['joint']['type'] != 'fixed':
            candidate['joint']['axis']['direction'] = neutral_axis(n['joint']).tolist()
        result['baseline'].append(state_metrics(n, com, asset['diagonal']))
        result['gravity_only'].append(state_metrics(candidate, com, asset['diagonal']))
        if candidate != n:
            old, new = unit(n['joint']['axis']['direction']), unit(candidate['joint']['axis']['direction'])
            result['changes'].append({'node_id': n['id'], 'type_changed': False,
                                      'axis_change_degrees': float(np.degrees(np.arccos(np.clip(old@new, -1, 1)))),
                                      'old_direction': old.tolist(), 'new_direction': new.tolist()})
        # Competing types are diagnostics, not silently selected corrections.
        # Both have nonzero stroke; revolute sweep uses an explicit 45deg probe,
        # slider stroke is 0.25D. These are not GT motion ranges.
        if n['parent'] != -1:
            pivot = mesh.bounds.mean(0)
            pivot[0] = mesh.bounds[0, 0]
            for typ, direction, limits in [('prismatic', [0.,0.,1.], [0., .25*asset['diagonal']]),
                                            ('revolute', UP.tolist(), [0.,45.])]:
                probe = copy.deepcopy(n)
                probe['joint'] = {'type': typ, 'range': limits,
                                  'axis': {'direction': direction, 'origin': pivot.tolist()}}
                metrics = state_metrics(probe, com, asset['diagonal'])
                result['type_probes'].append({'node_id': n['id'], 'candidate_joint': probe['joint'],
                                             'max_abs_residual': metrics['max_abs_residual'],
                                             'selection': 'not identifiable from gravity alone'})
    active = [i for i,n in enumerate(baseline) if n['joint']['type'] != 'fixed']
    result['loss_values'] = {k:float(np.mean([result[k][i]['mean_squared_residual'] for i in active]))
                             for k in ['baseline', 'gravity_only']}
    result['geometry_files'] = frozen
    result['geometry_frozen'] = all(sha256(x['path']) == x['sha256'] for x in frozen)
    result['type_identified'] = False
    result['pivot_identified'] = False
    result['fixed_root_failure_mode'] = 'Root is a fixed generated cabinet body; gravity cannot decide that a fixed part should move. Fixed joints trivially satisfy the assumption.'
    # Geometry links reference original bytes. Only object.json is new.
    target = out/'gravity_only'
    target.mkdir()
    for folder in ['ply', 'gs']:
        source = asset['object_path'].parent/folder
        if source.exists():
            (target/folder).symlink_to(source, target_is_directory=True)
    (target/'object.json').write_text(json.dumps(obj, indent=2)+'\n')
    result['candidate_object_sha256'] = sha256(target/'object.json')
    # Seal before any GT evaluation.
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    assert result['geometry_frozen']
    assert max(x['max_finite_difference_error'] for k in ['baseline','gravity_only'] for x in result[k]) < 1e-8
    assert max(x['max_abs_residual'] for x in result['gravity_only']) < 1e-10
    assert all(sha256(x['path']) == x['sha256'] for x in frozen)
    print(json.dumps({'out': str(out), 'loss': result['loss_values'], 'changes':result['changes']}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--out', required=True)
    a = parser.parse_args()
    out = pathlib.Path(a.out).resolve()
    out.mkdir(exist_ok=False, parents=True)
    evaluate(a.run, out)


if __name__ == '__main__':
    main()
