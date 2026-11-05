"""Evaluate sealed automatic joint edits with a fixed raw-geometry GT matching.

The correspondence and static-body similarity transform come from the earlier
post-fit evaluator's *original* raw source-node assessment.  Those were formed
without semantic labels or axis errors and are identical for both candidates.
No scores in this file are used to select or revise an axis rule.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def score_joint(node, raw_row, alignment):
    result = dict(raw_row)
    result.update(predicted_owner=node['id'], predicted_type=node['joint']['type'],
                  matched=bool(raw_row['matched'] and node['joint']['type'] != 'fixed'),
                  type_correct=False, axis_correct=False, angle_degrees=None,
                  pivot_percent_D=None)
    if not result['matched']:
        result['reason'] = 'Automatic export is fixed' if raw_row['matched'] else raw_row.get('reason', 'No raw geometry match')
        return result
    joint = node['joint']
    result['type_correct'] = joint['type'] == raw_row['GT_type']
    if not result['type_correct']:
        result['reason'] = 'Wrong motion type'
        return result
    R = np.asarray(alignment['R_export_to_GT'], float)
    scale = float(alignment['scale'])
    shift = np.asarray(alignment['translation'], float)
    direction = R @ np.asarray(joint['axis']['direction'], float)
    origin = scale * (R @ np.asarray(joint['axis']['origin'], float)) + shift
    truth_direction = np.asarray(raw_row['GT_axis_direction'], float)
    truth_origin = np.asarray(raw_row['GT_axis_origin'], float)
    if np.linalg.norm(direction) < 1e-12 or np.linalg.norm(truth_direction) < 1e-12:
        result['reason'] = 'Degenerate axis direction'
        return result
    direction /= np.linalg.norm(direction)
    truth_direction /= np.linalg.norm(truth_direction)
    result['angle_degrees'] = float(np.degrees(np.arccos(np.clip(abs(direction @ truth_direction), 0, 1))))
    if raw_row['GT_type'] == 'revolute':
        result['pivot_percent_D'] = float(100 * np.linalg.norm(np.cross(origin - truth_origin, truth_direction)) / alignment['GT_static_diagonal'])
    result['axis_correct'] = result['angle_degrees'] <= 5 and (result['pivot_percent_D'] is None or result['pivot_percent_D'] <= 5)
    return result


def summarize(rows, active_count):
    n = len(rows)
    typed = sum(r['type_correct'] for r in rows)
    return dict(GT_joint_count=n, predicted_active_joint_count=active_count,
                independent_joint_coverage=sum(r['matched'] for r in rows) / n,
                type_correct_GT_coverage=typed / n,
                type_correct_precision=typed / max(active_count, 1),
                axis_correct_count=sum(r['axis_correct'] for r in rows),
                axis_correct_rate=sum(r['axis_correct'] for r in rows) / n,
                conditional_mean_angle_degrees=mean(r['angle_degrees'] for r in rows),
                conditional_mean_pivot_percent_D=mean(r['pivot_percent_D'] for r in rows))


def run(auto_root, baseline_root, output):
    assert (auto_root / 'FINAL_SEAL.json').exists(), 'Automatic candidates must be sealed before GT reading'
    cases = json.loads((auto_root / 'protocol.json').read_text())['cases']
    rows = []
    for case in cases:
        base = json.loads((baseline_root / case / 'evaluation.json').read_text())
        original = base['joint_methods']['original']
        exported = json.loads((auto_root / case / 'axis_only_export/object.json').read_text())
        nodes = {n['id']: n for n in exported['diffuse_tree']}
        new_rows = []
        for raw_row in original['rows']:
            owner = raw_row.get('predicted_owner')
            if owner is None:
                new_rows.append(dict(raw_row))
            else:
                new_rows.append(score_joint(nodes[owner], raw_row, base['alignment']))
        auto = summarize(new_rows, sum(n['joint']['type'] != 'fixed' for n in nodes.values()))
        if original['axis_correct_count'] != sum(r['axis_correct'] for r in original['rows']):
            raise ValueError(f'Raw baseline mismatch: {case}')
        rows.append(dict(case=case, original={k: v for k, v in original.items() if k != 'rows'}, automatic=auto,
                         joint_rows=[dict(GT_part=r['GT_part'], origin=original['rows'][i], automatic=r) for i, r in enumerate(new_rows)],
                         common_matching='Original raw source geometry and static-body similarity alignment; frozen before automatic-axis scoring'))
    n = sum(x['automatic']['GT_joint_count'] for x in rows)
    summary = dict(case_count=len(rows), GT_joint_count=n,
                   original_axis_correct=sum(x['original']['axis_correct_count'] for x in rows),
                   automatic_axis_correct=sum(x['automatic']['axis_correct_count'] for x in rows))
    summary['original_type_correct'] = sum(sum(j['origin']['type_correct'] for j in x['joint_rows']) for x in rows)
    summary['automatic_type_correct'] = sum(sum(j['automatic']['type_correct'] for j in x['joint_rows']) for x in rows)
    summary['original_mean_input_IoU'] = mean(json.loads((baseline_root / c / 'evaluation.json').read_text())['fit_image_metrics']['original']['IoU'] for c in cases)
    summary['automatic_mean_input_IoU'] = summary['original_mean_input_IoU']
    output.mkdir(parents=True, exist_ok=False)
    (output / 'results.json').write_text(json.dumps(dict(summary=summary, cases=rows, protocol=dict(
        source_auto=str(auto_root), source_baseline=str(baseline_root),
        warning='The baseline alignment and raw geometry matching originate in the earlier human-prior experiment, but neither uses its semantic annotations, fitted geometry, or joint predictions to score these raw nodes. The auto export was frozen before evaluation.',
        axis_correct='GT joint denominator; independent, type-correct, undirected direction <=5deg and revolute pivot distance <=5% GT static-body diagonal',
        geometry='No automatic initial-frame geometry optimization is claimed by this axis-only result')), ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--auto-root', type=Path, required=True)
    parser.add_argument('--baseline-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.auto_root.resolve(), args.baseline_root.resolve(), args.output.resolve())
