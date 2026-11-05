"""Post-fit GT and held-out-view evaluation for auto-axis initial-frame fitting."""
import argparse
import json
import pathlib

import numpy as np

from collision_refinement_common import sha256
from multicase_closed_common import write_json
import evaluate_multicase_closed as existing
from fit_auto_semantic_initial_frame import AutoFitModel


def avg(values):
    values = [x for x in values if x is not None]
    return float(np.mean(values)) if values else None


def run(root, axis_eval):
    protocol = json.loads((root/'protocol.json').read_text())
    cases = protocol['cases']
    assert len(cases) == 11, 'Expected the complete prespecified batch'
    assert all((root/c/'fit_results.json').exists() for c in cases), 'All fits must finish before GT evaluation'
    existing.Model = AutoFitModel
    for case in cases:
        if not (root/case/'evaluation.json').exists():
            existing.evaluate_case(root/case)
    data = {c: json.loads((root/c/'evaluation.json').read_text()) for c in cases}
    axes = json.loads(axis_eval.read_text())
    assert set(data) == {c['case'] for c in axes['cases']}
    summary = dict(case_count=len(cases),
                   fit_input_IoU_origin=avg(d['fit_image_metrics']['original']['IoU'] for d in data.values()),
                   fit_input_IoU_optimized=avg(d['fit_image_metrics']['optimized']['IoU'] for d in data.values()),
                   fit_input_PSNR_origin=avg(d['fit_image_metrics']['original']['PSNR'] for d in data.values()),
                   fit_input_PSNR_optimized=avg(d['fit_image_metrics']['optimized']['PSNR'] for d in data.values()))
    summary['verified_part_mIoU_origin'] = avg(json.loads((root/c/'fit_results.json').read_text())['part_mask_metrics']['original']['mean_IoU'] for c in cases)
    summary['verified_part_mIoU_optimized'] = avg(json.loads((root/c/'fit_results.json').read_text())['part_mask_metrics']['optimized']['mean_IoU'] for c in cases)
    novel = [c for c in cases if c != '20411']
    summary['heldout_case_count'] = len(novel)
    for metric in ['IoU','PSNR']:
        summary[f'heldout_{metric}_origin'] = avg(data[c]['unfitted_open_mean']['original'][metric] for c in novel)
        summary[f'heldout_{metric}_optimized'] = avg(data[c]['unfitted_open_mean']['optimized'][metric] for c in novel)
    for metric in ['CD_percent_D','Fscore_1pct_D','Fscore_5pct_D']:
        summary[f'surface_{metric}_origin'] = avg(data[c]['surface_reconstruction']['original'][metric] for c in cases)
        summary[f'surface_{metric}_optimized'] = avg(data[c]['surface_reconstruction']['optimized'][metric] for c in cases)
    summary['axis_correct_origin'] = axes['summary']['original_axis_correct']
    summary['axis_correct_auto'] = axes['summary']['automatic_axis_correct']
    summary['type_correct_origin'] = axes['summary']['original_type_correct']
    summary['type_correct_auto'] = axes['summary']['automatic_type_correct']
    summary['GT_joint_count'] = axes['summary']['GT_joint_count']
    summary['input_IoU_improved_cases'] = sum(data[c]['fit_image_metrics']['optimized']['IoU'] > data[c]['fit_image_metrics']['original']['IoU'] for c in cases)
    summary['heldout_IoU_improved_cases'] = sum(data[c]['unfitted_open_mean']['optimized']['IoU'] > data[c]['unfitted_open_mean']['original']['IoU'] for c in novel)
    summary['surface_CD_improved_cases'] = sum(data[c]['surface_reconstruction']['optimized']['CD_percent_D'] < data[c]['surface_reconstruction']['original']['CD_percent_D'] for c in cases)
    write_json(root/'summary.json', dict(summary=summary, per_case={c:dict(
        fit_image_metrics=data[c]['fit_image_metrics'],
        verified_part_mask=json.loads((root/c/'fit_results.json').read_text())['part_mask_metrics'],
        heldout=data[c]['unfitted_open_mean'],
        surface=data[c]['surface_reconstruction'],
        source_files_unchanged=data[c]['source_files_unchanged']) for c in cases},
        protocol=dict(axis_eval=str(axis_eval), original_RGB_and_instance_mask_only=True,
                      manually_annotated_semantic_labels_used=False,
                      axis_frozen_during_geometry_fit=True,
                      verified_mask_subset=True,
                      registration='Reused original-RGB/silhouette registration from previous batch, which does not use human semantic labels',
                      GT_after_all_fit_seals=True)))
    write_json(root/'EVALUATION_SEAL.json',dict(summary_sha256=sha256(root/'summary.json'),
                                               fit_seals={c:sha256(root/c/'fit_results.json') for c in cases},
                                               evaluation_files={c:sha256(root/c/'evaluation.json') for c in cases}))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=pathlib.Path,required=True)
    p.add_argument('--axis-eval',type=pathlib.Path,required=True)
    a=p.parse_args()
    run(a.root.resolve(),a.axis_eval.resolve())
