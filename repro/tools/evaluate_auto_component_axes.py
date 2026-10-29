"""Post-seal evaluation of automatic component joints using frozen raw-body GT alignment."""
import argparse
import json
import pathlib

import numpy as np
import torch

import evaluate_multicase_closed as evaluation
from multicase_closed_common import Model, write_json
from collision_refinement_common import sha256


def read(path):
    return json.loads(path.read_text())


def run(root, reference_root):
    torch.set_num_threads(4)
    for case in read(root/'protocol.json')['cases']:
        out = root/case
        manifest = read(out/'motion_groups.json')
        assert sha256(out/'membership.npz') == manifest['membership_sha256']
        assert all(sha256(f['path']) == f['sha256'] for f in manifest['geometry_files'])
        model = Model(read(out/'config.json'), manifest, out)
        old = evaluation.gt_and_alignment
        def fixed_alignment(m):
            target, body, moving, _, _, _, _, _ = old(m)
            a = read(reference_root/case/'evaluation.json')['alignment']
            return (target, body, moving, np.asarray(a['R_export_to_GT']),
                    a['scale'], np.asarray(a['translation']), a['GT_static_diagonal'], a)
        evaluation.gt_and_alignment = fixed_alignment
        try:
            scores = evaluation.joint_scores(model)
        finally:
            evaluation.gt_and_alignment = old
        scores['protocol'] = dict(axis_candidates_sealed_before_GT=True,
            matching='Independent type/axis-blind GT surface assignment',
            alignment='Frozen static-body similarity from earlier raw-source evaluation',
            geometry='Unchanged original generated geometry',
            correct='direction <= 5 degrees; hinge pivot <= 5% GT static-body diagonal')
        write_json(out/'joint_evaluation.json', scores)
        orig = scores['joint_methods']['original']; prior = scores['joint_methods']['prior']
        print('EVAL', case, 'GT', prior['GT_joint_count'],
              'original', orig['axis_correct_count'], 'optimized', prior['axis_correct_count'], flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=pathlib.Path, required=True)
    p.add_argument('--reference-root', type=pathlib.Path, required=True)
    a = p.parse_args()
    run(a.root, a.reference_root)


if __name__ == '__main__':
    main()
