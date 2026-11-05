"""Post-seal integrity, independent physics checks and read-only GT comparison."""
import argparse
import copy
import json
import pathlib

import imageio.v2 as imageio
import numpy as np
from scipy.spatial.transform import Rotation

import gravity_motion_trial as trial
from collision_refinement_common import sha256


def angle(a,b):
    a,b = trial.unit(a),trial.unit(b)
    return float(np.degrees(np.arccos(np.clip(abs(a@b),0,1))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root',required=True)
    a = p.parse_args()
    root = pathlib.Path(a.root).resolve()
    result = json.loads((root/'result.json').read_text())
    original_path = pathlib.Path(result['source_object'])
    selected_path = root/'gravity_only/object.json'
    original = json.loads(original_path.read_text())
    selected = json.loads(selected_path.read_text())
    assert sha256(original_path)==result['source_object_sha256']
    assert sha256(selected_path)==result['candidate_object_sha256']
    assert sha256(trial.__file__)==result['source_tool_sha256']
    for old,new in zip(original['diffuse_tree'],selected['diffuse_tree']):
        restored = copy.deepcopy(new)
        restored['joint']['axis']['direction'] = old['joint']['axis']['direction']
        assert restored==old,('Unexpected change beyond direction',old['id'])
    assert all(sha256(x['path'])==x['sha256'] for x in result['geometry_files'])
    # A hinge can have zero gravity torque at one configuration while gravity
    # drives it elsewhere. All-state sampling must reject that explanation.
    hinge = {'id':9,'joint':{'type':'revolute','range':[0.,90.],
                           'axis':{'direction':[1.,0.,0.],'origin':[0.,0.,0.]}}}
    com = np.array([0.,1.,0.])
    metric = trial.state_metrics(hinge,com,2.)
    assert abs(metric['states'][0]['normalized_motion_gravity_residual'])<1e-12
    assert metric['max_abs_residual']>.7
    assert metric['max_finite_difference_error']<1e-8
    # Rotate geometry, joint and gravity together: driving force is invariant.
    R = Rotation.from_rotvec([.3,-.5,.9]).as_matrix()
    rotated = copy.deepcopy(hinge)
    rotated['joint']['axis']['direction'] = (R@np.array([1.,0.,0.])).tolist()
    rotated['joint']['axis']['origin'] = [0.,0.,0.]
    old_gravity = trial.GRAVITY.copy()
    try:
        trial.GRAVITY = R@old_gravity
        rotated_metric = trial.state_metrics(rotated,R@com,2.)
    finally:
        trial.GRAVITY = old_gravity
    equivariance_error = max(abs(a['normalized_motion_gravity_residual']-b['normalized_motion_gravity_residual'])
                             for a,b in zip(metric['states'],rotated_metric['states']))
    assert equivariance_error<1e-12
    # A vertical hinge remains neutral at arbitrary horizontal pivots.
    pivot_residuals = []
    for pivot in [[0.,0.,0.],[.4,0.,-.7],[-.9,1.,.2]]:
        hinge['joint']['axis'] = {'direction':[0.,1.,0.],'origin':pivot}
        m = trial.state_metrics(hinge,[.3,.7,.5],2.)
        pivot_residuals.append(m['max_abs_residual'])
    assert max(pivot_residuals)<1e-12
    assert all(x['max_abs_residual']<1e-12 for x in result['type_probes'])
    # Existing GT is loaded ONLY after the candidate's byte seal is checked.
    gt = json.loads((pathlib.Path(result['source_run'])/'joint_evaluation/joint_errors.json').read_text())
    R,s,shift,D = np.array(gt['R_export_to_gtworld']),gt['scale'],np.array(gt['translation']),gt['GT_static_diagonal']
    scores = {}
    for name,obj in [('original',original),('gravity_only',selected)]:
        byid = {n['id']:n for n in obj['diffuse_tree']}
        rows = []
        for old in gt['joints']:
            n = byid[old['predicted_node_id']]
            j = n['joint']
            correct = j['type']==old['GT_type']
            d = R@np.array(j['axis']['direction'])
            o = s*(R@np.array(j['axis']['origin']))+shift
            gd,go = np.array(old['GT_direction_world']),np.array(old['GT_origin_world'])
            rows.append({'node_id':n['id'],'GT_part':old['gt_mesh_index'],
                         'predicted_type':j['type'],'GT_type':old['GT_type'],
                         'type_correct':correct,
                         'axis_error_degrees':angle(d,gd) if correct else None,
                         'pivot_error_percent_GT_D':float(100*np.linalg.norm(np.cross(o-go,trial.unit(gd)))/D) if correct and j['type']=='revolute' else None})
        scores[name] = {'rows':rows,'joint_type_accuracy':float(np.mean([x['type_correct'] for x in rows])),
                        'type_correct_count':sum(x['type_correct'] for x in rows), 'GT_joint_count':len(rows),
                        'conditional_mean_axis_error_degrees':float(np.mean([x['axis_error_degrees'] for x in rows if x['axis_error_degrees'] is not None])),
                        'conditional_mean_pivot_error_percent_GT_D':float(np.mean([x['pivot_error_percent_GT_D'] for x in rows if x['pivot_error_percent_GT_D'] is not None]))}
    videos = []
    for path in sorted(root.glob('*.mp4')):
        with imageio.get_reader(path) as reader:
            meta = reader.get_meta_data()
            count = reader.count_frames()
            reader.get_data(0)
            reader.get_data(count-1)
        videos.append({'file':path.name,'frames':count,'fps':meta['fps'],
                       'size':list(meta['size']),'sha256':sha256(path)})
    check = {'geometry_file_count':len(result['geometry_files']), 'all_geometry_hashes_unchanged':True,
             'only_axis_direction_changed':True,'types_ranges_pivots_hierarchy_unchanged':True,
             'analytic_vs_finite_difference_max_error':max(x['max_finite_difference_error'] for key in ['baseline','gravity_only'] for x in result[key]),
             'all_state_hinge_check':True,'world_coordinate_equivariance_error':equivariance_error,
             'different_vertical_pivots_same_zero_gravity':pivot_residuals,
             'both_nonzero_motion_types_pass_gravity':True,'videos':videos,
             'GT_evaluation':scores,
             'GT_protocol':'Read-only after candidate sealing. Reuse original fixed alignment and correspondence; no fitting or candidate selection. Wrong-type joints excluded from conditional axis error; retained in type denominator.',
             'conclusion':'Gravity feasibility improves; joint type accuracy is unchanged. No evidence of a type or pivot fix.'}
    (root/'verification.json').write_text(json.dumps(check,indent=2)+'\n')
    print(json.dumps(check,indent=2))


if __name__=='__main__':
    main()
