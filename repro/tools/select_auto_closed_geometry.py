"""Choose automatic candidates using only their closed input image scores.

The geometric/contact-prior candidate is kept unless a second automatic fit
improves the observed silhouette IoU by more than two percentage points. No GT
3D mesh, GT joint, or manual-fit metric is read by this selector.
"""
import argparse
import pathlib

from auto_part_semantics import sha256,write_json
from fit_auto_closed_geometry import CASES,read


def choose(outputs,case):
    sources={
        'geometric_pose':outputs/'closed_auto_geometry_v8_geometric_pose_20261104'/case,
        'image_pose':outputs/'closed_auto_geometry_v5_20261104'/case,
        'closed_drawer_image_pose':outputs/'closed_auto_geometry_v6_closed_drawers_20261104'/case,
        'previous_auto':outputs/'closed_travel_reference_geometry_v011_20261031_v2'/case,
    }
    scores={}
    for label,root in sources.items():
        if label=='previous_auto':
            scores[label]=read(root/'fit_seal.json')['geometry_fit']['image']['IoU']
        elif (root/'auto_closed_fit.json').exists():
            scores[label]=read(root/'auto_closed_fit.json')['after']['image']['IoU']
    base=scores['geometric_pose']
    winner=max(scores,key=scores.get)
    chosen=winner if scores[winner]>base+.02 else 'geometric_pose'
    checkpoint=sources[chosen]/('observed_pose_geometry.pt' if chosen=='previous_auto' else 'auto_closed_geometry.pt')
    return dict(case=case,selected=chosen,selected_root=str(sources[chosen]),
                checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),
                axis_manifest_sha256=sha256(sources[chosen]/'motion_groups.json'),
                geometric_pose_2d_IoU=base,candidate_2d_IoU=scores,
                criterion='Prefer geometric closure/body-contact prior; switch only if another fully automatic candidate improves the observed closed silhouette IoU by >0.02. No manual or GT 3D/joint input.')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--outputs',type=pathlib.Path,required=True)
    p.add_argument('--selection-root',type=pathlib.Path,required=True)
    a=p.parse_args()
    outputs=a.outputs.resolve()
    root=a.selection_root.resolve()
    root.mkdir(parents=True,exist_ok=True)
    rows=[choose(outputs,case) for case in CASES]
    write_json(root/'SELECTION.json',dict(cases=rows,
        selection_inputs='Only closed-image silhouette IoU in sealed automatic fit results; excludes manual result and GT 3D/joint evaluation.',
        base='v8 geometric closed-pose and body-contact prior',margin=0.02))
    for row in rows:
        link=root/row['case']
        target=pathlib.Path(row['selected_root'])
        if link.is_symlink() and link.resolve()!=target:link.unlink()
        if not link.exists():link.symlink_to(target,target_is_directory=True)
        print(row['case'],row['selected'],round(row['candidate_2d_IoU'][row['selected']],3),flush=True)


if __name__=='__main__':main()
