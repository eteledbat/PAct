"""Resume-safe ten-case single-origin travel-only experiment."""
import argparse
import datetime
import json
import os
import pathlib
import subprocess
import sys

REPO=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'repro/tools'))
from auto_part_semantics import sha256,write_json
from fit_single_origin_extrema_travel import OLD_BATCH,REPAIRED,EXTREMES,source


def read(path):return json.loads(pathlib.Path(path).read_text())


def initialize(root):
    cases=read(OLD_BATCH/'protocol.json')['cases']
    sources={case:dict(four_version_root=str(source(case)[0]),single_image_model=str(source(case)[1])) for case in cases}
    write_json(root/'protocol.json',dict(cases=cases,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        published_release='v0.1.1-handle-attachment-axis',
        published_commit='bdf55de84df403bf1093eefb55b5b775382c5caf',
        original_four_version_sources=sources,
        release_scope='v0.1.1 updated the 100481 handle-axis artifact only; nine other existing single-image outputs remain from the v0.1.0 ten-case batch.',
        input_extremes=str(EXTREMES),
        fit='Freeze single-image generated geometry, registration, component ownership, joint type, axis direction and pivot. Search only each matched joint numeric q at the two observed extrema with projected instance-mask Dice; replace only numeric range endpoints.',
        video='Decode the existing four-column video without altering the source file; append a newly rendered fifth column using its exact two cameras and the unchanged single-image optimized geometry with fitted travel.',
        GT_3D_joint_used_for_fit=False,
        mask_provenance='Existing two-extrema oracle projected 2D part masks; not RGB-only segmentation.',
        evaluation='Post-fit comparison at both selected extreme frames: common fixed-body GT alignment, surface-voxel 3D IoU, plus per-joint type, direction and revolute pivot-line errors. All ten cases; no candidate-specific GT alignment.',
        code_sha256={name:sha256(REPO/'repro/tools'/name) for name in ['fit_single_origin_extrema_travel.py','render_single_origin_five_versions.py','evaluate_single_origin_travel_3d.py','summarize_single_origin_extrema_travel.py','summarize_single_origin_travel_3d.py','run_single_origin_extrema_travel.py']}))
    write_json(root/'controller_status.json',dict(state='initialized',cases={}))


def stage(root,which):
    if not (root/'protocol.json').exists():initialize(root)
    cases=read(root/'protocol.json')['cases'];status=read(root/'controller_status.json')
    script,marker={'fit':('fit_single_origin_extrema_travel.py','travel_fit.json'),
                   'render':('render_single_origin_five_versions.py','five_video_results.json'),
                   'evaluate':('evaluate_single_origin_travel_3d.py','three_way_3d_axis.json')}[which]
    for case in cases:
        out=root/case;out.mkdir(exist_ok=True,parents=True)
        if (out/marker).exists():status['cases'][f'{case}:{which}']='complete';continue
        if which in ('render','evaluate'):assert (out/'travel_fit.json').exists(),case
        args=[sys.executable,str(REPO/'repro/tools'/script),'--root',str(root),'--case',case]
        with (root/f'{case}_{which}.log').open('w') as log:
            result=subprocess.run(args,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4','OPENCV_IO_ENABLE_OPENEXR':'1'})
        status['cases'][f'{case}:{which}']='complete' if result.returncode==0 else 'failed'
        status['last_stage']=which;write_json(root/'controller_status.json',status)
        print(case,which,status['cases'][f'{case}:{which}'],flush=True)
        if result.returncode:raise RuntimeError(f'{case}:{which} failed; see {root/f"{case}_{which}.log"}')
    status['state']=which+'_complete';write_json(root/'controller_status.json',status)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--stage',choices=['fit','render','evaluate','all'],required=True);a=p.parse_args()
    root=a.root.resolve()
    for stage_name in (['fit','render','evaluate'] if a.stage=='all' else [a.stage]):stage(root,stage_name)
