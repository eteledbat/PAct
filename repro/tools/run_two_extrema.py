"""Resume-safe ten-case controller for the two-extrema v0.1.1 experiment."""
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

OLD=REPO/'repro/outputs/closed_multicase_20261028_v1'
RELEASE='bdf55de84df403bf1093eefb55b5b775382c5caf'
TOOLS=['extract_motion_extrema.py','prepare_two_extrema_inventory.py','generate_extrema_origins.py',
       'register_extrema_origin.py','recognize_semantics_two_stage.py',
       'recognize_door_pulls.py','prepare_two_extrema_axis_input.py','auto_component_axis_prior.py',
       'refine_peer_handle_axis.py','select_paired_handle_axis_from_extremes.py',
       'fit_two_extrema.py','evaluate_two_extrema.py','render_two_extrema.py',
       'summarize_two_extrema.py','run_two_extrema.py']


def read(path):return json.loads(pathlib.Path(path).read_text())


def initialize(root):
    cases=read(OLD/'protocol.json')['cases']
    axis=root/'axis';assert all((axis/c/'motion_groups.json').exists() for c in cases)
    protocol=dict(cases=cases,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        release_tag='v0.1.1-handle-attachment-axis',release_commit=RELEASE,
        origin='Official PAct generator rerun with fixed seed42 and published checkpoint on the first selected extreme of every case; second extreme used downstream',
        input='Two actual rendered motion-extreme frames maximizing Euclidean distance in normalized multi-joint state space; FreeArt 00/05',
        semantic='Pinned local Qwen3-VL-4B-Instruct rerun on both endpoint RGBs and 2D oracle part masks; local door-pull VLM; labels transferred by stable instance ID to Origin conditioning-view mask',
        mask='Oracle 2D GT mesh projection at known native cameras. This supervision uses GT geometry and joint trajectory for frame selection/mask projection; numeric GT joint parameters never guide fitting.',
        axis='Published v0.1.1 automatic generated-component matching and semantic prior; generated-contact peer fallback and two-extreme 2D mask selection of a shared direction for paired carrying handles; independent pivots and observed travel bounds',
        geometry='One shared generated-geometry deformation fitted to both endpoints, 240 Adam steps, RGB/silhouette/part Dice/interior CE, frozen original registration and axes',
        evaluation='Only after all fits: axis type/direction/pivot, input IoU/part mIoU, held-out state0 camera IoU/PSNR, 3D CD/F-score, 6-state surface motion; static-body alignment recomputed for each new Origin and frozen across its variants',
        video='84 frames, two common views, GT | Origin | VLM axis+observed travel | two-frame geometry',
        axis_root=str(axis),source_reference_root=str(OLD),
        registration_root=str(root/'registration'),
        code_sha256={name:sha256(REPO/'repro/tools'/name) for name in TOOLS})
    write_json(root/'protocol.json',protocol)
    write_json(root/'controller_status.json',dict(state='initialized',cases={}))


def stage(root,which):
    if not (root/'protocol.json').exists():initialize(root)
    protocol=read(root/'protocol.json');cases=protocol['cases']
    specs={'prepare':('fit_two_extrema.py','two_endpoint_prepared.json'),
           'fit':('fit_two_extrema.py','two_endpoint_fit.json'),
           'evaluate':('evaluate_two_extrema.py','evaluation.json'),
           'render':('render_two_extrema.py','video_results.json')}
    tool,marker=specs[which]
    if which in ('evaluate','render'):
        assert all((root/c/'two_endpoint_fit.json').exists() for c in cases)
    states=read(root/'controller_status.json')
    for case in cases:
        key=f'{case}:{which}'
        if (root/case/marker).exists():states['cases'][key]='complete';continue
        args=[sys.executable,str(REPO/'repro/tools'/tool),'--root',str(root),'--case',case]
        if which in ('prepare','fit'):args+=['--stage',which]
        log=root/f'{case}_{which}.log'
        with log.open('w') as stream:
            result=subprocess.run(args,cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,
                env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4','OPENCV_IO_ENABLE_OPENEXR':'1'})
        states['cases'][key]='complete' if result.returncode==0 else 'failed';states['last_stage']=which
        write_json(root/'controller_status.json',states)
        print(key,states['cases'][key],flush=True)
        if result.returncode:raise RuntimeError(f'{key} failed: {log}')
    states['state']=which+'_complete';write_json(root/'controller_status.json',states)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True)
    p.add_argument('--stage',choices=['prepare','fit','evaluate','render','all'],required=True)
    a=p.parse_args();root=a.root.resolve()
    for which in (['prepare','fit','evaluate','render'] if a.stage=='all' else [a.stage]):stage(root,which)


if __name__=='__main__':main()
