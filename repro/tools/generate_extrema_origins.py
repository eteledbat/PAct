"""Rerun official single-image PAct Origin on each selected extreme input.

The second extreme remains a held observation for semantic, travel and shape
fitting; the published PAct generator itself consumes one RGB/mask pair.
"""
import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
import cv2
import numpy as np
from PIL import Image

REPO=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
from modules.pact.datasets import ImageConditioned_dataset
sys.path.insert(0,str(REPO/'repro/tools'))
from auto_part_semantics import sha256,write_json


def read(path):return json.loads(pathlib.Path(path).read_text())


def prepare(root,case):
    selected=read(root/'extremes'/case/'manifest.json')['endpoints'][0]
    frame=selected['frame']
    target=REPO/'repro/inputs'/f'two_extrema_{case}_frame{frame:06d}'
    if (target/'input_provenance.json').exists():return target
    sample=target/'Sample';sample.mkdir(parents=True,exist_ok=False)
    name=f'{case}_{frame:06d}'
    image=sample/f'{name}_processed.png';shutil.copyfile(selected['rgb'],image)
    labels=np.load(selected['mask']).astype(np.float32)
    exr=sample/f'{name}_mask.exr';assert cv2.imwrite(str(exr),np.repeat(labels[:,:,None],3,axis=2))
    pack=ImageConditioned_dataset(str(target))[0]
    unordered,ordered=pack['unordered_masks'].numpy(),pack['masks'].numpy()
    mapping=[]
    for part in range(1,int(labels.max())+1):
        ids=np.unique(ordered[unordered==part]);ids=ids[ids>0]
        assert len(ids)==1,(case,part,ids)
        mapping.append(dict(gt_mesh_index=part-1,input_label=part,pact_node_id=int(ids[0]-1),visible_pixels=int((labels==part).sum())))
    record=dict(scene_id=case,frame_index=frame,source_image=selected['rgb'],source_data_root=read(root/'extremes'/case/'manifest.json')['source_scene'],
        protocol='One observed motion-extreme RGBA and oracle projected 2D part mask for PAct Origin. The second extreme and GT numeric joints are not passed to PAct generator.',
        selection='First frame of maximally separated actual motion-state pair; fixed seed42, official PAct checkpoint, no GT metric based checkpoint/seed selection',
        silhouette_iou=selected['silhouette_iou'],part_count=int(pack['num_parts']),part_correspondence=mapping,input_dir=str(target),
        source_extrema_manifest=str(root/'extremes'/case/'manifest.json'),
        files=[dict(path=str(p),sha256=sha256(p)) for p in [image,exr]])
    write_json(target/'input_provenance.json',record)
    print('ORIGIN INPUT',case,frame,'parts',record['part_count'],flush=True)
    return target


def run(root,case):
    out=root/'origin_runs';out.mkdir(exist_ok=True)
    pointer=out/f'{case}.json'
    if pointer.exists() and read(pointer).get('state')=='complete':return
    inputs=prepare(root,case)
    current=REPO/f'repro/current_gt_{case}.txt';original=current.read_text() if current.exists() else None
    try:
        cmd=[sys.executable,str(REPO/'repro/tools/run_gt_case.py'),'--scene',case,'--inputs',str(inputs)]
        log=out/f'{case}_origin.log'
        with log.open('w') as stream:proc=subprocess.run(cmd,cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,
            env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4','OPENCV_IO_ENABLE_OPENEXR':'1'})
        if proc.returncode:raise RuntimeError(f'Origin failed: {log}')
        generated=pathlib.Path(current.read_text().strip())
        status=read(generated/'status.json');assert status['state']=='complete'
        write_json(pointer,dict(state='complete',case=case,run=str(generated),origin_input=str(inputs),
            origin_input_sha256=sha256(inputs/'input_provenance.json'),
            source_frame=read(inputs/'input_provenance.json')['frame_index'],
            release_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
            generation_elapsed_seconds=status['elapsed_seconds']))
        print('ORIGIN COMPLETE',case,'seconds',round(status['elapsed_seconds'],1),flush=True)
    finally:
        if original is None:current.unlink(missing_ok=True)
        else:current.write_text(original)


def inventory(root):
    cases=read(REPO/'repro/outputs/closed_multicase_20261028_v1/protocol.json')['cases']
    assert all((root/'origin_runs'/f'{case}.json').exists() for case in cases)
    items=[]
    for case in cases:
        origin=read(root/'origin_runs'/f'{case}.json');input_dir=pathlib.Path(origin['origin_input'])
        prov=input_dir/'input_provenance.json';source=read(prov);frame=source['frame_index'];sample=input_dir/'Sample';name=f'{case}_{frame:06d}'
        end=read(root/'extremes'/case/'manifest.json')['endpoints']
        items.append(dict(case=case,run=origin['run'],provenance=str(prov),mask=str(sample/f'{name}_mask.exr'),
                          rgb=end[0]['rgb'],second_rgb=end[1]['rgb'],second_mask=end[1]['mask'],
                          source_origin_rgb=end[0]['rgb']))
    write_json(root/'two_extrema_inventory.json',items)
    print('NEW ORIGIN INVENTORY',len(items),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True)
    p.add_argument('--case');p.add_argument('--stage',choices=['prepare','infer','all','inventory'],default='all')
    a=p.parse_args();root=a.root.resolve()
    cases=[a.case] if a.case else read(REPO/'repro/outputs/closed_multicase_20261028_v1/protocol.json')['cases']
    if a.stage=='inventory':inventory(root);return
    if a.stage in ('prepare','all'):
        for case in cases:prepare(root,case)
    if a.stage in ('infer','all'):
        for case in cases:run(root,case)
    if a.stage=='all' and not a.case:inventory(root)


if __name__=='__main__':main()
