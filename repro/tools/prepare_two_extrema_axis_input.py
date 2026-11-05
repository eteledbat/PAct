"""Match two-view VLM semantics to the original PAct conditioning-view masks."""
import argparse
import json
import os
import pathlib
import shutil
import sys

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
import cv2
import numpy as np

REPO=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'repro/tools'))
from auto_part_semantics import sha256,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);a=p.parse_args()
    root=a.root.resolve();vlm=root/'vlm';source=root/'pulls' if (root/'pulls/PULL_RECOGNITION_SEAL.json').exists() else vlm
    cases=json.loads((REPO/'repro/outputs/closed_multicase_20261028_v1/protocol.json').read_text())['cases']
    inventory=json.loads((root/'two_extrema_inventory.json').read_text())
    target=root/'axis_input';target.mkdir(exist_ok=True)
    write_json(target/'protocol.json',dict(cases=cases,source_semantics=str(source),
        matching='Original PAct conditioning-view oracle 2D instance IDs; two-extreme VLM semantic labels are transferred by stable instance ID',
        GT_3D_joint_information_used=False))
    for case in cases:
        dest=target/case;dest.mkdir(exist_ok=True)
        item=next(x for x in inventory if x['case']==case)
        raw=cv2.imread(str(item['mask']),cv2.IMREAD_UNCHANGED)
        assert raw is not None
        ids=(raw if raw.ndim==2 else raw[:,:,0]).astype(np.int32)
        np.save(dest/'instance_ids.npy',ids)
        semantics=json.loads((source/case/'semantics.json').read_text())
        body_parts=[x for x in semantics['parts'] if x['semantic']=='base']
        corrections=[]
        if len(body_parts)>1:
            primary=max(body_parts,key=lambda x:int((ids==x['instance_id']).sum()))['instance_id']
            for part in body_parts:
                if part['instance_id']==primary:continue
                part.update(semantic='unknown',semantic_confidence='low',motion_role='unknown',motion_confidence='low')
                corrections.append(dict(instance_id=part['instance_id'],rule='Only one primary body; duplicate smaller VLM body becomes unknown pending generated-component evidence',original_label='base'))
        semantics['post_vlm_structural_validation']=corrections
        write_json(dest/'semantics.json',semantics)
        inputs=json.loads((vlm/case/'inputs.json').read_text())
        new_provenance=json.loads(pathlib.Path(item['provenance']).read_text())
        inputs.update(source_run=item['run'],source_image=item['rgb'],source_mask=item['mask'],
                      source_image_sha256=sha256(item['rgb']),source_mask_sha256=sha256(item['mask']),
                      input_label_to_native_node={str(x['input_label']):x['pact_node_id'] for x in new_provenance['part_correspondence']},
                      semantic_observation_extrema=[x['rgb'] for x in json.loads((root/'extremes'/case/'manifest.json').read_text())['endpoints']])
        write_json(dest/'inputs.json',inputs)
        write_json(dest/'transfer.json',dict(case=case,semantic_sha256=sha256(dest/'semantics.json'),
            original_conditioning_mask_sha256=sha256(dest/'instance_ids.npy'),
            semantic_source=str(source/case/'semantics.json'),stable_instance_ID_correspondence=True,
            post_vlm_structural_validation=corrections))
        print('AXIS INPUT',case,sorted(int(i) for i in np.unique(ids) if i),flush=True)


if __name__=='__main__':main()
