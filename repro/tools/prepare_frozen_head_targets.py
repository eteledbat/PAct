"""Prepare ONLY training-object GT targets with the released 24-channel encoder."""
import argparse
import copy
import json
import pathlib
import sys

import numpy as np
import torch

from frozen_joint_common import RAW_TO_EXPORT

REPO=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
from modules.pact.datasets.components import StandardDatasetBase_Part


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
    root=pathlib.Path(a.root);cfg=json.loads((root/'config.json').read_text());out=root/'training_targets';out.mkdir(exist_ok=False)
    encoder=object.__new__(StandardDatasetBase_Part);encoder.is_mask_primistic_axis_o=True
    for c in cfg['train_cases']:
        assert c not in cfg['test_cases']
        run=pathlib.Path(cfg['runs'][c]);base=json.loads((run/'joint_evaluation/joint_errors.json').read_text());obj=json.loads(pathlib.Path(json.loads((run/'status.json').read_text())['object']).read_text())
        R=np.array(base['R_raw_to_gtworld']);s=base['scale'];t=np.array(base['translation']);gts={g['predicted_node_id']:g for g in base['joints']}
        targets=[];masks=[];notes=[]
        for n in obj['diffuse_tree']:
            node=copy.deepcopy(n);center=RAW_TO_EXPORT.T@np.array(n['aabb']['center'])
            node['aabb']={'min_xyz':center.tolist(),'max_xyz':center.tolist()}
            node['joint']['axis']['direction']=(RAW_TO_EXPORT.T@np.array(n['joint']['axis']['direction'])).tolist()
            node['joint']['axis']['origin']=(RAW_TO_EXPORT.T@np.array(n['joint']['axis']['origin'])).tolist()
            gt=gts.get(n['id']);correct=gt is not None and gt['GT_type']==n['joint']['type']
            if correct:
                node['joint']['axis']['direction']=(R.T@np.array(gt['GT_direction_world'])).tolist()
                node['joint']['axis']['origin']=(R.T@(np.array(gt['GT_origin_world'])-t)/s).tolist()
                limits=gt['GT_limit'];span=float(limits['b']-limits['a'])
                if n['joint']['type']=='prismatic':span/=s
                node['joint']['range']=[0.,span]
            target,mask=encoder._prepare_node_data(node)
            selected=np.zeros(24,np.float32)
            if correct:
                selected[6:18]=1
            mask*=selected
            targets.append(target);masks.append(mask)
            notes.append({'node':n['id'],'type_correct':bool(correct),'target_from_GT':bool(correct),
                          'range_start_gauge':0,'GT_limits':None if gt is None else gt['GT_limit']})
        torch.save({'target':torch.tensor(np.array(targets)),'mask':torch.tensor(np.array(masks))},out/f'{c}.pt')
        (out/f'{c}.json').write_text(json.dumps({'case':c,'rows':notes,'target_source':'Existing verified training-object GT joint evaluation; evaluation-only static similarity is now explicitly GT-assisted TRAIN normalization','encoder':'Unchanged official StandardDatasetBase_Part._prepare_node_data','mask':'Official validity mask intersect requested continuous channels and unchanged predicted type'},indent=2))
    print('TRAINING TARGETS ONLY',cfg['train_cases'],flush=True)


if __name__=='__main__':main()
