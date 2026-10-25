"""Check sampled trilinear-field Jacobians; not a certificate of valid solid mesh."""
import argparse
import json
import pathlib

import numpy as np
import torch
from plyfile import PlyData

from collision_refinement_common import load_asset


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=pathlib.Path(a.root);cfg=json.loads((root/'config.json').read_text());bounds={}
    undo=np.array([[1,0,0],[0,0,1],[0,-1,0]])
    for case,run in cfg['cases'].items():
        asset=load_asset(run);rows=[]
        for part in asset['parts']:
            node=part['node'];v=PlyData.read(asset['object_path'].parent/node['gs'][0])['vertex'].data;xyz=np.column_stack([v[k] for k in ['x','y','z']])@undo.T;rows.append(np.array([np.minimum(xyz.min(0),part['mesh'].bounds[0]),np.maximum(xyz.max(0),part['mesh'].bounds[1])]))
        bounds[case]=rows
    results={};g=cfg['geometry_grid']
    for experiment in cfg['experiments']:
        name=experiment['name'];results[name]={}
        for arm in ['A','B']:
            state=torch.load(root/f'{name}_{arm}_train/checkpoint.pt',map_location='cpu',weights_only=False)['model'];parts=[]
            for part_index,bound in enumerate(bounds[experiment['case']]):
                field=state[f'geometry.{part_index}'].numpy().reshape(g,g,g,3);cell_size=(bound[1]-bound[0]).clip(1e-5)/(g-1);determinants=[]
                for i in range(g-1):
                    for j in range(g-1):
                        for k in range(g-1):
                            block=field[i:i+2,j:j+2,k:k+2];columns=[np.diff(block,axis=axis).mean(axis=(0,1,2))/cell_size[axis] for axis in range(3)];J=np.eye(3)+np.stack(columns,axis=1);determinants.append(float(np.linalg.det(J)))
                parts.append({'part_index':part_index,'min_cell_center_Jacobian_det':min(determinants),'nonpositive_cell_centers':sum(d<=0 for d in determinants),'sampled_cells':len(determinants)})
            results[name][arm]=parts
    result={'results':results,'interpretation':'Determinant<=0 at a sampled cell center indicates local field orientation reversal/degeneracy. Positive center samples do not certify no fold everywhere, self-intersection-free meshes, or solid validity.'};(root/'deformation_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
