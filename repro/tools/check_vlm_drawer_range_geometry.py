"""Post-seal nearness validation with independent Gaussian center samples."""
import argparse,pathlib,json
import numpy as np,trimesh
from scipy.spatial import cKDTree
from plyfile import PlyData
from fit_vlm_drawer_ranges import read,descendants,DEFAULT
from multicase_closed_common import write_json


def check(batch,case):
 root=batch/case;assert read(batch/'ALL_FITS_SEAL.json')['state']=='complete';fit=read(root/'range_fit.json');nodes=read(root/'native/object.json')['diffuse_tree'];undo=np.array([[1,0,0],[0,0,1],[0,-1,0]],float);clouds={};allpoints=[]
 for g in nodes:
  data=PlyData.read(g['gs'][0])['vertex'].data;alpha=1/(1+np.exp(-data['opacity'].astype(float)));x=np.column_stack([data[k] for k in ['x','y','z']]).astype(float)@undo.T;clouds[g['id']]=x[alpha>=.05];allpoints.append(x)
 bounds=np.vstack([trimesh.load(g['plys'][0],process=False,force='mesh').bounds for g in nodes]);D=float(np.linalg.norm(np.ptp(bounds,axis=0)));rng=np.random.default_rng(2026);body=np.concatenate([clouds[g['id']] for g in nodes if g.get('semantic_owner')==1]);body=body[rng.choice(len(body),min(32768,len(body)),replace=False)];tree=cKDTree(body);eps=.005*D;rows=[]
 for row in fit['rows']:
  gid=row['group'];g=next(g for g in nodes if g['id']==gid);d=np.array(g['joint']['axis']['direction']);d=d/np.linalg.norm(d);owned=descendants(nodes,gid);moving=np.concatenate([clouds[i] for i in sorted(owned)]);moving=moving[rng.choice(len(moving),min(8192,len(moving)),replace=False)];variants={}
  for name in ['native','full']:
   n=next(n for n in read(root/name/'object.json')['diffuse_tree'] if n['id']==gid);lo,hi=n['joint']['range'];curve=[]
   for t in np.linspace(0,1,41):
    q=(1-t)*lo+t*hi;distance=tree.query(moving+q*d,workers=4)[0];curve.append(float(np.exp(-distance**2/(2*eps**2)).mean()))
   variants[name]=dict(mean_nearness_proxy=float(np.mean(curve)),max_nearness_proxy=float(np.max(curve)),curve=curve)
  rows.append(dict(group=gid,variants=variants))
 write_json(root/'heldout_geometry_checks.json',dict(case=case,seed=2026,progress_samples=41,rows=rows,note='Independent Gaussian-center samples and41 progress points; proxy only, not solid volume or proof of no penetration. Axis/geometry fixed; intervals only.'))
 print('GEOMETRY CHECK',case,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);a=p.parse_args();batch=a.root.resolve()
 for case in read(batch/'protocol.json')['cases']:check(batch,case)
