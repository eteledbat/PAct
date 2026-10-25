"""Geometry qualification: global assembly plus pose-only per-part shape fit.

GT part meshes are at the input's articulated state. A per-part rigid fit is a
shape-only diagnostic, NOT a new joint alignment or an assembled reconstruction.
No per-part scale fitting. Score on fresh area samples after static scale fixed.
"""
import argparse,itertools,json,pathlib,time
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from collision_refinement_common import load_asset,surface,sha256

def metric(x,y,D):
    da=cKDTree(y).query(x)[0];db=cKDTree(x).query(y)[0]
    return {'CD_mean_percent_static_diagonal':float(50*(da.mean()+db.mean())/D),
            'RMS_percent_static_diagonal':float(100*np.sqrt((np.mean(da**2)+np.mean(db**2))/2)/D)}

def proper_pca(x):
    _,v=np.linalg.eigh(np.cov(x.T));v=v[:,::-1]
    if np.linalg.det(v)<0:v[:,-1]*=-1
    return v

def rigid_shape_fit(x,y,D):
    cx,cy=x.mean(0),y.mean(0);xx=(x-cx)/D;yy=(y-cy)/D;vx,vy=proper_pca(xx),proper_pca(yy);tree=cKDTree(yy);candidates=[]
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([-1,1],repeat=3):
            P=np.eye(3)[:,perm]@np.diag(signs)
            if np.linalg.det(P)<0:continue
            R=vy@P@vx.T;t=np.zeros(3)
            for iteration in range(35):
                q=xx@R.T+t;d,idx=tree.query(q);keep=d<=np.quantile(d,.9);src=q[keep];dst=yy[idx[keep]];a,b=src.mean(0),dst.mean(0)
                U,_,Vt=np.linalg.svd((src-a).T@(dst-b));dr=Vt.T@U.T
                if np.linalg.det(dr)<0:Vt[-1]*=-1;dr=Vt.T@U.T
                dt=b-dr@a;R=dr@R;t=dr@t+dt
                if np.linalg.norm(dt)+np.linalg.norm(dr-np.eye(3))<1e-7:break
            q=xx@R.T+t;score=float(np.mean(tree.query(q)[0]**2)+np.mean(cKDTree(q).query(yy)[0]**2))
            candidates.append((score,R,D*t+cy-R@cx))
    score,R,t=min(candidates,key=lambda c:c[0]);assert np.linalg.det(R)>.99999
    return R,t,score

p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--out',required=True);a=p.parse_args();start=time.monotonic()
asset=load_asset(a.run);run=asset['run'];baseline=json.loads((run/'joint_evaluation/joint_errors.json').read_text());prov=json.loads((run/'input_provenance.json').read_text())
scene=pathlib.Path(prov['source_data_root']);R,s,t=np.array(baseline['R_export_to_gtworld']),baseline['scale'],np.array(baseline['translation']);D=baseline['GT_static_diagonal']
nodes={p['node']['id']:p for p in asset['parts']};rows=[]
for item in prov['part_correspondence']:
    gi,pid=item['gt_mesh_index'],item['pact_node_id'];pred=nodes[pid]['mesh'];target_path=scene/f'gt/part_{gi}.ply';target=trimesh.load(target_path,process=False,force='mesh')
    source_fit=s*(surface(pred,4000,100+gi)@R.T)+t;target_fit=surface(target,4000,200+gi)
    xv=s*(surface(pred,8000,300+gi)@R.T)+t;yv=surface(target,8000,400+gi)
    row={'GT_mesh_index':gi,'predicted_node_id':pid,'GT_mesh_sha256':sha256(target_path),'predicted_mesh_sha256':sha256(nodes[pid]['mesh_path']),
         'global_frame':metric(xv,yv,D),'predicted_type':nodes[pid]['node']['joint']['type']}
    if gi:
        rr,tt,score=rigid_shape_fit(source_fit,target_fit,D);row.update(shape_only=metric(xv@rr.T+tt,yv,D),shape_fit_R=rr.tolist(),shape_fit_translation=tt.tolist(),fit_objective=score)
    rows.append(row);print('GEOMETRY',prov['scene_id'],gi,row.get('shape_only',row['global_frame']),flush=True)
result={'scene_id':prov['scene_id'],'run':str(run),'parts':rows,'GT_static_diagonal':D,'scale':s,
        'protocol':'Global static-AABB scale/pose remains fixed for joint scores. Independent per-moving-part rigid PCA24 + trimmed ICP35 (4000 fit / 8000 fresh evaluation samples), no scaling, gives shape-only lower-bound error at unknown rest pose. Not a corrected assembly and never passed to optimization.',
        'static_primary_RMS_percent_diagonal':baseline['static_surface_rms_percent_diagonal'],
        'elapsed_seconds':time.monotonic()-start}
pathlib.Path(a.out).write_text(json.dumps(result,indent=2,allow_nan=False))
