"""Evaluation-only local rigid registration of static geometry, never GT axes.
Primary fixed-convention results remain immutable; separate sensitivity output.
"""
import argparse,ast,json,pathlib,numpy as np,trimesh
from scipy.spatial import cKDTree
ROOT=pathlib.Path(__file__).resolve().parents[2];p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();run=pathlib.Path(a.run);base=json.loads((run/'joint_evaluation/joint_errors.json').read_text());provenance=json.loads((run/'input_provenance.json').read_text());obj_path=pathlib.Path(json.loads((run/'status.json').read_text())['object']);nodes=json.loads(obj_path.read_text())['diffuse_tree'];root_id=next(x['pact_node_id'] for x in provenance['part_correspondence'] if x['gt_mesh_index']==0);node=next(n for n in nodes if n['id']==root_id);gt=trimesh.load(pathlib.Path(provenance['source_data_root'])/'gt/part_0.ply',process=False,force='mesh');pred=trimesh.load(obj_path.parent/node['plys'][0],process=False,force='mesh');R0=np.array(base['R_export_to_gtworld']);s=base['scale'];t0=np.array(base['translation']);gdiag=base['GT_static_diagonal']
def surface(mesh,seed):
 rng=np.random.default_rng(seed);ids=rng.choice(len(mesh.faces),12000,p=mesh.area_faces/mesh.area);v=mesh.vertices[mesh.faces[ids]];u=rng.random((12000,2));flip=u.sum(1)>1;u[flip]=1-u[flip];return v[:,0]+u[:,0,None]*(v[:,1]-v[:,0])+u[:,1,None]*(v[:,2]-v[:,0])
x=s*(surface(pred,0)@R0.T)+t0;y=surface(gt,0);tree=cKDTree(y);rotation=np.eye(3);translation=np.zeros(3);history=[]
for iteration in range(50):
 d,idx=tree.query(x);keep=d<=np.quantile(d,.8);src=x[keep];target=y[idx[keep]];c=src.mean(0);ct=target.mean(0);U,S,Vt=np.linalg.svd((src-c).T@(target-ct));deltaR=Vt.T@U.T
 if np.linalg.det(deltaR)<0:Vt[-1]*=-1;deltaR=Vt.T@U.T
 dt=ct-deltaR@c;x=x@deltaR.T+dt;translation=deltaR@translation+dt;rotation=deltaR@rotation;objective=float(np.mean(d[keep]**2));history.append(objective)
 if iteration>0 and abs(history[-1]-history[-2])<1e-10:break
# Independent samples for reporting fit quality; not used to choose rotation.
xv=s*(surface(pred,1)@R0.T)+t0;yv=surface(gt,1)
def rms(a,b):return float(np.sqrt((np.mean(cKDTree(a).query(b)[0]**2)+np.mean(cKDTree(b).query(a)[0]**2))/2))
before=rms(xv,yv);after=rms(xv@rotation.T+translation,yv)
functions=[n for n in ast.parse((ROOT/'repro/tools/evaluate_gt_joints.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ['unit','angle','position']];env={'np':np,'gdiag':gdiag};exec(compile(ast.Module(body=functions,type_ignores=[]),'metric-functions','exec'),env)
rows=[]
for original in base['joints']:
 go=np.array(original['GT_origin_world']);gd=np.array(original['GT_direction_world']);row={'gt_mobility_id':original['gt_mobility_id'],'GT_type':original['GT_type'],'variants':{}}
 for label,v in original['variants'].items():
  direction=rotation@np.array(v['direction_world']) if v['direction_world'] is not None else np.zeros(3);origin=rotation@np.array(v['origin_world'])+translation;pos=env['position'](origin,direction,go,gd) if original['GT_type']=='revolute' and v['predicted_type']=='revolute' else {'pivot_to_gt_axis':None,'axis_line_distance':None,'pivot_to_gt_axis_percent_static_diagonal':None,'axis_line_distance_percent_static_diagonal':None};row['variants'][label]={'axis_direction_error_degrees':env['angle'](direction,gd),**pos,'predicted_type':v['predicted_type'],'type_correct':v['type_correct']}
 rows.append(row)
report={'scene_id':base['scene_id'],'protocol':'Evaluation-only static-geometry local rigid ICP from fixed canonical initialization, 12000 surface samples, 80% trimmed point-to-point, at most50iterations. No GT joint axis/type/origin in fitting, no multistart or score-based rotation selection. No change to generated geometry or model.','rotation':rotation.tolist(),'translation':translation.tolist(),'rotation_delta_degrees':float(np.degrees(np.arccos(np.clip((np.trace(rotation)-1)/2,-1,1)))),'iterations':len(history),'heldout_static_symmetric_rms_before_percent_diagonal':100*before/gdiag,'heldout_static_symmetric_rms_after_percent_diagonal':100*after/gdiag,'caveat':'Static geometry may be symmetric or imperfect; registration-dependent joint error is a sensitivity diagnostic, not a replacement for the primary fixed-frame score.','joints':rows};dest=run/'joint_evaluation/geometry_registration_sensitivity.json';assert not dest.exists();dest.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
