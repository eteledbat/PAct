"""GT-only post-seal interval/endpoint/trajectory evaluation. No fitting here."""
import argparse,copy,json,pathlib
import numpy as np,trimesh
from scipy.spatial import cKDTree
from collision_refinement_common import surface,sha256
from multicase_closed_common import REPO,write_json
from fit_vlm_drawer_ranges import read,descendants
from evaluate_multicase_closed import distance_scores


def area_centroid(mesh):return (np.asarray(mesh.triangles_center)*mesh.area_faces[:,None]).sum(0)/mesh.area

def target_truth(case,source,alignment,frozen):
 scene=pathlib.Path(read(source/'config.json')['source_scene']);R=np.array(alignment['R_export_to_GT']);front_world=R@np.array([0.,0.,1.]);truth={}
 if case=='20411':
  from evaluate_freeart_partnet_case import gt_parts
  states,delta=gt_parts(scene);truth[1]=dict(closed=states[0][1],opened=states[5][1],outward=delta/np.linalg.norm(delta),travel=float(np.linalg.norm(delta)),closed_source='GT GLB00',open_source='GT GLB05');return truth
 # All raw endpoint and frame0 values are used only in this post-seal evaluator.
 from add_gt_multicase_videos import GT
 # GT requires the case-level config and fixed alignment, supplied by evaluate().
 gt=GT(frozen)
 for record in gt.records:
  if record['type']!='prismatic':continue
  part=record['part'];mesh=gt.parts[part];d=np.array(record['axis_world']);initial=record['mesh_frame0_value'];a=record['reference_endpoint_a'];b=record['open_endpoint_b'];closed,opened=(a,b) if (b-a)*np.dot(d,front_world)>=0 else (b,a);cv=mesh.copy();ov=mesh.copy();cv.vertices=np.array(mesh.vertices)+(closed-initial)*d;ov.vertices=np.array(mesh.vertices)+(opened-initial)*d;delta=(opened-closed)*d;truth[part]=dict(closed=cv,opened=ov,outward=delta/np.linalg.norm(delta),travel=float(np.linalg.norm(delta)),closed_source=closed,open_source=opened)
 return truth


def evaluate(batch,case):
 assert read(batch/'ALL_FITS_SEAL.json')['state']=='complete';root=batch/case;fit=read(root/'range_fit.json');source=pathlib.Path(fit['source']);seal=read(root/'FIT_SEAL.json');assert sha256(root/'range_fit.json')==seal['range_fit_sha256'];nodes=read(root/'native/object.json')['diffuse_tree'];meshes={g['id']:trimesh.load(g['plys'][0],process=False,force='mesh') for g in nodes};reference=read(REPO/f'repro/outputs/vlm_component_geometry_20261029_v1/{case}/evaluation.json');alignment=read(REPO/f'repro/outputs/general_semantic_priors_20261029_v2/{case}/evaluation.json')['alignment'];R=np.array(alignment['R_export_to_GT']);s=alignment['scale'];shift=np.array(alignment['translation']);D=alignment['GT_static_diagonal'];write_json(root/'config.json',read(source/'config.json'));write_json(root/'evaluation.json',dict(alignment=alignment,state='evaluating'));truth=target_truth(case,source,alignment,root);priorrows=reference['joint_methods']['prior']['rows'];mapping={int(r['predicted_owner']):int(r['GT_part']) for r in priorrows if r.get('matched') and r.get('type_correct') and r.get('GT_type')=='prismatic'};results=[]
 for row in fit['rows']:
  gid=row['group'];g=next(v for v in nodes if v['id']==gid);GTpart=mapping.get(gid)
  if GTpart not in truth:results.append(dict(group=gid,evaluation_available=False,reason='Frozen pre-range type/axis-blind mapping has no GT prismatic part'));continue
  gt=truth[GTpart];gd=gt['outward'];gtclose=area_centroid(gt['closed']);gtopen=area_centroid(gt['opened']);ids=descendants(nodes,gid);mesh=trimesh.util.concatenate([meshes[i] for i in sorted(ids)]);sample=surface(mesh,4096,700+gid);center=area_centroid(mesh);base_world=s*(sample@R.T)+shift;center_world=s*(center@R.T)+shift;d=np.array(g['joint']['axis']['direction']);d/=np.linalg.norm(d);world_d=R@d;gt_closed_points=surface(gt['closed'],4096,800+gid);gt_delta=gtopen-gtclose;old=np.array(g['joint']['range']);variants={}
  for name in read(batch/'protocol.json')['variants']:
   tree=read(root/name/'object.json')['diffuse_tree'];joint=next(v['joint'] for v in tree if v['id']==gid);q=np.array(joint['range']);pc=center_world+s*q[0]*world_d;po=center_world+s*q[1]*world_d;predtravel=float(s*abs(q[1]-q[0]));close_along=float(abs(np.dot(pc-gtclose,gd)));open_along=float(abs(np.dot(po-gtopen,gd)));predinterval=sorted([float(np.dot(pc,gd)),float(np.dot(po,gd))]);gtinterval=sorted([float(np.dot(gtclose,gd)),float(np.dot(gtopen,gd))]);intersection=max(0.,min(predinterval[1],gtinterval[1])-max(predinterval[0],gtinterval[0]));union=max(predinterval[1],gtinterval[1])-min(predinterval[0],gtinterval[0]);trajectory=[]
   for t in np.linspace(0,1,21):
    pred=base_world+s*((1-t)*q[0]+t*q[1])*world_d;target=gt_closed_points+t*gt_delta;trajectory.append(dict(progress=float(t),**distance_scores(pred,target,D)))
   a=next(x for x in fit['rows'] if x['group']==gid);terms=a['native']['terms'] if name=='native' else a['variants'][name]['terms'];variants[name]=dict(joint_range=q.tolist(),predicted_travel_GT_units=predtravel,GT_travel_units=gt['travel'],travel_absolute_error_percent_D=100*abs(predtravel-gt['travel'])/D,travel_relative_error=abs(predtravel-gt['travel'])/gt['travel'],closed_along_axis_error_percent_D=100*close_along/D,open_along_axis_error_percent_D=100*open_along/D,mean_endpoint_along_error_percent_D=50*(close_along+open_along)/D,closed_centroid_error_percent_D=100*np.linalg.norm(pc-gtclose)/D,open_centroid_error_percent_D=100*np.linalg.norm(po-gtopen)/D,centroid_projected_interval_IoU=intersection/max(union,1e-12),trajectory_mean_CD_percent_D=float(np.mean([r['CD_percent_D'] for r in trajectory])),closed_CD_percent_D=trajectory[0]['CD_percent_D'],open_CD_percent_D=trajectory[-1]['CD_percent_D'],trajectory=trajectory,predicted_geometry_constraint_terms=terms)
  results.append(dict(group=gid,semantic_instance=row['semantic_instance'],GT_part=GTpart,evaluation_available=True,GT_closed_centroid_world=gtclose.tolist(),GT_open_centroid_world=gtopen.tolist(),GT_outward_direction=gd.tolist(),GT_travel=gt['travel'],GT_closed_source=gt['closed_source'],GT_open_source=gt['open_source'],source_generated_centroid_world=center_world.tolist(),matching='Reuse frozen PRE-range geometry-based one-to-one association; no interval/error rematching',variants=variants));print('EVAL',case,gid,'endpoint',variants['native']['mean_endpoint_along_error_percent_D'],'->',variants['full']['mean_endpoint_along_error_percent_D'],'travel',variants['native']['travel_absolute_error_percent_D'],'->',variants['full']['travel_absolute_error_percent_D'],flush=True)
 metrics=dict(case=case,alignment=alignment,GT_only_after_all_fit_seals=True,rows=results,axis_and_geometry_unchanged=True,definitions=dict(endpoint='Absolute surface-centroid position error projected on GT outward rail axis, divided by static GT D. Includes fixed shape/reference-origin bias; no per-part alignment.',interval_IoU='Intersection/union of endpoint surface-centroid projected intervals in common GT frame. Not volumetric IoU or independent physical-stop certification.',trajectory='21 common closed-to-open progress values; bidirectional nonsquared surfaceCD on4096 area samples; fixed raw-body similarity; frozen geometry, only drawer ranges differ.',travel='Absolute error of numeric stroke magnitude after fixed global scale, divided by static GT D. Independent of endpoint-reference gauge.'),frozen_mapping_sha256=sha256(REPO/f'repro/outputs/vlm_component_geometry_20261029_v1/{case}/evaluation.json'));write_json(root/'evaluation.json',metrics)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);a=p.parse_args();batch=a.root.resolve()
 for case in read(batch/'protocol.json')['cases']:evaluate(batch,case)
