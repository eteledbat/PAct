"""Compare PAct raw and exported joints with existing GT under fixed data conventions.
Only a uniform scale and translation from the static part are used for positions.
No rotation/axis fitting, ICP, GT-selected prediction, or joint refinement.
"""
import argparse,json,pathlib,sys,math
import numpy as np,trimesh
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=pathlib.Path(__file__).resolve().parents[2];p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();run=pathlib.Path(a.run).resolve();assert run.is_relative_to(ROOT/'repro/outputs');status=json.loads((run/'status.json').read_text());assert status['state']=='complete';provenance=json.loads((run/'input_provenance.json').read_text());scene=pathlib.Path(provenance['source_data_root']);obj_path=pathlib.Path(status['object']);nodes=json.loads(obj_path.read_text())['diffuse_tree'];pred={n['id']:n for n in nodes};raw=np.load(run/'joint_trace/raw_head_00.npy').astype(np.float64);parsed=json.loads((run/'joint_trace/decoded_before_bbox.json').read_text())[0];mapping=provenance['part_correspondence'];out=run/'joint_evaluation';out.mkdir(exist_ok=False)
# Fixed matrices taken directly from original exporters and VideoArtGS read_gt.
R_gt=np.array([[0,0,-1],[-1,0,0],[0,1,0]],float)
R_raw_to_export=np.array([[1,0,0],[0,0,1],[0,-1,0]],float)
R_raw_to_world=R_gt@R_raw_to_export
root_id=next(x['pact_node_id'] for x in mapping if x['gt_mesh_index']==0);gt_base=trimesh.load(scene/'gt/part_0.ply',process=False,force='mesh');pred_base=trimesh.load(obj_path.parent/pred[root_id]['plys'][0],process=False,force='mesh');pb=np.asarray(pred_base.vertices)@R_gt.T;gb=np.asarray(gt_base.vertices);gdiag=float(np.linalg.norm(np.ptp(gb,axis=0)));pdiag=float(np.linalg.norm(np.ptp(pb,axis=0)));scale=gdiag/pdiag;center=(gb.min(0)+gb.max(0))/2;translation=center-scale*(pb.min(0)+pb.max(0))/2
assert scale>0 and np.isfinite(scale)
records=json.loads((scene/'gt/mobility_v2.json').read_text());base_ids=[n['id'] for n in records if n['parent']==-1];gt=[]
for n in records:
 if n.get('parent') not in base_ids or n.get('joint') not in ['hinge','slider']:continue
 if n.get('name') in ['knob','tray','control_button','button']:continue
 if n.get('name')=='handle' and provenance['scene_id']!='100481':continue
 gt.append(n)
assert len(gt)==len(mapping)-1,'Dataset correspondence/filter mismatch'

def unit(d):
 d=np.asarray(d,float);norm=np.linalg.norm(d);return d/norm if norm>1e-10 else None

def angle(d,g):
 d=unit(d);g=unit(g)
 return float(np.degrees(np.arccos(np.clip(abs(np.dot(d,g)),0,1)))) if d is not None and g is not None else None

def position(o,d,go,gd):
 d=unit(d);gd=unit(gd)
 if d is None:return {'pivot_to_gt_axis':None,'axis_line_distance':None,'pivot_to_gt_axis_percent_static_diagonal':None,'axis_line_distance_percent_static_diagonal':None}
 delta=np.asarray(o)-go;cross=np.cross(d,gd);n=np.linalg.norm(cross);point=float(np.linalg.norm(np.cross(delta,gd)));line=float(abs(np.dot(delta,cross))/n) if n>1e-7 else point
 return {'pivot_to_gt_axis':point,'axis_line_distance':line,'pivot_to_gt_axis_percent_static_diagonal':100*point/gdiag,'axis_line_distance_percent_static_diagonal':100*line/gdiag}

rows=[]
for gt_mesh_index,g in enumerate(gt,1):
 correspondence=next(x for x in mapping if x['gt_mesh_index']==gt_mesh_index);pid=correspondence['pact_node_id'];node=pred[pid];target_type='revolute' if g['joint']=='hinge' else 'prismatic';go=R_gt@np.array(g['jointData']['axis']['origin']);gd=unit(R_gt@np.array(g['jointData']['axis']['direction']));r=raw[pid];raw_type={1:'fixed',2:'revolute',3:'prismatic',4:'screw',5:'continuous'}[int(np.clip((np.mean(r[:6])+.5)*5,1,5).round())]
 variants={'raw_head':{'type':raw_type,'direction':R_raw_to_world@r[6:9],'origin':scale*(R_raw_to_world@r[9:12])+translation},'axis_snapped_before_bbox':{'type':parsed[pid]['joint']['type'],'direction':R_raw_to_world@np.array(parsed[pid]['joint']['axis']['direction']),'origin':scale*(R_raw_to_world@np.array(parsed[pid]['joint']['axis']['origin']))+translation},'original_exported':{'type':node['joint']['type'],'direction':R_gt@np.array(node['joint']['axis']['direction']),'origin':scale*(R_gt@np.array(node['joint']['axis']['origin']))+translation}}
 result={'gt_mesh_index':gt_mesh_index,'gt_mobility_id':g['id'],'predicted_node_id':pid,'GT_type':target_type,'GT_direction_world':gd.tolist(),'GT_origin_world':go.tolist(),'GT_limit':g['jointData'].get('limit'),'predicted_exported_range':node['joint']['range'],'variants':{}}
 for label,v in variants.items():
  pos=position(v['origin'],v['direction'],go,gd) if target_type=='revolute' and v['type']=='revolute' else {'pivot_to_gt_axis':None,'axis_line_distance':None,'pivot_to_gt_axis_percent_static_diagonal':None,'axis_line_distance_percent_static_diagonal':None}
  result['variants'][label]={'predicted_type':v['type'],'type_correct':v['type']==target_type,'direction_world':unit(v['direction']).tolist() if unit(v['direction']) is not None else None,'origin_world':v['origin'].tolist(),'axis_direction_error_degrees':angle(v['direction'],gd),**pos}
 # Plot actual geometry and three axis lines in the shared GT-world frame.
 gt_moving=trimesh.load(scene/f'gt/part_{gt_mesh_index}.ply',process=False,force='mesh');pred_moving=trimesh.load(obj_path.parent/node['plys'][0],process=False,force='mesh');pv=scale*(np.asarray(pred_moving.vertices)@R_gt.T)+translation;pg=scale*pb+translation;rng=np.random.default_rng(0);fig=plt.figure(figsize=(15,5))
 for view,azim in enumerate([25,125,225]):
  ax=fig.add_subplot(1,3,view+1,projection='3d')
  for vertices,color,name in [(gb,'grey','GT base'),(np.asarray(gt_moving.vertices),'green','GT part'),(pg,'lightblue','PAct base'),(pv,'#efb474','PAct part')]:
   ids=rng.choice(len(vertices),min(4500,len(vertices)),replace=False);ax.scatter(*vertices[ids].T,s=.3,c=color,alpha=.15,rasterized=True,label=name)
  for origin,direction,color,label in [(go,gd,'red','GT'),(variants['raw_head']['origin'],variants['raw_head']['direction'],'orange','raw'),(variants['original_exported']['origin'],variants['original_exported']['direction'],'blue','exported')]:
   direction=unit(direction)
   if direction is None:continue
   on_line=origin+np.dot(center-origin,direction)*direction;segment=on_line[None,:]+np.array([-.45,.45])[:,None]*gdiag*direction[None,:];ax.plot(*segment.T,c=color,lw=2,label=label+' axis')
  rad=gdiag*.6;ax.set(xlim=(center[0]-rad,center[0]+rad),ylim=(center[1]-rad,center[1]+rad),zlim=(center[2]-rad,center[2]+rad),xlabel='GT world x',ylabel='y',zlabel='z');ax.set_box_aspect([1,1,1]);ax.view_init(elev=15,azim=azim)
  if view==0:ax.legend(fontsize=7)
 fig.suptitle(f"Case {provenance['scene_id']} GTpart{gt_mesh_index}: raw {result['variants']['raw_head']['axis_direction_error_degrees']:.3f}°, exported {result['variants']['original_exported']['axis_direction_error_degrees']}°");fig.tight_layout();fig.savefig(out/f'axes_gtpart{gt_mesh_index}.png',dpi=160);plt.close(fig);rows.append(result)
# Geometry-only sanity diagnostic, not a fitted rotation or a model-selection metric.
def surface(mesh,count=10000):
 rng=np.random.default_rng(0);ids=rng.choice(len(mesh.faces),count,p=mesh.area_faces/mesh.area);v=mesh.vertices[mesh.faces[ids]];u=rng.random((count,2));flip=u.sum(1)>1;u[flip]=1-u[flip];return v[:,0]+u[:,0,None]*(v[:,1]-v[:,0])+u[:,1,None]*(v[:,2]-v[:,0])
gs=surface(gt_base);ps=scale*(surface(pred_base)@R_gt.T)+translation;ds=cKDTree(gs).query(ps)[0];dg=cKDTree(ps).query(gs)[0];rms=float(np.sqrt((np.mean(ds**2)+np.mean(dg**2))/2))
summary={'scene_id':provenance['scene_id'],'protocol':provenance['protocol'],'input_silhouette_iou':provenance['silhouette_iou'],'selection':'Fixed frame0 and seed42; no GT axis passed to model; correspondence from input-mask reorder, not GT joint-error permutation.','direction_alignment':'Fixed PAct export canonical frame to VideoArtGS GT-world R, from official source conventions; no fitted rotation or ICP.','position_alignment':'One uniform scale from static-part AABB diagonals and translation from static AABB centers; uses GT static geometry for evaluation only, never GT joint axis.','R_export_to_gtworld':R_gt.tolist(),'R_raw_to_gtworld':R_raw_to_world.tolist(),'scale':scale,'translation':translation.tolist(),'GT_static_diagonal':gdiag,'static_surface_symmetric_rms_after_fixed_alignment':rms,'static_surface_rms_percent_diagonal':100*rms/gdiag,'unit_note':'Distances in GT dataset-coordinate units and percent static diagonal, not claimed physically calibrated centimeters.','angle_definition':'acos(abs(dot(unit directions))) in degrees, ignores axis sign.','pivot_definition':'Distance from predicted origin to GT infinite axis, plus shortest infinite-axis line distance. Along-axis origin offset not identifiable; nonparallel intersecting infinite lines can have zero line distance despite a poor pivot.','GT_joint_ranges_comparison':'Not scored: input is a nonzero articulated state and generated rest-state convention may differ. GT limits and predicted range retained for diagnosis.','joints':rows,'aggregate':{}}
for label in ['raw_head','axis_snapped_before_bbox','original_exported']:
 values=[x['variants'][label] for x in rows];angles=[x['axis_direction_error_degrees'] for x in values if x['axis_direction_error_degrees'] is not None];summary['aggregate'][label]={'joint_type_accuracy':float(np.mean([x['type_correct'] for x in values])),'mean_axis_direction_error_degrees':float(np.mean(angles)) if angles else None,'direction_available_count':len(angles),'joint_count':len(values)}
(out/'joint_errors.json').write_text(json.dumps(summary,indent=2,allow_nan=False));print(json.dumps({'scene':summary['scene_id'],'static_RMS_percent_diagonal':summary['static_surface_rms_percent_diagonal'],'aggregate':summary['aggregate'],'joint_errors':rows},indent=2))
