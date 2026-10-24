"""Evaluation-only PartNet joint/geometry comparison in the original GLB frame.

The source has actual six-state geometry, so translation direction/travel are
checked against mesh displacement rather than guessed URDF frame conversions.
One static-part scale/translation, and a separate static-only ICP sensitivity
transform, are reused for every candidate. No GT-based candidate selection.
"""
import argparse, json, pathlib
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from collision_refinement_common import load_asset, surface, transforms, moved, sha256, candidate_nodes


def angle(a,b):
    a,b = np.asarray(a,float),np.asarray(b,float)
    if min(np.linalg.norm(a),np.linalg.norm(b)) < 1e-12: return None
    return float(np.degrees(np.arccos(np.clip(abs(a@b/np.linalg.norm(a)/np.linalg.norm(b)),0,1))))


def gt_parts(source):
    meshes = [trimesh.load(source/'gt_mesh'/f'{i:02d}.glb',process=False,force='mesh') for i in range(6)]
    a,b = meshes[0],meshes[-1]
    assert all(np.array_equal(m.faces,a.faces) and m.vertices.shape==a.vertices.shape for m in meshes)
    displacement = b.vertices-a.vertices
    mag = np.linalg.norm(displacement,axis=1)
    moving = mag > .1*mag.max()
    counts = moving[a.faces].sum(1)
    assert not ((counts>0)&(counts<3)).any()
    delta = np.median(displacement[moving],axis=0)
    assert np.max(np.linalg.norm(displacement[moving]-delta,axis=1)) < 1e-4
    states = [tuple(m.submesh([counts==k],append=True,repair=False) for k in [0,3]) for m in meshes]
    for i,m in enumerate(meshes):
        assert np.max(abs(m.vertices[~moving]-a.vertices[~moving]))<1e-6
        assert np.max(abs(m.vertices[moving]-a.vertices[moving]-i/5*delta))<1e-5
    return states,delta


def quality(mesh):
    v,inv = np.unique(mesh.vertices,axis=0,return_inverse=True)
    m = trimesh.Trimesh(v,inv[mesh.faces],process=False)
    counts = np.bincount(m.edges_unique_inverse,minlength=len(m.edges_unique))
    return {'faces':len(m.faces),'exact_welded_vertices':len(m.vertices),'watertight':bool(m.is_watertight),
            'winding_consistent':bool(m.is_winding_consistent),'boundary_edges':int((counts==1).sum()),
            'nonmanifold_edges':int((counts>2).sum()),'components':len(m.split(only_watertight=False)),
            'degenerate_faces':int((m.area_faces<1e-14).sum()),'self_intersection':'not_verified',
            'solid_volume_status':'unavailable until closed, consistent and self-intersection-free surfaces are verified'}


def distances(a,b,D):
    da,db = cKDTree(b).query(a)[0],cKDTree(a).query(b)[0]
    return {'symmetric_mean_distance':float((da.mean()+db.mean())/2),
            'symmetric_mean_distance_percent_static_diagonal':float(100*(da.mean()+db.mean())/2/D),
            'symmetric_RMS_percent_static_diagonal':float(100*np.sqrt((np.mean(da**2)+np.mean(db**2))/2)/D),
            'samples_each':len(a),'note':'Surface Chamfer with unsquared Euclidean distances; not a solid occupancy score.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
    root=pathlib.Path(a.root);config=json.loads((root/'experiment.json').read_text())
    assert len(config['cases'])==1
    name,run=next(iter(config['cases'].items()));asset=load_asset(run)
    assert json.loads((asset['run']/'status.json').read_text())['state']=='complete'
    provenance=json.loads((asset['run']/'input_provenance.json').read_text());source=pathlib.Path(provenance['source_data_root'])
    gt,delta=gt_parts(source);gd=delta/np.linalg.norm(delta)
    # The joint label confirms prismatic type; axis agreement with actual motion
    # is recorded as a GT-only frame audit, never used to align a prediction.
    labels=json.loads((source/'joint_info.json').read_text());assert len(labels)==1 and labels[0]['type']=='prismatic'
    mapping={v['gt_mesh_index']:v['pact_node_id'] for v in provenance['part_correspondence']}
    pred={v['node']['id']:v for v in asset['parts']};body=pred[mapping[0]]['mesh'];moving=pred[mapping[1]]['mesh']
    gb=gt[0][0];D=float(np.linalg.norm(gb.extents));s=D/float(np.linalg.norm(body.extents))
    R=np.eye(3);t=gb.bounds.mean(0)-s*body.bounds.mean(0)
    gpoints=surface(gb,18000,300);ppoints=s*surface(body,18000,301)+t
    mesh_quality={'prediction':{str(v['node']['id']):quality(v['mesh']) for v in asset['parts']},'GT':{'fixed':quality(gb),'moving':quality(gt[0][1])}}
    raw=np.load(asset['run']/'joint_trace/raw_head_00.npy');raw_R=np.array([[1,0,0],[0,0,1],[0,-1,0]],float)
    joint=pred[mapping[1]]['node']['joint'];target={'gt_mesh_index':1,'gt_mobility_id':'joint_0','predicted_node_id':mapping[1],
        'GT_type':'prismatic','GT_direction_world':gd.tolist(),'GT_origin_world':[0,0,0],
        'GT_travel_from_actual_mesh_states':float(np.linalg.norm(delta)),
        'variants':{'original_exported':{'predicted_type':joint['type'],'type_correct':joint['type']=='prismatic',
        'axis_direction_error_degrees':angle(joint['axis']['direction'],gd) if joint['type']=='prismatic' else None},
        'raw_head':{'axis_direction_error_degrees':angle(raw_R@raw[mapping[1],6:9],gd)}}}
    out=asset['run']/'joint_evaluation';out.mkdir(exist_ok=True)
    summary={'scene_id':name,'R_export_to_gtworld':R.tolist(),'R_raw_to_gtworld':raw_R.tolist(),'scale':s,'translation':t.tolist(),
        'GT_static_diagonal':D,'direction_alignment':'PAct export and original PartNet GLB both Y-up, x lateral and z front; identity fixed frame. No GT joint fit.',
        'position_alignment':'One static-part AABB diagonal scale and center translation; evaluation only.',
        'static_geometry_distance':distances(ppoints,gpoints,D),'joints':[target],
        'GT_label_axis_error_to_actual_mesh_motion_degrees':angle(labels[0]['axis']['direction'],gd),
        'GT_files_sha256':{str(q):sha256(q) for q in [source/'joint_info.json',*sorted((source/'gt_mesh').glob('*.glb'))]},
        'mesh_quality':mesh_quality,'pivot_error':None,'pivot_reason':'Prismatic origin does not affect motion and is unidentifiable.'}
    (out/'joint_errors.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    # Local static-only ICP: same fixed AABB scale as primary, no axis or moving
    # geometry in the fit. Reused globally, never re-fit for a refined candidate.
    x=s*surface(body,12000,0)+t;y=surface(gb,12000,0);tree=cKDTree(y);rr=np.eye(3);tt=np.zeros(3);history=[]
    for iteration in range(50):
        d,idx=tree.query(x);keep=d<=np.quantile(d,.8);xx=x[keep];yy=y[idx[keep]];cx,cy=xx.mean(0),yy.mean(0)
        U,_,Vt=np.linalg.svd((xx-cx).T@(yy-cy));dr=Vt.T@U.T
        if np.linalg.det(dr)<0: Vt[-1]*=-1;dr=Vt.T@U.T
        dt=cy-dr@cx;x=x@dr.T+dt;rr=dr@rr;tt=dr@tt+dt
        history.append(float(np.mean(d[keep]**2)))
        if iteration and abs(history[-1]-history[-2])<1e-10: break
    before=s*surface(body,12000,1)+t;gy=surface(gb,12000,1)
    reg={'rotation':rr.tolist(),'translation':tt.tolist(),'iterations':len(history),
        'protocol':'Evaluation-only 80% trimmed static point-to-point ICP from fixed identity, 12000 points, max50 iterations, no GT axis/moving part fitting.',
        'heldout_before':distances(before,gy,D),'heldout_after':distances(before@rr.T+tt,gy,D),
        'rotation_delta_degrees':float(np.degrees(np.arccos(np.clip((np.trace(rr)-1)/2,-1,1)))),
        'caveat':'Imperfect or symmetric geometry can bias registration. Primary fixed-frame and sensitivity errors are both retained.'}
    (out/'geometry_registration_sensitivity.json').write_text(json.dumps(reg,indent=2))
    # Rest-state comparison against both observed GT endpoints is diagnostic;
    # closed-to-open trajectory scores below state the rest convention explicitly.
    mpoints=s*surface(moving,18000,302)+t
    geom={'fixed':distances(ppoints,gpoints,D),'moving_vs_GT_closed':distances(mpoints,surface(gt[0][1],18000,303),D),
          'moving_vs_GT_open_input':distances(mpoints,surface(gt[5][1],18000,303),D),
          'whole_vs_GT_closed':distances(s*surface(trimesh.util.concatenate([body,moving]),18000,304)+t,surface(trimesh.util.concatenate(gt[0]),18000,305),D)}
    methods={}
    for method in ['original',*config['variants']]:
        candidate=None if method=='original' else root/f'{name}_{method}/candidate.json'
        assert candidate is None or (candidate.parent/'prediction_seal.json').exists()
        nodes=candidate_nodes(asset,candidate);n=next(n for n in nodes if n['id']==mapping[1]);j=n['joint']
        v=surface(moving,4000,41).mean(0)
        w0=transforms(nodes,0)[mapping[1]];start=moved(v[None],w0)[0]
        rows=[]
        for progress in np.linspace(0,1,21):
            endpoint=moved(v[None],transforms(nodes,float(progress))[mapping[1]])[0]
            displacement=s*(endpoint-start)
            rows.append({'t':float(progress),'predicted_displacement_GT_frame':displacement.tolist(),
                         'GT_closed_to_open_displacement':(progress*delta).tolist(),
                         'displacement_error_percent_static_diagonal':float(100*np.linalg.norm(displacement-progress*delta)/D)})
        pred_end=np.array(rows[-1]['predicted_displacement_GT_frame'])
        methods[method]={'candidate_sha256':sha256(candidate or asset['object_path']),'type_correct':j['type']=='prismatic',
            'axis_direction_error_degrees':angle(j['axis']['direction'],gd) if j['type']=='prismatic' else None,
            'same_static_ICP_axis_error_degrees':angle(rr@np.array(j['axis']['direction']),gd) if j['type']=='prismatic' else None,
            'predicted_range_model_units':j['range'],'predicted_travel_GT_units':float(np.linalg.norm(pred_end)),
            'GT_travel_units':float(np.linalg.norm(delta)),'travel_absolute_error_GT_units':float(abs(np.linalg.norm(pred_end)-np.linalg.norm(delta))),
            'endpoint_displacement_error_percent_static_diagonal':rows[-1]['displacement_error_percent_static_diagonal'],
            'trajectory_displacement_RMSE_percent_static_diagonal':float(np.sqrt(np.mean([r['displacement_error_percent_static_diagonal']**2 for r in rows]))),
            'motion_error_convention':'GT state00 closed to state05 open, uniform progress; evaluates relative displacement, not rest-position error. PAct image05 is open but canonical mesh is generated at closed rest.',
            'states':rows,'pivot_error':None}
    result={'case':name,'protocol':provenance['protocol'],'baseline_joint_evaluation':str(out/'joint_errors.json'),'alignment':summary,
            'static_ICP_sensitivity':reg,'rest_geometry':geom,'methods':methods,
            'solid_collision':{'status':'unavailable','volume':None,'depth':None,'reason':'GT surfaces open, prediction self-intersection not certified; no repaired/convex meshes substituted.'}}
    (root/'GT_geometry_and_motion.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({'geometry':geom,'methods':{k:{kk:vv for kk,vv in v.items() if kk!='states'} for k,v in methods.items()}},indent=2),flush=True)

if __name__=='__main__': main()
