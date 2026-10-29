"""Post-seal reconstruction and articulation metrics; no fitting/selection here."""
import argparse
import copy
import json
import os
import pathlib
import subprocess
import sys

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
import torch
import trimesh

from collision_refinement_common import sha256,surface
from multicase_closed_common import Model,REPO,R_WORLD,image_metrics,tensor,write_json


def unit(x):
    x=np.asarray(x,float);return x/np.linalg.norm(x) if np.linalg.norm(x)>1e-12 else None


def direction_error(a,b):
    a,b=unit(a),unit(b)
    return None if a is None or b is None else float(np.degrees(np.arccos(np.clip(abs(a@b),0,1))))


def avg(values):
    values=[float(x) for x in values if x is not None]
    return float(np.mean(values)) if values else None


def distance_scores(pred,truth,D):
    a=cKDTree(truth).query(pred,workers=4)[0];b=cKDTree(pred).query(truth,workers=4)[0]
    values={'CD_percent_D':float(50*(a.mean()+b.mean())/D),'RMS_percent_D':float(100*np.sqrt((np.mean(a*a)+np.mean(b*b))/2)/D)}
    for threshold in [.01,.05]:
        precision=float((a<threshold*D).mean());recall=float((b<threshold*D).mean())
        values[f'Fscore_{int(threshold*100)}pct_D']=2*precision*recall/max(precision+recall,1e-12)
    return values


def subgroup_mesh(model,group,optimized=False):
    source=next(p['mesh'] for p in model.asset['parts'] if p['node']['id']==group['source_node'])
    prefix=group['membership_prefix'];vi=model.membership[prefix+'_vertices'];fi=model.membership[prefix+'_faces']
    if optimized:
        part=model.parts[model.indices[group['id']]]
        with torch.no_grad():vertices=model.deform(model.indices[group['id']],tensor(part['vertex_local'])).cpu().numpy()
    else:vertices=np.asarray(source.vertices)[vi]
    lut=np.full(len(source.vertices),-1,np.int64);lut[vi]=np.arange(len(vi))
    return trimesh.Trimesh(vertices,lut[source.faces[fi]],process=False)


def gt_and_alignment(model):
    cfg=model.config;scene=pathlib.Path(cfg['source_scene']);run=pathlib.Path(model.manifest['source_run'])
    old=json.loads((run/'joint_evaluation/joint_errors.json').read_text())
    if cfg['case']=='20411':
        from evaluate_freeart_partnet_case import gt_parts
        states,_=gt_parts(scene);gt_body=states[5][0];gt_moving=[states[5][1]];R=np.eye(3)
    else:
        gt_body=trimesh.load(scene/'gt/part_0.ply',process=False,force='mesh')
        gt_moving=[trimesh.load(scene/f'gt/part_{r["gt_mesh_index"]}.ply',process=False,force='mesh') for r in old['joints']]
        R=np.asarray(R_WORLD,float)
    body_group=next(g for g in model.groups if g['id']==10)
    body=next(p['mesh'] for p in model.asset['parts'] if p['node']['id']==body_group['source_node'])
    pred=np.asarray(body.vertices)@R.T;truth=np.asarray(gt_body.vertices)
    D=float(np.linalg.norm(np.ptp(truth,axis=0)));scale=D/np.linalg.norm(np.ptp(pred,axis=0))
    shift=(truth.min(0)+truth.max(0))/2-scale*(pred.min(0)+pred.max(0))/2
    alignment={'R_export_to_GT':R.tolist(),'scale':float(scale),'translation':shift.tolist(),'GT_static_diagonal':D,
               'source_static_node':body_group['source_node'],
               'policy':'One similarity from the ACTUAL generated fixed-body geometry and GT static bbox. Same original transform for all three stages; no rotation fitting, ICP, axis fitting or candidate refit.'}
    return old['joints'],gt_body,gt_moving,R,scale,shift,D,alignment


def joint_scores(model):
    targets,gt_body,gt_moving,R,s,shift,D,alignment=gt_and_alignment(model)
    active=[g for g in model.groups if g['joint']['type']!='fixed']
    base_meshes={g['id']:subgroup_mesh(model,g) for g in model.groups}
    opt_meshes={g['id']:subgroup_mesh(model,g,True) for g in model.groups}
    def descendant_mesh(g,meshes):
        ids={g['id']}
        while True:
            extended=ids|{n['id'] for n in model.groups if n['parent'] in ids}
            if extended==ids:break
            ids=extended
        return trimesh.util.concatenate([meshes[i] for i in sorted(ids)])
    pred_surfaces=[];gt_surfaces=[]
    for i,g in enumerate(active):pred_surfaces.append(s*(surface(descendant_mesh(g,base_meshes),2048,140+i)@R.T)+shift)
    for i,m in enumerate(gt_moving):gt_surfaces.append(surface(m,2048,240+i))
    cost=np.empty((len(active),len(targets)))
    for i,p in enumerate(pred_surfaces):
        for j,t in enumerate(gt_surfaces):cost[i,j]=(cKDTree(t).query(p)[0].mean()+cKDTree(p).query(t)[0].mean())/(2*D)
    ii,jj=linear_sum_assignment(cost);mapping={int(j):int(i) for i,j in zip(ii,jj)}
    raw_byid={n['id']:n for n in model.asset['nodes']}
    raw_owner={j:active[i]['source_node'] for j,i in mapping.items()}
    # A raw shared joint cannot earn credit for two independent GT instances.
    keep={}
    for j,i in mapping.items():
        source=raw_owner[j]
        if source not in keep or cost[i,j]<cost[mapping[keep[source]],keep[source]]:keep[source]=j
    methods={};allgroups={g['id']:g for g in model.groups}
    for method in ['original','prior','optimized']:
        rows=[];used=set()
        for j,tgt in enumerate(targets):
            row={'GT_part':tgt['gt_mesh_index'],'GT_type':tgt['GT_type'],'matched':False,'type_correct':False,'axis_correct':False,
                 'angle_degrees':None,'pivot_percent_D':None,'predicted_owner':None}
            if j not in mapping:
                row['reason']='No predicted independent group';rows.append(row);continue
            i=mapping[j];g=active[i];row['prior_group']=g['id'];row['matching_CD_percent_D']=float(100*cost[i,j])
            if method=='original':
                node=raw_byid[g['source_node']];owner=node['id']
                if keep[owner]!=j:
                    row['reason']='Shared original joint already assigned to another GT instance';rows.append(row);continue
            else:node=g;owner=g['id']
            joint=node['joint'];row['predicted_owner']=owner;row['predicted_type']=joint['type']
            if joint['type']=='fixed':
                row['reason']='Predicted fixed';rows.append(row);continue
            used.add(owner);row['matched']=True;row['type_correct']=joint['type']==tgt['GT_type']
            d=R@np.array(joint['axis']['direction']);o=s*(R@np.array(joint['axis']['origin']))+shift
            gd=np.asarray(tgt['GT_direction_world']);go=np.asarray(tgt['GT_origin_world'])
            row.update(predicted_axis_direction=d.tolist(),predicted_axis_origin=o.tolist(),GT_axis_direction=gd.tolist(),GT_axis_origin=go.tolist())
            if row['type_correct']:
                row['angle_degrees']=direction_error(d,gd)
                if tgt['GT_type']=='revolute':row['pivot_percent_D']=float(100*np.linalg.norm(np.cross(o-go,unit(gd)))/D)
                row['axis_correct']=row['angle_degrees'] is not None and row['angle_degrees']<=5 and (row['pivot_percent_D'] is None or row['pivot_percent_D']<=5)
            meshes=opt_meshes if method=='optimized' else base_meshes
            points=s*(surface(descendant_mesh(g,meshes),2048,740+i)@R.T)+shift
            cover=float((cKDTree(points).query(gt_surfaces[j])[0]<.05*D).mean())
            row['GT_surface_coverage_at_5pct_D']=cover
            row['geometry_supported_axis_correct']=row['axis_correct'] and cover>=.3
            rows.append(row)
        predicted=[n for n in (model.asset['nodes'] if method=='original' else model.groups) if n['joint']['type']!='fixed']
        n=len(targets);typed=sum(r['type_correct'] for r in rows);correct=sum(r['axis_correct'] for r in rows)
        methods[method]={'GT_joint_count':n,'predicted_active_joint_count':len(predicted),
                         'independent_joint_coverage':sum(r['matched'] for r in rows)/n,
                         'type_correct_GT_coverage':typed/n,'type_correct_precision':typed/max(len(predicted),1),
                         'axis_correct_rate':correct/n,'axis_correct_count':correct,
                         'geometry_supported_axis_correct_rate':sum(r.get('geometry_supported_axis_correct',False) for r in rows)/n,
                         'conditional_mean_angle_degrees':avg(r['angle_degrees'] for r in rows),
                         'conditional_mean_pivot_percent_D':avg(r['pivot_percent_D'] for r in rows),
                         'extra_active_joint_count':len(predicted)-len(used),'rows':rows}
    # Evaluate original source rigid bodies as their OWN prediction set. In
    # particular, 31249 has a wrongly typed independently moving duplicate door;
    # inheriting only the corrected primary leaf's owner would discard that
    # original prediction and bias the baseline. All costs remain type/axis-blind.
    inherited_original=copy.deepcopy(methods['original'])
    raw_active=[n for n in model.asset['nodes'] if n['joint']['type']!='fixed']
    source_meshes={p['node']['id']:p['mesh'] for p in model.asset['parts']}
    raw_points=[s*(surface(source_meshes[n['id']],2048,440+i)@R.T)+shift for i,n in enumerate(raw_active)]
    raw_cost=np.empty((len(raw_active),len(targets)))
    for i,p in enumerate(raw_points):
        for j,t in enumerate(gt_surfaces):raw_cost[i,j]=(cKDTree(t).query(p)[0].mean()+cKDTree(p).query(t)[0].mean())/(2*D)
    ri,rj=linear_sum_assignment(raw_cost);raw_mapping={int(j):int(i) for i,j in zip(ri,rj)}
    rows=[]
    for j,tgt in enumerate(targets):
        row={'GT_part':tgt['gt_mesh_index'],'GT_type':tgt['GT_type'],'matched':False,'type_correct':False,
             'axis_correct':False,'angle_degrees':None,'pivot_percent_D':None,'predicted_owner':None}
        if j not in raw_mapping:
            row['reason']='No independent original active joint';rows.append(row);continue
        i=raw_mapping[j];node=raw_active[i];joint=node['joint']
        row.update(matched=True,predicted_owner=node['id'],predicted_type=joint['type'],
                   matching_CD_percent_D=float(100*raw_cost[i,j]),type_correct=joint['type']==tgt['GT_type'])
        d=R@np.asarray(joint['axis']['direction']);o=s*(R@np.asarray(joint['axis']['origin']))+shift
        gd=np.asarray(tgt['GT_direction_world']);go=np.asarray(tgt['GT_origin_world'])
        row.update(predicted_axis_direction=d.tolist(),predicted_axis_origin=o.tolist(),GT_axis_direction=gd.tolist(),GT_axis_origin=go.tolist())
        if row['type_correct']:
            row['angle_degrees']=direction_error(d,gd)
            if tgt['GT_type']=='revolute':row['pivot_percent_D']=float(100*np.linalg.norm(np.cross(o-go,unit(gd)))/D)
            row['axis_correct']=row['angle_degrees'] is not None and row['angle_degrees']<=5 and (row['pivot_percent_D'] is None or row['pivot_percent_D']<=5)
        coverage=float((cKDTree(raw_points[i]).query(gt_surfaces[j])[0]<.05*D).mean())
        row['GT_surface_coverage_at_5pct_D']=coverage;row['geometry_supported_axis_correct']=row['axis_correct'] and coverage>=.3
        rows.append(row)
    n=len(targets);typed=sum(r['type_correct'] for r in rows);correct=sum(r['axis_correct'] for r in rows)
    methods['original']={'GT_joint_count':n,'predicted_active_joint_count':len(raw_active),
                         'independent_joint_coverage':sum(r['matched'] for r in rows)/n,
                         'type_correct_GT_coverage':typed/n,'type_correct_precision':typed/max(len(raw_active),1),
                         'axis_correct_rate':correct/n,'axis_correct_count':correct,
                         'geometry_supported_axis_correct_rate':sum(r.get('geometry_supported_axis_correct',False) for r in rows)/n,
                         'conditional_mean_angle_degrees':avg(r['angle_degrees'] for r in rows),
                         'conditional_mean_pivot_percent_D':avg(r['pivot_percent_D'] for r in rows),
                         'extra_active_joint_count':len(raw_active)-len(raw_mapping),'rows':rows}
    # Original and prior have exactly the same rest geometry. Reuse their metric.
    truth=surface(trimesh.util.concatenate([gt_body,*gt_moving]),16000,177)
    geometry={}
    for method,meshes in [('prior',base_meshes),('optimized',opt_meshes)]:
        full=trimesh.util.concatenate([meshes[g['id']] for g in model.groups]);p=s*(surface(full,16000,178)@R.T)+shift
        geometry[method]=distance_scores(p,truth,D)
    geometry['original']=copy.deepcopy(geometry['prior'])
    return {'alignment':alignment,'correspondence':{'cost_percent_D':(100*cost).tolist(),'prior_active_groups':[g['id'] for g in active],
             'GT_parts':[t['gt_mesh_index'] for t in targets],'assignment_GT_index_to_prior_index':mapping,
             'original_active_sources':[n['id'] for n in raw_active],'original_cost_percent_D':(100*raw_cost).tolist(),
             'original_assignment_GT_index_to_source_index':raw_mapping,
             'definition':'Each original source rigid body is matched once using its full source geometry. Prior/optimized share a separate assignment on PRE-optimization corrected physical groups. Both are type/axis-blind fixed-frame geometric assignments; no axis-error permutations.'},
            'joint_methods':methods,'surface_reconstruction':geometry,
            'baseline_inherited_ownership_sensitivity':inherited_original,
            'geometry_pose':'Generated conditioning reference pose, not a separate per-part rest-pose registration. Includes pose and shape errors.'}


def evaluate_case(root):
    cfg=json.loads((root/'config.json').read_text());manifest=json.loads((root/'motion_groups.json').read_text());seal=json.loads((root/'fit_results.json').read_text())
    assert sha256(root/'checkpoint.pt')==seal['checkpoint_sha256'] and sha256(root/'motion_groups.json')==seal['manifest_sha256']
    torch.set_num_threads(4);model=Model(cfg,manifest,root)
    model.load_state_dict(torch.load(root/'checkpoint.pt',map_location='cuda',weights_only=False)['model']);model.requires_grad_(False)
    geom={m:model.geometry(original=m!='optimized') for m in ['original','prior','optimized']}
    novel={};panels=[]
    if cfg['case']!='20411':
        scene=pathlib.Path(cfg['source_scene']);camera=json.loads((scene/'transforms.json').read_text());frames=[25,50,75,100,125,149]
        for index in frames:
            frame=camera['frames'][index];assert frame['state']==0 and frame['time']==0.
            image=Image.open(scene/frame['file_path']).convert('RGBA').resize((256,256),Image.Resampling.LANCZOS)
            rgba=np.asarray(image).astype(np.float32)/255.;target=tensor(rgba[:,:,:3]*rgba[:,:,3:4]+1-rgba[:,:,3:4]).permute(2,0,1);mask=tensor(rgba[:,:,3])
            c2w=np.asarray(frame['transform_matrix']);c2w[:3,1:3]*=-1;ext=tensor(np.linalg.inv(c2w))
            intr=tensor([[camera['focal_x']/camera['w'],0,camera['cx']/camera['w']],[0,camera['focal_y']/camera['h'],camera['cy']/camera['h']],[0,0,1]])
            images=[target]
            with torch.no_grad():
                for m in geom:
                    rgb=model.render(geom[m],method='original' if m=='original' else 'prior',extrinsics=ext,intrinsics=intr,reference=True)
                    alpha=model.render(geom[m],'silhouette',method='original' if m=='original' else 'prior',extrinsics=ext,intrinsics=intr,reference=True)
                    novel.setdefault(m,[]).append({'frame':index,**image_metrics(rgb,alpha,target,mask)});images.append(rgb)
            if index in [25,75,125]:panels.append(torch.cat(images,2).clamp(0,1).permute(1,2,0).cpu().numpy())
        Image.fromarray((255*np.concatenate(panels,0)).astype('uint8')).save(root/'unfitted_open_view_comparison.png')
    novel_mean={m:{k:avg(v[k] for v in rows) for k in ['IoU','PSNR','foreground_PSNR','foreground_L1']} for m,rows in novel.items()}
    axes=joint_scores(model)
    results={'case':cfg['case'],'fit_image_metrics':seal['image_metrics'],'unfitted_open_views':novel,
             'unfitted_open_mean':novel_mean,'unfitted_view_protocol':'Six predeclared cameras [25,50,75,100,125,149] at input open state0, not used for geometry fit. Reference pose held at original generated q=0. No validation pose/camera refit. Frame0 was original generation input, and is excluded.',
             'unfitted_unavailable_reason':'Only fixed-camera six-state input, no novel cameras' if cfg['case']=='20411' else None,
             **axes,'source_files_unchanged':all(sha256(x['path'])==x['sha256'] for x in manifest['geometry_files']),
             'fitting_seal_sha256':sha256(root/'fit_results.json'),'code_sha256':sha256(__file__)}
    write_json(root/'evaluation.json',results)
    print('EVALUATED',cfg['case'],json.dumps({'fit':results['fit_image_metrics'],'unfitted':novel_mean,'axes':{m:{k:v for k,v in r.items() if k!='rows'} for m,r in axes['joint_methods'].items()}}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--case');a=p.parse_args();root=pathlib.Path(a.root).resolve()
    if a.case:evaluate_case(root/a.case);return
    protocol=json.loads((root/'protocol.json').read_text());status=json.loads((root/'controller_status.json').read_text())
    assert status['state']=='fit_complete','GT evaluation must follow all candidate fitting/sealing'
    states={}
    for case in protocol['cases']:
        out=root/case
        if not (out/'fit_results.json').exists():states[case]='fit_failed';continue
        if (out/'evaluation.json').exists():states[case]='complete';continue
        env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4')
        with (out/'evaluation.log').open('w') as f:
            proc=subprocess.run([sys.executable,__file__,'--root',str(root),'--case',case],stdout=f,stderr=subprocess.STDOUT,env=env,cwd=REPO)
        states[case]='complete' if proc.returncode==0 else 'failed';write_json(root/'evaluation_status.json',states)
        print('CASE EVALUATE',case,states[case],flush=True)


if __name__=='__main__':main()
