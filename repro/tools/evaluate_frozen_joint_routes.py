"""Post-seal axis/endpoints and held-out-source motion analysis; no fitting."""
import argparse
import copy
import hashlib
import json
import pathlib
import os
import time

import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image,ImageDraw

from collision_refinement_common import transforms,sha256
from frozen_joint_common import (FrozenScene,atomic_json,continuous_from_raw,
                                 paths,state_from_nodes,tensor)


def angle(a,b):
    a,b=np.array(a),np.array(b)
    return float(np.degrees(np.arccos(np.clip(abs(a@b)/(np.linalg.norm(a)*np.linalg.norm(b)),0,1))))


def axis_metrics(scene,obj):
    run=scene.asset['run'];gt=json.loads((run/'joint_evaluation/joint_errors.json').read_text());byid={n['id']:n for n in obj['diffuse_tree']};initial={n['id']:n for n in scene.asset['nodes']};R=np.array(gt['R_export_to_gtworld']);s=gt['scale'];shift=np.array(gt['translation']);D=gt['GT_static_diagonal'];rows=[]
    for old in gt['joints']:
        n=byid[old['predicted_node_id']];j=n['joint'];j0=initial[n['id']]['joint'];d=R@np.array(j['axis']['direction']);o=s*(R@np.array(j['axis']['origin']))+shift;gd=np.array(old['GT_direction_world']);go=np.array(old['GT_origin_world']);correct=j['type']==old['GT_type'];active=n['id'] in scene.params.indices
        limits=np.array(j['range'],float);lim0=np.array(j0['range'],float)
        sign=-1 if active and np.dot(j['axis']['direction'],j0['axis']['direction'])<0 else 1
        comparable=sign*limits
        rows.append({'GT_part':old['gt_mesh_index'],'node_id':n['id'],'type':j['type'],'GT_type':old['GT_type'],'type_correct':correct,
                     'axis_error_degrees':angle(d,gd) if active and correct else None,
                     'pivot_error_percent_GT_D':float(100*np.linalg.norm(np.cross(o-go,gd))/D) if correct and j['type']=='revolute' else None,
                     'predicted_axis_origin_world':o.tolist(),'predicted_axis_direction_world':d.tolist(),'GT_axis_origin_world':go.tolist(),'GT_axis_direction_world':gd.tolist(),
                     'origin_change_percent_GT_D':float(100*s*np.linalg.norm(np.array(j['axis']['origin'])-np.array(j0['axis']['origin']))/D) if j['type']=='revolute' else None,
                     'axis_rotation_from_original_degrees':angle(j['axis']['direction'],j0['axis']['direction']) if active else None,
                     'q_start':float(comparable[0]) if active else None,'q_end':float(comparable[1]) if active else None,
                     'native_q_start':float(lim0[0]) if active else None,'native_q_end':float(lim0[1]) if active else None,
                     'absolute_range':float(abs(limits[1]-limits[0])) if active else None,
                     'range_ratio_vs_original':float(abs(limits[1]-limits[0])/abs(lim0[1]-lim0[0])) if active and abs(lim0[1]-lim0[0])>1e-8 else None,
                     'q_unit':'degrees' if j['type'] in ['revolute','continuous'] else 'export-model length units',
                     'GT_endpoint_error':None,'GT_endpoint_status':'unavailable: generated rest-state gauge is not aligned to GT limit reference; report predictions and signed GT span only',
                     'GT_limits_original_annotation':old['GT_limit']})
    return rows


def heldout_sources(scene,count=2048):
    class ParallelQueries:
        def __init__(self,tree):self.tree=tree
        def query(self,points,workers=4):return self.tree.query(points,workers=min(16,len(os.sched_getaffinity(0))))
    scene.objective.trees=[ParallelQueries(t) for t in scene.objective.trees]
    for p,data in zip(scene.asset['parts'],scene.objective.data):
        available=np.setdiff1d(p['retained_ids'],p['sample_ids']);rng=np.random.default_rng(2026+int(p['node']['id']));chosen=np.sort(rng.choice(available,min(count,len(available)),replace=False));assert not np.intersect1d(chosen,p['sample_ids']).size
        ids=torch.as_tensor(np.searchsorted(p['retained_ids'],chosen),device='cuda')
        for small,full in [('xyz','full_xyz'),('alpha','full_alpha'),('sigma','full_sigma')]:data[small]=data[full][ids]


def motion(scene,limits):
    result=[]
    for name,qs in paths(limits,41):
        # Bound transient arrays without changing the independent evaluation grid.
        values={k:[] for k in ['proxy','minimum_sampled_center_distance','minimum_sampled_one_sigma_gap']}
        for chunk in qs.split(41):
            block=scene.proxy_states(chunk)
            for k in values:values[k].extend(block[k].cpu().tolist())
        result.append({'path':name,**values,'mean_proxy':float(np.mean(values['proxy'])),'max_proxy':float(max(values['proxy']))})
    return {'paths':result,'mean_proxy':float(np.mean([r['mean_proxy'] for r in result])),'worst_proxy':float(max(r['max_proxy'] for r in result)),
            'source_sampling':'Up to2048 source centers/part, seed2026; optimizer sources excluded; all retained targets',
            'solid_overlap_volume':None,'penetration_depth':None,
            'solid_status':'unavailable: mesh closed-solid validity not established; Gaussian proxy/center gaps are not mesh penetration'}


def plot_axes(rows_by_method,directory,title):
    keys=[r['node_id'] for r in next(iter(rows_by_method.values())) if r['type']!='fixed'];fig,axes=plt.subplots(len(keys),2,figsize=(12,4*len(keys)),squeeze=False)
    colors={'R0':'#777777','A':'#2166ac','B':'#b2182b','R0_raw':'#888888','A_raw':'#2166ac','B_raw':'#b2182b','A_official':'#00a6a6','B_official':'#ef8a62'}
    for index,nid in enumerate(keys):
        first=next(r for r in next(iter(rows_by_method.values())) if r['node_id']==nid);gd=np.array(first['GT_axis_direction_world']);gd/=np.linalg.norm(gd);ref=np.eye(3)[np.argmin(abs(gd))];u=np.cross(gd,ref);u/=np.linalg.norm(u);v=np.cross(gd,u);go=np.array(first['GT_axis_origin_world'])
        pos,ori=axes[index];ori.scatter([0],[0],marker='*',s=130,c='black',label='GT')
        if not first['type_correct']:
            for ax in [pos,ori]:
                ax.clear();ax.text(.1,.5,f"node {nid}: predicted {first['type']} / GT {first['GT_type']}\nValid axis accuracy unavailable",transform=ax.transAxes);ax.set_axis_off()
            continue
        if first['type']=='revolute':pos.scatter([0],[0],marker='*',s=130,c='black',label='GT axis')
        for method,rows in rows_by_method.items():
            r=next(r for r in rows if r['node_id']==nid);d=np.array(r['predicted_axis_direction_world']);d/=np.linalg.norm(d)
            if d@gd<0:d=-d
            tilts=np.degrees(np.arctan2([d@u,d@v],d@gd));ori.scatter(*tilts,c=colors[method],label=method)
            if r['type']=='revolute':
                delta=np.array(r['predicted_axis_origin_world'])-go;pos.scatter(delta@u,delta@v,c=colors[method],label=method)
        pos.set_title(f'node {nid}: pivot in GT-normal plane' if first['type']=='revolute' else f'node {nid}: prismatic pivot N/A')
        pos.set_xlabel('u (GT data units)');pos.set_ylabel('v (GT data units)');ori.set_title(f'node {nid}: direction tilt from GT');ori.set_xlabel('tilt u (degrees)');ori.set_ylabel('tilt v (degrees)')
        if first['type']!='revolute':
            pos.text(.18,.5,'Prismatic axis origin is not identifiable.\nPosition error: unavailable.',transform=pos.transAxes);pos.set_axis_off()
        for ax in ([pos,ori] if first['type']=='revolute' else [ori]):ax.grid(alpha=.25);ax.legend(fontsize=8);ax.set_aspect('equal',adjustable='datalim')
    fig.suptitle(title+' | frozen local geometry');fig.tight_layout();fig.savefig(directory/'axes_vs_GT.png',dpi=170);fig.savefig(directory/'axes_vs_GT.pdf');plt.close(fig)


def render_movie(scene,objects,directory,title):
    from modules.pact.representations.mesh import MeshExtractResult
    from modules.pact.renderers.mesh_renderer import MeshRenderer
    import utils3d
    methods=list(objects);palette=[[.20,.52,.77],[.94,.53,.18],[.25,.70,.42],[.65,.26,.67],[.72,.65,.20]];renderer=MeshRenderer({'resolution':256,'near':.05,'far':20.})
    # Common camera bounds from all method trajectories; never from GT.
    samples=[]
    for obj in objects.values():
        for t in np.linspace(0,1,5):
            world=transforms(obj['diffuse_tree'],float(t))
            for part in scene.asset['parts']:
                x=np.array(part['mesh'].vertices)[::max(1,len(part['mesh'].vertices)//1000)];tr=world[part['node']['id']];samples.append(x@tr[:3,:3].T+tr[:3,3])
    samples=np.concatenate(samples);low,high=samples.min(0),samples.max(0);center=(low+high)/2;radius=float(np.linalg.norm(high-low));target=tensor(center);origin=target+tensor([1.3,.65,1.5])*radius
    ext=utils3d.torch.extrinsics_look_at(origin,target,tensor([0,1,0]));fov=tensor(np.deg2rad(45.));intr=utils3d.torch.intrinsics_from_fov_xy(fov,fov);movies=[]
    for state,t in enumerate(np.linspace(0,1,21)):
        panels=[]
        for method,obj in objects.items():
            world=transforms(obj['diffuse_tree'],float(t));vertices=[];faces=[];attrs=[];offset=0
            for i,part in enumerate(scene.asset['parts']):
                mesh=part['mesh'];tr=world[part['node']['id']];x=np.asarray(mesh.vertices)@tr[:3,:3].T+tr[:3,3];vertices.append(x);faces.append(mesh.faces+offset);offset+=len(x);attrs.append(np.tile(palette[i%len(palette)]+[0.,0.,0.],(len(x),1)))
            representation=MeshExtractResult(tensor(np.concatenate(vertices)),torch.tensor(np.concatenate(faces),device='cuda',dtype=torch.int64),tensor(np.concatenate(attrs)))
            image=renderer.render(representation,ext,intr,return_types=['color','mask','normal']);shade=.6+.4*torch.abs(2*image['normal'][2:3]-1);rgb=(image['color']*shade+1-image['mask'][None]).clamp(0,1);pic=Image.fromarray((255*rgb.permute(1,2,0).cpu().numpy()).astype(np.uint8));ImageDraw.Draw(pic).text((5,5),f'{method} t={t:.2f}',fill='black');panels.append(np.array(pic))
        frame=np.concatenate(panels,1);movies.append(frame)
        if state in [0,10,20]:Image.fromarray(frame).save(directory/f'rigid_motion_state{state:02d}.png')
    imageio.mimsave(directory/'rigid_motion.mp4',movies,fps=10)
    atomic_json(directory/'movie_protocol.json',{'states':21,'geometry':'Full unchanged original triangles; rigid FK only','repair':False,'camera':'Common across methods, bounds from prediction trajectories, no GT','title':title})


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=pathlib.Path(a.root);cfg=json.loads((root/'config.json').read_text());torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;started=time.monotonic();summary={}
    assert json.loads((root/'controller_status.json').read_text())['state']=='optimization_complete'
    atomic_json(root/'evaluation_status.json',{'state':'running'})
    try:
        for route in ['direct','head']:
            summary[route]={}
            for case in cfg['test_cases']:
                scene=FrozenScene(cfg['runs'][case],cfg['sources_per_part']);heldout_sources(scene);objects={'R0':copy.deepcopy(scene.asset['object'])};sources={}
                if route=='direct':
                    for arm in ['A','B']:
                        folder=root/f'direct_{case}_{arm}';seal=json.loads((folder/'seal.json').read_text());assert seal['frozen_assets_unchanged'] and seal['checkpoint_sha256']==sha256(folder/'checkpoint.pt');path=folder/f'{case}_candidate.json';assert sha256(path)==seal['candidates'][case+'_continuous'];objects[arm]=json.loads(path.read_text());sources[arm]=str(path)
                else:
                    cache=torch.load(root/'features'/case/'cache.pt',weights_only=False);scene.params.override=continuous_from_raw(cache['raw'].cuda(),scene);objects['R0_raw']=scene.params.export(scene.asset)
                    for arm in ['A','B']:
                        folder=root/f'head_shared_{arm}';seal=json.loads((folder/'seal.json').read_text());assert seal['frozen_assets_unchanged'] and seal['checkpoint_sha256']==sha256(folder/'checkpoint.pt')
                        for key,suffix in [('raw','candidate'),('official','official')]:
                            path=folder/f'{case}_{suffix}.json';assert sha256(path)==seal['candidates'][case+('_continuous' if key=='raw' else '_official')];objects[arm+'_'+key]=json.loads(path.read_text());sources[arm+'_'+key]=str(path)
                directory=root/f'eval_{route}_{case}';directory.mkdir(exist_ok=True);results={};motion_cache={}
                def motion_key(limits):
                    return hashlib.sha256(b''.join(x.detach().cpu().numpy().tobytes() for x in [*scene.params.axes(),limits])).hexdigest()
                def cached_motion(limits):
                    key=motion_key(limits)
                    if key not in motion_cache:motion_cache[key]=motion(scene,limits)
                    return copy.deepcopy(motion_cache[key])
                for method,obj in objects.items():
                    file=directory/f'{method}_metrics.json'
                    scene.params.override=state_from_nodes(scene,obj['diffuse_tree'])
                    if file.exists():
                        result=json.loads(file.read_text());results[method]=result
                        motion_cache[motion_key(scene.params.limits())]=result['own_range_motion']
                        motion_cache[motion_key(scene.params.limits0)]=result['common_original_range_motion']
                        continue
                    atomic_json(root/'evaluation_status.json',{'state':'running','route':route,'case':case,'method':method})
                    rows=axis_metrics(scene,obj)
                    with torch.no_grad():learned=cached_motion(scene.params.limits());common=cached_motion(scene.params.limits0)
                    result={'case':case,'route':route,'method':method,'joints':rows,'own_range_motion':learned,'common_original_range_motion':common,'geometry_frozen':True,'candidate':sources.get(method),'axis_alignment':'Same original-baseline GT static-AABB similarity for every method; no candidate refit','GT_test_used':'Post-seal evaluation only'}
                    atomic_json(file,result);results[method]=result
                    print('EVALUATED',route,case,method,flush=True)
                fig,axes=plt.subplots(1,2,figsize=(12,4));colors=plt.get_cmap('tab10')
                for i,(method,result) in enumerate(results.items()):
                    for ax,key in zip(axes,['own_range_motion','common_original_range_motion']):
                        curve=next(x for x in result[key]['paths'] if x['path']=='synchronized');ax.plot(np.linspace(0,1,41),curve['proxy'],label=method,c=colors(i))
                for ax,title in zip(axes,['Each candidate motion endpoints','COMMON original motion endpoints']):ax.set_title(title);ax.set_xlabel('Motion fraction');ax.set_ylabel('Gaussian overlap proxy, not solid volume');ax.grid(alpha=.25);ax.legend(fontsize=8)
                fig.suptitle(route+' '+case+' | unchanged local geometry');fig.tight_layout();fig.savefig(directory/'motion_overlap.png',dpi=170);plt.close(fig)
                plot_axes({m:r['joints'] for m,r in results.items()},directory,route+' '+case)
                movie_methods=['R0','A','B'] if route=='direct' else ['R0','A_official','B_official']
                with torch.no_grad():render_movie(scene,{m:objects[m] for m in movie_methods},directory,route+' '+case)
                summary[route][case]=results
                del scene;torch.cuda.empty_cache()
        atomic_json(root/'results.json',summary)
        atomic_json(root/'evaluation_status.json',{'state':'complete','elapsed_seconds':time.monotonic()-started})
    except BaseException as exc:atomic_json(root/'evaluation_status.json',{'state':'failed','error':f'{type(exc).__name__}: {exc}'});raise


if __name__=='__main__':main()
