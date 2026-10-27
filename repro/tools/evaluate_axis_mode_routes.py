"""Post-seal single-variable evaluation with exact-input collision-cache reuse."""
import argparse
import copy
import json
import pathlib
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from axis_mode_common import (ModeScene,atomic_json,sha256,assert_fields,state_from_nodes,
    continuous_from_raw,export_axis)
from evaluate_axis_fixed_routes import axis_metrics,signature
from evaluate_frozen_joint_routes import heldout_sources,motion,render_movie


def plot_axes(rows_by_method,directory,title):
    keys=[r['node_id'] for r in next(iter(rows_by_method.values())) if r['type']!='fixed'];fig,axes=plt.subplots(len(keys),2,figsize=(12,4*len(keys)),squeeze=False)
    colors={'R0':'#777777','H0':'#999999','A':'#2166ac','B':'#b2182b','A_raw':'#2166ac','B_raw':'#b2182b','A_official':'#00a6a6','B_official':'#ef8a62'}
    for index,nid in enumerate(keys):
        first=next(r for r in next(iter(rows_by_method.values())) if r['node_id']==nid);gd=np.array(first['GT_axis_direction_world']);gd/=np.linalg.norm(gd);ref=np.eye(3)[np.argmin(abs(gd))];u=np.cross(gd,ref);u/=np.linalg.norm(u);v=np.cross(gd,u);go=np.array(first['GT_axis_origin_world']);pos,ori=axes[index]
        if not first['type_correct']:
            for ax in [pos,ori]:ax.text(.1,.5,f"node {nid}: predicted {first['type']} / GT {first['GT_type']}\nValid axis accuracy unavailable",transform=ax.transAxes);ax.set_axis_off()
            continue
        ori.scatter([0],[0],marker='*',s=130,c='black',label='GT')
        if first['type']=='revolute':pos.scatter([0],[0],marker='*',s=130,c='black',label='GT axis')
        for method,rows in rows_by_method.items():
            r=next(r for r in rows if r['node_id']==nid);d=np.array(r['predicted_axis_direction_world']);d/=np.linalg.norm(d)
            if d@gd<0:d=-d
            tilts=np.degrees(np.arctan2([d@u,d@v],d@gd));ori.scatter(*tilts,c=colors[method],label=method)
            if r['type']=='revolute':
                delta=np.array(r['predicted_axis_origin_world'])-go;pos.scatter(delta@u,delta@v,c=colors[method],label=method)
        pos.set_title(f'node {nid}: pivot in GT-normal plane' if first['type']=='revolute' else f'node {nid}: prismatic pivot N/A');pos.set_xlabel('u (GT data units)');pos.set_ylabel('v (GT data units)');ori.set_title(f'node {nid}: direction tilt from GT');ori.set_xlabel('tilt u (degrees)');ori.set_ylabel('tilt v (degrees)')
        if first['type']!='revolute':pos.text(.18,.5,'Prismatic origin is not identifiable.\nPosition accuracy: unavailable.',transform=pos.transAxes);pos.set_axis_off()
        for ax in ([pos,ori] if first['type']=='revolute' else [ori]):ax.grid(alpha=.25);ax.legend(fontsize=8);ax.set_aspect('equal',adjustable='datalim')
    fig.suptitle(title+' | original geometry and endpoints fixed');fig.tight_layout();fig.savefig(directory/'axes_vs_GT.png',dpi=170);fig.savefig(directory/'axes_vs_GT.pdf');plt.close(fig)


def frozen_change(obj,scene):
    old={n['id']:n for n in scene.asset['nodes']};key='origin' if scene.mode=='direction' else 'direction'
    return max(float(np.max(abs(np.array(n['joint']['axis'][key])-np.array(old[n['id']]['joint']['axis'][key])))) for n in obj['diffuse_tree'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=pathlib.Path(a.root);top=json.loads((root/'config.json').read_text());parent=pathlib.Path(top['parent_root']);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;started=time.monotonic()
    assert json.loads((root/'controller_status.json').read_text())['state']=='optimization_complete'
    assert sha256(parent/'integrity.json')==top['parent_integrity_sha256'];pin=json.loads((parent/'integrity.json').read_text());assert sha256(parent/'results.json')==pin['results_sha256']
    old=json.loads((parent/'results.json').read_text());parent_cfg=json.loads((parent/'config.json').read_text());cache={};cache_sources={};reuse=[];results={}
    # All inputs, not a rounded metric or a nearby axis, must be byte-identical.
    for route,cases in old.items():
        for case,methods in cases.items():
            for method,result in methods.items():
                key=(case,result['motion_state_sha256'])
                if key not in cache:cache[key]=result['fixed_range_motion'];cache_sources[key]=f'parent/{route}/{case}/{method}'
    try:
        for mode in top['modes']:
            sub=root/mode;cfg=json.loads((sub/'config.json').read_text());assert cfg['runs']==parent_cfg['runs'] and cfg['sources_per_part']==parent_cfg['sources_per_part'];assert (sub/'frozen_assets.json').read_bytes()==(parent/'frozen_assets.json').read_bytes();results[mode]={}
            for route in ['direct','head']:
                results[mode][route]={}
                for case in cfg['test_cases']:
                    scene=ModeScene(cfg['runs'][case],cfg['sources_per_part'],mode,'head');heldout_sources(scene);objects={'R0':copy.deepcopy(scene.asset['object'])};candidates={}
                    if route=='head':
                        features=torch.load(sub/'features'/case/'cache.pt',weights_only=False);scene.params.override=continuous_from_raw(features['raw'].cuda(),scene);objects['H0']=export_axis(scene)
                    for arm in ['A','B']:
                        folder=sub/(f'direct_{case}_{arm}' if route=='direct' else f'head_shared_{arm}');seal=json.loads((folder/'seal.json').read_text());assert sha256(folder/'checkpoint.pt')==seal['checkpoint_sha256'];assert seal['frozen_counterpart_all_steps'] and seal['max_frozen_counterpart_delta']==0
                        for method,suffix in ([(arm,'candidate')] if route=='direct' else [(arm+'_raw','candidate'),(arm+'_official','official')]):
                            file=folder/f'{case}_{suffix}.json';assert sha256(file)==seal['candidates'][case+('_official' if suffix=='official' else '_continuous')];objects[method]=json.loads(file.read_text());candidates[method]=str(file)
                    directory=sub/f'eval_{route}_{case}';directory.mkdir(exist_ok=True);rows={}
                    for method,obj in objects.items():
                        atomic_json(root/'evaluation_status.json',{'state':'running','mode':mode,'route':route,'case':case,'method':method})
                        assert_fields(obj,scene);scene.params.override=state_from_nodes(scene,obj['diffuse_tree']);scene.assert_frozen();d,o=scene.params.axes();key=(case,signature((d,o,scene.params.limits0),scene.params.limits0));file=directory/f'{method}_metrics.json'
                        if file.exists():
                            result=json.loads(file.read_text());assert result['motion_state_sha256']==key[1]
                            cache[key]=result['fixed_range_motion'];cache_sources[key]=result['motion_cache_origin']
                        else:
                            if key not in cache:
                                with torch.no_grad():cache[key]=motion(scene,scene.params.limits0)
                                cache_sources[key]=f'computed/{mode}/{route}/{case}/{method}'
                            measured=copy.deepcopy(cache[key]);result={'mode':mode,'route':route,'case':case,'method':method,'joints':axis_metrics(scene,obj),'fixed_range_motion':measured,
                                'geometry_frozen':True,'endpoints_frozen':True,'frozen_counterpart':'origin' if mode=='direction' else 'direction',
                                'max_numeric_endpoint_change':0.,'max_frozen_counterpart_change':frozen_change(obj,scene),'candidate':candidates.get(method),
                                'motion_state_sha256':key[1],'motion_cache_origin':cache_sources[key],
                                'GT_test_used':'Only post-seal evaluation; no GT checkpoint/seed/weight selection',
                                'H0_note':'Selected original continuous output, other physical field fixed to original export; hybrid initialization, not the previous full-continuous R0_raw',
                                'official_note':'Official postprocessing followed by replay of requested frozen physical counterpart'}
                            assert result['max_frozen_counterpart_change']==0.;atomic_json(file,result)
                        rows[method]=result;reuse.append({'mode':mode,'route':route,'case':case,'method':method,'state_sha256':key[1],'origin':result['motion_cache_origin']});print('EVALUATED',mode,route,case,method,result['motion_cache_origin'],flush=True)
                    first=rows['R0']['fixed_range_motion']['paths'];n=len(first);fig,axes=plt.subplots((n+1)//2,2,figsize=(12,3.7*((n+1)//2)),squeeze=False)
                    for index,path in enumerate(first):
                        ax=axes.flat[index]
                        for method,result in rows.items():curve=next(p for p in result['fixed_range_motion']['paths'] if p['path']==path['path']);ax.plot(np.linspace(0,1,41),curve['proxy'],label=method)
                        ax.set(title=path['path'],xlabel='Fraction of fixed native range',ylabel='Gaussian overlap proxy (not volume)');ax.grid(alpha=.25);ax.legend(fontsize=8)
                    for ax in list(axes.flat)[n:]:ax.set_axis_off()
                    fig.suptitle(f'{mode}-only / {route} / {case}');fig.tight_layout();fig.savefig(directory/'motion_overlap.png',dpi=170);plt.close(fig)
                    plot_axes({m:r['joints'] for m,r in rows.items()},directory,f'{mode}-only / {route} / {case}')
                    selected=['R0','A','B'] if route=='direct' else ['R0','A_official','B_official']
                    with torch.no_grad():render_movie(scene,{m:objects[m] for m in selected},directory,f'{mode}-only / {route} / {case}')
                    results[mode][route][case]=rows;del scene;torch.cuda.empty_cache()
            atomic_json(sub/'results.json',results[mode]);atomic_json(sub/'evaluation_status.json',{'state':'complete'})
        atomic_json(root/'results.json',results);atomic_json(root/'motion_cache_provenance.json',reuse);atomic_json(root/'evaluation_status.json',{'state':'complete','elapsed_seconds':time.monotonic()-started})
    except BaseException as exc:atomic_json(root/'evaluation_status.json',{'state':'failed','error':f'{type(exc).__name__}: {exc}'});raise


if __name__=='__main__':main()
