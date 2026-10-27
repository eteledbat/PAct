"""Sealed axis-only evaluation, exact original endpoints, and exact-input cache reuse."""
import argparse
import copy
import hashlib
import json
import pathlib
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from axis_fixed_common import (FrozenScene,atomic_json,continuous_from_raw,paths,
    state_from_nodes,tensor,export_axis,assert_ranges,sha256)
from frozen_joint_common import continuous_from_raw as parent_decode
from evaluate_frozen_joint_routes import (axis_metrics as parent_axis_metrics,
    heldout_sources,motion,plot_axes,render_movie,angle)


def signature(state,limits):
    return hashlib.sha256(b''.join(x.detach().cpu().numpy().tobytes() for x in [state[0],state[1],limits])).hexdigest()


def axis_metrics(scene,obj):
    rows=parent_axis_metrics(scene,obj);byid={n['id']:n['joint'] for n in obj['diffuse_tree']};old={n['id']:n['joint'] for n in scene.asset['nodes']}
    assert_ranges(obj,scene)
    for row in rows:
        j,j0=byid[row['node_id']],old[row['node_id']]
        row['q_start_axis_sign_normalized']=row['q_start'];row['q_end_axis_sign_normalized']=row['q_end']
        active=row['q_start'] is not None
        row['q_start']=float(j['range'][0]) if active else None
        row['q_end']=float(j['range'][1]) if active else None
        row['max_numeric_endpoint_change']=float(np.max(np.abs(np.array(j['range'])-np.array(j0['range']))))
        assert row['max_numeric_endpoint_change']==0.
        if active:
            d,d0=np.array(j['axis']['direction']),np.array(j0['axis']['direction']);cos=np.clip(d@d0/(np.linalg.norm(d)*np.linalg.norm(d0)),-1,1)
            row['axis_sign_reversed']=bool(cos<0)
            row['oriented_axis_rotation_from_original_degrees']=float(np.degrees(np.arccos(cos)))
        else:row['axis_sign_reversed']=False;row['oriented_axis_rotation_from_original_degrees']=None
    return rows


def prime_parent_cache(scene,root,cfg,route,case,parent_results):
    parent=pathlib.Path(cfg['parent_root']);cache={};origins={}
    for method,old in parent_results[route][case].items():
        if method=='R0':obj=scene.asset['object']
        elif method=='R0_raw':
            features=torch.load(root/'features'/case/'cache.pt',weights_only=False)
            scene.params.override=parent_decode(features['raw'].cuda(),scene)
            obj=scene.params.export(scene.asset)
        else:
            file=pathlib.Path(old['candidate']);seal=json.loads((file.parent/'seal.json').read_text());suffix='_official' if method.endswith('_official') else '_continuous'
            assert sha256(file)==seal['candidates'][case+suffix]
            obj=json.loads(file.read_text())
        state=state_from_nodes(scene,obj['diffuse_tree'])
        for limits,label in [(state[2],'own_range_motion'),(scene.params.limits0,'common_original_range_motion')]:
            key=signature(state,limits)
            if key not in cache:
                cache[key]=old[label];origins[key]=f'parent/{route}/{case}/{method}/{label}'
    scene.params.override=None;scene.assert_limits()
    return cache,origins


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=pathlib.Path(a.root);cfg=json.loads((root/'config.json').read_text());torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;started=time.monotonic();results={};reuse=[]
    assert cfg['endpoints_frozen'] and json.loads((root/'controller_status.json').read_text())['state']=='optimization_complete'
    parent=pathlib.Path(cfg['parent_root']);assert sha256(parent/'integrity.json')==cfg['parent_integrity_sha256'];pin=json.loads((parent/'integrity.json').read_text());assert sha256(parent/'results.json')==pin['results_sha256']
    assert (root/'frozen_assets.json').read_bytes()==(parent/'frozen_assets.json').read_bytes()
    parent_cfg=json.loads((parent/'config.json').read_text());assert parent_cfg['runs']==cfg['runs'] and parent_cfg['sources_per_part']==cfg['sources_per_part']
    old_results=json.loads((parent/'results.json').read_text())
    atomic_json(root/'evaluation_status.json',{'state':'running'})
    try:
        for route in ['direct','head']:
            results[route]={}
            for case in cfg['test_cases']:
                scene=FrozenScene(cfg['runs'][case],cfg['sources_per_part']);heldout_sources(scene);objects={'R0':copy.deepcopy(scene.asset['object'])};sources={}
                if route=='head':
                    cache=torch.load(root/'features'/case/'cache.pt',weights_only=False);scene.params.override=continuous_from_raw(cache['raw'].cuda(),scene);objects['R0_raw']=export_axis(scene)
                for arm in ['A','B']:
                    folder=root/(f'direct_{case}_{arm}' if route=='direct' else f'head_shared_{arm}');seal=json.loads((folder/'seal.json').read_text())
                    assert seal['checkpoint_sha256']==sha256(folder/'checkpoint.pt') and seal['endpoints_frozen_all_steps'] and seal['max_endpoint_delta']==0.
                    for name,suffix in ([ (arm,'candidate') ] if route=='direct' else [(arm+'_raw','candidate'),(arm+'_official','official')]):
                        file=folder/f'{case}_{suffix}.json';key=case+('_official' if suffix=='official' else '_continuous');assert sha256(file)==seal['candidates'][key]
                        obj=json.loads(file.read_text());assert_ranges(obj,scene);objects[name]=obj;sources[name]=str(file)
                directory=root/f'eval_{route}_{case}';directory.mkdir(exist_ok=True);rows={}
                motion_cache,cache_origins=prime_parent_cache(scene,root,cfg,route,case,old_results)
                for method,obj in objects.items():
                    atomic_json(root/'evaluation_status.json',{'state':'running','route':route,'case':case,'method':method})
                    file=directory/f'{method}_metrics.json'
                    if file.exists():rows[method]=json.loads(file.read_text());continue
                    scene.params.override=state_from_nodes(scene,obj['diffuse_tree']);scene.assert_limits();joint_rows=axis_metrics(scene,obj);key=signature(scene.params.override,scene.params.limits0)
                    if key not in motion_cache:
                        with torch.no_grad():motion_cache[key]=motion(scene,scene.params.limits0)
                        cache_origins[key]='computed_for_this_exact_axis_state'
                    measured=copy.deepcopy(motion_cache[key]);origin=cache_origins[key]
                    result={'case':case,'route':route,'method':method,'joints':joint_rows,'fixed_range_motion':measured,
                        'own_range_motion':copy.deepcopy(measured),'common_original_range_motion':copy.deepcopy(measured),
                        'geometry_frozen':True,'endpoints_frozen':True,'max_numeric_endpoint_change':0.,'candidate':sources.get(method),
                        'axis_alignment':'Same original baseline static-AABB similarity as previous round; never refit each candidate',
                        'motion_cache_origin':origin,'motion_state_sha256':key,'GT_test_used':'Post-seal evaluation only',
                        'range_note':'q_start/q_end are actual serialized unchanged values; sign-normalized diagnostics stored separately; direction reversal remains an axis change'}
                    atomic_json(file,result);rows[method]=result;reuse.append({'route':route,'case':case,'method':method,'state_sha256':key,'origin':origin})
                    print('EVALUATED',route,case,method,origin,flush=True)
                first=next(iter(rows.values()))['fixed_range_motion']['paths'];n=len(first);fig,axes=plt.subplots((n+1)//2,2,figsize=(12,3.7*((n+1)//2)),squeeze=False);colors=plt.get_cmap('tab10')
                for index,path in enumerate(first):
                    ax=axes.flat[index]
                    for i,(method,res) in enumerate(rows.items()):curve=next(p for p in res['fixed_range_motion']['paths'] if p['path']==path['path']);ax.plot(np.linspace(0,1,41),curve['proxy'],label=method,color=colors(i))
                    ax.set(title=path['path'],xlabel='Fraction of FIXED native range',ylabel='Gaussian overlap proxy (not volume)');ax.grid(alpha=.25);ax.legend(fontsize=8)
                for ax in list(axes.flat)[n:]:ax.set_axis_off()
                fig.suptitle(f'{route} {case}: frozen geometry AND numeric endpoints');fig.tight_layout();fig.savefig(directory/'motion_overlap.png',dpi=170);plt.close(fig)
                plot_axes({m:r['joints'] for m,r in rows.items()},directory,f'{route} {case}: axes only')
                selected=['R0','A','B'] if route=='direct' else ['R0','A_official','B_official']
                with torch.no_grad():render_movie(scene,{m:objects[m] for m in selected},directory,f'{route} {case}: endpoints fixed')
                results[route][case]=rows;del scene;torch.cuda.empty_cache()
        atomic_json(root/'results.json',results);atomic_json(root/'motion_cache_provenance.json',reuse)
        atomic_json(root/'evaluation_status.json',{'state':'complete','elapsed_seconds':time.monotonic()-started})
    except BaseException as exc:atomic_json(root/'evaluation_status.json',{'state':'failed','error':f'{type(exc).__name__}: {exc}'});raise


if __name__=='__main__':main()
