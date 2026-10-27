"""New held-out objects: same fixed-endpoint direct trials, sealed head transfer.

Original training sources and previous experiments remain unchanged. Test GT is
read only after all direct candidates and transferred head predictions are sealed.
"""
import argparse
import copy
import datetime
import json
import math
import os
import pathlib
import subprocess
import sys
import time

import numpy as np
import torch

from axis_fixed_common import (FrozenScene,atomic_json,sha256,paths,tensor,
    export_axis,assert_ranges,fixed_raw,continuous_from_raw,original_postprocessed,
    state_from_nodes)
from collision_refinement_common import transforms
from run_axis_fixed_trial import make_head,mlp
from run_axis_fixed_routes import compare

REPO=pathlib.Path(__file__).resolve().parents[2]


def inventory(run,case):
    status=json.loads((pathlib.Path(run)/'status.json').read_text());assert status['state']=='complete'
    file=pathlib.Path(status['object']);obj=json.loads(file.read_text())
    files=[file]+[file.parent/rel for n in obj['diffuse_tree'] for key in ['gs','plys'] for rel in n[key]]
    return [{'case':case,'path':str(f),'sha256':sha256(f)} for f in files]


def finalize(root,cfg):
    assert not list(root.glob('direct_*/checkpoint.pt'))
    frozen=[]
    for case in cfg['test_cases']:
        status=json.loads((root/'features'/case/'status.json').read_text())
        assert status['same_run_feature_asset_association']
        frozen+=inventory(cfg['runs'][case],case)
    atomic_json(root/'frozen_assets.json',frozen)
    parent=pathlib.Path(cfg['parent_root']);old=json.loads((parent/'frozen_assets.json').read_text())
    old+=inventory(cfg['source_runs']['20411'],'old_20411')
    atomic_json(root/'older_source_assets.json',old)
    cfg['baseline_association']='Same-run new original assets and frozen cached features; previous trained A/B are NEVER direct initializers'
    cfg['head_targets']='No new targets read: reuse previous sealed axis-only A/B head weights; test objects excluded from that training'
    cfg['head_training_for_this_extension']='NONE; previous 200-step A/B checkpoints transferred without updates'
    atomic_json(root/'config.json',cfg)
    atomic_json(root/'inputs_finalized.json',{'config_sha256':sha256(root/'config.json'),'files':len(frozen)})


def preflight(root,cfg):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;rows={}
    for case in cfg['test_cases']:
        scene=FrozenScene(cfg['runs'][case],cfg['sources_per_part']);p=scene.params
        assert len(p.active)>0,'No predicted movable joint: retain this failure rather than inventing an axis'
        assert {k for k,v in p.named_parameters() if v.requires_grad}=={'theta','phi'}
        qs=dict(paths(p.limits(),21))['synchronized']
        with torch.no_grad():
            world=p.world(qs);error=max(float(np.max(abs(world[n['id']].cpu().numpy()-np.array([transforms(scene.asset['nodes'],float(t))[n['id']] for t in np.linspace(0,1,21)])))) for n in scene.asset['nodes'])
            scene.objective.times=qs;parity=abs(float(scene.objective())-float(scene.proxy(qs)))
            rigid=0.
            for part in scene.asset['parts']:
                x=tensor(part['xyz'][:20]);tr=world[part['node']['id']];y=torch.einsum('nj,bij->bni',x,tr[:,:3,:3])+tr[:,None,:3,3]
                rigid=max(rigid,float((torch.cdist(y,y)-torch.cdist(x,x)[None]).abs().max()))
        collision=scene.proxy(qs[[2,10,20]])/scene.normalization
        gs=torch.autograd.grad(collision,[p.theta,p.phi],allow_unused=True);norms=[float(g.norm()) if g is not None else 0. for g in gs]
        assert all(torch.isfinite(g).all() for g in gs if g is not None)
        if scene.normalization>1e-8:assert norms[0]>0
        pivot_identifiable=bool(p.pivot_mask.any())
        if pivot_identifiable and scene.normalization>1e-8:assert norms[1]>0
        if not pivot_identifiable:assert norms[1]==0
        assert max(error,parity,rigid)<2e-6
        scene.assert_limits();assert_ranges(export_axis(scene),scene)
        cache=torch.load(root/'features'/case/'cache.pt',weights_only=False);head=make_head(cache['mlp_state']);pred=mlp(head,cache['pooled']['average']);repro=float((pred.detach().cpu()-cache['raw']).abs().max());assert repro<1e-4
        raw=fixed_raw(pred,cache,scene);scene.params.override=continuous_from_raw(raw,scene);scene.assert_limits()
        perturbed=pred.detach().clone();perturbed[:,12:18]+=.3;assert torch.equal(fixed_raw(perturbed,cache,scene),raw.detach())
        official=original_postprocessed(raw,scene);axis_error=0.
        for a,b in zip(scene.asset['nodes'],official['diffuse_tree']):
            if a['id'] not in p.indices:continue
            for key in ['direction','origin']:
                if key=='origin' and a['joint']['type']=='prismatic':continue
                axis_error=max(axis_error,float(np.max(abs(np.array(a['joint']['axis'][key])-np.array(b['joint']['axis'][key])))))
        assert axis_error<1e-5
        rows[case]={'collision_gradient_direction':norms[0],'collision_gradient_pivot':norms[1],
            'pivot_identifiable':pivot_identifiable,'FK_max_error':error,'rigid_distance_max_error':rigid,
            'proxy_parity_error':parity,'original_MLP_reproduction_error':repro,'official_axis_reproduction_error':axis_error,
            'max_endpoint_delta':0.,'unused_range_perturbation_has_no_effect':True,'asset_normalization':scene.normalization}
        del head,scene;torch.cuda.empty_cache()
    atomic_json(root/'gradient_and_rigidity_checks.json',{'status':'passed','cases':rows})


def export_heads(root,cfg):
    """No optimizer, no GT: only forward the already-sealed A/B output MLPs."""
    parent=pathlib.Path(cfg['parent_root']);assert sha256(parent/'integrity.json')==cfg['parent_integrity_sha256']
    original_cache=torch.load(parent/'features'/cfg['train_cases'][0]/'cache.pt',map_location='cpu',weights_only=False)
    for arm in ['A','B']:
        out=root/f'head_shared_{arm}'
        if (out/'seal.json').exists():continue
        out.mkdir(exist_ok=True);source=parent/f'head_shared_{arm}';oldseal=json.loads((source/'seal.json').read_text())
        assert oldseal['endpoints_frozen_all_steps'] and sha256(source/'checkpoint.pt')==oldseal['checkpoint_sha256']
        checkpoint=torch.load(source/'checkpoint.pt',map_location='cpu',weights_only=False);assert checkpoint['step']==cfg['steps']
        model=make_head(original_cache['mlp_state']);model.load_state_dict(checkpoint['model'],strict=True);model.eval();candidates={}
        for case in cfg['test_cases']:
            scene=FrozenScene(cfg['runs'][case],cfg['sources_per_part']);cache=torch.load(root/'features'/case/'cache.pt',weights_only=False)
            assert all(torch.equal(v,cache['mlp_state'][k]) for k,v in original_cache['mlp_state'].items())
            with torch.no_grad():
                pred=mlp(model,cache['pooled']['average']);raw=fixed_raw(pred,cache,scene);assert torch.isfinite(raw).all()
                scene.params.override=continuous_from_raw(raw,scene);scene.assert_limits();obj=export_axis(scene);official=original_postprocessed(raw,scene)
            for suffix,value,kind in [('candidate',obj,'continuous'),('official',official,'official')]:
                file=out/f'{case}_{suffix}.json';atomic_json(file,value);candidates[case+'_'+kind]=sha256(file)
            np.save(out/f'{case}_raw.npy',raw.detach().cpu().numpy());del scene
        checkpoint_file=out/'checkpoint.pt'
        if not checkpoint_file.exists():checkpoint_file.symlink_to(os.path.relpath(source/'checkpoint.pt',out))
        atomic_json(out/'protocol.json',{'checkpoint_reused_from':str(source),'checkpoint_sha256':oldseal['checkpoint_sha256'],
            'original_training_protocol_sha256':sha256(source/'protocol.json'),'config_sha256':sha256(root/'config.json'),
            'new_optimizer_steps':0,'GT_test_in_optimizer':False,'test_cases':cfg['test_cases'],'native_endpoints_frozen':True})
        atomic_json(out/'status.json',{'state':'complete','step':200,'new_optimizer_steps':0,'route':'head','arm':arm,'checkpoint_reused':True})
        atomic_json(out/'seal.json',{'checkpoint_sha256':oldseal['checkpoint_sha256'],'candidates':candidates,
            'frozen_assets_unchanged':True,'endpoints_frozen_all_steps':True,'max_endpoint_delta':0.,'GT_reads_blocked':[],
            'checkpoint_reused_from':str(source),'new_optimizer_steps':0})
        del model;torch.cuda.empty_cache()


def prepare_gt(root,cfg):
    """Called only after all optimization/prediction seals exist."""
    for case in cfg['test_cases']:
        run=pathlib.Path(cfg['runs'][case]);out=run/'joint_evaluation';file=out/'joint_errors.json'
        if file.exists():continue
        if case!='20411':
            with (root/f'GT_baseline_{case}.log').open('wb') as log:subprocess.run([sys.executable,str(REPO/'repro/tools/evaluate_gt_joints.py'),'--run',str(run)],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
            continue
        from evaluate_freeart_partnet_case import gt_parts
        import trimesh
        provenance=json.loads((run/'input_provenance.json').read_text());source=pathlib.Path(provenance['source_data_root']);states,delta=gt_parts(source)
        labels=json.loads((source/'joint_info.json').read_text());assert len(labels)==1 and labels[0]['type']=='prismatic'
        objfile=pathlib.Path(json.loads((run/'status.json').read_text())['object']);obj=json.loads(objfile.read_text());byid={n['id']:n for n in obj['diffuse_tree']}
        mapping={r['gt_mesh_index']:r['pact_node_id'] for r in provenance['part_correspondence']};body=trimesh.load(objfile.parent/byid[mapping[0]]['plys'][0],process=False,force='mesh');gtbody=states[0][0]
        D=float(np.linalg.norm(gtbody.extents));scale=D/np.linalg.norm(body.extents);shift=gtbody.bounds.mean(0)-scale*body.bounds.mean(0)
        summary={'scene_id':case,'R_export_to_gtworld':np.eye(3).tolist(),'scale':float(scale),'translation':shift.tolist(),'GT_static_diagonal':D,
            'direction_alignment':'Fixed identity between PAct export and original GLB; verified six-state translation, no axis fitting',
            'joints':[{'gt_mesh_index':1,'predicted_node_id':mapping[1],'GT_type':'prismatic','GT_direction_world':(delta/np.linalg.norm(delta)).tolist(),
                'GT_origin_world':[0,0,0],'GT_limit':labels[0]['range']}],
            'GT_use':'Post-seal evaluation, same six-state translation extraction as earlier 20411, no candidate selection',
            'GT_mesh_travel':float(np.linalg.norm(delta)),'pivot_status':'unavailable: prismatic origin does not affect kinematics'}
        out.mkdir(exist_ok=True);atomic_json(file,summary)


def evaluate(root,cfg):
    from evaluate_axis_fixed_routes import axis_metrics,signature
    from evaluate_frozen_joint_routes import heldout_sources,motion,plot_axes,render_movie
    import matplotlib.pyplot as plt
    assert json.loads((root/'controller_status.json').read_text())['state']=='optimization_complete'
    prepare_gt(root,cfg);results={'direct':{},'head':{}};started=time.monotonic()
    for case in cfg['test_cases']:
        scene=FrozenScene(cfg['runs'][case],cfg['sources_per_part']);heldout_sources(scene);motion_cache={}
        for route in ['direct','head']:
            directory=root/f'eval_{route}_{case}';directory.mkdir(exist_ok=True);objects={'R0':copy.deepcopy(scene.asset['object'])};candidate_paths={}
            if route=='head':
                cache=torch.load(root/'features'/case/'cache.pt',weights_only=False);scene.params.override=continuous_from_raw(cache['raw'].cuda(),scene);objects['R0_raw']=export_axis(scene)
            for arm in ['A','B']:
                folder=root/(f'direct_{case}_{arm}' if route=='direct' else f'head_shared_{arm}');seal=json.loads((folder/'seal.json').read_text());assert sha256(folder/'checkpoint.pt')==seal['checkpoint_sha256']
                for method,suffix in ([(arm,'candidate')] if route=='direct' else [(arm+'_raw','candidate'),(arm+'_official','official')]):
                    file=folder/f'{case}_{suffix}.json';assert sha256(file)==seal['candidates'][case+('_official' if suffix=='official' else '_continuous')]
                    objects[method]=json.loads(file.read_text());candidate_paths[method]=str(file)
            rows={}
            for method,obj in objects.items():
                atomic_json(root/'evaluation_status.json',{'state':'running','route':route,'case':case,'method':method})
                file=directory/f'{method}_metrics.json';scene.params.override=state_from_nodes(scene,obj['diffuse_tree']);scene.assert_limits();key=signature(scene.params.override,scene.params.limits0)
                if file.exists():
                    result=json.loads(file.read_text());motion_cache[key]=result['fixed_range_motion'];rows[method]=result;continue
                joints=axis_metrics(scene,obj)
                if key not in motion_cache:
                    with torch.no_grad():motion_cache[key]=motion(scene,scene.params.limits0)
                measured=copy.deepcopy(motion_cache[key]);result={'case':case,'route':route,'method':method,'joints':joints,'fixed_range_motion':measured,
                    'geometry_frozen':True,'endpoints_frozen':True,'max_numeric_endpoint_change':0.,'candidate':candidate_paths.get(method),'motion_state_sha256':key,
                    'GT_test_used':'Post-seal evaluation only; previous case GT has been inspected historically, not a blind test'}
                atomic_json(file,result);rows[method]=result;print('EVALUATED',route,case,method,flush=True)
            first=rows['R0']['fixed_range_motion']['paths'];fig,axes=plt.subplots((len(first)+1)//2,2,figsize=(12,3.7*((len(first)+1)//2)),squeeze=False)
            for index,path in enumerate(first):
                ax=axes.flat[index]
                for method,result in rows.items():curve=next(p for p in result['fixed_range_motion']['paths'] if p['path']==path['path']);ax.plot(np.linspace(0,1,41),curve['proxy'],label=method)
                ax.set(title=path['path'],xlabel='Fraction of fixed native range',ylabel='Gaussian overlap proxy (not volume)');ax.legend(fontsize=8);ax.grid(alpha=.25)
            for ax in list(axes.flat)[len(first):]:ax.set_axis_off()
            fig.suptitle(f'{route}/{case}: geometry and endpoints fixed');fig.tight_layout();fig.savefig(directory/'motion_overlap.png',dpi=170);plt.close(fig)
            plot_axes({m:r['joints'] for m,r in rows.items()},directory,f'{route}/{case}')
            selected=['R0','A','B'] if route=='direct' else ['R0','A_official','B_official']
            with torch.no_grad():render_movie(scene,{m:objects[m] for m in selected},directory,f'{route}/{case}')
            results[route][case]=rows
        del scene;torch.cuda.empty_cache()
    atomic_json(root/'results.json',results);atomic_json(root/'evaluation_status.json',{'state':'complete','elapsed_seconds':time.monotonic()-started})


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--phase',choices=['run','evaluate'],default='run');a=p.parse_args();root=pathlib.Path(a.root);cfg=json.loads((root/'config.json').read_text());torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    parent=pathlib.Path(cfg['parent_root']);assert sha256(parent/'integrity.json')==cfg['parent_integrity_sha256'];assert sha256(parent/'config.json')==cfg['parent_config_sha256']
    if a.phase=='evaluate':evaluate(root,cfg);return
    def status(state,**extra):atomic_json(root/'controller_status.json',{'state':state,'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra})
    def trial(case,arm,label,smoke=False,zero=False,stop=None):
        out=root/label
        if (out/'seal.json').exists():return
        cmd=[sys.executable,str(REPO/'repro/tools/run_axis_fixed_trial.py'),'--root',str(root),'--route','direct','--case',case,'--arm',arm,'--tag',label]
        if smoke:cmd+=['--steps','30']
        if zero:cmd+=['--zero-weight']
        if (out/'checkpoint.pt').exists():cmd+=['--resume']
        if stop:cmd+=['--stop-after',str(stop)]
        status('running',job=label)
        with (root/'commands.jsonl').open('a') as f:f.write(json.dumps(cmd)+'\n')
        with (root/f'{label}.log').open('ab') as log:subprocess.run(cmd,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
    try:
        if not (root/'inputs_finalized.json').exists():finalize(root,cfg)
        else:assert json.loads((root/'inputs_finalized.json').read_text())['config_sha256']==sha256(root/'config.json')
        if not (root/'gradient_and_rigidity_checks.json').exists():status('preflight');preflight(root,cfg)
        assert json.loads((root/'gradient_and_rigidity_checks.json').read_text())['status']=='passed'
        for case in cfg['test_cases']:
            for arm in ['A','B']:trial(case,arm,f'direct_{case}_{arm}_smoke',smoke=True)
            trial(case,'B',f'direct_{case}_zero_control',smoke=True,zero=True)
            compare(root/f'direct_{case}_A_smoke/checkpoint.pt',root/f'direct_{case}_zero_control/checkpoint.pt',root/f'{case}_zero_control.json')
        case=cfg['test_cases'][0];label='resume_control'
        if not (root/label/'checkpoint.pt').exists():trial(case,'B',label,smoke=True,stop=10)
        trial(case,'B',label,smoke=True)
        compare(root/f'direct_{case}_B_smoke/checkpoint.pt',root/label/'checkpoint.pt',root/'resume_control.json')
        for case in cfg['test_cases']:
            for arm in ['A','B']:trial(case,arm,f'direct_{case}_{arm}')
        status('exporting_sealed_heads');export_heads(root,cfg)
        frozen=json.loads((root/'frozen_assets.json').read_text());assert all(sha256(f['path'])==f['sha256'] for f in frozen)
        status('optimization_complete',head_new_optimizer_steps=0)
    except BaseException as exc:status('failed',error=f'{type(exc).__name__}: {exc}');raise


if __name__=='__main__':main()
