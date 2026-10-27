"""Axis-only matched trials: both original endpoints and all geometry frozen."""
import argparse
import datetime
import json
import os
import pathlib
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from axis_fixed_common import (FrozenScene,atomic_json,continuous_from_raw,
                                 original_postprocessed,paths,sha256,tensor,
                                 fixed_raw,export_axis,assert_ranges,output_mask)

REPO=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))


def make_head(state):
    from modules.pact.models.structured_latent_flow import ArticulationRegressionHead
    head=ArticulationRegressionHead(base_model=None,feat_dim=1024,out_channel=24,input_layer_num=2,out_layer_num=3,use_fp16=False)
    head.art_out_layers.load_state_dict(state,strict=True)
    # Only the original released output MLP is continued, not a newly designed net.
    return head.art_out_layers.float().cuda()


def mlp(head,pooled):
    from modules.pact.modules.sparse import SparseTensor
    x=SparseTensor(coords=torch.zeros((len(pooled),1),dtype=torch.int32,device='cuda'),feats=pooled.cuda())
    for i,layer in enumerate(head):
        x=layer(x)
        if i!=len(head)-1:x=x.replace(F.silu(x.feats))
    return x.feats[1:]


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--route',choices=['direct','head'],required=True);p.add_argument('--case');p.add_argument('--arm',choices=['A','B'],required=True);p.add_argument('--steps',type=int);p.add_argument('--tag');p.add_argument('--resume',action='store_true');p.add_argument('--zero-weight',action='store_true');p.add_argument('--stop-after',type=int);a=p.parse_args()
    root=pathlib.Path(a.root).resolve();cfg=json.loads((root/'config.json').read_text());steps=a.steps or cfg['steps'];label=a.tag or f'{a.route}_{a.case or "shared"}_{a.arm}';out=root/label;out.mkdir(exist_ok=a.resume)
    start=time.monotonic();random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    blocked=[]
    def guard(event,values):
        if event!='open' or not isinstance(values[0],(str,bytes)):return
        name=os.fsdecode(values[0])
        if any(v in name for v in ['/VideoArtGS/','/FreeArt3D/','/Part2GS-data/','/gt/','joint_evaluation','mobility_v2']):
            blocked.append(name);raise PermissionError('Source/GT files are forbidden during optimization: '+name)
        if '/training_targets/' in name:
            allowed=a.route=='head' and pathlib.Path(name).stem in cfg['train_cases']
            if not allowed:blocked.append(name);raise PermissionError('Non-training target read: '+name)
    sys.addaudithook(guard)
    cases=[a.case] if a.route=='direct' else cfg['train_cases']
    scenes={c:FrozenScene(cfg['runs'][c],cfg['sources_per_part']) for c in cases}
    if a.route=='direct':
        assert a.case in cfg['direct_cases'];model=scenes[a.case].params
        optimizer=torch.optim.Adam([{'params':[model.theta],'lr':.002},{'params':[model.phi],'lr':.0005*model.diagonal}])
        weight=cfg['direct_collision_weight']
        caches=None
    else:
        caches={c:torch.load(root/'features'/c/'cache.pt',map_location='cpu',weights_only=False) for c in cases}
        first=caches[cases[0]]['mlp_state']
        assert all(all(torch.equal(v,caches[c]['mlp_state'][k]) for k,v in first.items()) for c in cases)
        model=make_head(first);optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['head_lr'],weight_decay=0)
        targets={c:torch.load(root/'training_targets'/f'{c}.pt',map_location='cuda',weights_only=False) for c in cases}
        weight=cfg['head_collision_weight']
        # Fixed type/semantic channels; continuous outputs alone may change.
        output_masks={}
        for c in cases:
            output_masks[c]=output_mask(caches[c],scenes[c])
            assert torch.count_nonzero(targets[c]['mask'][:,12:18])==0
    if a.arm=='A' or a.zero_weight:weight=0.
    sources=[pathlib.Path(__file__),pathlib.Path(__file__).with_name('axis_fixed_common.py'),pathlib.Path(__file__).with_name('frozen_joint_common.py'),pathlib.Path(__file__).with_name('run_frozen_joint_trial.py'),pathlib.Path(__file__).with_name('refine_collision_axes.py'),pathlib.Path(__file__).with_name('collision_refinement_common.py'),pathlib.Path(__file__).with_name('reconstruction_ablation_common.py')]
    protocol={'route':a.route,'cases':cases,'arm':a.arm,'steps':steps,'weight':weight,'config_sha256':sha256(root/'config.json'),
              'sources':{s.name:sha256(s) for s in sources},'geometry_frozen':True,'types_tree_frozen':True,'seed':cfg['seed'],
              'GT_training_labels':cases if a.route=='head' else [],'GT_test_in_optimizer':False,
              'selection':'Fixed final step, never GT best checkpoint',
              'endpoint_updates':'NONE: original numeric start/end frozen for every object, including head outputs',
              'endpoint_values':{c:{str(n['id']):n['joint']['range'] for n in scenes[c].asset['nodes']} for c in cases},
              'baseline':'Only prediction-retention prior (stationary at initialization)' if a.route=='direct' else 'Official masked MSE on effective direction/pivot channels only; original range outputs replayed; GT range targets stripped',
              'feature_targets_sha256':{} if caches is None else {c:{'features':sha256(root/'features'/c/'cache.pt'),'target':sha256(root/'training_targets'/f'{c}.pt')} for c in cases}}
    history=[];begin=0
    if a.resume:
        assert json.loads((out/'protocol.json').read_text())==protocol,'Resume protocol/source mismatch'
        checkpoint=torch.load(out/'checkpoint.pt',map_location='cuda',weights_only=False);model.load_state_dict(checkpoint['model']);optimizer.load_state_dict(checkpoint['optimizer']);history=checkpoint['history'];begin=checkpoint['step']
    else:atomic_json(out/'protocol.json',protocol)

    def save(state,**extra):
        tmp=out/'checkpoint.tmp';torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'history':history,'step':len(history)},tmp);tmp.replace(out/'checkpoint.pt')
        atomic_json(out/'history.json',history)
        atomic_json(out/'status.json',{'state':state,'route':a.route,'arm':a.arm,'step':len(history),'total_steps':steps,'elapsed_seconds':time.monotonic()-start,'peak_gpu_MiB':torch.cuda.max_memory_allocated()/2**20,'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra})
    try:
        save('running')
        for step in range(begin,steps):
            optimizer.zero_grad(set_to_none=True);case=cases[step%len(cases)];scene=scenes[case]
            if a.route=='direct':
                base=cfg['direct_prior_weight']*model.prior();collision=scene.minibatch(step)
            else:
                cache=caches[case];key=[k for k in cache['pooled'] if k!='average'][(step//len(cases))%4]
                pred=mlp(model,cache['pooled'][key]);mask=output_masks[case];raw=fixed_raw(pred,cache,scene)
                target=targets[case];base=F.mse_loss(raw*target['mask'],target['target']*target['mask'])
                scene.params.override=continuous_from_raw(raw,scene)
                scene.assert_limits()
                collision=scene.minibatch(step//len(cases))
            loss=base+weight*collision/scene.normalization
            assert torch.isfinite(loss), 'Nonfinite objective'
            loss.backward();norms={n:float(v.grad.norm()) for n,v in model.named_parameters() if v.grad is not None}
            assert norms and all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None),'Invalid gradient'
            optimizer.step()
            if a.route=='direct':model.project()
            scene.assert_limits()
            row={'step':step+1,'case':case,'base_loss':float(base.detach()),'motion_proxy':float(collision.detach()),'loss':float(loss.detach()),'gradient_norms':norms,'max_endpoint_delta':0.0,'range_grad_none':scene.params.range_delta.grad is None}
            history.append(row)
            if (step+1)%10==0 or step==0:save('running',latest=row);print(label,step+1,row['base_loss'],row['motion_proxy'],flush=True)
            if a.stop_after and step+1==a.stop_after:
                save('interrupted_for_resume_test');return
        candidates={}
        with torch.no_grad():
            if a.route=='direct':
                obj=export_axis(scenes[a.case]);atomic_json(out/f'{a.case}_candidate.json',obj);candidates[a.case+'_continuous']=sha256(out/f'{a.case}_candidate.json')
            else:
                # Inference for all cases reads cached features only, never test labels.
                for case in cfg['train_cases']+cfg['test_cases']:
                    cache=torch.load(root/'features'/case/'cache.pt',map_location='cpu',weights_only=False)
                    scene=scenes.get(case) or FrozenScene(cfg['runs'][case],cfg['sources_per_part'])
                    raw=fixed_raw(mlp(model,cache['pooled']['average']),cache,scene)
                    scene.params.override=continuous_from_raw(raw,scene)
                    atomic_json(out/f'{case}_candidate.json',export_axis(scene))
                    atomic_json(out/f'{case}_official.json',original_postprocessed(raw,scene))
                    np.save(out/f'{case}_raw.npy',raw.cpu().numpy())
                    candidates[case+'_continuous']=sha256(out/f'{case}_candidate.json');candidates[case+'_official']=sha256(out/f'{case}_official.json')
        frozen=json.loads((root/'frozen_assets.json').read_text());assert all(sha256(x['path'])==x['sha256'] for x in frozen),'Original asset changed'
        assert not blocked
        save('complete')
        atomic_json(out/'seal.json',{'checkpoint_sha256':sha256(out/'checkpoint.pt'),'candidates':candidates,'frozen_assets_unchanged':True,'local_geometry_appearance_trainable':False,'GT_reads_blocked':blocked,'endpoints_frozen_all_steps':True,'max_endpoint_delta':0.0,'protocol_sha256':sha256(out/'protocol.json')})
    except BaseException as exc:
        save('failed',error=f'{type(exc).__name__}: {exc}',blocked=blocked);raise


if __name__=='__main__':main()
