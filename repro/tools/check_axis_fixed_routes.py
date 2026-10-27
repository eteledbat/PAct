"""Real-data proof: axes receive gradients, endpoints neither vary nor receive gradients."""
import argparse
import json
import pathlib

import numpy as np
import torch
import torch.nn.functional as F

from axis_fixed_common import (FrozenScene,atomic_json,continuous_from_raw,
    original_postprocessed,paths,tensor,fixed_raw,export_axis,assert_ranges)
from collision_refinement_common import transforms
from run_axis_fixed_trial import make_head,mlp


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=pathlib.Path(a.root);cfg=json.loads((root/'config.json').read_text())
    torch.set_num_threads(4);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False
    report={'status':'running','direct':{},'head':{}}
    try:
        for case in cfg['test_cases']:
            scene=FrozenScene(cfg['runs'][case],cfg['sources_per_part']);params=scene.params
            assert {n for n,p in params.named_parameters() if p.requires_grad}=={'theta','phi'}
            qs=dict(paths(params.limits(),21))['synchronized']
            with torch.no_grad():
                world=params.world(qs)
                error=max(float(np.max(np.abs(world[n['id']].cpu().numpy()-np.array([transforms(scene.asset['nodes'],float(t))[n['id']] for t in np.linspace(0,1,21)])))) for n in scene.asset['nodes'])
                scene.objective.times=qs;parity=abs(float(scene.objective())-float(scene.proxy(qs)));rigid=0.
                for part in scene.asset['parts']:
                    x=tensor(part['xyz'][:20]);tr=world[part['node']['id']];y=torch.einsum('nj,bij->bni',x,tr[:,:3,:3])+tr[:,None,:3,3]
                    rigid=max(rigid,float((torch.cdist(y,y)-torch.cdist(x,x)[None]).abs().max()))
            loss=scene.proxy(qs[[2,10,20]])/scene.normalization
            grads=torch.autograd.grad(loss,[params.theta,params.phi]);norms=[float(g.norm()) for g in grads]
            assert all(torch.isfinite(g).all() for g in grads) and min(norms)>0
            assert error<2e-6 and parity<2e-6 and rigid<2e-6
            scene.assert_limits();assert not params.limits().requires_grad
            assert not any(v.requires_grad for d in scene.objective.data for v in d.values())
            assert_ranges(export_axis(scene),scene)
            raw=tensor(np.load(pathlib.Path(cfg['runs'][case])/'joint_trace/raw_head_00.npy'))
            obj=original_postprocessed(raw,scene);maximum=0.
            for old,new in zip(scene.asset['nodes'],obj['diffuse_tree']):
                if old['id'] not in params.indices:continue
                for key in ['direction','origin']:
                    if key=='origin' and old['joint']['type']=='prismatic':continue
                    maximum=max(maximum,float(np.max(np.abs(np.array(old['joint']['axis'][key])-np.array(new['joint']['axis'][key])))))
            assert maximum<1e-5
            report['direct'][case]={'trainable_parameters':['theta','phi'],'endpoint_requires_grad':False,'endpoint_gradient':None,'max_endpoint_change':0.,'FK_max_error':error,'proxy_parity_error':parity,'rigid_pair_distance_max_error':rigid,'axis_collision_gradient_norms':dict(zip(['direction','pivot'],norms)),'official_axis_postprocessing_reproduction_error':maximum}
            del scene
        for case in cfg['train_cases']:
            cache=torch.load(root/'features'/case/'cache.pt',weights_only=False);head=make_head(cache['mlp_state']);scene=FrozenScene(cfg['runs'][case],cfg['sources_per_part'])
            pred=mlp(head,cache['pooled']['average']);reproduction=float((pred.detach().cpu()-cache['raw']).abs().max());assert reproduction<1e-4
            raw=fixed_raw(pred,cache,scene);scene.params.override=continuous_from_raw(raw,scene);scene.assert_limits()
            qs=dict(paths(scene.params.limits(),21))['synchronized'][[2,10,20]];collision=scene.proxy(qs)/scene.normalization
            outputs=torch.autograd.grad(collision,scene.params.override[:2],retain_graph=True)
            physical_grads=torch.autograd.grad(collision,tuple(head.parameters()),retain_graph=True)
            target=torch.load(root/'training_targets'/f'{case}.pt',weights_only=False,map_location='cuda')
            assert torch.count_nonzero(target['mask'][:,12:18])==0 and torch.count_nonzero(target['target'][:,12:18])==0
            base=F.mse_loss(raw*target['mask'],target['target']*target['mask'])
            base_grads=torch.autograd.grad(base,tuple(head.parameters()),retain_graph=True)
            pred_grad=torch.autograd.grad(base+cfg['head_collision_weight']*collision,pred)[0]
            assert torch.count_nonzero(pred_grad[:,12:18])==0
            assert torch.count_nonzero(pred_grad[:,:6])==0 and torch.count_nonzero(pred_grad[:,18:])==0
            norm=lambda gs:float(torch.stack([g.square().sum() for g in gs]).sum().sqrt())
            assert all(torch.isfinite(g).all() for g in (*physical_grads,*base_grads,*outputs,pred_grad))
            assert norm(physical_grads)>0 and norm(base_grads)>0 and min(float(g.norm()) for g in outputs)>0
            changed=pred.detach().clone();changed[:,12:18]+=.3
            assert torch.equal(fixed_raw(changed,cache,scene),raw.detach()),'Unused range predictions leaked through shared MLP'
            assert_ranges(export_axis(scene),scene);assert_ranges(original_postprocessed(raw,scene),scene)
            report['head'][case]={'initial_MLP_reproduction_error':reproduction,'collision_MLP_gradient_norm':norm(physical_grads),'axis_MSE_gradient_norm':norm(base_grads),'axis_output_collision_gradient_norms':dict(zip(['direction','pivot'],[float(g.norm()) for g in outputs])),'range_output_gradient_max':float(pred_grad[:,12:18].abs().max()),'perturbed_unused_range_output_has_no_effect':True,'numeric_endpoint_change':0.,'GT_range_targets_present':False,'joint_limits_requires_grad':False}
            del head,scene
        report['status']='passed';atomic_json(root/'gradient_and_rigidity_checks.json',report)
    except BaseException as exc:report.update(status='failed',error=f'{type(exc).__name__}: {exc}');atomic_json(root/'gradient_and_rigidity_checks.json',report);raise
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
