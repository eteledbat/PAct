"""Check the selected variable gets gradients and the counterpart is truly fixed."""
import argparse
import json
import pathlib

import numpy as np
import torch
import torch.nn.functional as F

from axis_mode_common import (ModeScene,atomic_json,paths,tensor,fixed_raw,
    continuous_from_raw,export_axis,original_postprocessed,assert_fields,output_mask)
from run_axis_mode_trial import make_head,mlp
from collision_refinement_common import transforms


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=pathlib.Path(a.root);cfg=json.loads((root/'config.json').read_text());mode=cfg['mode'];torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;report={'status':'running','mode':mode,'direct':{},'head':{}}
    try:
        for case in cfg['test_cases']:
            scene=ModeScene(cfg['runs'][case],cfg['sources_per_part'],mode,'direct');param=scene.params.theta if mode=='direction' else scene.params.phi
            expected='theta' if mode=='direction' else 'phi';assert {n for n,p in scene.params.named_parameters() if p.requires_grad}=={expected}
            qs=dict(paths(scene.params.limits0,21))['synchronized']
            with torch.no_grad():
                world=scene.params.world(qs);fk=max(float(np.max(abs(world[n['id']].cpu().numpy()-np.array([transforms(scene.asset['nodes'],float(t))[n['id']] for t in np.linspace(0,1,21)])))) for n in scene.asset['nodes'])
                scene.objective.times=qs;parity=abs(float(scene.objective())-float(scene.proxy(qs)));rigid=0.
                for part in scene.asset['parts']:
                    x=tensor(part['xyz'][:20]);tr=world[part['node']['id']];y=torch.einsum('nj,bij->bni',x,tr[:,:3,:3])+tr[:,None,:3,3];rigid=max(rigid,float((torch.cdist(y,y)-torch.cdist(x,x)[None]).abs().max()))
            loss=scene.proxy(qs[[2,10,20]])/scene.normalization;g=torch.autograd.grad(loss,param)[0]
            assert torch.isfinite(g).all() and float(g.norm())>0
            assert max(fk,parity,rigid)<2e-6
            assert not any(v.requires_grad for data in scene.objective.data for v in data.values())
            scene.assert_frozen();assert_fields(export_axis(scene),scene)
            report['direct'][case]={'trainable':[expected],'selected_collision_gradient_norm':float(g.norm()),'FK_max_error':fk,'proxy_parity_error':parity,'rigid_distance_max_error':rigid,'frozen_counterpart_max_change':0.,'max_endpoint_change':0.,'normalization':scene.normalization}
            del scene
        for case in cfg['train_cases']:
            scene=ModeScene(cfg['runs'][case],cfg['sources_per_part'],mode,'head');cache=torch.load(root/'features'/case/'cache.pt',weights_only=False);model=make_head(cache['mlp_state']);pred=mlp(model,cache['pooled']['average']);error=float((pred.detach().cpu()-cache['raw']).abs().max());assert error<1e-4
            raw=fixed_raw(pred,cache,scene);scene.params.override=continuous_from_raw(raw,scene);scene.assert_frozen()
            target=torch.load(root/'training_targets'/f'{case}.pt',map_location='cuda',weights_only=False);lo,hi=(6,9) if mode=='direction' else (9,12)
            for key in ['target','mask']:
                check=target[key].clone();check[:,lo:hi]=0;assert not torch.count_nonzero(check)
            qs=dict(paths(scene.params.limits0,21))['synchronized'][[2,10,20]];collision=scene.proxy(qs)/scene.normalization
            base=F.mse_loss(raw*target['mask'],target['target']*target['mask'])
            collision_grads=torch.autograd.grad(collision,tuple(model.parameters()),retain_graph=True);base_grads=torch.autograd.grad(base,tuple(model.parameters()),retain_graph=True);pred_grad=torch.autograd.grad(base+cfg['head_collision_weight']*collision,pred)[0]
            mask=output_mask(cache,scene);assert not torch.count_nonzero(pred_grad[mask==0]);assert torch.count_nonzero(pred_grad[mask==1])>0
            norm=lambda gs:float(torch.stack([g.square().sum() for g in gs]).sum().sqrt())
            assert norm(collision_grads)>0 and norm(base_grads)>0
            assert all(torch.isfinite(g).all() for g in (*collision_grads,*base_grads,pred_grad))
            altered=pred.detach().clone();altered[mask==0]+=.3;assert torch.equal(fixed_raw(altered,cache,scene),raw.detach())
            obj=original_postprocessed(raw,scene);assert_fields(obj,scene);assert_fields(export_axis(scene),scene);maximum=0.
            for old,new in zip(scene.asset['nodes'],obj['diffuse_tree']):
                for key in ['direction','origin']:maximum=max(maximum,float(np.max(abs(np.array(old['joint']['axis'][key])-np.array(new['joint']['axis'][key])))))
            assert maximum<1e-5
            report['head'][case]={'original_MLP_reproduction_error':error,'official_initial_axis_reproduction_error':maximum,'selected_collision_MLP_gradient_norm':norm(collision_grads),'selected_MSE_MLP_gradient_norm':norm(base_grads),'non_selected_raw_output_gradient_max':float(pred_grad[mask==0].abs().max()),'perturbed_non_selected_outputs_have_no_effect':True,'frozen_counterpart_max_change':0.,'max_endpoint_change':0.,'only_selected_GT_target_channels_present':True}
            del scene,model
        report['status']='passed';atomic_json(root/'gradient_and_freeze_checks.json',report)
    except BaseException as exc:report.update(status='failed',error=f'{type(exc).__name__}: {exc}');atomic_json(root/'gradient_and_freeze_checks.json',report);raise
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
