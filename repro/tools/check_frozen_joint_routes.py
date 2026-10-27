"""Real-data gradient, rigid-FK, proxy parity, and official-decoder checks."""
import argparse
import json
import pathlib

import numpy as np
import torch
import torch.nn.functional as F

from frozen_joint_common import (FrozenScene,atomic_json,continuous_from_raw,
                                 original_postprocessed,paths,tensor)
from collision_refinement_common import transforms
from run_frozen_joint_trial import make_head,mlp


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
    root=pathlib.Path(a.root);cfg=json.loads((root/'config.json').read_text());torch.set_num_threads(4);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False
    report={'status':'running','direct':{},'head':{}}
    try:
        for c in cfg['test_cases']:
            scene=FrozenScene(cfg['runs'][c],cfg['sources_per_part']);p=scene.params
            qs=dict(paths(p.limits0,21))['synchronized']
            with torch.no_grad():
                world=p.world(qs)
                fk=max(float(np.max(np.abs(world[n['id']].cpu().numpy()-np.array([transforms(scene.asset['nodes'],float(t))[n['id']] for t in np.linspace(0,1,21)])))) for n in scene.asset['nodes'])
                scene.objective.times=qs
                proxy=float(scene.objective());new=float(scene.proxy(qs));parity=abs(proxy-new)
                rigid=0.
                for part in scene.asset['parts']:
                    x=tensor(part['xyz'][:20]);tr=world[part['node']['id']];y=torch.einsum('nj,bij->bni',x,tr[:,:3,:3])+tr[:,None,:3,3]
                    rigid=max(rigid,float((torch.cdist(y,y)-torch.cdist(x,x)[None]).abs().max()))
            # Three nonzero points on a synchronized trajectory exercise all joints.
            coll=scene.proxy(qs[[2,10,20]])/scene.normalization
            grads=torch.autograd.grad(coll,[p.theta,p.phi,p.range_delta])
            norms=[float(g.norm()) for g in grads]
            assert all(torch.isfinite(g).all() for g in grads) and min(norms)>0
            assert fk<2e-6 and parity<2e-6 and rigid<2e-6,(fk,parity,rigid)
            assert not any(v.requires_grad for d in scene.objective.data for v in d.values())
            report['direct'][c]={'fk_max_error':fk,'proxy_absolute_error':parity,'rigid_pair_distance_max_error':rigid,'collision_gradient_norms':dict(zip(['axis_direction','pivot','both_endpoints'],norms)),
                                 'start_gradient_norm':float(grads[2][:,0].norm()),'end_gradient_norm':float(grads[2][:,1].norm()),'normalization':scene.normalization}
            # Verify baseline official decoding on frozen GS before comparing edits.
            raw=tensor(np.load(pathlib.Path(cfg['runs'][c])/'joint_trace/raw_head_00.npy'))
            obj=original_postprocessed(raw,scene);original=scene.asset['nodes'];error=0.
            for old,new in zip(original,obj['diffuse_tree']):
                if old['id'] not in p.indices:continue
                for key in ['direction','origin']:
                    if key=='origin' and old['joint']['type']=='prismatic':continue
                    error=max(error,float(np.max(np.abs(np.array(old['joint']['axis'][key])-np.array(new['joint']['axis'][key])))))
                error=max(error,float(np.max(np.abs(np.array(old['joint']['range'])-np.array(new['joint']['range'])))))
            report['direct'][c]['official_postprocess_reproduction_error']=error
            assert error<1e-5,('Official postprocessing reproduction failed',c,error)
            del scene
        for c in cfg['train_cases']:
            cache=torch.load(root/'features'/c/'cache.pt',weights_only=False);head=make_head(cache['mlp_state']);scene=FrozenScene(cfg['runs'][c],cfg['sources_per_part'])
            raw=mlp(head,cache['pooled']['average']);reproduction=float((raw.detach().cpu()-cache['raw']).abs().max())
            scene.params.override=continuous_from_raw(raw,scene)
            q=dict(paths(scene.params.limits(),21))['synchronized'][[2,10,20]]
            collision=scene.proxy(q)/scene.normalization
            output_gradients=torch.autograd.grad(collision,scene.params.override,retain_graph=True)
            gradient=torch.autograd.grad(collision,tuple(head.parameters()),retain_graph=True)
            target=torch.load(root/'training_targets'/f'{c}.pt',weights_only=False,map_location='cuda');base=F.mse_loss(raw*target['mask'],target['target']*target['mask'])
            base_gradient=torch.autograd.grad(base,tuple(head.parameters()))
            norm=lambda gs:float(torch.stack([g.square().sum() for g in gs]).sum().sqrt())
            assert all(torch.isfinite(g).all() for g in gradient+base_gradient)
            assert norm(gradient)>0 and norm(base_gradient)>0 and reproduction<1e-4
            assert all(torch.isfinite(g).all() for g in output_gradients) and min(float(g.norm()) for g in output_gradients)>0
            report['head'][c]={'released_MLP_reproduction_max_error':reproduction,'collision_gradient_norm':norm(gradient),'official_MSE_gradient_norm':norm(base_gradient),'official_MSE':float(base),'target_nonzero_channels':int((target['mask']!=0).sum()),'collision_output_gradient_norms':dict(zip(['direction','pivot','motion_limits'],[float(g.norm()) for g in output_gradients])),'start_trainable':False}
            del scene,head
        report['status']='passed';atomic_json(root/'gradient_and_rigidity_checks.json',report)
    except BaseException as exc:
        report['status']='failed';report['error']=f'{type(exc).__name__}: {exc}';atomic_json(root/'gradient_and_rigidity_checks.json',report);raise
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
