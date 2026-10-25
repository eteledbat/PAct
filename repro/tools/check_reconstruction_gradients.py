"""Actual-asset gradients, renderer projection, and shared geometry checks."""
import argparse
import json
import pathlib

import numpy as np
import torch

from reconstruction_ablation_common import Reconstruction, RenderGaussians, tensor
from collision_refinement_common import transforms


def norms(model):
    groups={'geometry':list(model.geometry),'direction':[model.joints.theta],'pivot':[model.joints.phi]};out={}
    for name,params in groups.items():
        grads=[p.grad for p in params if p.grad is not None];assert grads and all(torch.isfinite(g).all() for g in grads)
        out[name]=float(torch.sqrt(sum(g.square().sum() for g in grads)))
        assert out[name]>1e-12,(name,out[name])
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--case',required=True);a=p.parse_args();root=pathlib.Path(a.root);config=json.loads((root/'config.json').read_text());torch.set_num_threads(4)
    model=Reconstruction(config['cases'][a.case],config,np.load(root/'data'/a.case/'observations.npz'));saved=torch.load(root/f'prealign_{a.case}/checkpoint.pt',map_location='cuda',weights_only=False);model.load_state_dict(saved['model']);model.configure('train')
    # Compare all hierarchy transforms against previously validated NumPy FK.
    maximum=0.
    for t in [0.,.37,1.]:
        qs=tensor([[n['joint']['range'][0]+t*(n['joint']['range'][1]-n['joint']['range'][0]) for n in model.joints.active]])
        world,_=model.joints.world_q(qs);reference=transforms(model.asset['nodes'],t)
        maximum=max(maximum,max(float(np.abs(world[n['id']][0].detach().cpu().numpy()-reference[n['id']]).max()) for n in model.asset['nodes']))
    assert maximum<3e-6
    # A single Gaussian must project to the analytic camera pixel, ruling out a flip.
    view=0;ext=model.data['extrinsics'][view];intr=model.data['intrinsics'][view];camera_point=tensor([[.3,-.2,3.]]);w2c_inv=torch.linalg.inv(ext);world_point=camera_point@w2c_inv[:3,:3].T+w2c_inv[:3,3]
    g=RenderGaussians(world_point,tensor([[.02,.02,.02]]),tensor([[1.,0,0,0]]),tensor([[.99]]));image=model.renderer.render(g,ext,intr,colors_overwrite=tensor([[0.,0,0.]]))['color'];weight=1-image.mean(0);yy,xx=torch.meshgrid(torch.arange(config['resolution'],device='cuda'),torch.arange(config['resolution'],device='cuda'),indexing='ij');center=tensor([0.,0.]);center[0]=(weight*xx).sum()/weight.sum();center[1]=(weight*yy).sum()/weight.sum();analytic=tensor([(float(intr[0,0])*.3/3+float(intr[0,2]))*config['resolution']-.5,(float(intr[1,1])*(-.2)/3+float(intr[1,2]))*config['resolution']-.5]);projection_error=float((center-analytic).abs().max());assert projection_error<1.,(center,analytic)
    model.zero_grad(set_to_none=True);base,_,_=model.image_loss(0);base.backward();image_gradients=norms(model)
    model.zero_grad(set_to_none=True);qs=model.motion_q('signed90_assumption',21)[0][1][[7,13,20]];collision=model.collision(model.deformed(),qs).mean();collision.backward();collision_gradients=norms(model)
    # The field must move the exported mesh too, rather than just the render splats.
    before=model.deformed_mesh(0).vertices.copy()
    with torch.no_grad():model.geometry[0][:,0]+=1e-4*model.D
    after=model.deformed_mesh(0).vertices;mesh_field_error=float(np.max(np.abs((after-before)-np.array([1e-4*model.D,0,0]))));assert mesh_field_error<1e-7
    result={'case':a.case,'FK_max_abs_error':maximum,'analytic_projection_max_pixel_error':projection_error,'image_loss':float(base),'image_loss_gradient_norms':image_gradients,'motion_loss':float(collision),'motion_loss_gradient_norms':collision_gradients,'mesh_shared_field_translation_error':mesh_field_error,'status':'passed'}
    (root/f'gradient_{a.case}.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
