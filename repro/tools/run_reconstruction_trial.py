"""Matched no-collision / motion-collision image reconstruction refinement."""
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
from PIL import Image

from reconstruction_ablation_common import Reconstruction
from collision_refinement_common import sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--case');p.add_argument('--experiment');p.add_argument('--stage',choices=['prealign','train','smoke'],required=True);p.add_argument('--arm',choices=['A','B'],default='A');p.add_argument('--steps',type=int);p.add_argument('--resume',action='store_true');p.add_argument('--weight-override',type=float);p.add_argument('--tag');a=p.parse_args()
    root=pathlib.Path(a.root).resolve();config=json.loads((root/'config.json').read_text());experiment=next((x for x in config['experiments'] if x['name']==a.experiment),None)
    case=experiment['case'] if experiment else a.case;assert case in config['cases']
    label=a.tag or (f'{a.stage}_{case}' if a.stage=='prealign' else f"{a.experiment}_{a.arm}_{a.stage}")
    out=root/label;out.mkdir(exist_ok=a.resume);start=time.monotonic();blocked=[]
    def guard(event,values):
        if event=='open' and isinstance(values[0],(str,bytes)):
            name=os.fsdecode(values[0])
            if any(s in name for s in ['/VideoArtGS/','/FreeArt3D/','/Part2GS-data/','/gt/','joint_evaluation','mobility_v2','part_info.json']):blocked.append(name);raise PermissionError('GT/source 3D input forbidden in reconstruction fitting: '+name)
    sys.addaudithook(guard);random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    data=np.load(root/'data'/case/'observations.npz');model=Reconstruction(config['cases'][case],config,data);weight=0. if a.arm=='A' else config['collision_weight'];weight=weight if a.weight_override is None else a.weight_override
    steps=a.steps or (config['prealign_steps'] if a.stage=='prealign' else config['steps']);history=[];begin=0
    if a.stage=='prealign':
        model.configure('prealign');optimizer=torch.optim.Adam([{'params':[model.log_scale,model.translation],'lr':.005},{'params':[model.input_q],'lr':.7}]);schedule=None;normalization=None
    else:
        assert experiment is not None;shared=torch.load(root/f'prealign_{case}'/'checkpoint.pt',map_location='cuda',weights_only=False);assert shared['step']==config['prealign_steps'];model.load_state_dict(shared['model']);model.configure('train')
        optimizer=torch.optim.Adam([{'params':list(model.geometry),'lr':.0003*model.D},{'params':[model.joints.theta],'lr':.001},{'params':[model.joints.phi],'lr':.0003*model.D}])
        schedule=torch.cat([q for _,q in model.motion_q(experiment['range'],21)])
        with torch.no_grad():normalization=float(model.collision(model.deformed(),schedule[::max(1,len(schedule)//21)]).mean())
        assert np.isfinite(normalization) and normalization>1e-10
    source_paths=[pathlib.Path(__file__),pathlib.Path(__file__).with_name('reconstruction_ablation_common.py'),pathlib.Path(__file__).with_name('refine_collision_axes.py'),pathlib.Path(__file__).with_name('collision_refinement_common.py')]
    protocol={'case':case,'experiment':experiment,'stage':a.stage,'arm':a.arm,'steps':steps,'weight':weight,'normalization':normalization,
              'observation_sha256':sha256(root/'data'/case/'observations.npz'),'seed':42,'source_sha256':{p.name:sha256(p) for p in source_paths},
              'shared_initialization_sha256':None if a.stage=='prealign' else sha256(root/f'prealign_{case}'/'checkpoint.pt'),
              'objective':'RGB weighted L1 + 2*semantic-palette weighted MSE + .001 deformation amplitude + .001 deformation smoothness + .001 normalized axis displacement + lambda*(motion Gaussian proxy / initial proxy)',
              'geometry_trainable':a.stage!='prealign','geometry':'Shared 4x4x4 trilinear vector field for Gaussian centers and mesh vertices, displacement<=.03 initial asset diagonal',
              'frozen':'Gaussian opacity/scales/color, mesh topology, joint types and evaluation ranges; camera similarity and observed-pose q frozen after common image-only registration',
              'motion_sampling':'21 points/path; all single-joint paths plus synchronized path; deterministic cyclic mini-batches of3 states, every predefined state visited before repeats',
              'GT_axis_mesh_targets':False,'checkpoint_selection':'Fixed final step, no GT/best-validation selection'}
    if a.resume:
        assert json.loads((out/'protocol.json').read_text())==protocol;saved=torch.load(out/'checkpoint.pt',map_location='cuda',weights_only=False);model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);history=saved['history'];begin=saved['step']
    else:(out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    def save(state,step,**kwargs):
        status={'state':state,'step':step,'total_steps':steps,'case':case,'experiment':a.experiment,'arm':a.arm,'elapsed_seconds':time.monotonic()-start,'peak_allocated_gpu_MiB':torch.cuda.max_memory_allocated()/2**20,'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**kwargs}
        (out/'status.tmp').write_text(json.dumps(status,indent=2));(out/'status.tmp').replace(out/'status.json')
        torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'history':history,'step':step},out/'checkpoint.tmp');(out/'checkpoint.tmp').replace(out/'checkpoint.pt');(out/'history.json').write_text(json.dumps(history,indent=2))
    def preview(name):
        with torch.no_grad():rgb,seg=model.render(0)
        panels=[model.data['rgb'][0],rgb,model.target_labels[0],seg];array=torch.cat(panels,2).clamp(0,1).permute(1,2,0).cpu().numpy();Image.fromarray((255*array).astype(np.uint8)).save(out/f'{name}.png')
    try:
        if begin==0:preview('initial');save('running',0)
        for step in range(begin,steps):
            optimizer.zero_grad(set_to_none=True);geometry=model.deformed();base,photo,semantic=model.image_loss(step%6,geometry);regularizer=model.regularizer() if a.stage!='prealign' else base*0
            if schedule is not None:
                ids=(torch.arange(3,device='cuda')+step*3)%len(schedule);collision=model.collision(geometry,schedule[ids]).mean();loss=base+regularizer+weight*collision/normalization
            else:collision=base*0;loss=base
            assert torch.isfinite(loss);loss.backward()
            norms={}
            for name,params in [('geometry',model.geometry),('axis_direction',[model.joints.theta]),('pivot',[model.joints.phi])]:
                grads=[p.grad for p in params if p.grad is not None];assert all(torch.isfinite(g).all() for g in grads);norms[name]=float(sum(g.square().sum() for g in grads).sqrt()) if grads else None
            optimizer.step();model.project();row={'step':step+1,'view':step%6,'loss':float(loss),'photo':float(photo),'semantic':float(semantic),'regularizer':float(regularizer),'collision':float(collision),'gradient_norms':norms};history.append(row)
            if (step+1)%10==0 or step+1==steps:save('running',step+1,latest=row);print(label,step+1,'photo',float(photo),'semantic',float(semantic),'collision',float(collision),flush=True)
        preview('final');save('complete',steps,GT_reads_blocked=blocked)
        frozen=[{'path':str(p[key]),'sha256':p[hk]} for p in model.parts for key,hk in [('mesh_path','mesh_sha256'),('gs_path','gs_sha256')]]
        assert all(sha256(x['path'])==x['sha256'] for x in frozen) and not blocked
        (out/'seal.json').write_text(json.dumps({'checkpoint_sha256':sha256(out/'checkpoint.pt'),'GT_reads_blocked':blocked,'original_files_unchanged':True,'frozen_files':frozen,'sources':protocol['source_sha256']},indent=2))
    except BaseException as exc:save('failed',len(history),error=f'{type(exc).__name__}: {exc}',GT_reads_blocked=blocked);raise


if __name__=='__main__':main()
