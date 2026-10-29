"""Shared-camera three-stage animations and native optimized assets per case."""
import argparse
import copy
import json
import os
import pathlib
import subprocess
import sys
import time

import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch

from collision_refinement_common import moved,sha256,transforms
from multicase_closed_common import Model,REPO,CovGaussians,tensor,write_json
from render_closed_image_45194 import export_parts
from modules.pact.renderers.gaussian_render import GaussianRenderer
import utils3d


def render_case(root):
    cfg=json.loads((root/'config.json').read_text());manifest=json.loads((root/'motion_groups.json').read_text());seal=json.loads((root/'fit_results.json').read_text())
    assert sha256(root/'checkpoint.pt')==seal['checkpoint_sha256'] and sha256(root/'motion_groups.json')==seal['manifest_sha256']
    torch.set_num_threads(4);model=Model(cfg,manifest,root)
    checkpoint=torch.load(root/'checkpoint.pt',map_location='cuda',weights_only=False);assert checkpoint['stage']=='complete' and checkpoint['step']==240
    model.load_state_dict(checkpoint['model']);model.requires_grad_(False)
    export=export_parts(model,root)
    methods=['original','prior','optimized'];trees={'original':model.asset['nodes'],'prior':model.groups,'optimized':model.groups}
    geometry={'original':model.geometry(original=True),'prior':model.geometry(original=True),'optimized':model.geometry()}
    renderer=GaussianRenderer({'resolution':384,'near':.05,'far':20.,'bg_color':(1.,1.,1.)});renderer.pipe.compute_cov3D_python=True
    cloud=[]
    for method in methods:
        for t in np.linspace(0,1,9):
            tr=transforms(trees[method],t)
            for part,(xyz,_) in zip(model.parts,geometry[method]):
                gid=part['group']['source_node'] if method=='original' else part['group']['id']
                cloud.append(moved(xyz[::197].cpu().numpy(),tr[gid]))
    cloud=np.concatenate(cloud);center=(cloud.min(0)+cloud.max(0))/2;radius=1.55*np.linalg.norm(np.ptp(cloud,axis=0))
    intr=utils3d.torch.intrinsics_from_fov_xy(tensor(np.deg2rad(43.)),tensor(np.deg2rad(43.)))
    cameras=[]
    for yaw in [.35,-.7]:
        eye=center+radius*np.array([np.sin(yaw)*np.cos(.22),np.sin(.22),np.cos(yaw)*np.cos(.22)])
        cameras.append(utils3d.torch.extrinsics_look_at(tensor(eye),tensor(center),tensor([0,1,0])))
    colors=torch.cat([p['colors'] for p in model.parts]).contiguous();alpha=torch.cat([p['alpha'] for p in model.parts]).contiguous()
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    titles={'original':'Original PAct','prior':'Semantic motion prior','optimized':'Closed-image shape fit'}

    @torch.no_grad()
    def frame(method,t,view=0,diagnostic=False):
        nodes=trees[method]
        if diagnostic:
            nodes=copy.deepcopy(nodes)
            for n in nodes:
                if n['joint']['type'] in ['revolute','continuous']:
                    lo,hi=n['joint']['range'];n['joint']['range']=[lo,lo+np.sign(hi-lo or 1)*90.]
        tr=transforms(nodes,t);xyzs,covs=[],[]
        for p,(x,c) in zip(model.parts,geometry[method]):
            gid=p['group']['source_node'] if method=='original' else p['group']['id'];T=tr[gid];R=tensor(T[:3,:3])
            xyzs.append(x@R.T+tensor(T[:3,3]));covs.append(R[None]@c@R.T[None])
        gs=CovGaussians(torch.cat(xyzs).contiguous(),torch.cat(covs).contiguous(),alpha)
        rgb=renderer.render(gs,cameras[view],intr,colors_overwrite=colors)['color']
        pic=Image.fromarray((255*rgb.clamp(0,1).permute(1,2,0).cpu().numpy()).astype('uint8'));draw=ImageDraw.Draw(pic)
        draw.rectangle((0,0,383,47),fill='white');draw.text((8,6),cfg['case']+' | '+titles[method],fill='black',font=font)
        draw.text((8,27),('ASSUMED 90deg diagnostic' if diagnostic else f'motion fraction {t:.2f}'),fill='#555555',font=font)
        return np.asarray(pic)
    # Fit-view comparison has its own 256px renderer/camera, identical to fitting.
    with torch.no_grad():
        images=[model.rgb]
        for m in methods:images.append(model.render(geometry[m],method='original' if m=='original' else 'prior'))
        panel=torch.cat(images,2).clamp(0,1).permute(1,2,0).cpu().numpy()
        Image.fromarray((255*panel).astype('uint8')).save(root/'closed_target_and_three_versions.png')
    seq=np.r_[np.zeros(6),np.linspace(0,1,33),np.ones(6),np.linspace(1,0,33),np.zeros(6)]
    files={m:root/f'{cfg["case"]}_{m}.mp4' for m in methods};comparison=root/f'{cfg["case"]}_three_versions.mp4'
    writers={m:imageio.get_writer(path,fps=16,codec='libx264',quality=8) for m,path in files.items()}
    cmp=imageio.get_writer(comparison,fps=16,codec='libx264',quality=8);start=time.monotonic()
    try:
        for i,t in enumerate(seq):
            panels={m:[frame(m,float(t),v) for v in [0,1]] for m in methods}
            for m in methods:writers[m].append_data(np.concatenate(panels[m],1))
            panel=np.concatenate([np.concatenate([panels[m][v] for m in methods],1) for v in [0,1]],0);cmp.append_data(panel)
            if i in [6,22,38]:Image.fromarray(panel).save(root/f'three_versions_t{t:.2f}.png')
            if i%28==0:print('RENDER',cfg['case'],i,flush=True)
    finally:
        cmp.close()
        for w in writers.values():w.close()
    paths=[comparison,*files.values()]
    spans=[abs(np.diff(g['joint']['range'])[0]) for g in model.groups if g['joint']['type']=='revolute']
    if spans and min(spans)<10:
        diag=root/f'{cfg["case"]}_assumed90_diagnostic.mp4'
        with imageio.get_writer(diag,fps=12,codec='libx264',quality=8) as w:
            for t in np.r_[np.linspace(0,1,25),np.linspace(1,0,25)]:w.append_data(np.concatenate([frame(m,float(t),diagnostic=True) for m in methods],1))
        paths.append(diag)
    videos=[]
    for path in paths:
        with imageio.get_reader(path) as r:
            n=r.count_frames();meta=r.get_meta_data();r.get_data(0);r.get_data(n-1)
        videos.append({'file':path.name,'frames':n,'fps':meta['fps'],'size':list(meta['size']),'sha256':sha256(path)})
    maxrigid=0.
    for p,(x,_) in zip(model.parts,geometry['optimized']):
        x=x[::199].cpu().numpy().astype(np.float64);d=np.linalg.norm(x-x[0],axis=1)
        for t in np.linspace(0,1,41):
            y=moved(x,transforms(model.groups,t)[p['group']['id']]);maxrigid=max(maxrigid,float(abs(np.linalg.norm(y-y[0],axis=1)-d).max()))
    assert maxrigid<1e-12 and all(sha256(f['path'])==f['sha256'] for f in manifest['geometry_files'])
    write_json(root/'render_results.json',{'videos':videos,'shared_camera_center':center.tolist(),'shared_camera_radius':float(radius),
               'methods':methods,'same_prior_and_optimized_motion':True,'max_rigid_error':maxrigid,
               'native_export':export,'seconds':time.monotonic()-start,
               'range_note':'Original retains released interval. Prior and optimized use geometric closed pose plus original absolute span; newly unlocked fixed fragments explicitly assume 90deg.'})
    print('RENDER COMPLETE',cfg['case'],flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--case');a=p.parse_args();root=pathlib.Path(a.root).resolve()
    if a.case:render_case(root/a.case);return
    protocol=json.loads((root/'protocol.json').read_text());states={}
    for case in protocol['cases']:
        out=root/case
        if not (out/'fit_results.json').exists():states[case]='fit_failed';continue
        if (out/'render_results.json').exists():states[case]='complete';continue
        env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4')
        with (out/'render.log').open('w') as f:
            proc=subprocess.run([sys.executable,__file__,'--root',str(root),'--case',case],stdout=f,stderr=subprocess.STDOUT,env=env,cwd=REPO)
        states[case]='complete' if proc.returncode==0 else 'failed'
        write_json(root/'render_status.json',states);print('CASE RENDER',case,states[case],flush=True)


if __name__=='__main__':main()
