"""Append two-frame travel to the exact old four-column single-image video."""
import argparse
import json
import pathlib
import sys

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont

REPO=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'repro/tools'))
from auto_part_semantics import sha256,write_json
from fit_observed_drawer_pose import load,read
from fit_single_origin_extrema_travel import source
from collision_refinement_common import transforms
from multicase_closed_common import CovGaussians,tensor
from add_gt_multicase_videos import SEQ
from modules.pact.renderers.gaussian_render import GaussianRenderer
import utils3d

FONT=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
TITLE='Single-image geometry | two-frame travel'


def render(batch,case):
    root=batch/case;fit=read(root/'travel_fit.json')
    oldroot,modelroot=source(case)
    oldvideo=pathlib.Path(fit['source_four_panel'])
    assert sha256(oldvideo)==fit['source_four_panel_sha256']
    assert sha256(modelroot/'observed_pose_geometry.pt')==fit['source_checkpoint_sha256']
    assert sha256(root/'travel_motion_groups.json')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    model=load(modelroot,checkpoint='observed_pose_geometry.pt');model.requires_grad_(False)
    tree=read(root/'travel_motion_groups.json')['groups']
    assert len(tree)==len(model.groups)
    oldmeta=read(oldroot/'video_results.json')
    assert oldmeta['size']==[1536,768] and oldmeta['frames']==84
    assert oldmeta['sha256']==fit['source_four_panel_sha256']
    center=np.asarray(oldmeta['cameras']['center']);radius=float(oldmeta['cameras']['radius'])
    intr=utils3d.torch.intrinsics_from_fov_xy(tensor(np.deg2rad(43.)),tensor(np.deg2rad(43.)))
    cameras=[]
    for yaw in [.35,-.7]:
        eye=center+radius*np.array([np.sin(yaw)*np.cos(.22),np.sin(.22),np.cos(yaw)*np.cos(.22)])
        cameras.append(utils3d.torch.extrinsics_look_at(tensor(eye),tensor(center),tensor([0,1,0])))
    renderer=GaussianRenderer(dict(resolution=384,near=.05,far=20.,bg_color=(1.,1.,1.)))
    renderer.pipe.compute_cov3D_python=True
    geometry=model.geometry()
    alpha=torch.cat([p['alpha'] for p in model.parts]).contiguous()
    colors=torch.cat([p['colors'] for p in model.parts]).contiguous()

    @torch.inference_mode()
    def prediction(progress,view):
        transforms_now=transforms(tree,progress)
        positions=[];covariances=[]
        for part,(points,cov) in zip(model.parts,geometry):
            T=transforms_now[part['group']['id']];R=tensor(T[:3,:3])
            positions.append(points@R.T+tensor(T[:3,3]))
            covariances.append(R[None]@cov@R.T[None])
        gaussians=CovGaussians(torch.cat(positions).contiguous(),torch.cat(covariances).contiguous(),alpha)
        rgb=renderer.render(gaussians,cameras[view],intr,colors_overwrite=colors)['color'].clamp(0,1)
        image=Image.fromarray((rgb.permute(1,2,0).cpu().numpy()*255).astype('uint8'))
        d=ImageDraw.Draw(image);d.rectangle((0,0,383,47),fill='white')
        d.text((7,4),TITLE,font=FONT,fill='black')
        d.text((7,26),f'frames {fit["frames"][0]} -> {fit["frames"][1]} | {progress:.2f}',font=FONT,fill='#555555')
        return np.asarray(image)

    path=root/f'{case}_GT_Origin_VLMAxis_VLMPoseGeo_ExtremaTravel.mp4'
    count=0
    with imageio.get_reader(oldvideo) as reader,imageio.get_writer(path,fps=16,codec='libx264',quality=8) as writer:
        for i,(original,t) in enumerate(zip(reader,SEQ)):
            assert original.shape==(768,1536,3)
            upper=np.concatenate([original[:384],prediction(float(t),0)],axis=1)
            lower=np.concatenate([original[384:],prediction(float(t),1)],axis=1)
            output=np.concatenate([upper,lower],axis=0)
            writer.append_data(output);count+=1
            if i in (6,22,38):Image.fromarray(output).save(root/f'five_versions_t{float(t):.2f}.png')
            if i%28==0:print('RENDER',case,i,'/',len(SEQ),flush=True)
    assert count==84
    with imageio.get_reader(path) as reader:
        actual=sum(frame.shape==(768,1920,3) for frame in reader)
        meta=reader.get_meta_data()
    assert actual==84
    write_json(root/'five_video_results.json',dict(video=path.name,sha256=sha256(path),frames=actual,
        fps=meta['fps'],size=list(meta['size']),all_frames_decoded=True,
        columns=[*oldmeta['versions'],TITLE],source_four_panel=str(oldvideo),
        source_four_panel_sha256=fit['source_four_panel_sha256'],
        camera_policy='The fifth column uses the exact center, radius, yaw, pitch, FOV and two view rows from the existing four-column video; original four columns are decoded from that video and inserted unchanged before final encode.',
        chronology_caveat='Original GT column retains its existing source-to-old-endpoint timeline. New fifth column interpolates poses estimated at the two selected extreme images. Progress fractions are shared but physical endpoints differ for most VideoArtGS cases.',
        change='Numeric travel ranges only; same published single-image axis and optimized geometry.'))
    print('VIDEO COMPLETE',path,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--case',required=True);a=p.parse_args()
    render(a.root.resolve(),a.case)
