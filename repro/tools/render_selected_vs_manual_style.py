"""Render selected and uniform manual-style automatic geometry side by side.

The two generated columns use the published two virtual cameras and each
asset's own native joint ranges. GT/Origin are decoded from the sealed source
video; their timeline is retained verbatim.
"""
import argparse
import pathlib
import sys

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

REPO=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'repro/tools'))
from add_gt_multicase_videos import SEQ
from auto_part_semantics import sha256, write_json
from collision_refinement_common import transforms
from fit_closed_extrema_geometry import CASES
from fit_observed_drawer_pose import load, read
from fit_single_origin_extrema_travel import source
from multicase_closed_common import CovGaussians, tensor
from modules.pact.renderers.gaussian_render import GaussianRenderer
import utils3d

FONT=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
TITLES=('Selected automatic','New automatic | manual-style fit')


def selected_checkpoint(root):
    if (root/'auto_closed_geometry.pt').exists():
        return 'auto_closed_geometry.pt'
    if (root/'observed_pose_geometry.pt').exists():
        return 'observed_pose_geometry.pt'
    raise FileNotFoundError('Selected geometry checkpoint missing: '+str(root))


def cameras(meta):
    center=np.asarray(meta['cameras']['center'])
    radius=float(meta['cameras']['radius'])
    intr=utils3d.torch.intrinsics_from_fov_xy(tensor(np.deg2rad(43.)),tensor(np.deg2rad(43.)))
    views=[]
    for yaw in (.35,-.7):
        eye=center+radius*np.array([np.sin(yaw)*np.cos(.22),np.sin(.22),np.cos(yaw)*np.cos(.22)])
        views.append(utils3d.torch.extrinsics_look_at(tensor(eye),tensor(center),tensor([0,1,0])))
    return views,intr


class Scene:
    def __init__(self,root,checkpoint,title):
        self.root=root
        self.checkpoint=checkpoint
        self.title=title
        self.model=load(root,checkpoint=checkpoint)
        self.model.requires_grad_(False)
        self.geometry=self.model.geometry()
        self.alpha=torch.cat([p['alpha'] for p in self.model.parts]).contiguous()
        self.colors=torch.cat([p['colors'] for p in self.model.parts]).contiguous()

    @torch.inference_mode()
    def panel(self,progress,view,camera,intr,renderer):
        track=transforms(self.model.groups,progress)
        xs=[];cs=[]
        for part,(points,cov) in zip(self.model.parts,self.geometry):
            T=track[part['group']['id']]
            R=tensor(T[:3,:3])
            xs.append(points@R.T+tensor(T[:3,3]))
            cs.append(R[None]@cov@R.T[None])
        gaussians=CovGaussians(torch.cat(xs).contiguous(),torch.cat(cs).contiguous(),self.alpha)
        rgb=renderer.render(gaussians,camera,intr,colors_overwrite=self.colors)['color'].clamp(0,1)
        im=Image.fromarray((rgb.permute(1,2,0).cpu().numpy()*255).astype(np.uint8))
        draw=ImageDraw.Draw(im)
        draw.rectangle((0,0,383,46),fill='white')
        draw.text((7,4),self.title,font=FONT,fill='black')
        draw.text((7,25),f'View {view+1} | native motion {progress:.2f}',font=FONT,fill='#555555')
        return np.asarray(im)


def render_case(outputs,out,case):
    root=out/case
    root.mkdir(parents=True,exist_ok=True)
    selected=outputs/'closed_auto_geometry_selected_20261104'/case
    experiment=outputs/'closed_auto_manual_style_geometric_pose_20261104_v1'/case
    selected_ckpt=selected_checkpoint(selected)
    experiment_ckpt='manual_style_geometry.pt'
    fit=read(experiment/'manual_style_fit.json')
    assert sha256(experiment/experiment_ckpt)==fit['checkpoint_sha256']
    choice=next(row for row in read(outputs/'closed_auto_geometry_selected_20261104'/'SELECTION.json')['cases'] if row['case']==case)
    assert sha256(selected/selected_ckpt)==choice['checkpoint_sha256']
    oldroot,_=source(case)
    oldvideo=oldroot/f'{case}_GT_Origin_VLMAxis_VLMPoseGeo.mp4'
    oldmeta=read(oldroot/'video_results.json')
    assert sha256(oldvideo)==oldmeta['sha256']
    assert oldmeta['frames']==len(SEQ)==84 and oldmeta['size']==[1536,768]
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    views,intr=cameras(oldmeta)
    renderer=GaussianRenderer(dict(resolution=384,near=.05,far=20.,bg_color=(1.,1.,1.)))
    renderer.pipe.compute_cov3D_python=True
    scenes=[Scene(selected,selected_ckpt,TITLES[0]),Scene(experiment,experiment_ckpt,TITLES[1])]
    four=root/f'{case}_GT_Origin_Selected_NewAuto.mp4'
    two=root/f'{case}_Selected_vs_NewAuto.mp4'
    count=0
    with imageio.get_reader(oldvideo) as reader, \
         imageio.get_writer(four,fps=16,codec='libx264',quality=8) as four_writer, \
         imageio.get_writer(two,fps=16,codec='libx264',quality=8) as two_writer:
        for i,(old,progress) in enumerate(zip(reader,SEQ)):
            assert old.shape==(768,1536,3),(case,i,old.shape)
            comparison=[];all_columns=[]
            for view in range(2):
                panels=[scene.panel(float(progress),view,views[view],intr,renderer) for scene in scenes]
                comparison.append(np.concatenate(panels,axis=1))
                start=view*384;stop=start+384
                all_columns.append(np.concatenate([old[start:stop,:768],*panels],axis=1))
            two_frame=np.concatenate(comparison,axis=0)
            four_frame=np.concatenate(all_columns,axis=0)
            two_writer.append_data(two_frame)
            four_writer.append_data(four_frame)
            count+=1
            if i in (0,41,83):
                Image.fromarray(four_frame).save(root/f'comparison_frame_{i:03d}.png')
            if i%14==0:print('RENDER',case,i,'/',len(SEQ),flush=True)
    assert count==84
    verified={}
    for label,path,shape in [('four',four,(768,1536,3)),('two',two,(768,768,3))]:
        with imageio.get_reader(path) as reader:
            decoded=sum(frame.shape==shape for frame in reader)
            meta=reader.get_meta_data()
        assert decoded==count,(case,label,decoded)
        verified[label]=dict(path=str(path),sha256=sha256(path),frames=decoded,fps=meta['fps'],size=list(meta['size']))
    write_json(root/'video_results.json',dict(case=case,videos=verified,
        columns=['GT','Origin',*TITLES],views=['published yaw +0.35','published yaw -0.70'],
        selected_candidate=choice['selected'],selected_source=str(selected.resolve()),
        selected_checkpoint=str(selected/selected_ckpt),selected_checkpoint_sha256=choice['checkpoint_sha256'],
        new_source=str(experiment),new_checkpoint=str(experiment/experiment_ckpt),
        new_checkpoint_sha256=fit['checkpoint_sha256'],
        source_video=str(oldvideo),source_video_sha256=oldmeta['sha256'],
        camera_policy='Same published center, radius, two yaw/pitch values and 43-degree FOV for both generated models.',
        motion_policy='Both generated models use their own native joint ranges over the same 84 progress samples; fitted closed poses are only for training. GT/Origin columns are decoded from the published source video.',
        GT_timeline_caveat='GT source-to-endpoint motion can differ in physical angle and endpoints from both generated native tracks.'))
    print('VIDEO COMPLETE',case,str(four),str(two),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--outputs',type=pathlib.Path,required=True)
    p.add_argument('--out',type=pathlib.Path,required=True)
    p.add_argument('--case')
    a=p.parse_args()
    outputs=a.outputs.resolve();out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    for case in ([a.case] if a.case else CASES):render_case(outputs,out,case)


if __name__=='__main__':main()
