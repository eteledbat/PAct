"""GT | published Origin | published axis-only | newly closed-frame geometry.

The first three columns are decoded verbatim from the published single-image
four-panel video. The fourth uses the exact same two virtual cameras and native
axis range. This isolates the new closed-frame pose/geometry fit visually.
"""
import argparse,pathlib,sys
import imageio.v2 as imageio,numpy as np,torch
from PIL import Image,ImageDraw,ImageFont
REPO=pathlib.Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO/'repro/tools'))
from auto_part_semantics import sha256,write_json
from fit_single_origin_extrema_travel import source
from fit_observed_drawer_pose import load,read
from collision_refinement_common import transforms
from multicase_closed_common import CovGaussians,tensor
from add_gt_multicase_videos import SEQ
from modules.pact.renderers.gaussian_render import GaussianRenderer
import utils3d
FONT=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
TITLE='VLM axis + closed-frame pose/geometry'

def render(batch,case):
    root=batch/case;fit=read(root/'fit_seal.json');oldroot,_=source(case)
    oldvideo=oldroot/f'{case}_GT_Origin_VLMAxis_VLMPoseGeo.mp4';oldmeta=read(oldroot/'video_results.json')
    assert sha256(oldvideo)==oldmeta['sha256']
    assert sha256(root/'motion_groups.json')==fit['frozen_axis_sha256']
    assert sha256(root/'observed_pose_geometry.pt')==fit['geometry_checkpoint_sha256']
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    model=load(root,checkpoint='observed_pose_geometry.pt');model.requires_grad_(False)
    center=np.asarray(oldmeta['cameras']['center']);radius=float(oldmeta['cameras']['radius'])
    intr=utils3d.torch.intrinsics_from_fov_xy(tensor(np.deg2rad(43.)),tensor(np.deg2rad(43.)))
    cameras=[]
    for yaw in [.35,-.7]:
        eye=center+radius*np.array([np.sin(yaw)*np.cos(.22),np.sin(.22),np.cos(yaw)*np.cos(.22)])
        cameras.append(utils3d.torch.extrinsics_look_at(tensor(eye),tensor(center),tensor([0,1,0])))
    renderer=GaussianRenderer(dict(resolution=384,near=.05,far=20.,bg_color=(1.,1.,1.)))
    renderer.pipe.compute_cov3D_python=True
    geom=model.geometry();alpha=torch.cat([p['alpha'] for p in model.parts]).contiguous()
    colors=torch.cat([p['colors'] for p in model.parts]).contiguous()
    @torch.inference_mode()
    def panel(t,view):
        tr=transforms(model.groups,t);xs=[];cs=[]
        for p,(x,c) in zip(model.parts,geom):
            T=tr[p['group']['id']];R=tensor(T[:3,:3]);xs.append(x@R.T+tensor(T[:3,3]));cs.append(R[None]@c@R.T[None])
        gs=CovGaussians(torch.cat(xs).contiguous(),torch.cat(cs).contiguous(),alpha)
        rgb=renderer.render(gs,cameras[view],intr,colors_overwrite=colors)['color'].clamp(0,1)
        im=Image.fromarray((rgb.permute(1,2,0).cpu().numpy()*255).astype(np.uint8));d=ImageDraw.Draw(im)
        d.rectangle((0,0,383,47),fill='white');d.text((7,4),TITLE,font=FONT,fill='black')
        d.text((7,26),f'closed reference frame {fit["closed_frame"]} | {t:.2f}',font=FONT,fill='#555555')
        return np.asarray(im)
    path=root/f'{case}_GT_Origin_VLMAxis_ClosedGeometry.mp4';count=0
    with imageio.get_reader(oldvideo) as reader,imageio.get_writer(path,fps=16,codec='libx264',quality=8) as writer:
        for i,(old,t) in enumerate(zip(reader,SEQ)):
            assert old.shape==(768,1536,3)
            out=np.concatenate([np.concatenate([old[:384,:1152],panel(float(t),0)],axis=1),
                                np.concatenate([old[384:,:1152],panel(float(t),1)],axis=1)],axis=0)
            writer.append_data(out);count+=1
            if i in (6,22,38):Image.fromarray(out).save(root/f'four_versions_t{float(t):.2f}.png')
            if i%28==0:print('RENDER',case,i,'/',len(SEQ),flush=True)
    assert count==84
    with imageio.get_reader(path) as reader:
        actual=sum(frame.shape==(768,1536,3) for frame in reader);meta=reader.get_meta_data()
    assert actual==84
    write_json(root/'video_results.json',dict(video=str(path),sha256=sha256(path),frames=actual,fps=meta['fps'],
        size=list(meta['size']),all_frames_decoded=True,columns=['GT','Origin','VLM axis only','VLM axis + closed-frame geometry'],
        original_video=str(oldvideo),original_video_sha256=sha256(oldvideo),
        camera_policy='Published first three columns reused exactly before MP4 re-encode; fourth rendered with published center/radius/yaw/pitch/FOV.',
        motion_policy='All generated columns keep published native numeric ranges; closed-frame observed pose calibrates geometry only.',
        GT_timeline_caveat='Published GT timeline is source to its selected endpoint, which may differ from the closed travel-reference frame.'))
    print('VIDEO COMPLETE',case,path,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--case',required=True);a=p.parse_args();render(a.root.resolve(),a.case)
