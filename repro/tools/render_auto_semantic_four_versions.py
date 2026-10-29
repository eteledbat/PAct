"""GT | native PAct | automatic axes | automatic axes plus input-frame fit.

All four columns use the same two cameras and begin at the source conditioning
pose. GT travels from that pose to the farther annotated joint endpoint; model
columns retain their own predicted ranges. No GT information edits the models.
"""
import argparse
import json
import pathlib
import time

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from add_gt_multicase_videos import GT, MeshExtractResult, MeshRenderer
from collision_refinement_common import moved, sha256, transforms
from evaluate_freeart_partnet_case import gt_parts
from fit_auto_semantic_initial_frame import AutoFitModel
from multicase_closed_common import CovGaussians, tensor, write_json
from modules.pact.renderers.gaussian_render import GaussianRenderer
import utils3d

SEQ = np.r_[np.zeros(6), np.linspace(0,1,33), np.ones(6), np.linspace(1,0,33), np.zeros(6)]
TITLES = ['GT | source to endpoint', 'Origin | original range',
          'Auto axis | predicted range', 'Auto axis + frame fit']
FONT = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)


def read(path):
    return json.loads(path.read_text())


def gt_at_source(root):
    gt = GT(root)
    audit = []
    if root.name == '20411':
        states, delta = gt_parts(pathlib.Path(read(root/'config.json')['source_scene']))
        gt.parts = [states[5][0], states[5][1]]
        gt.delta = -delta
        audit.append(dict(type='prismatic', source_state='05', endpoint='00', displacement=(-delta).tolist()))
    else:
        for record in gt.records:
            initial = float(record['mesh_frame0_value'])
            candidates = [record['reference_endpoint_a'],record['open_endpoint_b']]
            differences = []
            for endpoint in candidates:
                difference = float(endpoint-initial)
                if record['type'] == 'revolute':
                    difference = float((difference+np.pi)%(2*np.pi)-np.pi)
                differences.append(difference)
            selected = int(np.argmax(np.abs(differences)))
            record['source_annotated_endpoints'] = candidates
            record['source_condition_value'] = initial
            record['selected_endpoint_index'] = selected
            record['reference_endpoint_a'] = initial
            record['open_endpoint_b'] = initial+differences[selected]
            audit.append(dict(part=record['part'],type=record['type'],source_value=initial,
                              annotated_endpoints=candidates,selected_endpoint_index=selected,
                              displayed_delta=differences[selected]))
    return gt,audit


def render_case(root):
    started = time.monotonic()
    cfg = read(root/'config.json')
    manifest = read(root/'motion_groups.json')
    fit = read(root/'fit_results.json')
    evaluation = read(root/'evaluation.json')
    assert sha256(root/'checkpoint.pt') == fit['checkpoint_sha256']
    assert sha256(root/'motion_groups.json') == fit['manifest_sha256']
    assert evaluation['source_files_unchanged']
    torch.set_num_threads(4)
    model = AutoFitModel(cfg,manifest,root)
    model.load_state_dict(torch.load(root/'checkpoint.pt',map_location='cuda',weights_only=False)['model'])
    model.requires_grad_(False)
    gt,gt_motion = gt_at_source(root)
    methods = ['original','prior','optimized']
    trees = dict(original=model.asset['nodes'],prior=model.groups,optimized=model.groups)
    geometry = dict(original=model.geometry(original=True),prior=model.geometry(original=True),optimized=model.geometry())
    points = []
    for method in methods:
        for t in [0.,.5,1.]:
            motion = transforms(trees[method],t)
            for part,(xyz,_) in zip(model.parts,geometry[method]):
                gid = part['group']['source_node'] if method == 'original' else part['group']['id']
                points.append(moved(xyz[::max(1,len(xyz)//1000)].cpu().numpy(),motion[gid]))
    for t in [0.,.5,1.]:
        v = gt.vertices(t)
        points.append(v[::max(1,len(v)//3000)])
    cloud = np.concatenate(points)
    center = (cloud.min(0)+cloud.max(0))/2
    radius = 1.75*np.linalg.norm(np.ptp(cloud,axis=0))
    intr = utils3d.torch.intrinsics_from_fov_xy(tensor(np.deg2rad(43.)),tensor(np.deg2rad(43.)))
    cameras = []
    for yaw in [.35,-.7]:
        eye = center+radius*np.array([np.sin(yaw)*np.cos(.22),np.sin(.22),np.cos(yaw)*np.cos(.22)])
        cameras.append(utils3d.torch.extrinsics_look_at(tensor(eye),tensor(center),tensor([0,1,0])))
    pred_renderer = GaussianRenderer({'resolution':384,'near':.05,'far':20.,'bg_color':(1.,1.,1.)})
    pred_renderer.pipe.compute_cov3D_python = True
    gt_renderer = MeshRenderer({'resolution':384,'near':.05,'far':20.,'ssaa':2})
    colors = torch.cat([p['colors'] for p in model.parts]).contiguous()
    alpha = torch.cat([p['alpha'] for p in model.parts]).contiguous()

    def heading(pixels,index,t):
        image=Image.fromarray(pixels)
        draw=ImageDraw.Draw(image)
        draw.rectangle((0,0,383,47),fill='white')
        draw.text((8,5),TITLES[index],font=FONT,fill='black')
        draw.text((8,27),f'{cfg["case"]} | progress {t:.2f}',font=FONT,fill='#555555')
        return np.asarray(image)

    @torch.inference_mode()
    def gt_frame(t,view):
        vertices = tensor(gt.vertices(t))
        mesh = MeshExtractResult(vertices,gt.faces,gt.attrs)
        out = gt_renderer.render(mesh,cameras[view],intr,return_types=['color','mask','normal'])
        shade = .65+.35*abs(2*out['normal'][2:3]-1)
        rgb = (out['color']*shade+1-out['mask'].unsqueeze(0)).clamp(0,1)
        return heading((255*rgb.permute(1,2,0).cpu().numpy()).astype('uint8'),0,t)

    @torch.inference_mode()
    def pred_frame(method,t,view):
        tr = transforms(trees[method],t)
        xyzs=[];covs=[]
        for part,(xyz,cov) in zip(model.parts,geometry[method]):
            gid = part['group']['source_node'] if method == 'original' else part['group']['id']
            T=tr[gid];R=tensor(T[:3,:3]);shift=tensor(T[:3,3])
            xyzs.append(xyz@R.T+shift)
            covs.append(R[None]@cov@R.T[None])
        gs=CovGaussians(torch.cat(xyzs).contiguous(),torch.cat(covs).contiguous(),alpha)
        rgb=pred_renderer.render(gs,cameras[view],intr,colors_overwrite=colors)['color']
        return heading((255*rgb.clamp(0,1).permute(1,2,0).cpu().numpy()).astype('uint8'),methods.index(method)+1,t)

    path=root/f'{cfg["case"]}_GT_Origin_AutoAxis_FrameFit.mp4'
    snapshots=[]
    with imageio.get_writer(path,fps=16,codec='libx264',quality=8) as writer:
        for i,t in enumerate(SEQ):
            rows=[]
            for view in range(2):
                frames=[gt_frame(float(t),view)]+[pred_frame(m,float(t),view) for m in methods]
                rows.append(np.concatenate(frames,axis=1))
            panel=np.concatenate(rows,axis=0)
            writer.append_data(panel)
            if i in [6,22,38]:
                still=root/f'four_versions_auto_t{float(t):.2f}.png'
                Image.fromarray(panel).save(still)
                snapshots.append(still.name)
            if i%28==0:print('RENDER',cfg['case'],i,'/',len(SEQ),flush=True)
    with imageio.get_reader(path) as video:
        count=video.count_frames();meta=video.get_meta_data()
        assert count==len(SEQ)
        for i in range(count):
            frame=video.get_data(i)
            assert frame.shape[:2]==(768,1536)
    assert all(sha256(f['path'])==f['sha256'] for f in manifest['geometry_files'])
    write_json(root/'auto_four_video_results.json',dict(case=cfg['case'],video=path.name,
        sha256=sha256(path),frames=count,fps=meta['fps'],size=list(meta['size']),
        all_frames_decoded=True,snapshots=snapshots,GT_source_to_endpoint=gt_motion,
        predicted_range_policy='Native Origin and sealed automatic-axis ranges, no GT angle replacement',
        camera_center=center.tolist(),camera_radius=float(radius),two_common_views=True,
        GT_source_files={str(p):sha256(p) for p in gt.source_paths},
        fit_seal_sha256=sha256(root/'fit_results.json'),evaluation_sha256=sha256(root/'evaluation.json'),
        seconds=time.monotonic()-started))
    print('COMPLETE',cfg['case'],path,flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=pathlib.Path,required=True)
    p.add_argument('--case')
    a=p.parse_args();root=a.root.resolve()
    cases=[a.case] if a.case else read(root/'protocol.json')['cases']
    for case in cases:
        if (root/case/'auto_four_video_results.json').exists():continue
        render_case(root/case)


if __name__=='__main__':main()
