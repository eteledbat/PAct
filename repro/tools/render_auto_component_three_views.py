"""Render GT | original PAct | automatically matched component axes, three cameras."""
import argparse
import json
import pathlib
import time

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from add_gt_multicase_videos import MeshExtractResult, MeshRenderer
from collision_refinement_common import moved, sha256, transforms
from multicase_closed_common import CovGaussians, Model, tensor, write_json
from render_auto_semantic_four_versions import gt_at_source
from modules.pact.renderers.gaussian_render import GaussianRenderer
import utils3d


SEQ = np.r_[np.zeros(3), np.linspace(0, 1, 13), np.ones(3), np.linspace(1, 0, 13)]
SIZE = 320
FONT = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)
TITLES = ('GT', 'Origin', 'Automatic component axes')
VIEWS = (('left oblique', -.65), ('front oblique', .35), ('right oblique', 1.10))


def read(path):
    return json.loads(path.read_text())


def render_case(root, reference_root):
    started = time.monotonic()
    cfg = read(root/'config.json')
    manifest = read(root/'motion_groups.json')
    assert sha256(root/'membership.npz') == manifest['membership_sha256']
    assert all(sha256(f['path']) == f['sha256'] for f in manifest['geometry_files'])
    torch.set_num_threads(4)
    model = Model(cfg, manifest, root)
    model.requires_grad_(False)
    # The GT alignment was frozen earlier from raw generated static-body bounds.
    # It is read only after candidate joints are sealed, exclusively for render.
    gt, gt_motion = gt_at_source(reference_root)
    raw_geometry = model.geometry(original=True)
    trees = {'origin': model.asset['nodes'], 'axis': model.groups}
    points = []
    for method in trees:
        for t in (0., .5, 1.):
            motion = transforms(trees[method], t)
            for part, (xyz, _) in zip(model.parts, raw_geometry):
                gid = part['group']['source_node'] if method == 'origin' else part['group']['id']
                sample = xyz[::max(1, len(xyz)//500)].detach().cpu().numpy()
                points.append(moved(sample, motion[gid]))
    for t in (0., .5, 1.):
        v = gt.vertices(t)
        points.append(v[::max(1, len(v)//3000)])
    cloud = np.concatenate(points)
    center = (cloud.min(0)+cloud.max(0))/2
    radius = 1.8*np.linalg.norm(np.ptp(cloud, axis=0))
    intr = utils3d.torch.intrinsics_from_fov_xy(tensor(np.deg2rad(43.)), tensor(np.deg2rad(43.)))
    cameras = []
    for _, yaw in VIEWS:
        eye = center + radius*np.array([np.sin(yaw)*np.cos(.22), np.sin(.22), np.cos(yaw)*np.cos(.22)])
        cameras.append(utils3d.torch.extrinsics_look_at(tensor(eye), tensor(center), tensor([0, 1, 0])))
    gr = GaussianRenderer({'resolution': SIZE, 'near': .05, 'far': 20., 'bg_color': (1., 1., 1.)})
    gr.pipe.compute_cov3D_python = True
    mr = MeshRenderer({'resolution': SIZE, 'near': .05, 'far': 20., 'ssaa': 2})
    color = torch.cat([p['colors'] for p in model.parts]).contiguous()
    alpha = torch.cat([p['alpha'] for p in model.parts]).contiguous()

    def label(rgb, column, view, t):
        pic = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pic)
        draw.rectangle((0, 0, SIZE-1, 43), fill='white')
        draw.text((7, 4), TITLES[column], font=FONT, fill='black')
        draw.text((7, 23), f'{cfg["case"]} | {view} | {t:.2f}', font=FONT, fill='#555555')
        return np.asarray(pic)

    @torch.inference_mode()
    def gt_frame(t, view):
        mesh = MeshExtractResult(tensor(gt.vertices(t)), gt.faces, gt.attrs)
        out = mr.render(mesh, cameras[view], intr, return_types=['color', 'mask', 'normal'])
        shade = .65+.35*abs(2*out['normal'][2:3]-1)
        rgb = (out['color']*shade+1-out['mask'].unsqueeze(0)).clamp(0, 1)
        return label((255*rgb.permute(1, 2, 0).cpu().numpy()).astype('uint8'), 0, VIEWS[view][0], t)

    @torch.inference_mode()
    def pred_frame(method, t, view):
        tr = transforms(trees[method], t)
        xyzs, covs = [], []
        for part, (xyz, cov) in zip(model.parts, raw_geometry):
            gid = part['group']['source_node'] if method == 'origin' else part['group']['id']
            T = tr[gid]
            R = tensor(T[:3, :3]); shift = tensor(T[:3, 3])
            xyzs.append(xyz@R.T+shift)
            covs.append(R[None]@cov@R.T[None])
        gs = CovGaussians(torch.cat(xyzs).contiguous(), torch.cat(covs).contiguous(), alpha)
        rgb = gr.render(gs, cameras[view], intr, colors_overwrite=color)['color']
        return label((255*rgb.clamp(0, 1).permute(1, 2, 0).cpu().numpy()).astype('uint8'),
                     1 if method == 'origin' else 2, VIEWS[view][0], t)

    path = root/f'{cfg["case"]}_GT_Origin_AutoComponentAxis_3views.mp4'
    with imageio.get_writer(path, fps=8, codec='libx264', quality=8) as writer:
        for i, value in enumerate(SEQ):
            t = float(value)
            panel = np.concatenate([
                np.concatenate([gt_frame(t, view), pred_frame('origin', t, view),
                                pred_frame('axis', t, view)], axis=1)
                for view in range(3)], axis=0)
            writer.append_data(panel)
            if i in (3, 9, 15):
                Image.fromarray(panel).save(root/f'three_views_t{t:.2f}.png')
            if i % 8 == 0:
                print('RENDER', cfg['case'], i, '/', len(SEQ), flush=True)
    with imageio.get_reader(path) as video:
        count = video.count_frames()
        meta = video.get_meta_data()
        assert count == len(SEQ)
        for i in range(count):
            assert video.get_data(i).shape[:2] == (SIZE*3, SIZE*3)
    write_json(root/'video_results.json', dict(case=cfg['case'], video=path.name,
        sha256=sha256(path), frames=count, fps=meta['fps'], size=list(meta['size']),
        all_frames_decoded=True, views=[x[0] for x in VIEWS], columns=list(TITLES),
        GT_source_to_endpoint=gt_motion,
        predicted_range_policy='Origin retains native range; automatic axes retain compatible native range or marked assumed range; GT range is not copied',
        camera_center=center.tolist(), camera_radius=float(radius),
        source_files_unchanged=True, GT_alignment_source=str(reference_root/'evaluation.json'),
        seconds=time.monotonic()-started))
    print('COMPLETE', cfg['case'], path, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=pathlib.Path, required=True)
    p.add_argument('--reference-root', type=pathlib.Path, required=True)
    p.add_argument('--case')
    a = p.parse_args()
    cases = [a.case] if a.case else read(a.root/'protocol.json')['cases']
    for case in cases:
        root = a.root/case
        if not (root/'video_results.json').exists():
            render_case(root, a.reference_root/case)


if __name__ == '__main__':
    main()
