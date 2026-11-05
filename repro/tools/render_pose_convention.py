"""Render selected automatic candidates over the unified reference-pose span.

Each movable joint is played from its fitted observed closed pose to the native
open extreme, so the rendered sequence always contains the state used for
geometry fitting and CD evaluation.  This is the visual half of
``pose_convention.py``; it changes no sealed checkpoint or joint manifest.

The native range is replaced only in the in-memory copy used for rendering.
"""
import argparse
import copy
import pathlib
import sys

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'repro/tools'))
from add_gt_multicase_videos import SEQ
from auto_part_semantics import sha256, write_json
from collision_refinement_common import transforms
from fit_auto_closed_geometry import CASES, read
from fit_observed_drawer_pose import load
from fit_single_origin_extrema_travel import source
from multicase_closed_common import CovGaussians, tensor
from modules.pact.renderers.gaussian_render import GaussianRenderer
import utils3d

FONT = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)
DEFAULT_CONVENTION = 'pose_convention_20261105/pose_convention.json'


def selected_checkpoint(root):
    if (root / 'auto_closed_geometry.pt').exists():
        return 'auto_closed_geometry.pt'
    if (root / 'observed_pose_geometry.pt').exists():
        return 'observed_pose_geometry.pt'
    raise FileNotFoundError('Selected geometry checkpoint missing: ' + str(root))


def cameras(meta):
    center = np.asarray(meta['cameras']['center'])
    radius = float(meta['cameras']['radius'])
    intr = utils3d.torch.intrinsics_from_fov_xy(tensor(np.deg2rad(43.)), tensor(np.deg2rad(43.)))
    views = []
    for yaw in (.35, -.7):
        eye = center + radius * np.array([np.sin(yaw) * np.cos(.22), np.sin(.22), np.cos(yaw) * np.cos(.22)])
        views.append(utils3d.torch.extrinsics_look_at(tensor(eye), tensor(center), tensor([0, 1, 0])))
    return views, intr


def playback_groups(groups, joints):
    """Return a copy of the joint tree whose ranges encode the convention."""
    record = {j['group']: j for j in joints}
    out = copy.deepcopy(groups)
    for g in out:
        rec = record.get(g['id'])
        if rec is not None and rec.get('status') == 'resolved':
            g['joint']['range'] = [float(rec['playback'][0]), float(rec['playback'][1])]
    return out


class Scene:
    def __init__(self, root, checkpoint, title):
        self.root = root
        self.title = title
        self.model = load(root, checkpoint=checkpoint)
        self.model.requires_grad_(False)
        self.geometry = self.model.geometry()
        self.alpha = torch.cat([p['alpha'] for p in self.model.parts]).contiguous()
        self.colors = torch.cat([p['colors'] for p in self.model.parts]).contiguous()
        self.playback = copy.deepcopy(self.model.groups)

    def set_convention(self, joints):
        self.playback = playback_groups(self.playback, joints)

    @torch.inference_mode()
    def panel(self, progress, view, camera, intr, renderer):
        track = transforms(self.playback, progress)
        xs = []
        cs = []
        for part, (points, cov) in zip(self.model.parts, self.geometry):
            T = track[part['group']['id']]
            R = tensor(T[:3, :3])
            xs.append(points @ R.T + tensor(T[:3, 3]))
            cs.append(R[None] @ cov @ R.T[None])
        gaussians = CovGaussians(torch.cat(xs).contiguous(), torch.cat(cs).contiguous(), self.alpha)
        rgb = renderer.render(gaussians, camera, intr, colors_overwrite=self.colors)['color'].clamp(0, 1)
        im = Image.fromarray((rgb.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
        draw = ImageDraw.Draw(im)
        draw.rectangle((0, 0, 383, 46), fill='white')
        draw.text((7, 4), self.title, font=FONT, fill='black')
        draw.text((7, 25), f'View {view + 1} | convention progress {progress:.2f}', font=FONT, fill='#555555')
        return np.asarray(im)


def render_case(outputs, convention, out, case):
    row = convention[case]
    root = pathlib.Path(row['selected_root'])
    checkpoint = selected_checkpoint(root)
    oldroot, _ = source(case)
    oldmeta = read(oldroot / 'video_results.json')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    views, intr = cameras(oldmeta)
    renderer = GaussianRenderer(dict(resolution=384, near=.05, far=20., bg_color=(1., 1., 1.)))
    renderer.pipe.compute_cov3D_python = True
    scene = Scene(root, checkpoint, f"{case} | {row['selected_candidate']}")
    scene.set_convention(row['joints'])
    dest = out / case
    dest.mkdir(parents=True, exist_ok=True)
    video = dest / f'{case}_pose_convention.mp4'
    frames = []
    with imageio.get_writer(video, fps=16, codec='libx264', quality=8) as writer:
        for progress in SEQ:
            panels = [scene.panel(float(progress), v, views[v], intr, renderer) for v in range(2)]
            frame = np.concatenate(panels, axis=1)
            writer.append_data(frame)
            frames.append(frame)
    Image.fromarray(frames[len(frames) // 2]).save(dest / 'middle_frame.png')
    return {
        'case': case,
        'selected_candidate': row['selected_candidate'],
        'checkpoint_sha256': sha256(root / checkpoint),
        'video': str(video),
        'video_sha256': sha256(video),
        'frames': len(frames),
        'fps': 16,
        'size': [768, 768],
        'pose_source': row['pose_source'],
        'flags': row['flags'],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--outputs', type=pathlib.Path, required=True)
    p.add_argument('--convention', type=pathlib.Path)
    p.add_argument('--out', type=pathlib.Path)
    p.add_argument('--case', action='append')
    a = p.parse_args()
    outputs = a.outputs.resolve()
    convention_path = (a.convention or outputs / DEFAULT_CONVENTION).resolve()
    convention = {r['case']: r for r in read(convention_path)['cases']}
    out = (a.out or outputs / 'pose_convention_videos_20261105').resolve()
    out.mkdir(parents=True, exist_ok=True)
    cases = a.case or CASES
    rows = [render_case(outputs, convention, out, case) for case in cases]
    write_json(out / 'render_manifest.json', {'convention': str(convention_path), 'cases': rows})
    print('Rendered', len(rows), 'cases to', out)


if __name__ == '__main__':
    main()
