"""Render unchanged original triangles under each sealed joint candidate."""
import argparse
import json
import pathlib
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import torch

from collision_refinement_common import candidate_nodes, load_asset, moved, transforms

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from modules.pact.representations.mesh import MeshExtractResult
from modules.pact.renderers.mesh_renderer import MeshRenderer
import utils3d

parser = argparse.ArgumentParser()
parser.add_argument('--root', required=True)
parser.add_argument('--case', required=True)
args = parser.parse_args()
root = pathlib.Path(args.root)
run = json.loads((root / 'experiment.json').read_text())['cases'][args.case]
asset = load_asset(run)
methods = ['original', 'position', 'direction', 'both']
nodes = {m: candidate_nodes(asset, None if m == 'original' else root / f'{args.case}_{m}/candidate.json') for m in methods}
out = root / f'previews_{args.case}'
out.mkdir(exist_ok=False)
torch.set_num_threads(2)
renderer = MeshRenderer({'resolution': 384, 'near': .05, 'far': 20.})
fov = torch.deg2rad(torch.tensor(50., device='cuda'))
intr = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
corners = []
for method in methods:
    for t in np.linspace(0, 1, 41):
        world = transforms(nodes[method], t)
        for part in asset['parts']:
            bmin, bmax = part['mesh'].bounds
            c = np.array([[x, y, z] for x in [bmin[0], bmax[0]] for y in [bmin[1], bmax[1]] for z in [bmin[2], bmax[2]]])
            corners.append(moved(c, world[part['node']['id']]))
bounds = np.concatenate(corners)
center = (bounds.min(0) + bounds.max(0)) / 2
radius = 1.15 * np.linalg.norm(bounds.max(0) - bounds.min(0))
palette = np.array([[.94, .53, .18], [.25, .70, .42], [.55, .55, .60], [.60, .38, .78], [.8, .2, .35]])


@torch.inference_mode()
def render(method, t, yaw):
    world = transforms(nodes[method], t)
    vertices, faces, attrs = [], [], []
    offset = 0
    for part in asset['parts']:
        node, mesh = part['node'], part['mesh']
        vertices.append(moved(mesh.vertices, world[node['id']]))
        faces.append(mesh.faces + offset)
        offset += len(mesh.vertices)
        color = np.array([.55, .55, .60]) if node['joint']['type'] == 'fixed' else palette[node['id'] % len(palette)]
        attrs.append(np.tile(np.r_[color, 0., 0., 0.], (len(mesh.vertices), 1)))
    rep = MeshExtractResult(torch.tensor(np.concatenate(vertices), device='cuda', dtype=torch.float32),
                            torch.tensor(np.concatenate(faces), device='cuda', dtype=torch.int64),
                            torch.tensor(np.concatenate(attrs), device='cuda', dtype=torch.float32))
    eye = center + np.array([np.sin(yaw) * radius, .45 * radius, np.cos(yaw) * radius])
    ext = utils3d.torch.extrinsics_look_at(torch.tensor(eye, device='cuda', dtype=torch.float32),
                                       torch.tensor(center, device='cuda', dtype=torch.float32),
                                       torch.tensor([0., 1., 0.], device='cuda'))
    result = renderer.render(rep, ext, intr, return_types=['color', 'mask', 'normal'])
    shade = .65 + .35 * torch.abs(2 * result['normal'][2:3] - 1)
    rgb = (result['color'] * shade + 1 - result['mask'].unsqueeze(0)).clamp(0, 1)
    pic = Image.fromarray((rgb.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8'))
    ImageDraw.Draw(pic).text((8, 8), f'{method} | t={t:.2f}', fill='black')
    return np.asarray(pic)


frames = []
for state, t in enumerate(np.linspace(0, 1, 21)):
    views = []
    for yaw in [.55, 2.0]:
        views.append(np.concatenate([render(m, t, yaw) for m in methods], axis=1))
    panel = np.concatenate(views, axis=0)
    frames.append(panel)
    if state in [0, 10, 20]:
        Image.fromarray(panel).save(out / f'mesh_state{state:02d}_comparison.png')
    print('RENDERED', args.case, state, flush=True)
imageio.mimsave(out / 'original_position_direction_both.mp4', frames, fps=10)
(out / 'render_info.json').write_text(json.dumps({'states': 21, 'methods': methods, 'actual_original_triangles': True,
                                               'geometry_repair': False, 'range_change': False,
                                               'shared_camera_center': center.tolist(), 'shared_camera_radius': radius,
                                               'motion': 'All retain original predicted range; no GT range override'}, indent=2))
