"""Render original frozen Gaussian bytes with sealed motion-only candidates."""
import argparse
import copy
import json
import pathlib
import sys
import time

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from plyfile import PlyData
from scipy.spatial.transform import Rotation
import torch

from collision_refinement_common import load_asset, moved, sha256, transforms

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from modules.pact.renderers.gaussian_render import GaussianRenderer
import utils3d


class Gaussians:
    active_sh_degree = 0

    def __init__(self, xyz, scales, rotations, opacity):
        self.get_xyz = xyz
        self.get_scaling = scales
        self.get_rotation = rotations
        self.get_opacity = opacity


def tensor(x):
    return torch.as_tensor(x, device='cuda', dtype=torch.float32)


def quaternion_product(a, b):
    w,x,y,z = a.unbind(-1)
    v,i,j,k = b.unbind(-1)
    return torch.stack([w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j,
                        w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v], -1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    a = p.parse_args()
    out = pathlib.Path(a.root).resolve()
    result = json.loads((out/'result.json').read_text())
    asset = load_asset(result['source_run'])
    selected_path = out/'gravity_only/object.json'
    assert sha256(selected_path) == result['candidate_object_sha256']
    candidate = json.loads(selected_path.read_text())
    methods = {'original': asset['nodes'], 'gravity_only':candidate['diffuse_tree']}
    torch.set_num_threads(4)
    renderer = GaussianRenderer({'resolution':512, 'near':.05, 'far':20., 'bg_color':(1.,1.,1.)})
    undo = np.array([[1,0,0],[0,0,1],[0,-1,0]], float)
    parts = []
    for part in asset['parts']:
        path = asset['object_path'].parent/part['node']['gs'][0]
        v = PlyData.read(path)['vertex'].data
        xyz = np.column_stack([v[k] for k in ['x','y','z']]).astype(np.float32) @ undo.T
        scales = np.exp(np.column_stack([v[k] for k in ['scale_0','scale_1','scale_2']]).astype(np.float32))
        alpha = (1/(1+np.exp(-v['opacity'].astype(np.float32))))[:,None]
        q = np.column_stack([v[k] for k in ['rot_0','rot_1','rot_2','rot_3']])
        R = Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix()
        q = Rotation.from_matrix(undo[None]@R).as_quat()[:,[3,0,1,2]].copy()
        colors = (.28209479177387814*np.column_stack([v[k] for k in ['f_dc_0','f_dc_1','f_dc_2']])+.5).clip(0,1)
        parts.append({'id':part['node']['id'], 'xyz':tensor(xyz), 'scales':tensor(scales),
                      'q':tensor(q), 'alpha':tensor(alpha), 'colors':tensor(colors)})
    clouds = []
    for nodes in methods.values():
        for t in np.linspace(0,1,21):
            tr = transforms(nodes, t)
            for part in asset['parts']:
                b = part['mesh'].bounds
                corners = np.array([[x,y,z] for x in b[:,0] for y in b[:,1] for z in b[:,2]])
                clouds.append(moved(corners, tr[part['node']['id']]))
    cloud = np.concatenate(clouds)
    center = (cloud.min(0)+cloud.max(0))/2
    # Generous shared frame includes full original and candidate trajectories.
    radius = 1.65*np.linalg.norm(np.ptp(cloud, axis=0))
    intr = utils3d.torch.intrinsics_from_fov_xy(tensor(np.deg2rad(42.)), tensor(np.deg2rad(42.)))
    cameras = []
    for yaw in [.38, -0.72]:
        eye = center + radius*np.array([np.sin(yaw)*np.cos(.22), np.sin(.22), np.cos(yaw)*np.cos(.22)])
        cameras.append(utils3d.torch.extrinsics_look_at(tensor(eye), tensor(center), tensor([0.,1.,0.])))
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    colors = torch.cat([part['colors'] for part in parts])
    scales = torch.cat([part['scales'] for part in parts])
    alphas = torch.cat([part['alpha'] for part in parts])

    @torch.no_grad()
    def frame(nodes, camera, label):
        tr = transforms(nodes, 1.)
        xyz, quats = [], []
        for part in parts:
            T = tr[part['id']]
            xyz.append(part['xyz']@tensor(T[:3,:3]).T+tensor(T[:3,3]))
            quat = Rotation.from_matrix(T[:3,:3]).as_quat()[[3,0,1,2]].copy()
            quats.append(quaternion_product(tensor(quat), part['q']))
        g = Gaussians(torch.cat(xyz), scales, torch.cat(quats), alphas)
        rgb = renderer.render(g, cameras[camera], intr, colors_overwrite=colors)['color']
        pic = Image.fromarray((rgb.clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8))
        draw = ImageDraw.Draw(pic)
        draw.rectangle((0,0,511,56), fill='white')
        draw.text((12,8), label, fill='black', font=font)
        draw.text((12,31), '45194 | original frozen geometry', fill='#555555', font=font)
        return np.asarray(pic)

    def pose(nodes, t, only=None):
        posed = copy.deepcopy(nodes)
        for n in posed:
            lo,hi = n['joint']['range']
            value = lo+(hi-lo)*(t if only is None or n['id']==only else 0.)
            n['joint']['range'] = [value,value]
        return posed

    started = time.monotonic()
    video = out/'45194_gravity_motion_only.mp4'
    comparison = out/'45194_original_vs_gravity.mp4'
    with imageio.get_writer(video, fps=20, codec='libx264', quality=8) as main_writer, imageio.get_writer(comparison, fps=20, codec='libx264', quality=8) as cmp_writer:
        sequence = np.r_[np.zeros(10), np.linspace(0,1,41), np.ones(10), np.linspace(1,0,41), np.zeros(10)]
        for i,t in enumerate(sequence):
            panels = {}
            for method,nodes in methods.items():
                posed = pose(nodes, float(t))
                panels[method] = [frame(posed, camera, f'{method} | t={t:.2f}') for camera in [0,1]]
            main_writer.append_data(np.concatenate(panels['gravity_only'], axis=1))
            panel = np.concatenate([np.concatenate([panels[m][camera] for m in methods], axis=1) for camera in [0,1]], axis=0)
            cmp_writer.append_data(panel)
            if i in [10,30,50]:
                Image.fromarray(panel).save(out/f'comparison_t{t:.2f}.png')
            if i%20==0:
                print('RENDER synchronized',i,len(sequence),flush=True)
    active = [n['id'] for n in asset['nodes'] if n['joint']['type']!='fixed']
    independent = out/'45194_individual_joints.mp4'
    with imageio.get_writer(independent, fps=15, codec='libx264', quality=8) as writer:
        for nid in active:
            for t in np.r_[np.linspace(0,1,21), np.linspace(1,0,21)]:
                panels = []
                for method,nodes in methods.items():
                    panels.append(frame(pose(nodes,float(t),nid), 0, f'{method} | node {nid} | t={t:.2f}'))
                writer.append_data(np.concatenate(panels,axis=1))
            print('RENDER individual',nid,flush=True)
    assert all(sha256(x['path'])==x['sha256'] for x in result['geometry_files'])
    info = {'source_gaussians_unchanged':True, 'source_meshes_unchanged':True,
            'renderer':'Original PAct GaussianRenderer; fixed DC appearance, all original points/scales/opacities',
            'gaussian_serialization_rotation_undone':undo.tolist(),
            'shared_camera_center':center.tolist(), 'shared_camera_radius':float(radius),
            'motion':'Original native numeric ranges; analytic gravity-only axis projection. Types and pivots retained.',
            'files':[str(video),str(comparison),str(independent)],
            'elapsed_seconds':time.monotonic()-started,
            'gpu_peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20}
    (out/'render_info.json').write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(info,indent=2),flush=True)


if __name__ == '__main__':
    main()
