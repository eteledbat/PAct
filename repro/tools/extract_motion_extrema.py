"""Extract the two most separated observed articulation states for each case.

The joint trajectory is used only to select image indices and project oracle
visible 2D instance masks. Numeric GT joints never enter the candidate model.
For several independently moving leaves, no two frames can necessarily place
every leaf at both of its individual extrema; the maximally separated pair in
normalized joint-state space is an explicit assembly-level convention.
"""
import argparse
import json
import os
import pathlib
import shutil
import sys

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
import cv2
import nvdiffrast.torch as dr
import numpy as np
import torch
import trimesh
from PIL import Image
from scipy.spatial.transform import Rotation

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from modules.pact.renderers.mesh_renderer import intrinsics_to_projection
from auto_part_semantics import sha256, write_json
from evaluate_freeart_partnet_case import gt_parts

OLD = REPO/'repro/outputs/closed_multicase_20261028_v1'
INVENTORY = REPO/'repro/outputs/original_mask_multicase_20261029_v1/input_inventory.json'
R_GLB = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
R_WORLD = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], float)


def video_states(scene):
    info = json.loads((scene/'gt/part_info.json').read_text())
    n = len(json.loads((scene/'transforms.json').read_text())['frames'])
    columns = []
    for record in info:
        move = np.array([record['part_move'][str(i)] for i in range(n)], float)
        channel = int(np.argmax(np.ptp(move, axis=0)))
        values = move[:, channel]
        span = np.ptp(values)
        assert span > 1e-8
        columns.append((values-values.min())/span)
    return np.stack(columns, axis=1), info


def most_separated(states):
    distance = ((states[:, None, :]-states[None, :, :])**2).sum(-1)
    first, last = np.unravel_index(np.argmax(distance), distance.shape)
    assert first < last and distance[first, last] > 0
    return [int(first), int(last)], float(np.sqrt(distance[first, last]))


def video_meshes(scene, frame, infos):
    mobility = json.loads((scene/'gt/mobility_v2.json').read_text())
    base_ids = {x['id'] for x in mobility if x['parent'] == -1}
    selected = [x for x in mobility if x['parent'] in base_ids
                and x['joint'] in ['hinge', 'slider']
                and x['name'] not in ['knob', 'tray', 'control_button', 'button']
                and (x['name'] != 'handle' or scene.name == '100481')]
    assert len(infos) == len(selected)
    meshes = []
    for part in range(len(selected)+1):
        mesh = trimesh.load(scene/f'gt/part_{part}.ply', process=False, force='mesh')
        vertices = np.asarray(mesh.vertices).copy()
        if part:
            raw, info = selected[part-1], infos[part-1]
            assert raw['id'] == info['idx']
            move = np.array([info['part_move'][str(i)] for i in [0, frame]], float)
            channel = int(np.argmax(np.max(abs(np.array(list(info['part_move'].values()))), axis=0)))
            axis = R_WORLD @ np.array(raw['jointData']['axis']['direction'], float)
            axis /= np.linalg.norm(axis)
            q = (move[1, channel]-move[0, channel])*np.sign(axis[channel])
            if raw['joint'] == 'hinge':
                origin = R_WORLD @ np.array(raw['jointData']['axis']['origin'], float)
                matrix = Rotation.from_rotvec(q*axis).as_matrix()
                vertices = (vertices-origin) @ matrix.T+origin
            else:
                vertices = vertices+q*axis
        meshes.append((vertices, np.asarray(mesh.faces)))
    return meshes


def freeart_meshes(scene, frame):
    states, _ = gt_parts(scene)
    return [(np.asarray(m.vertices)@R_GLB.T, np.asarray(m.faces)) for m in states[frame]]


def camera(scene, frame, is_freeart, size):
    w, h = size
    if is_freeart:
        old = REPO/'repro/inputs/freeart_partnet_20411_frame05/camera_check.json'
        data = json.loads(old.read_text())
        return np.array(data['extrinsic']), np.array(data['intrinsic_normalized'])
    data = json.loads((scene/'transforms.json').read_text())
    f = data['frames'][frame]
    c2w = np.array(f['transform_matrix'])
    c2w[:3, 1:3] *= -1
    intr = np.array([[data['focal_x']/w, 0, data['cx']/w],
                     [0, data['focal_y']/h, data['cy']/h], [0, 0, 1]], float)
    return np.linalg.inv(c2w), intr


def rasterize(meshes, ext, intr, size):
    w, h = size
    vertices = np.concatenate([v for v, _ in meshes])
    faces, owners, offset = [], [], 0
    for owner, (v, f) in enumerate(meshes, 1):
        faces.append(f+offset)
        owners.extend([owner]*len(f))
        offset += len(v)
    v = torch.as_tensor(vertices, dtype=torch.float32, device='cuda')
    f = torch.as_tensor(np.concatenate(faces), dtype=torch.int32, device='cuda')
    e = torch.as_tensor(ext, dtype=torch.float32, device='cuda')
    projection = intrinsics_to_projection(torch.as_tensor(intr, dtype=torch.float32, device='cuda'), .1, 20.)
    clip = (torch.cat([v, torch.ones((len(v), 1), device='cuda')], 1)@(projection@e).T)[None]
    with torch.no_grad(): rast, _ = dr.rasterize(dr.RasterizeCudaContext(), clip, f, (h, w))
    triangle = rast[0, :, :, 3].cpu().numpy().astype(np.int32)
    labels = np.zeros((h, w), np.int32)
    valid = triangle > 0
    labels[valid] = np.asarray(owners)[triangle[valid]-1]
    return labels


def prepare_case(root, item):
    case = item['case']
    scene = pathlib.Path(item['scene'])
    is_freeart = case == '20411'
    if is_freeart:
        frames, distance, infos = [0, 5], 1., None
    else:
        states, infos = video_states(scene)
        frames, distance = most_separated(states)
    dest = root/'extremes'/case
    dest.mkdir(parents=True)
    results = []
    for end, frame in enumerate(frames):
        source = scene/(f'{frame:02d}_pure.png' if is_freeart else f'images/{frame:06d}.png')
        rgb_path = dest/f'endpoint_{end}_frame{frame:06d}.png'
        shutil.copyfile(source, rgb_path)
        image = Image.open(rgb_path).convert('RGBA')
        meshes = freeart_meshes(scene, frame) if is_freeart else video_meshes(scene, frame, infos)
        ext, intr = camera(scene, frame, is_freeart, image.size)
        labels = rasterize(meshes, ext, intr, image.size)
        foreground = np.asarray(image)[:, :, 3] >= 128
        visible = labels > 0
        iou = float((foreground & visible).sum()/max((foreground | visible).sum(), 1))
        assert iou > .975, (case, frame, iou)
        mask = dest/f'endpoint_{end}_mask.npy'
        np.save(mask, labels)
        preview = np.full((*labels.shape, 3), 255, np.uint8)
        palette = np.array([[140, 140, 150], [240, 130, 40], [70, 180, 100],
                            [200, 80, 160], [150, 120, 220], [130, 170, 40]])
        for owner in range(1, len(meshes)+1): preview[labels == owner] = palette[(owner-1) % len(palette)]
        Image.fromarray(preview).save(dest/f'endpoint_{end}_mask.png')
        results.append(dict(frame=frame, rgb=str(rgb_path), rgb_sha256=sha256(rgb_path),
                            mask=str(mask), mask_sha256=sha256(mask),
                            silhouette_iou=iou, extrinsics=ext.tolist(), intrinsics=intr.tolist(),
                            instance_pixels={str(i): int((labels == i).sum()) for i in range(1, len(meshes)+1)}))
    manifest = dict(case=case, source_scene=str(scene), endpoints=results,
                    normalized_joint_state_distance=distance,
                    extraction='Farthest observed pair in Euclidean normalized multi-joint state space' if not is_freeart else 'FreeArt six-state endpoints 00 and 05',
                    GT_use='Frame selection and projected 2D instance masks only; no GT numeric joints passed to VLM, axis router or geometry fitter',
                    original_origin_rgb=item['rgb'], original_origin_mask=item['mask'], source_run=item['run'])
    write_json(dest/'manifest.json', manifest)
    print(case, frames, [round(x['silhouette_iou'], 5) for x in results], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=pathlib.Path, required=True)
    parser.add_argument('--case')
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    items = json.loads(INVENTORY.read_text())
    cases = json.loads((OLD/'protocol.json').read_text())['cases']
    assert set(cases) == {x['case'] for x in items if x['case'] in cases}
    for item in items:
        if item['case'] not in cases or args.case and item['case'] != args.case: continue
        if not (root/'extremes'/item['case']/'manifest.json').exists(): prepare_case(root, item)


if __name__ == '__main__': main()
