"""Append animated dataset GT to every existing closed-multicase video.

Prediction pixels and motion are read from the sealed videos. GT uses the same
camera and the previously frozen static-body evaluation similarity, never ICP.
No model loading, geometry fitting, joint editing, or old-file writes occur.
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation
import torch
import trimesh

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from collision_refinement_common import sha256
from evaluate_freeart_partnet_case import gt_parts
from modules.pact.representations.mesh import MeshExtractResult
from modules.pact.renderers.mesh_renderer import MeshRenderer
import utils3d

R_WORLD = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], float)
SEQ = np.r_[np.zeros(6), np.linspace(0, 1, 33), np.ones(6),
            np.linspace(1, 0, 33), np.zeros(6)]
TITLES = ['GT (dataset mesh)', 'Origin / Original PAct', 'Motion prior', 'Closed-image fit']
FONT = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def tensor(x):
    return torch.tensor(np.array(x, copy=True), dtype=torch.float32, device='cuda')


class GT:
    def __init__(self, root):
        cfg = read(root / 'config.json')
        self.scene = pathlib.Path(cfg['source_scene'])
        align = read(root / 'evaluation.json')['alignment']
        self.R = np.array(align['R_export_to_GT'])
        self.s = align['scale']
        self.shift = np.array(align['translation'])
        self.parts = []
        self.records = []
        self.source_paths = []
        if cfg['case'] == '20411':
            states, delta = gt_parts(self.scene)
            self.parts = [states[0][0], states[0][1]]
            self.delta = delta
            self.source_paths = sorted((self.scene / 'gt_mesh').glob('*.glb'))
            self.records = [dict(part=1, type='prismatic',
                                 range_source='Actual six GLB states 00 (closed) to 05 (open)',
                                 displacement=delta.tolist())]
        else:
            self.delta = None
            mobility = read(self.scene / 'gt/mobility_v2.json')
            infos = read(self.scene / 'gt/part_info.json')
            base_ids = {x['id'] for x in mobility if x['parent'] == -1}
            selected = [x for x in mobility if x['parent'] in base_ids
                        and x['joint'] in ['hinge', 'slider']
                        and x['name'] not in ['knob', 'tray', 'control_button', 'button']
                        and (x['name'] != 'handle' or cfg['case'] == '100481')]
            assert len(infos) == len(selected)
            self.source_paths = [self.scene / 'gt/mobility_v2.json', self.scene / 'gt/part_info.json']
            for part in range(len(selected) + 1):
                path = self.scene / f'gt/part_{part}.ply'
                self.parts.append(trimesh.load(path, process=False, force='mesh'))
                self.source_paths.append(path)
            for part, (raw, info) in enumerate(zip(selected, infos), 1):
                assert info['idx'] == raw['id'], 'Review GT joint ordering rather than guessing'
                kind = 'revolute' if raw['joint'] == 'hinge' else 'prismatic'
                assert info['type'] == kind
                axis = R_WORLD @ np.array(raw['jointData']['axis']['direction'])
                axis /= np.linalg.norm(axis)
                origin = R_WORLD @ np.array(raw['jointData']['axis']['origin'])
                ordered = sorted(info['part_move'], key=int)
                motion = np.array([info['part_move'][k] for k in ordered], float)
                # part_info stores one nonzero world Euler-axis channel, even
                # when the actual annotated hinge is slightly tilted (101808).
                # Keep its signed ANGLE, not its projection onto the tilted
                # axis, which would shrink the true 90-degree motion range.
                channel = int(np.argmax(np.max(abs(motion), axis=0)))
                other = np.delete(motion, channel, axis=1)
                assert np.max(abs(other)) < 1e-6
                assert abs(axis[channel]) > .99
                scalar = motion[:, channel] * np.sign(axis[channel])
                channel_axis_error = float(np.degrees(np.arccos(np.clip(abs(axis[channel]), 0, 1))))
                limits = raw['jointData']['limit']
                a, b = float(limits['a']), float(limits['b'])
                if kind == 'revolute':
                    a, b = np.deg2rad([a, b])
                # Meshes are stored at frame 0, NOT at joint value zero. Values
                # may differ by full turns (100481); Rodrigues handles this.
                initial = float(scalar[0])
                span = abs(b - a)
                assert abs(np.ptp(scalar) - span) < 2e-5
                # Both annotated endpoints must be observed, modulo full turns.
                for q in [a, b]:
                    error = scalar - q
                    if kind == 'revolute':
                        error = (error + np.pi) % (2 * np.pi) - np.pi
                    assert np.min(abs(error)) < 2e-5
                self.records.append(dict(part=part, mobility_id=raw['id'], type=kind,
                                         axis_world=axis.tolist(), origin_world=origin.tolist(),
                                         mesh_frame0_value=initial, trajectory_channel_axis_error_degrees=channel_axis_error,
                                         trajectory_angle_convention='Signed active world Euler channel; magnitude preserved, sign matched to annotated axis', reference_endpoint_a=a,
                                         open_endpoint_b=b,
                                         range_source='Annotated PartNet endpoints a -> b; both verified against part_info trajectory',
                                         units='radians' if kind == 'revolute' else 'dataset coordinate units'))
        self.faces = []
        self.attrs = []
        offset = 0
        for mesh in self.parts:
            self.faces.append(np.asarray(mesh.faces) + offset)
            # PLYs contain UVs but no texture image; a loader-generated 2x2 gray
            # material is not source appearance. Explicitly render neutral GT.
            color = np.array([.67, .70, .74])
            self.attrs.append(np.tile(np.r_[color, 0., 0., 0.], (len(mesh.vertices), 1)))
            offset += len(mesh.vertices)
        self.faces = torch.as_tensor(np.concatenate(self.faces), dtype=torch.int64, device='cuda')
        self.attrs = tensor(np.concatenate(self.attrs))

    def vertices(self, t):
        result = []
        for index, mesh in enumerate(self.parts):
            v = np.asarray(mesh.vertices)
            if index:
                if self.delta is not None:
                    v = v + t * self.delta
                else:
                    record = self.records[index - 1]
                    q = ((1 - t) * record['reference_endpoint_a'] + t * record['open_endpoint_b']
                         - record['mesh_frame0_value'])
                    d = np.array(record['axis_world'])
                    if record['type'] == 'revolute':
                        o = np.array(record['origin_world'])
                        rot = Rotation.from_rotvec(q * d).as_matrix()
                        v = (v - o) @ rot.T + o
                    else:
                        v = v + q * d
            result.append(((v - self.shift) / self.s) @ self.R)
        return np.concatenate(result)


def label(array, index, t, diagnostic=False):
    pic = Image.fromarray(array.copy())
    draw = ImageDraw.Draw(pic)
    draw.rectangle((0, 0, 383, 47), fill='white')
    draw.text((8, 5), TITLES[index], fill='black', font=FONT)
    detail = ('GT annotated range' if index == 0 else 'Models: ASSUMED 90deg') if diagnostic else (
        'GT annotated range' if index == 0 else 'Original predicted range' if index == 1 else 'Prior range')
    draw.text((8, 27), f'{detail} | {t:.2f}', fill='#555555', font=FONT)
    return np.asarray(pic)


def video_info(path):
    with imageio.get_reader(path) as reader:
        count = reader.count_frames()
        meta = reader.get_meta_data()
        for i in range(count):
            frame = reader.get_data(i)
            assert frame.shape[:2] == (meta['size'][1], meta['size'][0])
        assert np.isfinite(frame).all()
    return dict(file=path.name, frames=count, fps=meta['fps'], size=list(meta['size']), sha256=sha256(path), all_frames_decoded=True)


def render_case(root):
    start = time.monotonic()
    cfg = read(root / 'config.json')
    previous = read(root / 'render_results.json')
    protected = {root / x['file']: x['sha256'] for x in previous['videos']}
    for name in ['checkpoint.pt', 'motion_groups.json', 'fit_results.json', 'optimized_parts.npz',
                 'config.json', 'evaluation.json', 'render_results.json']:
        protected[root / name] = sha256(root / name)
    assert all(sha256(p) == digest for p, digest in protected.items())
    case = cfg['case']
    torch.set_num_threads(4)
    gt = GT(root)
    renderer = MeshRenderer({'resolution': 384, 'near': .05, 'far': 20., 'ssaa': 2})
    center = np.array(previous['shared_camera_center'])
    radius = previous['shared_camera_radius']
    intr = utils3d.torch.intrinsics_from_fov_xy(tensor(np.deg2rad(43.)), tensor(np.deg2rad(43.)))
    cameras = []
    for yaw in [.35, -.7]:
        eye = center + radius * np.array([np.sin(yaw) * np.cos(.22), np.sin(.22), np.cos(yaw) * np.cos(.22)])
        cameras.append(utils3d.torch.extrinsics_look_at(tensor(eye), tensor(center), tensor([0, 1, 0])))
    cache = {}
    projection_margin = 1.

    @torch.inference_mode()
    def gt_frame(t, view):
        nonlocal projection_margin
        key = (round(float(t), 8), view)
        if key not in cache:
            vertices = tensor(gt.vertices(t))
            mesh = MeshExtractResult(vertices, gt.faces, gt.attrs)
            out = renderer.render(mesh, cameras[view], intr, return_types=['color', 'mask', 'normal'])
            shade = .65 + .35 * abs(2 * out['normal'][2:3] - 1)
            rgb = (out['color'] * shade + 1 - out['mask'].unsqueeze(0)).clamp(0, 1)
            cache[key] = (rgb.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            # Audit every vertex against shared camera frustum, not sampled bounds.
            cam = vertices @ cameras[view][:3, :3].T + cameras[view][:3, 3]
            proj = cam @ intr.T
            uv = proj[:, :2] / proj[:, 2:3]
            margin = torch.minimum(uv, 1 - uv).min().item()
            projection_margin = min(projection_margin, margin)
            assert cam[:, 2].min() > .05 and cam[:, 2].max() < 20
            assert margin > .01, f'GT cropped in original shared camera: {margin}'
        return cache[key]

    paths = []
    main = root / f'{case}_four_versions_with_GT.mp4'
    paths.append(main)
    paired = {m: root / f'{case}_{m}_with_GT.mp4' for m in ['original', 'prior', 'optimized']}
    gt_only = root / f'{case}_GT.mp4'
    paths.extend([*paired.values(), gt_only])
    writers = {m: imageio.get_writer(p, fps=16, codec='libx264', quality=8) for m, p in paired.items()}
    cmp = imageio.get_writer(main, fps=16, codec='libx264', quality=8)
    gw = imageio.get_writer(gt_only, fps=16, codec='libx264', quality=8)
    source = imageio.get_reader(root / f'{case}_three_versions.mp4')
    try:
        assert source.count_frames() == len(SEQ)
        for i, t in enumerate(SEQ):
            old = source.get_data(i)
            rows = []
            pairs = {m: [] for m in paired}
            gtviews = []
            for view in range(2):
                gtview = label(gt_frame(t, view), 0, t)
                gtviews.append(gtview)
                panels = [gtview]
                for j, method in enumerate(paired, 1):
                    predicted = label(old[view * 384:(view + 1) * 384, (j - 1) * 384:j * 384], j, t)
                    panels.append(predicted)
                    pairs[method].append(np.concatenate([gtview, predicted], axis=1))
                rows.append(np.concatenate(panels, axis=1))
            panel = np.concatenate(rows, axis=0)
            cmp.append_data(panel)
            gw.append_data(np.concatenate(gtviews, axis=1))
            for method in paired:
                writers[method].append_data(np.concatenate(pairs[method], axis=0))
            if i in [6, 22, 38]:
                Image.fromarray(panel).save(root / f'four_versions_with_GT_t{t:.2f}.png')
            if i % 28 == 0:
                print('GT RENDER', case, i, flush=True)
    finally:
        source.close()
        cmp.close()
        gw.close()
        for writer in writers.values():
            writer.close()
    diagnostic = root / f'{case}_assumed90_diagnostic.mp4'
    if diagnostic.exists():
        dst = root / f'{case}_assumed90_with_GT.mp4'
        paths.append(dst)
        seq = np.r_[np.linspace(0, 1, 25), np.linspace(1, 0, 25)]
        with imageio.get_reader(diagnostic) as src, imageio.get_writer(dst, fps=12, codec='libx264', quality=8) as writer:
            assert src.count_frames() == len(seq)
            for i, t in enumerate(seq):
                old = src.get_data(i)
                panels = [label(gt_frame(t, 0), 0, t, True)]
                panels += [label(old[:, j * 384:(j + 1) * 384], j + 1, t, True) for j in range(3)]
                writer.append_data(np.concatenate(panels, axis=1))
    videos = [video_info(path) for path in paths]
    assert all(sha256(p) == digest for p, digest in protected.items())
    sources = {str(path): sha256(path) for path in gt.source_paths}
    write(root / 'gt_video_results.json', dict(
        case=case, videos=videos, camera_center=center.tolist(), camera_radius=radius,
        cameras_extrinsic=[c.cpu().numpy().tolist() for c in cameras],
        intrinsics=intr.cpu().numpy().tolist(), min_GT_vertex_frustum_margin=projection_margin,
        fixed_alignment=read(root / 'evaluation.json')['alignment'], GT_joints=gt.records,
        GT_source_sha256=sources, protected_original_files_sha256={str(p): v for p, v in protected.items()},
        original_files_unchanged=True, prediction_images='Reused decoded sealed original triple/diagnostic video pixels; only headers and encoding changed',
        appearance='GT: neutral shaded actual source triangles (PLYs do not include texture images). Predictions: original Gaussian renders.',
        phase='All four use common normalized progress. GT follows dataset annotated a -> b endpoints (20411 actual GLB 00 -> 05); each prediction keeps its existing interval/rest-pose convention. Angles/displacements are not forced equal.',
        diagnostic='GT retains annotated range; only prediction columns retain the explicitly assumed 90 degree diagnostic.',
        seconds=time.monotonic() - start, code_sha256=sha256(pathlib.Path(__file__))))
    print('GT COMPLETE', case, len(videos), flush=True)


def finish(root, cases):
    records = [read(root / case / 'gt_video_results.json') for case in cases]
    videos = [dict(case=r['case'], **v) for r in records for v in r['videos']]
    assert all(v['all_frames_decoded'] for v in videos)
    for record in records:
        assert all(sha256(pathlib.Path(p)) == digest for p, digest in record['protected_original_files_sha256'].items())
        assert all(sha256(pathlib.Path(p)) == digest for p, digest in record['GT_source_sha256'].items())
        assert all(sha256(root / record['case'] / v['file']) == v['sha256'] for v in record['videos'])
    lines = ['# GT / Origin / Motion prior / Closed-image fit', '',
             '共 10 个 case；每个主视频从左至右为 **GT、Origin、Motion prior、Closed-image fit**，上下两行使用原对比视频的两台共同相机。', '',
             'GT 使用数据集真实三角网格和独立部件运动，统一转入已有评估坐标；复用已冻结的静态柜体缩放/平移。未重新拟合模型、几何、轴或对齐。预测画面直接来自原三路视频；原文件保留。', '',
             'GT 以中性灰着色，便于查看几何（源 PLY 不含真实纹理图）。三组预测保留原 Gaussian 外观，因此本视频用于几何及运动对比，不作同渲染器外观评分。', '',
             '**同步含义：统一开合进度，不保证相同角度/位移。** GT 使用标注端点 a→b；20411 使用真实 GLB 00→05。各预测保留原始/先验行程与参考姿态。原始预测的起始姿态可能已经打开；部分 case 的标注端点也不一定是完全闭合，不能将所有 t=0 画面视为同一物理姿态。', '',
             '10489、8961 的附加诊断视频只对预测使用假设 90°，GT 始终保留标注行程，标题中已区分。', '',
             '| Case | 四路主视频 | GT + Origin | GT + Motion prior | GT + Closed-image fit | GT 单独参考 |',
             '|---|---|---|---|---|---|']
    for case in cases:
        names = [f'{case}_four_versions_with_GT.mp4', f'{case}_original_with_GT.mp4',
                 f'{case}_prior_with_GT.mp4', f'{case}_optimized_with_GT.mp4', f'{case}_GT.mp4']
        lines.append('| ' + case + ' | ' + ' | '.join(f'[播放]({case}/{name})' for name in names) + ' |')
    lines += ['', '附加 90° 诊断：' + '、'.join(f'[{case}]({case}/{case}_assumed90_with_GT.mp4)' for case in ['10489', '8961']), '',
              '逐 case 的相机、GT 轴/初始值/范围、来源校验和、视频帧数和逐帧解码结果见 `gt_video_results.json`。']
    audit_path = root / 'GT_NATIVE_CAMERA_POSE_AUDIT.json'
    audit = read(audit_path) if audit_path.exists() else None
    if audit:
        scores = [row['silhouette_IoU'] for rows in audit['cases'].values() for row in rows]
        lines += ['', f'GT 坐标/姿态核验：将真实 GT 放回原相机，在 {len(scores)} 个源图状态上复核，轮廓 IoU 范围为 {100*min(scores):.2f}%–{100*max(scores):.2f}%。这是 GT 参考的姿态核验，不是新的重建指标。',
                  '', '[GT 姿态核验数据](GT_NATIVE_CAMERA_POSE_AUDIT.json)；[所有 GT 起始状态](GT_all_cases_t0.00.png)；[所有 GT 终止状态](GT_all_cases_t1.00.png)。']
    (root / 'GT_VIDEO_COMPARISON.md').write_text('\n'.join(lines) + '\n')
    write(root / 'GT_VIDEO_FINAL_SEAL.json', dict(cases=cases, video_count=len(videos), videos=videos,
          all_frames_decoded=True, all_original_experiment_files_unchanged=True,
          original_video_count=sum(len(read(root / case / 'render_results.json')['videos']) for case in cases),
          report_sha256=sha256(root / 'GT_VIDEO_COMPARISON.md'),
          GT_native_camera_pose_audit_sha256=sha256(audit_path) if audit else None,
          gt_pose_verification_code_sha256=sha256(REPO / 'repro/tools/verify_gt_video_pose.py'),
          per_case_manifest_sha256={case: sha256(root / case / 'gt_video_results.json') for case in cases},
          code_sha256=sha256(pathlib.Path(__file__))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--case')
    args = parser.parse_args()
    root = pathlib.Path(args.root).resolve()
    if args.case:
        render_case(root / args.case)
        return
    cases = read(root / 'protocol.json')['cases']
    states = {}
    for case in cases:
        manifest = root / case / 'gt_video_results.json'
        if not manifest.exists():
            env = os.environ.copy()
            env.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4')
            with (root / case / 'gt_render.log').open('w') as log:
                proc = subprocess.run([sys.executable, __file__, '--root', str(root), '--case', case],
                                      cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
            states[case] = 'complete' if proc.returncode == 0 else 'failed'
        else:
            states[case] = 'complete'
        write(root / 'gt_render_status.json', states)
        print('GT CASE', case, states[case], flush=True)
    if all(v == 'complete' for v in states.values()):
        finish(root, cases)
    else:
        raise RuntimeError(f'Incomplete GT videos: {states}')


if __name__ == '__main__':
    main()
