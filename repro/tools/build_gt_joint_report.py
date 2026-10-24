"""Build a report from completed, immutable case outputs; never run/select predictions."""
import hashlib
import json
import os
import pathlib

import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageOps

ROOT = pathlib.Path(__file__).resolve().parents[2]
REPRO = ROOT / 'repro'
DEST = REPRO / 'GT_JOINT_EVALUATION.md'
SCENES = ['10489', '31249', '101808']


def read(path):
    return json.loads(path.read_text())


def hashed(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def relative(path):
    return os.path.relpath(path, DEST.parent)


def link(path, label):
    assert path.is_file(), path
    return f'[{label}]({relative(path)})'


def number(value, digits=3):
    return 'unavailable' if value is None else f'{value:.{digits}f}'


def degrees(value):
    return 'unavailable' if value is None else f'{value:.3f}°'


def conditional_summary(joints, variant):
    values = [j['variants'][variant] for j in joints]
    correct = [v for v in values if v['type_correct']]
    angles = [v['axis_direction_error_degrees'] for v in correct
              if v['axis_direction_error_degrees'] is not None]
    return {'type_correct_count': len(correct), 'joint_count': len(values),
            'type_accuracy': len(correct) / len(values),
            'correct_type_axis_count': len(angles),
            'correct_type_mean_axis_error_degrees': sum(angles) / len(angles) if angles else None}


split = read(REPRO / 'evidence/singapo-data-split.json')
split_audit = {}
results = {'scope': 'Three existing GT-mask cases, fixed frame0/seed42, inference only',
           'wrong_type_vectors': 'Diagnostic only; excluded from conditional angle means but retained as type failures.',
           'cases': []}
lines = ['# PAct：已有数据集的关节 GT 对照', '',
         '三个案例均已完成实际 GPU 推理和评估；没有从头训练、关节 refinement、GT 误差选 seed 或重新配对部件。Part²GS 保持暂停。', '',
         '## 数据和协议', '',
         '使用本机 VideoArtGS/SAPIEN 的 10489、31249、101808，每例固定第 0 帧。'
         'RGB 来自原数据；由于没有现成二维部件标签，将已有 GT 部件 Mesh 按标定相机投影为 EXR mask。'
         '**这是 GT-mask / oracle segmentation 条件**：提供了真实部件划分及部件数，不能称为无 GT 的自动单图结果。'
         '原图不做内容编辑。GT 轴、GT 类型、GT origin 未送入 PAct；GT Mesh 不作为三维生成输入，但用于 mask 和事后评估。', '',
         'GT 来自各场景 `gt/mobility_v2.json`；不使用根目录估计的 joint_infos 文件。'
         '过滤规则及固定坐标旋转来自已有 VideoArtGS `utils/metrics.py:read_gt`。'
         '部件对应由原版输入 mask 的排序确定，不按 GT 轴误差找最优排列。'
         '101808 原 mobility 含旋钮关节，按此数据协议只评估两个门，旋钮并入静态几何，不把它写成全物体九关节评估。', '',
         '官方代码 SHA `a5ccec5056e7b27f39ddb5b83613c0efba9b2861`，'
         '模型 revision `8c6b56c7011454bbfb45a83e2dbfa6bbbfde63e9`。'
         '三例都用 seed42、SS25/SLAT25、CFG7、最后20步 feature 聚合；每例只运行一个预测。'
         '独立环境和运行兼容修复见 [CHANGES.md](CHANGES.md)；没有修改原版轴吸附、bbox origin 后处理或 Loss。', '',
         '## 主评估：固定坐标规范', '',
         '方向只使用源码规定的固定旋转；没有 ICP 或 GT 轴拟合。'
         '位置使用静态部件 AABB 对角线的单一尺度与中心平移，GT 静态几何仅在评估中使用。'
         '角度为 `acos(abs(dot(unit(pred), unit(GT))))`，单位度，不检验轴正负或开合方向。'
         'Pivot 指预测 origin 到 GT 无限轴的垂距，以 GT 静态主体对角线百分比表示；'
         '不是厘米，也不惩罚不可辨识的沿轴位移。另存无限轴最短线距；非平行相交线可得到零线距，不能单靠该值判断 pivot。'
         'prismatic 和类型错误的 pivot 均为 unavailable。', '',
         '| Case | 活动关节类型正确 | 类型正确子集的 raw 平均角误差 | 同一子集原版导出平均角误差 | 有效角度数 / 全部活动关节 |',
         '|---|---:|---:|---:|---:|']

case_rows = []
for scene in SCENES:
    run = pathlib.Path((REPRO / f'current_gt_{scene}.txt').read_text().strip())
    status = read(run / 'status.json')
    assert status['state'] == 'complete'
    primary = read(run / 'joint_evaluation/joint_errors.json')
    sensitivity = read(run / 'joint_evaluation/geometry_registration_sensitivity.json')
    provenance = read(run / 'input_provenance.json')
    obj = read(pathlib.Path(status['object']))
    nodes = {n['id']: n for n in obj['diffuse_tree']}
    correspondence = [{**m, 'predicted_semantic': nodes[m['pact_node_id']]['name'],
                       'predicted_exported_type': nodes[m['pact_node_id']]['joint']['type']}
                      for m in provenance['part_correspondence']]
    summaries = {v: conditional_summary(primary['joints'], v)
                 for v in ['raw_head', 'axis_snapped_before_bbox', 'original_exported']}
    r, e = summaries['raw_head'], summaries['original_exported']
    # The three current cases retain the same correct-type subset before/after parsing.
    assert [j['variants']['raw_head']['type_correct'] for j in primary['joints']] == [
        j['variants']['original_exported']['type_correct'] for j in primary['joints']]
    lines.append(f'| {scene} | {e["type_correct_count"]}/{e["joint_count"]} | '
                 f'{degrees(r["correct_type_mean_axis_error_degrees"])} | '
                 f'{degrees(e["correct_type_mean_axis_error_degrees"])} | '
                 f'{e["correct_type_axis_count"]}/{e["joint_count"]} |')
    split_audit[scene] = {key: [item for item in split[key]
                              if item.rsplit('/', 1)[-1] == scene or item.rsplit('/', 1)[-1].startswith(scene + '-')]
                          for key in ['train', 'test']}
    gt_file = pathlib.Path(provenance['source_data_root']) / 'gt/mobility_v2.json'
    checkpoint_loads = [json.loads(row) for row in (run / 'checkpoint_loads.jsonl').read_text().splitlines()]
    assert checkpoint_loads and all(not x['missing_keys'] and not x['unexpected_keys'] for x in checkpoint_loads)
    result = {'scene_id': scene, 'run': str(run), 'status': status,
              'conditional_summary': summaries, 'primary': primary, 'registration_sensitivity': sensitivity,
              'input_provenance': provenance, 'part_correspondence_with_predicted_semantics': correspondence,
              'GT_annotation': hashed(gt_file), 'exported_object': hashed(pathlib.Path(status['object'])),
              'checkpoint_load_count': len(checkpoint_loads),
              'evidence_files': [hashed(run / p) for p in ['command.json', 'source.patch', 'joint_trace/raw_head_00.npy',
                                                          'joint_evaluation/joint_errors.json',
                                                          'joint_evaluation/geometry_registration_sensitivity.json']]}
    results['cases'].append(result)
    case_rows.append((scene, run, primary, sensitivity, provenance, correspondence))

lines += ['', '**平均角度只统计类型正确的子集，必须与覆盖数一起阅读。**'
          '总共5/8个活动关节类型正确；这不是5根轴代表8根关节都成功。'
          '原始 joint_errors.json 中原有 mean 字段统计所有非零向量，包含错误类型；'
          '它们保留用于审计，不能当作类型正确的轴精度。上表由完整记录另算，未删除失败样本。', '',
          '| Case / GT mobility id（Mesh index） | GT | raw 类型 / 方向诊断角 | 原版导出类型 / 方向诊断角 | raw → 导出 Pivot（%主体对角线） |',
          '|---|---|---|---|---|']
for scene, run, primary, sensitivity, provenance, correspondence in case_rows:
    for j in primary['joints']:
        raw, exported = j['variants']['raw_head'], j['variants']['original_exported']
        marker = '' if exported['type_correct'] else ' **类型错误；非有效轴分数**'
        lines.append(f'| {scene} / {j["gt_mobility_id"]}（{j["gt_mesh_index"]}） | {j["GT_type"]} | '
                     f'{raw["predicted_type"]} / {degrees(raw["axis_direction_error_degrees"])} | '
                     f'{exported["predicted_type"]} / {degrees(exported["axis_direction_error_degrees"])}{marker} | '
                     f'{number(raw["pivot_to_gt_axis_percent_static_diagonal"])} → '
                     f'{number(exported["pivot_to_gt_axis_percent_static_diagonal"])} |')
lines += ['', '10489：两个连续 raw 方向误差为2.718°、3.211°；原版 `rescale_axis` 将 revolute 吸附到 X/Y/Z 后均为0°。'
          '因此0°不是连续回归恰好精确；bbox origin 后处理后仍有3.489%、3.522%的轴线位置偏差。', '',
          '31249：两抽屉方向误差约0.537°、0.676°；一个门 raw 为0.217°，吸附后0°，'
          '其 pivot 从15.302%降到0.596%。另一个真实旋转门被预测成 prismatic，方向向量相差80.810°；'
          '此数仅用于定位失败，不是有效 revolute 轴误差。', '',
          '101808：两门最终均为 fixed，活动关节类型0/2正确，没有有效轴精度或 pivot。'
          'GT 静态主体却被标为 door/revolute，两门被标为 base。第二门 raw 类型为 prismatic，'
          '被原版 `parse_tree` 的 base 语义强制设成 fixed，却仍留下非零方向字段；'
          '其76.967°是残留向量诊断，不能当作 fixed 关节存在可用轴。'
          '该例几何/整体姿态也失败，不能将大方向偏差全部归因于局部轴回归。', '',
          '## 只用静态几何的配准敏感性', '',
          '额外做一个固定初始化的局部刚体 ICP：静态表面12000点、80%截尾、最多50次迭代；'
          '不使用 GT 轴、类型、origin，不多起点挑分数。用独立seed1表面点衡量几何残差。'
          '只在评估中变换预测，不覆盖模型输出；主评估仍保留固定坐标结果。'
          '静态形状不完整或对称时局部配准也不唯一，因此它只是敏感性诊断。', '',
          '| Case | 配准旋转变化 | 独立静态表面 RMS（%对角线）前 → 后 | 类型正确子集 raw 角均值前 → 后 | 导出角均值前 → 后 |',
          '|---|---:|---:|---:|---:|']
for item in results['cases']:
    ss = item['registration_sensitivity']
    before = item['conditional_summary']
    raw = conditional_summary(ss['joints'], 'raw_head')
    exp = conditional_summary(ss['joints'], 'original_exported')
    item['registration_conditional_summary'] = {'raw_head': raw, 'original_exported': exp}
    lines.append(f'| {item["scene_id"]} | {number(ss["rotation_delta_degrees"])}° | '
                 f'{number(ss["heldout_static_symmetric_rms_before_percent_diagonal"])} → '
                 f'{number(ss["heldout_static_symmetric_rms_after_percent_diagonal"])} | '
                 f'{number(before["raw_head"]["correct_type_mean_axis_error_degrees"])} → '
                 f'{number(raw["correct_type_mean_axis_error_degrees"])} | '
                 f'{number(before["original_exported"]["correct_type_mean_axis_error_degrees"])} → '
                 f'{number(exp["correct_type_mean_axis_error_degrees"])} |')
lines += ['', '10489 配准后的导出 pivot 为6.664%、6.611%，比固定 AABB 归一化下更大，'
          '说明位置量对整体平移有敏感性。31249 几何残差略变差，却可旋转约2.99°，'
          '说明不能通过选一个配准来宣称亚度级精度。101808 配准旋转约45.34°后残差仍约12%，'
          '且类型仍全错；未把配准当成修复成功。', '',
          '## 输入、几何及轴叠加', '',
          '下面左侧为实际原图、中间为投影标签、右侧为原版 Gaussian 动画第一帧。'
          '右侧相机与输入不同，不做像素指标。轴图是完整 Mesh 顶点的抽样多视角叠加：'
          'GT 红、raw 橙、导出蓝。错误类型的非零字段只表示诊断向量。']
for scene, run, primary, sensitivity, provenance, correspondence in case_rows:
    video = next(run.rglob('00__articulation_animation.mp4'))
    reader = imageio.get_reader(str(video))
    generated = Image.fromarray(reader.get_data(0)).convert('RGB')
    reader.close()
    source = Image.open(provenance['source_image']).convert('RGB')
    mask = Image.open(pathlib.Path(provenance['input_dir']) / 'gt_part_mask_preview.png').convert('RGB')
    canvas = Image.new('RGB', (1440, 520), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, (img, label) in enumerate([(source, 'Original RGB'), (mask, 'GT-mask / oracle input'),
                                       (generated, 'PAct frame0 - different camera')]):
        canvas.paste(ImageOps.pad(img, (480, 480), color='white'), (480 * i, 40))
        draw.text((480 * i + 12, 12), f'{scene}: {label}', fill='black')
    comparison = run / 'joint_evaluation/input_mask_generated.jpg'
    canvas.save(comparison, quality=90)
    lines += ['', f'### {scene}', '', f'![{scene} 实际输入、标签和生成结果]({relative(comparison)})', '',
              link(video, '原版关节动画') + '；' + link(run / 'joint_evaluation/joint_errors.json', '逐关节原始指标') + '；' +
              link(run / 'joint_evaluation/geometry_registration_sensitivity.json', '配准敏感性')]
    if scene != '101808':
        image = run / 'joint_evaluation' / ('axes_gtpart1.png' if scene == '10489' else 'axes_gtpart3.png')
        lines += ['', f'![{scene} 类型正确的旋转关节与 GT 叠加]({relative(image)})']
    if scene == '101808':
        lines += ['', '本例输出将原静态主体识别成活动门，两门识别成 base；'
                  '不将残留 fixed 方向字段画成有效关节轴。完整输入部件与输出 node/语义对应保存在汇总 JSON。']

lines += ['', '## 划分、耗时与可复查性', '',
          '固定版本 SINGAPO split：10489 和101808在 train；31249 基础对象在 test，'
          '但 train 包含3个31249-*增强实例。因此本报告不是严格未见对象泛化评估。'
          'PAct 实际预训练数据 manifest 未独立核实，这份公开划分只能提示潜在重叠。'
          '[固定 split 文件](evidence/singapo-data-split.json)；[匹配条目](evidence/gt-case-split-audit.json)。', '',
          '| Case | RGBA前景/投影mask IoU | 推理+导出+校验秒数 | 采样观察到的显存 MiB |',
          '|---|---:|---:|---:|']
for item in results['cases']:
    st = item['status']
    lines.append(f'| {item["scene_id"]} | {item["input_provenance"]["silhouette_iou"]:.6f} | '
                 f'{st["elapsed_seconds"]:.2f} | {st["peak_observed_gpu_mib"]} |')
lines += ['', '显存每10秒采样，不是精确峰值；耗时不含准备mask和后续评估。所有模型 state_dict 加载均无缺失或多余 key。', '',
          '生成结果目录各自保存 command、源码差异、日志、raw head、后处理前后 joint JSON、Mesh、GS、视频和评估。'
          'GT annotation、原始图像、标定、Mesh 和相关评估证据均记录 SHA256。'
          '指标公式通过解析构造的符号反转、30°、平行轴偏移、相交轴、零向量和刚体/尺度协变检查；'
          '这些是公式测试，不是新实验结果。[公式测试证据](evidence/joint-metric-math.json)。', '',
          '按阶段复查或运行新案例（新目录拒绝覆盖）：', '', '```bash',
          'cd /root/autodl-tmp/PAct',
          'PYTHONDONTWRITEBYTECODE=1 /root/autodl-tmp/conda_envs/pact/bin/python repro/tools/prepare_video_gt_case.py --scene SCENE_ID',
          'PYTHONDONTWRITEBYTECODE=1 /root/autodl-tmp/conda_envs/pact/bin/python repro/tools/run_gt_case.py --scene SCENE_ID',
          '# 将 run 替换为该次输出目录',
          '/root/autodl-tmp/conda_envs/pact/bin/python repro/tools/evaluate_gt_joints.py --run /absolute/run',
          '/root/autodl-tmp/conda_envs/pact/bin/python repro/tools/evaluate_registration_sensitivity.py --run /absolute/run',
          '```', '', '运行中可直接读取 `repro/current_gt_SCENE_ID.txt` 指向目录下的 `status.json` 和 `inference.log`；'
          '也可独立监控（其他case替换数字）：', '', '```bash',
          '/root/autodl-tmp/conda_envs/pact/bin/python repro/tools/run_status.py --gt-scene 31249 --watch',
          '```', '', '已完成三例均无需继续训练，进程已退出。汇总报告可从保存的输出重建：', '', '```bash',
          '/root/autodl-tmp/conda_envs/pact/bin/python repro/tools/build_gt_joint_report.py', '```', '',
          '## 解释边界', '',
          '当前结果支持：部分正确类型关节 raw 轴已较准，原版坐标轴吸附进一步压低方向误差；'
          '轴位置、类型与部件语义仍会失败。不能由0°方向误差推出 pivot 正确或全运动无穿模。'
          '本轮没有对三例计算新碰撞曲线、标准 Chamfer 或图像重建指标，也没有关节范围对齐后的误差，均不冒充已测。'
          '原始 GT limit 与预测 range 保留在 JSON；输入处于非零开合状态，未强行比较不同 rest-state 的范围。'
          '10489原版动画看似不动的原因见 [运动幅度检查](ANIMATION_10489_DIAGNOSIS.md)：'
          '两门仅预测约6.53°和9.95°的运动，不能用方向0°误差代表完整运动成功。'
          '原版官方 Dishwasher_001 的21状态代理分析仍单独见 [原 smoke 报告](REPRODUCTION_REPORT.md)，不能移植为三例的碰撞结果。', '',
          '所有工具和小型证据提交独立分支；大型模型、输入/输出Mesh和视频仅保留本机，不向官方上游提交或推送。']

results['GT_coordinate_reference'] = hashed(pathlib.Path('/root/autodl-tmp/VideoArtGS/utils/metrics.py'))
results['PAct_articulation_reference'] = hashed(ROOT / 'modules/utils/articulation_utils.py')
results['split_reference'] = hashed(REPRO / 'evidence/singapo-data-split.json')
results['evaluation_tools'] = [hashed(REPRO / 'tools' / filename) for filename in [
    'prepare_video_gt_case.py', 'run_gt_case.py', 'guarded_infer.py',
    'evaluate_gt_joints.py', 'evaluate_registration_sensitivity.py', 'build_gt_joint_report.py']]
results['split_audit'] = split_audit
(REPRO / 'evidence/gt-case-split-audit.json').write_text(json.dumps(split_audit, indent=2) + '\n')
(REPRO / 'evidence/gt-joint-results.json').write_text(json.dumps(results, indent=2, allow_nan=False) + '\n')
DEST.write_text('\n'.join(lines) + '\n')
print(DEST)
