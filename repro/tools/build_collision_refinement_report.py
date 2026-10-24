"""Report only completed, sealed refinements and their independent measurements."""
import datetime
import json
import os
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from collision_refinement_common import sha256

REPO = pathlib.Path(__file__).resolve().parents[2]
ROOT = pathlib.Path((REPO / 'repro/current_refinement.txt').read_text().strip())
DEST = REPO / 'repro/COLLISION_REFINEMENT_RESULTS.md'
CONFIG = json.loads((ROOT / 'experiment.json').read_text())
METHODS = ['original', 'position', 'direction', 'both']
LABELS = {'original': '原版', 'position': '只调位置', 'direction': '只调方向', 'both': '位置+方向'}
summary = {'experiment_root': str(ROOT), 'config': CONFIG, 'cases': {},
           'scope': 'Independent bounded Gaussian-proxy axis refinement. Not strict nonpenetration, no solid volume claims.'}


def read(path):
    return json.loads(path.read_text())


def link(path, label):
    assert path.is_file(), path
    return f'[{label}]({os.path.relpath(path, DEST.parent)})'


def figure(path, label):
    return '!' + link(path, label)


def num(value, digits=3):
    return 'unavailable' if value is None else f'{value:.{digits}f}'


def gt_mean(metric):
    return metric['GT'].get('mean_valid_axis_error_degrees')


lines = ['# PAct 固定几何后的碰撞引导轴优化', '',
         f'完成记录更新：{datetime.datetime.now(datetime.timezone.utc).isoformat()}。', '',
         '[完整汇总数值、环境、配置及封存SHA](evidence/collision-refinement-results.json)。', '',
         '本轮新运行2个已有PAct案例、每例3项200步后处理消融，另保留原版对照。'
         '不从头训练PAct，不更新Gaussian/Mesh，不改关节类型、行程或初始姿态。'
         '所有候选保留并在固定最终步封存，GT只作事后统计，不按新GT成绩挑参数、checkpoint或案例。', '',
         '## 之前哪些案例有明显冲突', '',
         '既有FreeArt3D的26657确有严重封闭几何重叠，Original的21状态平均重叠约76.73%，'
         '详见[旧FreeArt3D报告](/root/autodl-tmp/FreeArt3D/GT_NONPENETRATION_COMPARISON.md)。'
         '那是不同方法和场景，旧冻结joint修改几何的实验不能作为本轮轴优化成绩。', '',
         'PAct官方Dishwasher_001在运动后半程出现明显增多的完整三角面相交；31249起始状态已经有较多相交。'
         '这些是重建表面的几何冲突证据，但表面交线也可能含合法接触。'
         'Dishwasher主体开放；31249至少两个部件自交，其余实体合法性未全部验证，所以真实实体交叠体积及精确穿透深度均unavailable。'
         '没有补洞、做凸包或将undefined填0。10489运动仅约6–10°，只保留筛查，不据此证明全运动修复；101808类型/部件语义失败不由局部轴优化修复。', '',
         '## 优化与独立验证', '',
         '[运行前协议及调度补充](COLLISION_REFINEMENT_DESIGN.md)。每部件2048个训练源中心、seed0；'
         '最近邻目标使用全部opacity≥0.05中心。21个均匀运动状态，对所有跨部件对求opacity加权Gaussian核均值，'
         '除以原版核均值，再加0.05倍参数位移正则。方向最多改变5°，旋转轴origin在初始轴垂直平面内最多移动资产对角线5%。'
         'Prismatic只调方向，origin不调。两例使用同一组设置，Adam固定200步；不吸附新轴、不按碰撞最低步选结果。', '',
         '优化核是重叠代理，可能处罚正常接触，不能称为严格不穿模约束。'
         '独立评估41个状态，含20个训练状态之间的插值状态；每部件至多12000个源中心与训练源中心不重叠，'
         '共享完整目标密度。训练用CPU KDTree，验证用独立Kaolin CUDA最近邻并在真实点子集与CPU核验，再用float64算核。'
         '另用VTK全体原始三角面独立测21状态的相交线长度。横切线排除两面法向夹角≤5°，仍不是真实体积。', '',
         't=0的几何与轴无关，初始重叠无法靠轴优化消除。所有曲线包含该状态，不能把它删掉再报告“零穿模”。', '',
         '## 定量对照', '',
         'Δ为相对同案例原版的降低百分比，负值表示恶化。Gaussian核无量纲；交线长度除以资产对角线D。'
         'GT方向均值只含类型正确关节，31249始终为3/4；pivot为预测origin到GT无限轴的垂距，占GT静态主体对角线的百分比。', '',
         '| Case / 方法 | 41状态Gaussian核均值 | 降低% | 21状态表面交线/D均值 | 降低% | GT有效轴均误差° | 旋转关节Pivot% |',
         '|---|---:|---:|---:|---:|---:|---:|']

for name, run in CONFIG['cases'].items():
    allmetrics = {m: read(ROOT / f'eval_{name}_{m}/metrics.json') for m in METHODS}
    allmesh = {m: read(ROOT / f'mesh_{name}_{m}/metrics.json') for m in METHODS}
    original = allmetrics['original']
    original_mesh = allmesh['original']
    case = {'methods': {}}
    for method in METHODS:
        metric, mesh = allmetrics[method], allmesh[method]
        reduction = 100 * (1 - metric['mean_all_state_pair_proxy'] / original['mean_all_state_pair_proxy'])
        mesh_reduction = 100 * (1 - mesh['mean_total_line_length_over_diagonal'] / original_mesh['mean_total_line_length_over_diagonal'])
        pivots = [j['pivot_to_GT_axis_percent_static_diagonal'] for j in metric['GT'].get('joints', []) if j['pivot_to_GT_axis_percent_static_diagonal'] is not None]
        pivot = float(np.mean(pivots)) if pivots else None
        lines.append(f'| {name} / {LABELS[method]} | {metric["mean_all_state_pair_proxy"]:.7f} | {reduction:.2f} | '
                     f'{mesh["mean_total_line_length_over_diagonal"]:.5f} | {mesh_reduction:.2f} | {num(gt_mean(metric))} | {num(pivot)} |')
        info = {'metric': {k: v for k, v in metric.items() if k not in ['states', 'independent_GPU_CPU_NN_checks']},
                'mesh': {k: v for k, v in mesh.items() if k != 'states'},
                'gaussian_mean_reduction_percent': reduction, 'surface_line_mean_reduction_percent': mesh_reduction,
                'pivot_mean_percent_GT_static_diagonal': pivot,
                'evidence': {str(p): sha256(p) for p in [ROOT / f'eval_{name}_{method}/metrics.json', ROOT / f'mesh_{name}_{method}/metrics.json']}}
        if method != 'original':
            directory = ROOT / f'{name}_{method}'
            seal, status = read(directory / 'prediction_seal.json'), read(directory / 'status.json')
            assert status['state'] == 'complete' and status['step'] == 200
            assert seal['candidate_sha256'] == metric['candidate_sha256']
            assert not seal['GT_reads_blocked'] and seal['frozen_files_unchanged']
            info['training_status'] = status
            info['protocol'] = read(directory / 'protocol.json')
            info['wall_seconds_since_protocol_write_including_resume_or_contention'] = datetime.datetime.fromisoformat(status['updated_utc']).timestamp() - (directory / 'protocol.json').stat().st_mtime
        case['methods'][method] = info
        assert np.isclose(metric['initial_state_mean_pair_proxy'], original['initial_state_mean_pair_proxy'], rtol=1e-7, atol=1e-12)
    summary['cases'][name] = case

lines += ['', '实测联合优化的效果：', '']
for name, case in summary['cases'].items():
    old, refined = case['methods']['original'], case['methods']['both']
    statement = (f'- {name}：Gaussian核均值降低{refined["gaussian_mean_reduction_percent"]:.2f}%，'
                 f'完整表面交线均值降低{refined["surface_line_mean_reduction_percent"]:.2f}%。')
    if refined['metric']['GT']['status'] == 'measured':
        delta = gt_mean(refined['metric']) - gt_mean(old['metric'])
        statement += (f' 主协议GT有效轴误差{num(gt_mean(old["metric"]))}° → {num(gt_mean(refined["metric"]))}°'
                      f'（变化{delta:+.3f}°，正值为恶化）；Pivot '
                      f'{num(old["pivot_mean_percent_GT_static_diagonal"])}% → {num(refined["pivot_mean_percent_GT_static_diagonal"])}%。')
    else:
        statement += ' 无可靠GT，不能报告轴或pivot精度提升。'
    lines.append(statement)
lines += ['', '完整几何和外观在rest-state保持原值，不能将本轮结果写成重建几何/Chamfer提升。'
          '关节类型和range也原样保留，误分类没有通过轴优化被改正。', '', '## 原版与优化后的运动曲线', '']

for name in CONFIG['cases']:
    case = summary['cases'][name]
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    for method in METHODS:
        metric = read(ROOT / f'eval_{name}_{method}/metrics.json')
        mesh = read(ROOT / f'mesh_{name}_{method}/metrics.json')
        axes[0].plot([s['t'] for s in metric['states']], [s['mean_pair_proxy'] for s in metric['states']], label=method)
        axes[1].plot([s['t'] for s in mesh['states']], [sum(p['line_length_over_diagonal'] for p in s['pairs']) for s in mesh['states']], label=method)
        axes[2].plot([s['t'] for s in mesh['states']], [sum(p['transverse_line_length_over_diagonal'] for p in s['pairs']) for s in mesh['states']], label=method)
    for ax, title in zip(axes, ['Gaussian NN overlap\n(held-out sources, 41 states)',
                                'Surface intersection\nlength / D (21 states)',
                                'Transverse intersection\nlength / D (angle >5deg)']):
        ax.set_ylabel(title, fontsize=9)
        ax.grid(alpha=.25)
        ax.legend()
    axes[-1].set_xlabel('Original predicted motion range, normalized t (NOT GT range)')
    fig.suptitle(name + ': fixed geometry and range; no solid volume claim')
    fig.tight_layout()
    path = ROOT / f'collision_curves_{name}.png'
    fig.savefig(path, dpi=160)
    plt.close(fig)
    lines += [figure(path, f'{name} 运动曲线'), '']
    original = case['methods']['original']['metric']
    both = case['methods']['both']['metric']
    lines += [f'{name}：原版 → 联合优化，插值留出状态均值 '
              f'{original["mean_interleaved_heldout_time_proxy"]:.7f} → {both["mean_interleaved_heldout_time_proxy"]:.7f}；'
              f'最坏状态的跨部件核均值 {original["max_state_mean_pair_proxy"]:.7f} → {both["max_state_mean_pair_proxy"]:.7f}。', '']
    preview = ROOT / f'previews_{name}'
    lines += [link(preview / 'original_position_direction_both.mp4', f'{name} 原版/位置/方向/联合，两视角运动视频'), '']
    for state in [0, 10, 20]:
        lines += [figure(preview / f'mesh_state{state:02d}_comparison.png', f'{name} t={state/20:.1f} 同一实际Mesh对照'), '']
    for state in [0, 10, 20]:
        lines += [figure(ROOT / f'hotspots_{name}/intersections_state{state:02d}.png', f'{name} 实际表面横切交线X-ray'), '']
    lines += [link(ROOT / f'hotspots_{name}/regions.json', '相交线区域坐标与显示采样说明'), '']

lines += ['## 相交发生在哪些部件', '',
          '下表保留全部部件对，包括零交线的对。数值是VTK表面交线长度/D；'
          '零交线只是此指标的零测量，不是实体碰撞体积为零的证明。', '',
          '| Case / node对 | 原版均值 | 联合优化均值 | 原版末端 | 联合优化末端 |',
          '|---|---:|---:|---:|---:|']
for name, case in summary['cases'].items():
    original = read(ROOT / f'mesh_{name}_original/metrics.json')['states']
    refined = read(ROOT / f'mesh_{name}_both/metrics.json')['states']
    pair_summary = []
    for index, pair in enumerate(original[0]['pairs']):
        before = float(np.mean([row['pairs'][index]['line_length_over_diagonal'] for row in original]))
        after = float(np.mean([row['pairs'][index]['line_length_over_diagonal'] for row in refined]))
        before_end = original[-1]['pairs'][index]['line_length_over_diagonal']
        after_end = refined[-1]['pairs'][index]['line_length_over_diagonal']
        pair_summary.append({'ids': pair['ids'], 'original_mean_line_over_D': before, 'refined_mean_line_over_D': after,
                             'original_end_line_over_D': before_end, 'refined_end_line_over_D': after_end})
        lines.append(f'| {name}/{pair["ids"]} | {before:.5f} | {after:.5f} | {before_end:.5f} | {after_end:.5f} |')
    case['original_vs_both_pair_surface_intersections'] = pair_summary
lines += ['', 'Dishwasher的交线主要位于门与柜体前下沿连接区域；优化后末端非零长度横切线消失，'
          '仍不能据此认证实体无碰撞。31249中node0为正确类型的旋转门、node1为被错判成平移关节的另一门、'
          'node2为柜体、node3/4为抽屉。联合优化减轻node0/柜体的相交，'
          '但node1/柜体的交线均值反而增加（约1.16207→1.20440），抽屉相关相交也大量残留。'
          '这是全体原始Mesh的独立检测，并非只展示训练代理最有利的区域。', '']

lines += ['## GT轴与位置：逐关节和对齐敏感性', '',
          '31249沿用旧评估的固定坐标、尺度和平移；额外沿用原版静态几何ICP变换，全部候选共享，不重新按GT配准。'
          '原数据使用GT投影mask，且公开划分中有增强对象家族重叠；这不是无GT mask或严格未见对象泛化测试。', '',
          '| GT mobility id / 类型 | 方法 | 类型正确 | 主协议轴误差° | 主协议Pivot% | 同一旧静态配准的轴误差° | 同一配准Pivot% |',
          '|---|---|---|---:|---:|---:|---:|']
for method in METHODS:
    gt = summary['cases']['31249']['methods'][method]['metric']['GT']
    assert gt['joint_type_correct_count'] == 3 and gt['joint_count'] == 4
    for joint in gt['joints']:
        lines.append(f'| {joint["gt_mobility_id"]} / {joint["GT_type"]} | {LABELS[method]} | {joint["type_correct"]} | '
                     f'{num(joint["valid_axis_error_degrees"])} | {num(joint["pivot_to_GT_axis_percent_static_diagonal"])} | '
                     f'{num(joint.get("same_baseline_registration_axis_error_degrees"))} | {num(joint.get("same_baseline_registration_pivot_percent_diagonal"))} |')

registered = {}
for method in METHODS:
    gt = summary['cases']['31249']['methods'][method]['metric']['GT']
    valid = [j for j in gt['joints'] if j['type_correct']]
    registered[method] = float(np.mean([j['same_baseline_registration_axis_error_degrees'] for j in valid]))
summary['cases']['31249']['same_baseline_registration_mean_axis_error_degrees'] = registered
lines += ['', f'同一旧静态配准下，有效轴均误差原版{registered["original"]:.3f}° → 联合优化{registered["both"]:.3f}°，'
          '与主协议的变化方向不同；这是必须保留的对齐敏感性。该配准下旋转pivot也由约0.426%恶化到4.200%。'
          '因此不能笼统宣称所有GT指标都更好或都更差，更不能选择有利配准宣称整体关节恢复成功。', '',
          '主协议原版的旋转轴方向已经被启发式吸附到0°误差，因此旋转轴方向不能在该主指标上进一步下降。'
          '错误类型的残留方向仍保存在完整指标里，只作诊断，不当作正确旋转轴成绩。'
          '固定几何配准存在歧义，主/补充协议结论可能不同，不能挑选更有利的一组宣称恢复GT。', '',
          '## 参数偏移及连接风险', '',
          '原pivot锚点随新运动产生的位移是连接漂移诊断，不能直接当作实际接触间隙或穿透深度。', '',
          '| Case / 方法 / node | 类型 | 方向偏离原版° | origin偏移%资产D | 原pivot锚点最大运动%资产D |',
          '|---|---|---:|---:|---:|']
axis_figures = [ROOT / 'GT_axis_previews' / f'GT_mobility_{idx}.png' for idx in [0, 1, 3]]
for name, case in summary['cases'].items():
    for method in METHODS[1:]:
        for row in case['methods'][method]['metric']['axis_changes']:
            assert row['direction_change_degrees'] <= 5.001
            assert row['origin_shift_percent_asset_diagonal'] is None or row['origin_shift_percent_asset_diagonal'] <= 5.001
            lines.append(f'| {name}/{LABELS[method]}/{row["node_id"]} | {row["type"]} | {num(row["direction_change_degrees"])} | '
                         f'{num(row["origin_shift_percent_asset_diagonal"])} | {num(row["max_original_pivot_anchor_displacement_percent_asset_diagonal"])} |')

lines += ['', '三根类型正确关节的GT/原版/全部消融轴叠加（同一旧对齐和固定几何）：', '']
for path in axis_figures:
    lines += [figure(path, path.stem), '']
lines += ['', '本轮新增loss只对轴方向切平面和pivot参数求导；Gaussian/Mesh不是优化参数，'
          '实际梯度检查和文件SHA共同验证了这一点。它是新的后处理实验，不是原版PAct已经包含的物理Joint优化。'
          '联合优化中原pivot锚点最大位移约为31249资产D的2.75%、Dishwasher资产D的1.47%；'
          '它提示连接可能偏离原估计，但不是已测真实间隙。只调位置的31249还触及5%偏移上限。'
          '避碰可能把正确轴推向错误的空闲区域，以补偿不准确几何或行程；本实验未分离这些潜在原因。', '',
          '## 可追溯性与限制', '',
          '独立分支 `research/collision-axis-refinement-20261024`；初始实现commit `fc7fe30`。'
          '原baseline分支及输出未覆盖。正式六项固定200步，输出原candidate、完整Adam checkpoint、history、冻结文件SHA和封存SHA。'
          '优化程序audit hook禁止打开GT、旧GT评估和其他项目数据；源码/配置保留。'
          '旧GT基线分数此前已经看过，因此不声称盲测。', '',
          '初次不冻结NN分支的有限差分检查遇到邻居切换未通过；之后明确检查所选邻居分支的解析梯度，生产优化仍每次刷新NN。'
          '实际导出数据上的位置/方向梯度与NumPy运动一致性检查通过。'
          '5步GPU smoke和中断恢复检查证据见汇总JSON；调度器切换时第一项从50步恢复，未选择其他checkpoint。'
          '并发运行的wall耗时不能解释为单项独占速度；Torch峰值分配不包含CUDA context。', '',
          '本轮只有两个案例、一个有GT、单一固定初始化；不能证明方法普适，也不能保证连续运动中严格无碰撞。'
          'Gaussian重叠、完整三角面表面相交、真实实体碰撞是三个不同量。'
          '降低其中一个不能自动证明其他量改善，也不能自动证明轴恢复更准确。', '',
          '独立监控（无需Codex持续调用）：', '', '```bash',
          '/root/autodl-tmp/conda_envs/pact/bin/python /root/autodl-tmp/PAct/repro/tools/monitor_collision_refinement.py --watch',
          '```', '', '全部工具和小型证据提交本分支；大型模型和实验输出留在本机，未推送官方上游。']

lines += ['', '耗时为从protocol写入至完成的wall时间，包含并发竞争和中断恢复；不作为独占性能benchmark。', '',
          '| Case / 方法 | 最终步数 | wall分钟 | Torch峰值分配MiB（不含context） |',
          '|---|---:|---:|---:|']
for name, case in summary['cases'].items():
    for method in METHODS[1:]:
        entry = case['methods'][method]
        lines.append(f'| {name}/{LABELS[method]} | 200 | {entry["wall_seconds_since_protocol_write_including_resume_or_contention"]/60:.2f} | '
                     f'{entry["training_status"]["gpu_peak_allocated_MiB"]:.2f} |')
lines += ['', '新增候选用本地ply/gs符号链接引用原始几何，candidate JSON及原始几何SHA均未变化；'
          '可直接按相对路径检查，未复制或上传大模型。相机首版空白较多，保留在previews_camera_v1_*；'
          '最终使用更近的共同相机，全部方法和状态共享，不改变任何预测。', '',
          '数值后端复核：真实运动子集的CPU/GPU最近邻距离通过容差检查，所检子集Gaussian核逐项一致；'
          '这是数值验证，不是新改善样本。环境、具体版本和所有原始指标的SHA见汇总证据。']

summary['gradient_check'] = read(ROOT / 'dense-gradient-check.json')
summary['resume_check'] = read(ROOT / 'resume-check.json')
summary['geometry_audit'] = read(ROOT / 'solid_audit_partial.json')
summary['environment'] = read(ROOT / 'environment.json')
summary['nn_backend_check'] = {k:v for k,v in read(ROOT / 'nn-backend-check.json').items() if k != 'rows'}
summary['geometry_links'] = read(ROOT / 'geometry_links.json')
summary['camera_changes'] = read(ROOT / 'camera_changes.json')
summary['sources'] = {str(path): sha256(path) for path in (REPO / 'repro/tools').glob('*.py')}
(REPO / 'repro/evidence/collision-refinement-results.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
DEST.write_text('\n'.join(lines) + '\n')
print(DEST)
