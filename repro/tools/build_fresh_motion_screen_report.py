"""Report every predeclared fresh case, including screening failures."""
import argparse
import hashlib
import json
import pathlib
import subprocess

REPO = pathlib.Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    root = pathlib.Path(args.root).resolve()
    config = json.loads((root / 'screen_protocol.json').read_text())
    status = json.loads((root / 'status.json').read_text())
    assert status['state'] == 'complete'
    evidence = {'root': str(root), 'protocol': config, 'cases': {}, 'optimization_started': False}
    lines = ['# 新场景的运动碰撞筛选结果', '',
             '新增案例为此前未做过 PAct 推理的 45194 和 8961。两者均使用真实本地数据、固定首帧及 seed42、官方模型与原版 25+25 步推理。GT Mesh 投影的 oracle 2D 部件 mask 是输入；GT 轴和关节类型不输入模型。新场景指本轮之前没有复现，不代表官方模型训练时未见过。', '',
             f'原始产物目录：[{root.name}]({root})。所有尝试均保留；本轮使用 GT 做诊断筛选，不能作为无偏泛化指标。', '',
             '| 场景 | 类型正确 | 静态 RMS / GT D | 合格 |',
             '| --- | --- | --- | --- |']
    details = []
    for scene in config['new_cases']:
        result = status['results'][scene]
        if result['state'] != 'complete':
            evidence['cases'][scene] = result
            lines.append(f'| {scene} | unavailable | unavailable | 推理/评价失败 |')
            continue
        run = pathlib.Path(result['run'])
        original = json.loads((run / 'joint_evaluation/joint_errors.json').read_text())
        geometry = json.loads((root / f'geometry_{scene}.json').read_text())
        motion_dir = root / f'motion_{scene}'
        motion = json.loads((motion_dir / 'metrics.json').read_text())
        run_status = json.loads((run / 'status.json').read_text())
        loads = [json.loads(row) for row in (run / 'checkpoint_loads.jsonl').read_text().splitlines() if row.strip()]
        assert loads and all(not row.get('missing_keys') and not row.get('unexpected_keys') and not row.get('missing_parameters') for row in loads)
        correct = 0
        rows, reasons = [], []
        limits = config['criteria']
        if original['static_surface_rms_percent_diagonal'] > limits['static_RMS_percent_GT_D_max']:
            reasons.append('静态几何 RMS 超过 3%')
        for joint in original['joints']:
            v = joint['variants']['original_exported']
            correct += int(v['type_correct'])
            amplitude = abs(joint['predicted_exported_range'][1] - joint['predicted_exported_range'][0])
            part = next(p for p in geometry['parts'] if p['GT_mesh_index'] == joint['gt_mesh_index'])
            shape_cd = part['shape_only']['CD_mean_percent_static_diagonal']
            row = {'GT_part': joint['gt_mesh_index'], 'predicted_node_id': joint['predicted_node_id'],
                   'GT_type': joint['GT_type'], 'predicted_type': v['predicted_type'], 'type_correct': v['type_correct'],
                   'axis_error_degrees': v['axis_direction_error_degrees'] if v['type_correct'] else None,
                   'raw_head_axis_error_degrees': joint['variants']['raw_head']['axis_direction_error_degrees'],
                   'pivot_percent_GT_D': v['pivot_to_gt_axis_percent_static_diagonal'],
                   'native_amplitude': amplitude, 'amplitude_unit': 'degrees' if v['predicted_type'] in ['revolute', 'continuous'] else 'model units',
                   'shape_only_CD_percent_GT_D': shape_cd}
            rows.append(row)
            if not v['type_correct']:
                reasons.append(f"GTpart{row['GT_part']} 类型错误")
            if shape_cd > limits['moving_rigid_shape_CD_percent_GT_D_max']:
                reasons.append(f"GTpart{row['GT_part']} shape-only CD 超过 3%")
            if joint['GT_type'] == 'revolute' and v['type_correct']:
                if v['axis_direction_error_degrees'] is None or v['axis_direction_error_degrees'] > limits['axis_degrees_max']:
                    reasons.append(f"GTpart{row['GT_part']} 方向误差超过 3°")
                if v['pivot_to_gt_axis_percent_static_diagonal'] is None or v['pivot_to_gt_axis_percent_static_diagonal'] < limits['pivot_percent_GT_D_min']:
                    reasons.append(f"GTpart{row['GT_part']} 无明显 pivot 缺陷")
                if amplitude < limits['motion_degrees_min']:
                    reasons.append(f"GTpart{row['GT_part']} 原生运动小于 20°")
        qualified = not reasons
        evidence['cases'][scene] = {'run': str(run), 'joint_rows': rows,
                                   'static_RMS_percent_GT_D': original['static_surface_rms_percent_diagonal'],
                                   'qualified': qualified, 'rejection_reasons': reasons,
                                   'original_inference_seconds': run_status['elapsed_seconds'],
                                   'peak_observed_gpu_mib': run_status['peak_observed_gpu_mib'],
                                   'motion_paths': [{k: p[k] for k in ['name', 'moving_predicted_node_ids', 'initial_state_proxy', 'mean_motion_only_proxy', 'max_state_proxy', 'max_state_minus_initial_proxy']} for p in motion['paths']],
                                   'motion_checks': motion['checks'], 'strict_checkpoint_loads': len(loads),
                                   'solid_volume': None, 'penetration_depth': None}
        lines.append(f"| {scene} | {correct}/{len(rows)} | {original['static_surface_rms_percent_diagonal']:.3f}% | {'是' if qualified else '否'} |")
        details += ['', f'## {scene}', '', f'筛选结论：{"符合预设阈值" if qualified else "；".join(reasons)}。', '',
                    '| GT 部件 / 预测 node | GT → 预测类型 | 有效方向误差 | Pivot / GT D | 原生幅度 | Shape-only CD / GT D |',
                    '| --- | --- | --- | --- | --- | --- |']
        for row in rows:
            angle = 'unavailable' if row['axis_error_degrees'] is None else f"{row['axis_error_degrees']:.3f}°"
            pivot = 'N/A' if row['pivot_percent_GT_D'] is None else f"{row['pivot_percent_GT_D']:.3f}%"
            details.append(f"| {row['GT_part']} / {row['predicted_node_id']} | {row['GT_type']} → {row['predicted_type']} | {angle} | {pivot} | {row['native_amplitude']:.3f} {row['amplitude_unit']} | {row['shape_only_CD_percent_GT_D']:.3f}% |")
        videos = sorted(run.rglob('*articulation_animation.mp4'))
        details += ['', f"推理 {run_status['elapsed_seconds']:.1f} 秒，峰值观测显存 {run_status['peak_observed_gpu_mib']} MiB。严格 checkpoint 加载审计通过。",
                    '', f'[逐关节运动代理曲线]({motion_dir / "motion_paths.png"})；[21 状态/轨迹的原始数据]({motion_dir / "metrics.json"})。']
        if videos:
            details += ['', f'[官方原始运动视频]({videos[0]})。']
        details += ['', f'[GT 轴叠加图]({run / "joint_evaluation/axes_gtpart1.png"})；[逐部件几何结果]({root / ("geometry_" + scene + ".json")})。']
    evidence['git_sha'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    lines += details + ['', '## 解释边界', '',
                        '运动诊断逐关节采样 21 个状态，其余关节固定在模型原生 t=0，并单独保留全关节同步路径。21 点采样不保证连续轨迹绝无碰撞，也没有穷举所有多关节组合。只使用预测范围，没有用 GT 开合角度扩大范围。', '',
                        '曲线左图是各状态的绝对 Gaussian 最近中心重叠代理，右图是相对起始状态的变化，仅用于诊断运动带来的变化，没有修改优化 Loss。它不是穿透体积；sigma gap 也不是实体穿透深度。模型 Mesh 的实体合法性未建立，真实体积和穿透深度为 unavailable。', '',
                        '共同全局对齐下的几何误差保持不变；shape-only 指每个活动部件独立刚体配准、不改变尺度后的形状诊断，不能看成修正后的整物体装配。关节评分仍用原有固定坐标变换，另保留静态 ICP 敏感性结果，不按 GT 关节误差重新排列部件或修正类型。', '',
                        '8961 的门片形状误差较小且 pivot 缺陷明显，因此是下一轮规定运动范围诊断的候选。但窗框 RMS 为 4.20%，未达到预设 3% 几何阈值，原生转角也不足 1°。若之后人为规定较大开合范围，必须单列为运动范围假设实验，不能称为原版预测范围结果，也不能把 GT 角度悄悄输入优化。', '',
                        '表中方向误差取原版导出的轴；revolute 在官方后处理中吸附到最近坐标轴，0° 不代表原始 head 已准确到 0°。原始 head 的方向误差保留在原始 joint_errors.json 与本轮 evidence 中。', '',
                        '本轮完成新场景的原始推理、GT/几何检查和运动轨迹诊断，没有启动新的碰撞优化。若没有合格案例，不把复杂几何或类型错误包装成只需校正轴的问题。']
    evidence['analysis_source_sha256'] = {name: hashlib.sha256((REPO / 'repro/tools' / name).read_bytes()).hexdigest() for name in ['build_fresh_motion_screen_report.py', 'inspect_independent_motion_paths.py', 'evaluate_pivot_case_geometry.py', 'collision_refinement_common.py']}
    content = '\n'.join(lines) + '\n'
    (root / 'results.md').write_text(content)
    (REPO / 'repro/NEW_MOTION_CASE_RESULTS.md').write_text(content)
    (REPO / 'repro/evidence/new-motion-case-screen.json').write_text(json.dumps(evidence, indent=2, allow_nan=False))
    print(json.dumps({scene: {'qualified': row.get('qualified'), 'rejection_reasons': row.get('rejection_reasons')} for scene, row in evidence['cases'].items()}, indent=2))


if __name__ == '__main__':
    main()
