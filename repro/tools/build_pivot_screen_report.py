"""Validate actual pivot-only results and retain the complete diagnostic screen."""
import argparse,copy,json,pathlib,re,subprocess
import imageio.v2 as imageio
import numpy as np
from PIL import Image
from collision_refinement_common import load_asset,sha256,transforms

REPO=pathlib.Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();screen=pathlib.Path(a.root).resolve();protocol=json.loads((screen/'screen_protocol.json').read_text())
trial=pathlib.Path((screen/'selected_trial.txt').read_text().strip());config=json.loads((trial/'experiment.json').read_text());case,run=next(iter(config['cases'].items()));assert case=='10489'
asset=load_asset(run);base=json.loads((asset['run']/'joint_evaluation/joint_errors.json').read_text());refinement=json.loads((trial/'10489_position/protocol.json').read_text());seal=json.loads((trial/'10489_position/prediction_seal.json').read_text());candidate=json.loads((trial/'10489_position/candidate.json').read_text())
assert json.loads((trial/'controller_status.json').read_text())['state']=='complete'
assert not seal['GT_reads_blocked'] and seal['frozen_files_unchanged'] and sha256(trial/'10489_position/candidate.json')==seal['candidate_sha256']
for file,digest in refinement['source_sha256'].items():assert sha256(file)==digest
for row in refinement['frozen_files']:
    for k in ['mesh','gs']:assert sha256(row[k])==row[k+'_sha256']
clean=copy.deepcopy(candidate);clean.pop('geometry_source_object');oldbyid={n['id']:n for n in asset['nodes']}
for node in clean['diffuse_tree']:
    old=oldbyid[node['id']];assert node['joint']['axis']['direction']==old['joint']['axis']['direction']
    delta=np.array(node['joint']['axis']['origin'])-np.array(old['joint']['axis']['origin'])
    assert np.linalg.norm(delta)<=.050001*asset['diagonal']
    if old['joint']['type']=='revolute':assert abs(delta@np.array(old['joint']['axis']['direction']))<1e-7
    node['joint']['axis']['origin']=old['joint']['axis']['origin']
assert clean==asset['object']
assert all(np.array_equal(transforms(asset['nodes'],0)[nid],transforms(candidate['diffuse_tree'],0)[nid]) for nid in oldbyid)
history=json.loads((trial/'10489_position/history.json').read_text());assert [r['step'] for r in history]==list(range(1,201))
assert all(np.isfinite([r['loss'],r['gaussian_proxy'],r['gradient_norms']['pivot']]).all() and r['gradient_norms']['direction'] is None for r in history)
metrics={m:json.loads((trial/f'eval_10489_{m}/metrics.json').read_text()) for m in ['original','position']};tri={m:json.loads((trial/f'mesh_10489_{m}/metrics.json').read_text()) for m in metrics}
for m in metrics:
    assert len(metrics[m]['states'])==41 and len(tri[m]['states'])==21
    assert all(j['valid_axis_error_degrees']==0 for j in metrics[m]['GT']['joints'])
    assert metrics[m]['solid_collision']['volume'] is None
    assert all(x['max_GPU_NN_distance_excess_over_CPU_on_31_real_points']<2e-6 for x in metrics[m]['independent_GPU_CPU_NN_checks'])
assert metrics['original']['initial_state_mean_pair_proxy']==metrics['position']['initial_state_mean_pair_proxy']
assert tri['original']['states'][0]==tri['position']['states'][0]
image=np.array(Image.open(trial/'previews_10489/mesh_state00_comparison.png'));h,w=image.shape[0]//2,image.shape[1]//2
assert all(np.array_equal(image[row*h+30:(row+1)*h,:w],image[row*h+30:(row+1)*h,w:]) for row in [0,1])
reader=imageio.get_reader(str(trial/'previews_10489/original_position.mp4'));count=reader.count_frames();reader.close();assert count==21
assert (trial/'diagnostics/figure_info.json').is_file()
integrity={'passed':True,'only_pivot_changed':True,'directions_types_ranges_mesh_GS_unchanged':True,'fixed_final_steps':200,'finite_loss_and_pivot_gradients':True,'GT_reads_during_optimization':seal['GT_reads_blocked'],'original_refined_t0_metrics_and_unlabelled_pixels_identical':True,'Gaussian_states_each':41,'triangle_states_each':21,'video_frames':count,'solid_volume_depth':'unavailable, no zero filling'}
(trial/'integrity.json').write_text(json.dumps(integrity,indent=2))
cases={}
for name in ['10489',*protocol['new_cases']]:
    run=pathlib.Path((REPO/f'repro/current_gt_{name}.txt').read_text().strip());b=json.loads((run/'joint_evaluation/joint_errors.json').read_text());geometry=json.loads((screen/f'geometry_{name}.json').read_text())
    rows=[]
    for j in b['joints']:
        v=j['variants']['original_exported'];rows.append({'GT_part':j['gt_mesh_index'],'GT_type':j['GT_type'],'predicted_type':v['predicted_type'],'type_correct':v['type_correct'],'axis_deg':v['axis_direction_error_degrees'] if v['type_correct'] else None,'pivot_percent_GT_D':v['pivot_to_gt_axis_percent_static_diagonal'],'native_range':j['predicted_exported_range']})
    shape=[r['shape_only']['CD_mean_percent_static_diagonal'] for r in geometry['parts'] if 'shape_only' in r]
    type_ok=all(j['type_correct'] for j in rows);geo_ok=b['static_surface_rms_percent_diagonal']<=3 and max(shape)<=3
    signal_ok=all(abs(j['native_range'][1]-j['native_range'][0])>=20 for j in rows);axes_ok=all(j['axis_deg'] is not None and j['axis_deg']<=3 for j in rows);pivot_bad=any(j['pivot_percent_GT_D'] is not None and j['pivot_percent_GT_D']>=1 for j in rows)
    cases[name]={'run':str(run),'static_RMS_percent_GT_D':b['static_surface_rms_percent_diagonal'],'moving_shape_CD_percent_GT_D':shape,'joints':rows,'type_ok':type_ok,'geometry_ok':geo_ok,'signal_ok':signal_ok,'axes_ok':axes_ok,'pivot_bad':pivot_bad,'strict_pass':type_ok and geo_ok and signal_ok and axes_ok and pivot_bad,'shape_protocol':geometry['protocol']}
ga=metrics['original']['mean_all_state_pair_proxy'];gb=metrics['position']['mean_all_state_pair_proxy'];ta=tri['original']['mean_total_line_length_over_diagonal'];tb=tri['position']['mean_total_line_length_over_diagonal']
summary={'scope':'GT-assisted diagnostic screen, not an unbiased benchmark','cases':cases,'strict_pass_count':sum(c['strict_pass'] for c in cases.values()),'selected_limited_trial':str(trial),'limitation':config['qualification'],'integrity':integrity,'original_GT':metrics['original']['GT'],'refined_GT':metrics['position']['GT'],'Gaussian_mean_original':ga,'Gaussian_mean_refined':gb,'Gaussian_reduction_percent':100*(1-gb/ga),'triangle_line_mean_original':ta,'triangle_line_mean_refined':tb,'triangle_line_reduction_percent':100*(1-tb/ta),'axis_changes':metrics['position']['axis_changes'],'optimization_protocol':refinement,'screen_protocol':protocol,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()}
(screen/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False));(REPO/'repro/evidence/pivot-case-screen-results.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
table='| 场景 | 固定部件 RMS / GT D | 活动部件 shape CD / GT D | 类型正确 | ≥20°运动信号 | 筛查结论 |\n|---|---:|---|---:|---|---|\n'
for name,c in cases.items():
    reason='低运动幅度的有限诊断' if name=='10489' else '部件类型/角色、几何不合格'
    table+=f'| {name} | {c["static_RMS_percent_GT_D"]:.3f}% | '+', '.join(f'{x:.3f}%' for x in c['moving_shape_CD_percent_GT_D'])+f' | {sum(j["type_correct"] for j in c["joints"])}/{len(c["joints"])} | {c["signal_ok"]} | {reason} |\n'
gt_table='| GT 部件 | 原始 pivot 误差 | 优化后 | 同一静态 ICP：原始→优化 | 轴方向 |\n|---|---:|---:|---|---|\n'
for a,b in zip(metrics['original']['GT']['joints'],metrics['position']['GT']['joints']):
    gt_table+=f'| {a["gt_mobility_id"]} | {a["pivot_to_GT_axis_percent_static_diagonal"]:.3f}% | {b["pivot_to_GT_axis_percent_static_diagonal"]:.3f}% | {a["same_baseline_registration_pivot_percent_diagonal"]:.3f}% → {b["same_baseline_registration_pivot_percent_diagonal"]:.3f}% | 0° → 0° |\n'
rel=screen.relative_to(REPO/'repro');tr=trial.relative_to(REPO/'repro')
text=f'''# Pivot 缺陷案例筛查与10489仅轴位置优化实测

目前最接近“几何相对较好、轴方向准确但 pivot 有误差”的实际 PAct 案例是 **10489**。四个新增场景未找到严格满足条件的干净案例。对10489的200步只改pivot实测中，重叠代理降低，**GT pivot反而变差**，不能宣称该方法已验证有效。

## 全部筛查结果

{table}

static RMS 为同一固定 canonical/static-AABB 对齐下、以GT静态部件对角线D归一的双向表面RMS。活动部件的 shape CD 是**独立刚性姿态配准后的形状误差下界**，4000点拟合、8000个新面积采样点评分，PCA24个proper rotations + 最多35步90% trimmed ICP，不允许额外scale。它仅排除输入打开姿态与生成rest不同的影响，**不是**装配结果更正，也未用于轴评分或优化。未配准的global误差也完整保留在 `geometry_CASE.json`。形状分数较小不能单独证明关节附近几何足够准确。

10489的两扇门原始类型正确，方向0°，pivot约3.5%D，shape CD约1.83%/2.18%D，故适合有限诊断；但原始range仅−6.525°/+9.948°，不满足预先规定≥20°条件。未将range改成GT或人为大角度来制造优化成功。其GT静态ICP也有配准敏感性，两个对齐协议都保留。

45503、10655、9016、100481均出现GT活动部件被预测为fixed，并伴随部件角色/形状不匹配。它们的大pivot数字混有语义/对应与重建错误，不能当作纯轴位置缺陷。未强制改为GT类型、重排部件以提高GT指标或换seed。旧31249有一扇门类型错误且正确门pivot仅0.596%D；官方Dishwasher没有可靠GT；20411是prismatic，origin不影响运动，均不作为这次主pivot案例。

## 10489：固定几何和方向，只改pivot

{gt_table}

误差为预测pivot到GT无限轴线的垂直距离 / GT static diagonal，并非任意沿轴origin点的欧氏距离。方向始终不变，表中主要固定对齐与复用同一静态ICP均显示pivot恶化。未针对候选重新配准。

41状态、训练source不重合的独立Gaussian均值 {ga:.9g} → {gb:.9g}，下降 **{summary['Gaussian_reduction_percent']:.2f}%**。21状态全部原始三角形的平均交线/D {ta:.6f} → {tb:.6f}，下降 **{summary['triangle_line_reduction_percent']:.2f}%**。它们是重叠/表面交线代理，不是真实实体体积。两个pivot均达到5% asset diagonal的位移上限；原始pivot锚点在运动中的位移变化上限分别约0.867%/0.569% asset diagonal，反映小角度下较大的轴位置变化也可能只带来较小的表面运动差异。

![独立碰撞曲线及GT pivot误差]({tr}/diagnostics/pivot_and_collision.png)

![GT铰链垂直平面截面与原始/优化轴位置]({tr}/diagnostics/GT_pivot_sections.png)

红叉是GT轴穿过该切面的位置，蓝点是原始轴，紫色加号是优化轴。曲线为GT/原始预测Mesh与同一切面的实际交线；两套PAct几何共享且未改变，不是把几何拟合到GT轴。保留GT开口，不填充实体。平面与坐标基记录在 [figure_info.json]({tr}/diagnostics/figure_info.json)。

[原始/优化双视角21状态动画]({tr}/previews_10489/original_position.mp4) · [运动中点交线位置]({tr}/hotspots_10489/intersections_state10.png) · [打开终点对照]({tr}/previews_10489/mesh_state20_comparison.png)

这个例子说明：即使形状相对较好且方向固定，当前纯排斥目标也可通过把pivot推离GT来减少代理。小角度、未施加铰链接触/装配锚定以及残余几何误差都可能影响该结果，不能仅凭本例将失败唯一归因于其中某项；但也不能把所有效果不佳都归因于20411几何差。**本轮没有找到或制造一个干净的成功案例。**

## 运行与证据

沿用原来的position-only目标和超参数：冻结所有Mesh/GS、方向、type和range；21个训练状态，2048 source/part、全部opacity≥.05 Gaussian targets；pivot≤5% assetD，正则.05，lr=.0005D；固定最后200步，GT不参与梯度或checkpoint选择。实际资产梯度和5步GPU smoke先通过，再用tmux正式运行。严格校验200步loss与pivot梯度finite、方向梯度未启用、Mesh/GS SHA和全部非origin字段不变，t=0变换/指标/去文字图像像素一致，41/21状态和21帧视频完整。所有验证通过。

案例选择本身明确使用GT，是诊断筛选，不能当无偏测试集均值；四个新增场景和全部失败保留。输入为单RGB+GT Mesh投影得到的oracle 2D part mask，silhouette IoU>.98；GT joint数值不进入模型或优化。10489、45503、10655在本地公开SINGAPO train清单中；实际PAct完整训练清单未核实，不作未见物体泛化声称。真实实体体积/穿透深度保持unavailable。

新增重建固定frame0/seed42、25+25步、CFG7、arti_mean_num20、同一官方权重，无重新训练网络。原始输出路径和严格模型load审计保留在各 `repro/current_gt_CASE.txt` 指向的目录。筛查控制日志、阶段命令、耗时、完整GT/形状数值、优化checkpoint/Adam/history/seal在 `{screen}`；仅轴位置实验在 `{trial}`。旧实验未覆盖，Part²GS没有恢复。

[预先协议及扩展记录](PIVOT_CASE_SCREEN_PROTOCOL.md) · [完整筛查与实测JSON](evidence/pivot-case-screen-results.json) · [完整性检查]({tr}/integrity.json) · [固定shape诊断]({rel}/geometry_10489.json)

复查或中断恢复：

```bash
PYTHONDONTWRITEBYTECODE=1 /root/autodl-tmp/conda_envs/pact/bin/python repro/tools/run_pivot_only_case.py --root "$(cat repro/current_pivot_screen.txt)/10489_native_position"
python repro/tools/monitor_collision_refinement.py --root "$(cat repro/current_pivot_screen.txt)/10489_native_position" --watch
```

仅本地独立分支提交代码、工具和文档，不上传权重、不推送官方上游。
'''
path=REPO/'repro/PIVOT_CASE_SCREEN_RESULTS.md';path.write_text(text)
for link in re.findall(r'\]\(([^)]+)\)',text):assert (path.parent/link).exists(),link
(trial/'artifact_status.json').write_text(json.dumps({'state':'complete','integrity_passed':True,'report':str(path)},indent=2))
print(json.dumps({'strict_pass_count':summary['strict_pass_count'],'report':str(path),'Gaussian_reduction_percent':summary['Gaussian_reduction_percent'],'triangle_reduction_percent':summary['triangle_line_reduction_percent']},indent=2))
