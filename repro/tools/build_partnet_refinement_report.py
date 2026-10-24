"""Build a measured-results report only after all required artifacts pass checks."""
import argparse,csv,datetime,json,pathlib,subprocess
import numpy as np
REPO=pathlib.Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=pathlib.Path(a.root).resolve()
config=json.loads((root/'experiment.json').read_text());name,run=next(iter(config['cases'].items()));run=pathlib.Path(run)
assert name=='20411' and config['variants']==['direction']
integrity=json.loads((root/'integrity.json').read_text());assert integrity['passed']
gt=json.loads((root/'GT_geometry_and_motion.json').read_text());env=json.loads((root/'environment.json').read_text());status=json.loads((run/'status.json').read_text());input_info=json.loads((run/'input_provenance.json').read_text())
gs={m:json.loads((root/f'eval_{name}_{m}/metrics.json').read_text()) for m in ['original','direction']}
mesh={m:json.loads((root/f'mesh_{name}_{m}/metrics.json').read_text()) for m in gs}
def improvement(old,new):return 100*(1-new/old) if old else None
def num(x,d=6):return f'{x:.{d}f}' if x is not None else 'unavailable'
A,B=gt['methods']['original'],gt['methods']['direction'];D=gt['alignment']['GT_static_diagonal']
quality=gt['alignment']['mesh_quality'];audit=json.loads((root/'mesh_self_intersections.json').read_text())
formal=json.loads((root/'20411_direction/status.json').read_text());protocol=json.loads((root/'20411_direction/protocol.json').read_text())
rel=root.relative_to(REPO/'repro');runrel=run.relative_to(REPO/'repro')
curve_rows=[]
for i in range(41):
    row={'state':i,'t':gs['original']['states'][i]['t'],'heldout_time':i%2==1}
    for m in gs:
        row[m+'_gaussian_proxy']=gs[m]['states'][i]['mean_pair_proxy']
        row[m+'_sampled_center_min_distance']=min(p['min_heldout_source_to_full_target_center_distance'] for p in gs[m]['states'][i]['pairs'])
        row[m+'_triangle_line_over_diagonal']=sum(p['line_length_over_diagonal'] for p in mesh[m]['states'][i//2]['pairs']) if i%2==0 else None
    curve_rows.append(row)
with (root/'motion_metrics.csv').open('w',newline='') as file:
    writer=csv.DictWriter(file,fieldnames=curve_rows[0]);writer.writeheader();writer.writerows(curve_rows)
main_rows=[
 ('关节类型','prismatic，1/1 正确','prismatic，1/1 正确','保持不变'),
 ('GT 轴方向误差，固定 canonical frame',num(A['axis_direction_error_degrees'],3)+'°',num(B['axis_direction_error_degrees'],3)+'°','误差增加 '+num(B['axis_direction_error_degrees']-A['axis_direction_error_degrees'],3)+'°'),
 ('同一个静态 ICP 下的 GT 方向误差',num(A['same_static_ICP_axis_error_degrees'],3)+'°',num(B['same_static_ICP_axis_error_degrees'],3)+'°','同样恶化'),
 ('Gaussian 重叠代理，41 状态均值',f"{gs['original']['mean_all_state_pair_proxy']:.9g}",f"{gs['direction']['mean_all_state_pair_proxy']:.9g}",num(improvement(gs['original']['mean_all_state_pair_proxy'],gs['direction']['mean_all_state_pair_proxy']),2)+'% 下降'),
 ('Gaussian 重叠代理，20 个 held-out 时间点',f"{gs['original']['mean_interleaved_heldout_time_proxy']:.9g}",f"{gs['direction']['mean_interleaved_heldout_time_proxy']:.9g}",num(improvement(gs['original']['mean_interleaved_heldout_time_proxy'],gs['direction']['mean_interleaved_heldout_time_proxy']),2)+'% 下降'),
 ('Gaussian 最坏状态代理',f"{gs['original']['max_state_mean_pair_proxy']:.9g}",f"{gs['direction']['max_state_mean_pair_proxy']:.9g}",'t=0，不变'),
 ('全三角形交线长度 / asset diagonal，21 状态均值',num(mesh['original']['mean_total_line_length_over_diagonal']),num(mesh['direction']['mean_total_line_length_over_diagonal']),num(improvement(mesh['original']['mean_total_line_length_over_diagonal'],mesh['direction']['mean_total_line_length_over_diagonal']),2)+'% 下降'),
 ('非平行交线长度 / asset diagonal，21 状态均值',num(mesh['original']['mean_transverse_line_length_over_diagonal']),num(mesh['direction']['mean_transverse_line_length_over_diagonal']),num(improvement(mesh['original']['mean_transverse_line_length_over_diagonal'],mesh['direction']['mean_transverse_line_length_over_diagonal']),2)+'% 下降'),
 ('GT 闭合→打开相对位移 RMSE / GT static diagonal',num(A['trajectory_displacement_RMSE_percent_static_diagonal'],3)+'%',num(B['trajectory_displacement_RMSE_percent_static_diagonal'],3)+'%','恶化'),
 ('终点相对位移误差 / GT static diagonal',num(A['endpoint_displacement_error_percent_static_diagonal'],3)+'%',num(B['endpoint_displacement_error_percent_static_diagonal'],3)+'%','恶化'),
 ('GT 尺度下行程绝对误差',num(A['travel_absolute_error_GT_units']),num(B['travel_absolute_error_GT_units']),'行程冻结'),
 ('Pivot / origin 误差','N/A','N/A','平移关节不具可识别 pivot'),
 ('真实实体交叠体积 / 穿透深度','unavailable','unavailable','没有以未定义值填 0')]
table='| 指标 | 原始 PAct | 减少重叠方向优化 | 变化 / 说明 |\n|---|---:|---:|---|\n'+'\n'.join('| '+' | '.join(row)+' |' for row in main_rows)
geom=gt['rest_geometry'];geom_table='| 原始静止几何 | 对称平均表面距离 / GT static diagonal |\n|---|---:|\n'+'\n'.join(f'| {label} | {geom[key]["symmetric_mean_distance_percent_static_diagonal"]:.3f}% |' for label,key in [('固定桌体','fixed'),('移动抽屉，对 GT closed','moving_vs_GT_closed'),('整体，对 GT closed','whole_vs_GT_closed'),('移动抽屉，对 GT open input，仅诊断','moving_vs_GT_open_input')])
qtable='| Mesh | 三角形数 | 精确焊接后边界边 | 非流形边 | 连通分量 | watertight / winding |\n|---|---:|---:|---:|---:|---|\n'
for label,q in [('PAct 桌体',quality['prediction']['0']),('PAct 抽屉',quality['prediction']['1']),('GT 桌体',quality['GT']['fixed']),('GT 抽屉',quality['GT']['moving'])]:
    qtable+=f'| {label} | {q["faces"]} | {q["boundary_edges"]} | {q["nonmanifold_edges"]} | {q["components"]} | {q["watertight"]} / {q["winding_consistent"]} |\n'
inference_video,=run.rglob('00__articulation_animation.mp4')
timings={q.stem:json.loads(q.read_text()) for q in root.glob('*_timing.json')}
timing_rows='\n'.join(f'| {k.removesuffix("_timing")} | {v["seconds"]:.2f} |' for k,v in timings.items())
direction_change=gs['direction']['axis_changes'][0]['direction_change_degrees']
old=np.array(json.loads(pathlib.Path(status['object']).read_text())['diffuse_tree'][1]['joint']['axis']['direction']);new=np.array(json.loads((root/'20411_direction/candidate.json').read_text())['diffuse_tree'][1]['joint']['axis']['direction'])
q=A['predicted_range_model_units'][1];displacement_delta=gt['alignment']['scale']*q*(new/np.linalg.norm(new)-old/np.linalg.norm(old))
summary={'case':name,'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'branch':subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip(),
    'original_run':str(run),'experiment_root':str(root),'input_protocol':input_info['protocol'],'selection':'Fixed seed42 original / fixed step200 refinement; no GT selection',
    'original':{'GT_axis_deg':A['axis_direction_error_degrees'],'same_static_ICP_GT_axis_deg':A['same_static_ICP_axis_error_degrees'],'Gaussian_proxy_mean':gs['original']['mean_all_state_pair_proxy'],'triangle_line_over_D_mean':mesh['original']['mean_total_line_length_over_diagonal'],'trajectory_RMSE_percent_GT_D':A['trajectory_displacement_RMSE_percent_static_diagonal']},
    'refined':{'GT_axis_deg':B['axis_direction_error_degrees'],'same_static_ICP_GT_axis_deg':B['same_static_ICP_axis_error_degrees'],'Gaussian_proxy_mean':gs['direction']['mean_all_state_pair_proxy'],'triangle_line_over_D_mean':mesh['direction']['mean_total_line_length_over_diagonal'],'trajectory_RMSE_percent_GT_D':B['trajectory_displacement_RMSE_percent_static_diagonal']},
    'Gaussian_proxy_reduction_percent':improvement(gs['original']['mean_all_state_pair_proxy'],gs['direction']['mean_all_state_pair_proxy']),
    'triangle_line_reduction_percent':improvement(mesh['original']['mean_total_line_length_over_diagonal'],mesh['direction']['mean_total_line_length_over_diagonal']),
    'max_Gaussian_proxy_unchanged':gs['original']['max_state_mean_pair_proxy']==gs['direction']['max_state_mean_pair_proxy'],
    'refinement_direction_change_degrees':direction_change,'endpoint_refined_minus_original_GT_frame':displacement_delta.tolist(),
    'pivot_error':None,'true_solid_volume':None,'true_penetration_depth':None,'GT_registration':gt['static_ICP_sensitivity'],
    'rest_geometry':geom,'mesh_audit':audit,'integrity':integrity,'environment':env,'inference_seconds':status['elapsed_seconds'],'formal_seconds':formal['elapsed_seconds'],'timings':timings,
    'triangle_evaluation_acceleration':json.loads((root/'evaluation_acceleration.json').read_text()),
    'experiment_configuration':config,'refinement_protocol':protocol,'input_provenance':input_info,'original_inference_command':json.loads((run/'command.json').read_text())}
(root/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
evidence=REPO/'repro/evidence';(evidence/f'pact-{name}-collision-results.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
text=f'''# PAct 20411：原始重建与减少重叠轴优化实测

20411 已完成独立 PAct 重建、真实资产梯度检查、5 步 smoke、200 步固定最终方向优化、41 状态 Gaussian 评估、21 状态全三角形检查和三方动画。**原始轴已经很准确；当前减少重叠 loss 没有改善 GT 轴，而是用偏斜的抽屉轨迹换取更低重叠。** 所有原始结果和负面结果均保留，没有事后挑选 seed、步数或调参。

## 主要实测

{table}

角度采用 acos(abs(dot(unit axes)))，不区分轴正负。行程和相对位移诊断同时保留真实运动符号。Gaussian 数字是 opacity 加权最近中心核，并非实体占比；三角形交线包含合法接触，并非实体交叠体积。优化改变方向 {direction_change:.3f}°，小于相对于原始轴的 5° 上限；5.092° 是相对于 GT 的误差，两者定义不同。

原始 range 为 {A['predicted_range_model_units']}；经固定桌体尺度归一后，预测移动 {A['predicted_travel_GT_units']:.6f}，实际 GT 六状态端点移动 {A['GT_travel_units']:.6f}。两方法行程完全相同。优化相对原始终点额外位移，GT 坐标 xyz 为 {displacement_delta.tolist()}；y 为向上，故抽屉额外下沉 {abs(displacement_delta[1]):.6f}，约为固定部件对角线的 {100*abs(displacement_delta[1])/D:.3f}%。这是一条偏离真实平移方向的直线，不是抽屉刚体几何发生形变。

## 输入、GT 使用与选择协议

真实来源为 `/root/autodl-tmp/FreeArt3D/refinement/data/inputs/20411`，类别 Table。预先固定 05_pure.png，单 RGB、两个二维部件 mask、seed42、25+25 diffusion steps、CFG7、arti_mean_num20，沿用同一官方预训练模型。现有 *_seg.png 含蓝色地面，不是部件 ID；未将它误当 mask。复现 FreeArt3D 原版相机参数，将真实 GT Mesh 的可见三角形投影为二维标签；silhouette IoU={input_info['silhouette_iou']:.9f}，原版 37×37 reader 两个部件均可见，mask reorder 对应固定桌体 node0、抽屉 node1。

**这是 GT-mask / oracle segmentation 输入。GT Mesh 和端点部件划分经二维 mask 进入了 PAct，不能称完全无 GT 或 RGB-only。GT 关节 type、axis、origin、range 没有输入推理或本次优化。** 数据核查期间查看了 GT 元信息、六状态位移，并做过原始轴方向的 sanity check；不能声称 GT 文件首次访问发生在 seal 后。正式对照评分在最终候选 seal 后运行，GT 不用于参数更新、超参数调整、对齐方案下的候选挑选或 checkpoint 选择。方向与行程还由实际六状态 Mesh 位移独立核实。未使用 FreeArt3D 预测替代 PAct 输出。FreeArt3D、VideoArtGS 均未修改，已暂停的 Part²GS 没有恢复。

Original Protocol 为固定官方模型、固定单次 seed 的原始导出（含原版 postprocess）；本例 prismatic 不发生 revolute axis snapping。GT-independent Selection 为完全相同几何和关节类型/行程上的最终第 200 步候选。两者都没有以 GT joint error 挑选 checkpoint。position-only 对平移关节严格无效，both 等价于 direction-only，因此没有重复训练或把它们伪装成独立实测。

## 几何与可视化

{geom_table}

采用每侧 18000 个确定性面积采样点，Chamfer 为双向**未平方 Euclidean 距离均值**，以 GT static diagonal 归一。不是论文未知单位下 CD 数字的直接对照。本轮几何冻结，因此上述静止几何指标在优化前后完全相同。输入是打开状态，PAct 输出是闭合 canonical rest；rest 对 closed 的距离显著小于对 open input 的距离，支持闭合→打开的运动对照约定。相对位移 RMSE 不包含 rest 几何和初始位置误差，也不等于完整 surface trajectory error。

实际纹理动画能看到完整桌体与抽屉正常开合，主要木纹和面板外观保留；抽屉前面板有弯曲/边缘不平整，GT 与 PAct 的抽屉尺寸及细节仍有差异。下列三方图片使用相同相机、实际原始三角形和原始行程，未补洞、凸化或缩短行程。

![输入 RGB 与部件 mask（这张是输入，不是重建）]({runrel}/{next(run.rglob('00__exploded_part.png')).relative_to(run)})

![PAct 原始纹理运动采样]({rel}/original_textured_animation_samples.png)

![GT / 原始 / 优化，打开状态，双视角]({rel}/GT_mesh_previews/GT_original_refined_state20.png)

[三方 21 状态视频]({rel}/GT_mesh_previews/GT_original_refined.mp4) · [原始与优化的双视角视频]({rel}/previews_20411/original_direction.mp4) · [原始 PAct 纹理视频]({inference_video.relative_to(REPO/'repro')}) · [闭合]({rel}/GT_mesh_previews/GT_original_refined_state00.png) · [中间]({rel}/GT_mesh_previews/GT_original_refined_state10.png)

![GT 方向三维叠加]({rel}/figures/GT_direction_overlay.png)

方向图的三条线使用同一**显示锚点**；平移关节没有可识别 pivot，不能根据这张图宣称轴位置精度。

## 全运动范围的重叠与交线

![41 状态 Gaussian、21 状态全三角形交线、采样最小中心距离]({rel}/figures/collision_curves.png)

Gaussian 优化 source 每部件 2048，seed0；评估 source 每部件 12000，seed2026，严格与训练 source 不重合。两方向 NN 的 targets 均为全部 opacity≥.05 Gaussian（桌体 {gs['original']['full_counts']['0']}，抽屉 {gs['original']['full_counts']['1']}），不是把 targets 稀疏化后假装碰撞消失。41 个评估状态中有 20 个时间间隔中点，训练只用 21 个均匀状态。CUDA Kaolin NN 与 CPU cKDTree 在真实 source points 上通过独立核查。最小距离仅是 held-out Gaussian source 到全 targets 的采样中心距离，不是全 Mesh clearance。

全三角形检查使用原始 728084+362228 个三角形、VTK AllContacts、零 box/cell tolerance；长度≤1e-6 asset diagonal 的线段不计入长度，normal angle>5° 的非平行交线单独报告。保留各状态 JSON、交线数量、长度、t=0/.5/1 的空间交线和 [逐状态 CSV]({rel}/motion_metrics.csv)。红线是实际表面交线，包含可能合法接触，不能直接标记为实体穿透区域。

旧版整网格 VTK 每状态约两分钟，原始/优化各完成六状态后保存了 partial outputs。为完成全范围而做的**评估效率修复**：按三角形质心递归分块，每块≤12000 个，所有原始三角形严格各属于一块；保守 AABB 只排除不相交块对，余下继续执行同样 VTK 判定。两方法各六个实际旧状态交叉核查，contact pairs / nonzero segments 计数完全一致，归一长度差≤1.52e-7，随后才运行完整 21 状态。没有 decimation 或改变几何，训练 loss、参数、结果选择未受影响。源提交 `55a37a0`，验证见 [完整性证据]({rel}/integrity.json)。旧评估结果保存在 `legacy_unpartitioned_mesh_*` 目录。

![中间状态的交线空间分布，红色]({rel}/hotspots_20411/intersections_state10.png)

[闭合交线]({rel}/hotspots_20411/intersections_state00.png) · [打开交线]({rel}/hotspots_20411/intersections_state20.png) · [交线区域坐标]({rel}/hotspots_20411/regions.json)

闭合状态 t=0 的变换与轴方向无关：两结果的 Gaussian 最大重叠代理、闭合交线及无文字几何渲染像素经检查一致。平均代理下降不代表整个范围或最坏姿态已无碰撞。优化后抽屉向下偏移，会带来与真实导轨不一致的轨迹/间隙；没有 geometry loss 或观测运动信号阻止它以这种方式降低代理。

三视角红线定位主要在抽屉两侧上沿与桌面下方的接合区域；优化后通过向下偏移将这些表面分开。交线图颜色按部件索引（桌体橙色、抽屉绿色），三方实体 Mesh 预览则统一桌体灰色、抽屉橙色。两类图的红线含义不变，仍不能直接换算为实体穿透深度。

## Mesh 合法性与指标边界

{qtable}

PAct 的两个部件虽拓扑封闭且 winding consistent，但封闭不自动意味着没有自交或多层壳。Open3D 原始精确焊接三角形的 self-intersection 检查分别在 45 秒预算内未完成，结果为 **unknown**；未将 timeout 当 False，也没有宣布实体合法。GT 两部件还存在明显开口及非流形边。连通分量数只说明拓扑，不能把所有分量直接认定为视觉碎片。因此真实交叠体积及穿透深度保持 unavailable，未以 undefined 充当 0。

PSNR / SSIM / LPIPS 本轮未计算：当前任务使用单 RGB 生成式重建，没有建立可审计的相同相机/曝光/背景的 photometric evaluation，不能把不同视角截图当定量重建指标。

## 对物理约束机制的判断

本轮追加的是独立 **冻结几何后的轴优化 loss**，没有重训 PAct 网络，也没有改变原始模型算法和 loss。目标为 21 状态 nearest-center Gaussian proxy / 初始 proxy + .05 × 归一方向位移二次正则。生产优化每次刷新 NN；离散邻居选择的分段边界不做梯度，而选中分支内的刚体变换对轴方向可微。

方向梯度检查使用实际资产，固定 NN 分支的 finite difference 相对误差为 7.57e-5；200 步 loss/方向梯度均 finite。方向梯度确实优化 Joint，不更新任何 Gaussian center、scale、opacity 或 Mesh。平移 origin 完全不进入 kinematics，pivot 梯度为 N/A。没有增加 Contact、Velocity、GT axis 或其他约束。

这个 case 的实测回答是：**原始 PAct 轴已经足够准确；单独降低 Gaussian 重叠，在此损害了轴方向和真实运动误差。** 固定 canonical frame 与复用同一静态 ICP 都得到这一结论。局部静态 ICP 旋转 {gt['static_ICP_sensitivity']['rotation_delta_degrees']:.3f}°，独立静态采样 RMS 从 {gt['static_ICP_sensitivity']['heldout_before']['symmetric_RMS_percent_static_diagonal']:.3f}% 变为 {gt['static_ICP_sensitivity']['heldout_after']['symmetric_RMS_percent_static_diagonal']:.3f}%，并未全面改善；保留原主结果，避免配准误差被隐去。未按某个对齐方案重选候选。

不能据此声称碰撞约束普遍无效，也不能将代理下降称为 GT joint refinement 成功。20411 更适合作为“轴原本准确、几何仍有偏差时，约束可能把运动推偏”的负面案例。下一轮研究应保留几何/轴/真实运动/碰撞四方面的独立指标；本轮没有根据失败结果另开调参搜索。

## 与既有 FreeArt3D 结果的关系

历史 FreeArt3D 报告位于 `/root/autodl-tmp/FreeArt3D/GT_NONPENETRATION_COMPARISON.md` 和 `NONPENETRATION_REFINEMENT.md`。它们使用的多状态图像、重建流程、冻结对象和配准协议不同；本轮为单 RGB + oracle 2D mask 的 PAct，不能将旧数字并成公平 head-to-head 排名，也没有复用旧 FreeArt3D 预测 Mesh。这里能确认 PAct 独立重建成功、类型正确、轴在本协议下很准，不能据单例断言所有几何效果都更好。

本地保存的 SINGAPO split 中，`Table/20411` 以及三个增强版本出现在 train；实际 PAct 预训练完整样本清单未核实。因此本例不能支持未见物体泛化结论。

## 环境、耗时与追溯

独立环境 `{env['executable']}`，Python3.10.22，torch {env['torch']}，CUDA runtime {env['CUDA_runtime']}，{env['GPU']}。原官方 PAct upstream commit 为 `a5ccec5056e7b27f39ddb5b83613c0efba9b2861`，保留 tag `upstream-original-a5ccec5`；官方权重 revision `8c6b56c7011454bbfb45a83e2dbfa6bbbfde63e9`，已核对 14 文件 SHA，所有 required model weights strict load，无缺失或 unexpected parameters。

推理命令记录的 Git SHA 为 `{json.loads((run/'command.json').read_text())['git_sha']}`，同时保留当时修改的 `source.patch`；准备器提交 `10198e7`，固定实验协议和 controller 提交 `164be67`，可视化/审计工具提交 `ec7a513`。优化器及 common helper 的 SHA、每个原始 Mesh/GS SHA 均写入 protocol，最终完整性检查再次核对；没有修改原版物理 loss。

PAct 推理 {status['elapsed_seconds']:.2f}s，轮询观察 GPU 峰值 {status['peak_observed_gpu_mib']}MiB；200 步优化 {formal['elapsed_seconds']:.2f}s，PyTorch GPU allocated 峰值 {formal['gpu_peak_allocated_MiB']:.2f}MiB。两种显存指标定义不同，不直接比较；优化包含 CPU 全 targets KD-tree 查询，不因 GPU 低利用率认定停止。表中独立阶段并行，耗时不能直接相加。

| 阶段 | 秒 |
|---|---:|
{timing_rows}

原始结果 `{run}`；完整对照 `{root}`。配置、环境、pip freeze、每阶段命令和日志、原始 raw head、checkpoint strict-load audit、Adam/checkpoint/history、seal、独立 metrics、图像和视频均在这些目录。优化器每 10 步保存参数/Adam，可按同一 protocol/source SHA 恢复；原始结果没有被覆盖。

[预先固定的协议](PACT_20411_PROTOCOL.md) · [完整性检查]({rel}/integrity.json) · [紧凑 JSON 结果](evidence/pact-20411-collision-results.json) · [环境]({rel}/environment.json) · [Mesh self-intersection 审计]({rel}/mesh_self_intersections.json)

分支为 `research/pact-20411-collision-20261025`，仅本地提交代码、评估工具与文档；没有推送官方上游，没有上传模型或大型输出。
'''
(REPO/'repro/PACT_20411_RESULTS.md').write_text(text)
print(json.dumps({'report':str(REPO/'repro/PACT_20411_RESULTS.md'),'Gaussian_proxy_reduction_percent':summary['Gaussian_proxy_reduction_percent'],'triangle_line_reduction_percent':summary['triangle_line_reduction_percent']},indent=2))
