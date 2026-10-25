# 新场景的运动碰撞筛选结果

新增案例为此前未做过 PAct 推理的 45194 和 8961。两者均使用真实本地数据、固定首帧及 seed42、官方模型与原版 25+25 步推理。GT Mesh 投影的 oracle 2D 部件 mask 是输入；GT 轴和关节类型不输入模型。新场景指本轮之前没有复现，不代表官方模型训练时未见过。

原始产物目录：[new_motion_case_screen_20261025T131141Z](/root/autodl-tmp/PAct/repro/outputs/new_motion_case_screen_20261025T131141Z)。所有尝试均保留；本轮使用 GT 做诊断筛选，不能作为无偏泛化指标。

| 场景 | 类型正确 | 静态 RMS / GT D | 合格 |
| --- | --- | --- | --- |
| 45194 | 3/4 | 13.118% | 否 |
| 8961 | 2/2 | 4.198% | 否 |

## 45194

筛选结论：静态几何 RMS 超过 3%；GTpart1 shape-only CD 超过 3%；GTpart2 类型错误；GTpart2 shape-only CD 超过 3%；GTpart3 shape-only CD 超过 3%；GTpart4 shape-only CD 超过 3%。

| GT 部件 / 预测 node | GT → 预测类型 | 有效方向误差 | Pivot / GT D | 原生幅度 | Shape-only CD / GT D |
| --- | --- | --- | --- | --- | --- |
| 1 / 0 | revolute → revolute | 0.000° | 56.029% | 43.135 degrees | 26.833% |
| 2 / 1 | revolute → fixed | unavailable | N/A | 0.000 model units | 32.650% |
| 3 / 3 | prismatic → prismatic | 0.997° | N/A | 0.442 model units | 6.528% |
| 4 / 4 | prismatic → prismatic | 0.645° | N/A | 0.494 model units | 5.105% |

推理 101.2 秒，峰值观测显存 25678 MiB。严格 checkpoint 加载审计通过。

[逐关节运动代理曲线](/root/autodl-tmp/PAct/repro/outputs/new_motion_case_screen_20261025T131141Z/motion_45194/motion_paths.png)；[21 状态/轨迹的原始数据](/root/autodl-tmp/PAct/repro/outputs/new_motion_case_screen_20261025T131141Z/motion_45194/metrics.json)。

[官方原始运动视频](/root/autodl-tmp/PAct/repro/outputs/gt_45194_20261025T131221Z/seed42_slatcfg7.0_sscfg7.0_sssteps25_slatsteps25_artioutmean_feature_regression_steps/00__articulation_animation.mp4)。

[GT 轴叠加图](/root/autodl-tmp/PAct/repro/outputs/gt_45194_20261025T131221Z/joint_evaluation/axes_gtpart1.png)；[逐部件几何结果](/root/autodl-tmp/PAct/repro/outputs/new_motion_case_screen_20261025T131141Z/geometry_45194.json)。

## 8961

筛选结论：静态几何 RMS 超过 3%；GTpart1 原生运动小于 20°；GTpart2 原生运动小于 20°。

| GT 部件 / 预测 node | GT → 预测类型 | 有效方向误差 | Pivot / GT D | 原生幅度 | Shape-only CD / GT D |
| --- | --- | --- | --- | --- | --- |
| 1 / 0 | revolute → revolute | 0.000° | 18.202% | 0.874 degrees | 0.868% |
| 2 / 1 | revolute → revolute | 0.000° | 17.498% | 0.485 degrees | 0.787% |

推理 70.9 秒，峰值观测显存 17306 MiB。严格 checkpoint 加载审计通过。

[逐关节运动代理曲线](/root/autodl-tmp/PAct/repro/outputs/new_motion_case_screen_20261025T131141Z/motion_8961/motion_paths.png)；[21 状态/轨迹的原始数据](/root/autodl-tmp/PAct/repro/outputs/new_motion_case_screen_20261025T131141Z/motion_8961/metrics.json)。

[官方原始运动视频](/root/autodl-tmp/PAct/repro/outputs/gt_8961_20261025T131422Z/seed42_slatcfg7.0_sscfg7.0_sssteps25_slatsteps25_artioutmean_feature_regression_steps/00__articulation_animation.mp4)。

[GT 轴叠加图](/root/autodl-tmp/PAct/repro/outputs/gt_8961_20261025T131422Z/joint_evaluation/axes_gtpart1.png)；[逐部件几何结果](/root/autodl-tmp/PAct/repro/outputs/new_motion_case_screen_20261025T131141Z/geometry_8961.json)。

## 解释边界

运动诊断逐关节采样 21 个状态，其余关节固定在模型原生 t=0，并单独保留全关节同步路径。21 点采样不保证连续轨迹绝无碰撞，也没有穷举所有多关节组合。只使用预测范围，没有用 GT 开合角度扩大范围。

曲线左图是各状态的绝对 Gaussian 最近中心重叠代理，右图是相对起始状态的变化，仅用于诊断运动带来的变化，没有修改优化 Loss。它不是穿透体积；sigma gap 也不是实体穿透深度。模型 Mesh 的实体合法性未建立，真实体积和穿透深度为 unavailable。

共同全局对齐下的几何误差保持不变；shape-only 指每个活动部件独立刚体配准、不改变尺度后的形状诊断，不能看成修正后的整物体装配。关节评分仍用原有固定坐标变换，另保留静态 ICP 敏感性结果，不按 GT 关节误差重新排列部件或修正类型。

8961 的门片形状误差较小且 pivot 缺陷明显，因此是下一轮规定运动范围诊断的候选。但窗框 RMS 为 4.20%，未达到预设 3% 几何阈值，原生转角也不足 1°。若之后人为规定较大开合范围，必须单列为运动范围假设实验，不能称为原版预测范围结果，也不能把 GT 角度悄悄输入优化。

表中方向误差取原版导出的轴；revolute 在官方后处理中吸附到最近坐标轴，0° 不代表原始 head 已准确到 0°。原始 head 的方向误差保留在原始 joint_errors.json 与本轮 evidence 中。

本轮完成新场景的原始推理、GT/几何检查和运动轨迹诊断，没有启动新的碰撞优化。若没有合格案例，不把复杂几何或类型错误包装成只需校正轴的问题。
