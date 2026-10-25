# 运动碰撞 Loss 的重建 A/B 实测

直接比较B与A：三组中 0 组 Whole Chamfer 降低，0 组留出 PSNR 提高，3 组 Gaussian 运动重叠代理降低。重建细化本身的收益应看A与R0；额外碰撞项的收益必须看B与A，不能混为一谈。

环境：NVIDIA GeForce RTX 4080 SUPER，Python 3.10.22，PyTorch 2.4.0+cu121，CUDA runtime 12.1。优化实现 commit `429c304f6a4a7a1fa43d2baedcb6667012bccf43`；完整环境、pip freeze、每组源码SHA与原始文件SHA均在产物目录。

本轮完成三组成对对照、共六次200步优化：两组都进行相同的图像重建细化，只有运动碰撞 Loss 权重不同。R0保留共同图像注册后的原始几何和轴。它是多视角后处理消融，不是 PAct 原始单视图训练或整个网络微调。

输入为每场景六个训练相机的 RGB 与 GT Mesh 投影的 oracle 2D 部件 mask，六个不同相机只用于最后评价。原始 PAct 推理仅首帧；细化使用额外视角，不能把额外观测带来的改进归因于碰撞 Loss。GT Mesh参与二维训练mask的制作；优化器不读取三维GT文件，不使用三维GT距离/法线或GT关节参数作监督，不按GT选checkpoint。GT已在此前基线诊断中被查看，因此不是盲测。

产物目录：[reconstruction_ablation_20261025_v1](/root/autodl-tmp/PAct/repro/outputs/reconstruction_ablation_20261025_v1)。

| 对照 | 留出 PSNR：A → B | Whole CD：A → B（% GT D） | 轴误差：A → B | Pivot：A → B（% GT D） | Gaussian 运动代理变化 | Mesh表面交线变化 |
| --- | --- | --- | --- | --- | --- | --- |
| 8961_native | 14.106 → 14.013 dB | 2.153 → 2.485 | 0.7145 → 0.5252° | 23.7819 → 22.9853 | -96.35% | -98.89% |
| 8961_sweep90 | 14.106 → 13.854 dB | 2.150 → 2.536 | 0.7169 → 0.6054° | 23.7805 → 23.0210 | -97.00% | -99.28% |
| 45194_native | 15.490 → 15.308 dB | 6.541 → 7.064 | 2.2259 → 2.5702° | 62.9842 → 62.9483 | -89.05% | -59.82% |

## 8961_native

| 指标 | R0 | A：重建 | B：重建＋运动碰撞 | B−A |
| --- | --- | --- | --- | --- |
| PSNR | 12.0349 | 14.1064 | 14.0131 | -0.0933 |
| foreground_PSNR | 11.0456 | 11.9617 | 11.9450 | -0.0167 |
| SSIM | 0.7651 | 0.8141 | 0.8113 | -0.0028 |
| LPIPS_vgg | 0.1914 | 0.1586 | 0.1599 | 0.0013 |
| fixed_CD_percent_GT_D | 5.2136 | 3.9669 | 4.1162 | 0.1493 |
| moving_CD_percent_GT_D | 1.9145 | 1.5533 | 1.9140 | 0.3606 |
| whole_CD_percent_GT_D | 3.2657 | 2.1531 | 2.4848 | 0.3317 |
| mean_axis_degrees | 0.0000 | 0.7145 | 0.5252 | -0.1893 |
| mean_pivot_percent_GT_D | 24.9207 | 23.7819 | 22.9853 | -0.7966 |
| Gaussian_motion_proxy | 0.0274 | 0.0155 | 0.0006 | -0.0149 |
| worst_Gaussian_motion_state | 0.0290 | 0.0184 | 0.0007 | -0.0177 |
| mesh_surface_intersection_line_over_D | 3.2252 | 3.9914 | 0.0444 | -3.9470 |
| max_motion_added_Gaussian_proxy_over_own_initial | 0.0000 | 0.0043 | 0.0000 | -0.0043 |
| max_motion_added_mesh_line_over_own_initial | 0.0000 | 0.9470 | 0.0006 | -0.9464 |

| GT部件 | 类型正确 | Axis A → B（°） | Pivot A → B（% GT D） |
| --- | --- | --- | --- |
| 1 | True | 0.9733 → 0.4531 | 24.8790 → 23.6420 |
| 2 | True | 0.4557 → 0.5974 | 22.6849 → 22.3286 |

[留出视角重建对比](/root/autodl-tmp/PAct/repro/outputs/reconstruction_ablation_20261025_v1/figures_8961_native/heldout_reconstruction.png) · [运动曲线](/root/autodl-tmp/PAct/repro/outputs/reconstruction_ablation_20261025_v1/figures_8961_native/motion_comparison.png) · [R0/A/B运动视频](/root/autodl-tmp/PAct/repro/outputs/reconstruction_ablation_20261025_v1/figures_8961_native/R0_A_B_motion.mp4) · [GT截面](/root/autodl-tmp/PAct/repro/outputs/reconstruction_ablation_20261025_v1/figures_8961_native/GT_sections.png)。

A/B 正式优化耗时：127.4 / 132.3秒；最终步均200。

[逐关节绝对重叠及相对起始状态的增量曲线](/root/autodl-tmp/PAct/repro/outputs/reconstruction_ablation_20261025_v1/figures_8961_native/motion_increment.png)。这些增量是proxy诊断，不是实体新增穿透体积；A/B起始几何不同，须与绝对峰值一起读。

## 8961_sweep90

| 指标 | R0 | A：重建 | B：重建＋运动碰撞 | B−A |
| --- | --- | --- | --- | --- |
| PSNR | 12.0349 | 14.1061 | 13.8541 | -0.2519 |
| foreground_PSNR | 11.0456 | 11.9596 | 11.8977 | -0.0619 |
| SSIM | 0.7651 | 0.8141 | 0.8086 | -0.0055 |
| LPIPS_vgg | 0.1914 | 0.1587 | 0.1635 | 0.0049 |
| fixed_CD_percent_GT_D | 5.2136 | 3.9671 | 4.2168 | 0.2498 |
| moving_CD_percent_GT_D | 1.9145 | 1.5515 | 1.9387 | 0.3872 |
| whole_CD_percent_GT_D | 3.2657 | 2.1500 | 2.5364 | 0.3864 |
| mean_axis_degrees | 0.0000 | 0.7169 | 0.6054 | -0.1115 |
| mean_pivot_percent_GT_D | 24.9207 | 23.7805 | 23.0210 | -0.7595 |
| Gaussian_motion_proxy | 0.0106 | 0.0090 | 0.0003 | -0.0087 |
| worst_Gaussian_motion_state | 0.0290 | 0.0310 | 0.0007 | -0.0302 |
| mesh_surface_intersection_line_over_D | 0.4972 | 2.0648 | 0.0148 | -2.0500 |
| max_motion_added_Gaussian_proxy_over_own_initial | 0.0000 | 0.0168 | 0.0000 | -0.0168 |
| max_motion_added_mesh_line_over_own_initial | 0.0000 | 3.3733 | 0.0352 | -3.3381 |

| GT部件 | 类型正确 | Axis A → B（°） | Pivot A → B（% GT D） |
| --- | --- | --- | --- |
| 1 | True | 0.9704 → 0.5067 | 24.8764 → 23.6782 |
| 2 | True | 0.4635 → 0.7042 | 22.6846 → 22.3638 |

[留出视角重建对比](/root/autodl-tmp/PAct/repro/outputs/reconstruction_ablation_20261025_v1/figures_8961_sweep90/heldout_reconstruction.png) · [运动曲线](/root/autodl-tmp/PAct/repro/outputs/reconstruction_ablation_20261025_v1/figures_8961_sweep90/motion_comparison.png) · [R0/A/B运动视频](/root/autodl-tmp/PAct/repro/outputs/reconstruction_ablation_20261025_v1/figures_8961_sweep90/R0_A_B_motion.mp4) · [GT截面](/root/autodl-tmp/PAct/repro/outputs/reconstruction_ablation_20261025_v1/figures_8961_sweep90/GT_sections.png)。

A/B 正式优化耗时：141.4 / 143.6秒；最终步均200。

[逐关节绝对重叠及相对起始状态的增量曲线](/root/autodl-tmp/PAct/repro/outputs/reconstruction_ablation_20261025_v1/figures_8961_sweep90/motion_increment.png)。这些增量是proxy诊断，不是实体新增穿透体积；A/B起始几何不同，须与绝对峰值一起读。

## 45194_native

| 指标 | R0 | A：重建 | B：重建＋运动碰撞 | B−A |
| --- | --- | --- | --- | --- |
| PSNR | 14.5463 | 15.4897 | 15.3076 | -0.1821 |
| foreground_PSNR | 11.3380 | 11.8300 | 11.7888 | -0.0412 |
| SSIM | 0.7192 | 0.7423 | 0.7376 | -0.0047 |
| LPIPS_vgg | 0.2529 | 0.2412 | 0.2444 | 0.0032 |
| fixed_CD_percent_GT_D | 21.1745 | 18.9371 | 19.7912 | 0.8541 |
| moving_CD_percent_GT_D | 14.9505 | 13.8299 | 14.5417 | 0.7118 |
| whole_CD_percent_GT_D | 7.7186 | 6.5413 | 7.0637 | 0.5224 |
| mean_axis_degrees | 0.5473 | 2.2259 | 2.5702 | 0.3443 |
| mean_pivot_percent_GT_D | 62.8617 | 62.9842 | 62.9483 | -0.0360 |
| Gaussian_motion_proxy | 0.0476 | 0.0192 | 0.0021 | -0.0171 |
| worst_Gaussian_motion_state | 0.0649 | 0.0230 | 0.0049 | -0.0182 |
| mesh_surface_intersection_line_over_D | 4.9708 | 8.5595 | 3.4390 | -5.1205 |
| max_motion_added_Gaussian_proxy_over_own_initial | 0.0000 | 0.0006 | 0.0027 | 0.0021 |
| max_motion_added_mesh_line_over_own_initial | 1.1046 | 0.0000 | 0.0000 | 0.0000 |

| GT部件 | 类型正确 | Axis A → B（°） | Pivot A → B（% GT D） |
| --- | --- | --- | --- |
| 1 | True | 1.2421 → 1.3992 | 62.9842 → 62.9483 |
| 2 | False | unavailable → unavailable | unavailable → unavailable |
| 3 | True | 4.6974 → 4.6887 | unavailable → unavailable |
| 4 | True | 0.7382 → 1.6228 | unavailable → unavailable |

[留出视角重建对比](/root/autodl-tmp/PAct/repro/outputs/reconstruction_ablation_20261025_v1/figures_45194_native/heldout_reconstruction.png) · [运动曲线](/root/autodl-tmp/PAct/repro/outputs/reconstruction_ablation_20261025_v1/figures_45194_native/motion_comparison.png) · [R0/A/B运动视频](/root/autodl-tmp/PAct/repro/outputs/reconstruction_ablation_20261025_v1/figures_45194_native/R0_A_B_motion.mp4) · [GT截面](/root/autodl-tmp/PAct/repro/outputs/reconstruction_ablation_20261025_v1/figures_45194_native/GT_sections.png)。

A/B 正式优化耗时：384.8 / 379.4秒；最终步均200。

[逐关节绝对重叠及相对起始状态的增量曲线](/root/autodl-tmp/PAct/repro/outputs/reconstruction_ablation_20261025_v1/figures_45194_native/motion_increment.png)。这些增量是proxy诊断，不是实体新增穿透体积；A/B起始几何不同，须与绝对峰值一起读。

## 验证、解释与限制

两场景的真实数据梯度检查通过：图像项与运动项均对几何、方向和pivot有非零有限梯度；渲染投影误差<0.004像素，FK与已有实现误差<4e-8，共享形变场同时改变Mesh。30步 smoke 均完成。lambda=0两臂最大参数差异 1.53e-07，通过1e-5门槛。native/sweep两次A正式复跑最大差异 0.000932，严格1e-5检查结果为 False。后者是长程可重复性诊断，未达到逐参数一致性，不能将小幅A/B差异解释为确定收益。两份原始A结果均保留，没有改用更有利的一份。

每个正式结果从相同共同初始化开始，200步固定最终checkpoint，无GT最佳步选择。两组同时开放每部件4x4x4控制点形变、方向和pivot；Gaussian opacity/尺度/颜色固定，位移约束、正则、学习率和样本路径相同。运动项使用每关节21点和同步路径的固定循环状态，每步3状态。所有保留Gaussian为最近邻目标，每部件1024源点；对双方几何可微。

独立运动评价：每关节与同步路径各41状态，源点最多4096/部件，排除训练源点；全密度目标。Mesh独立核查完整变形后的原始拓扑三角形，在同步路径21状态统计交线，采用相同初始资产对角线。尚未对全部多关节组合和每个独立路径计算Mesh交线；离散采样不能保证连续无碰撞。

90°实验是按原预测方向规定的假设范围，不是GT机械极限或原版预测完整运动；它不能证明真实机械全范围内安全。8961 native不足1°的限制仍在。45194原有类型错误保留，不能把类型错误部件的轴向量分数当有效关节精度。

图像指标来自留出相机；几何用候选自己的观察姿态FK，与GT对应部件在共同相机标定世界坐标直接比较，不重新对齐、不进行活动部件shape-only拟合。轴方向是忽略正负的夹角，pivot是预测原点到GT轴线的垂距，沿轴平移不作有效误差。这里使用图像注册对齐，与旧报告的GT静态AABB对齐不同，绝对数值不能跨协议直接比较；本轮A/B对齐完全相同。

Gaussian重叠是soft proxy；Mesh表面交线含合法接触，也受表面网格影响，都不等于实体穿透体积。Mesh实体合法性未认证，因此真实体积和穿透深度均 unavailable，不填0。

固定颜色/尺度和低维有界形变限制了可达到的重建质量。只有两个物体、一个种子与一个预声明权重，结果用于机制验证，不能据此宣称普适有效。碰撞下降而几何、外观或轴恶化属于权衡，不能单独称为重建改善。

监控：`/root/autodl-tmp/conda_envs/pact/bin/python repro/tools/monitor_reconstruction_ablation.py --root /root/autodl-tmp/PAct/repro/outputs/reconstruction_ablation_20261025_v1 --watch`。原模型和源数据未覆盖；不推送官方上游，不提交大型产物。

## 运动过程与几何变化的解释

8961的90°假设范围中，各独立/同步路径的最大相对起始状态proxy增量由A的 0.0168346 降到B的 0.0000000。因此检查包含运动中新增加的重叠，而非只比较静态起始图。该0只表示本次离散Gaussian proxy未超过自身起始值，不代表实体无碰撞；对应Mesh交线增量仍非零。

同时，B的起始重叠也大幅改变，且几何可训练，所以不能将全程重叠下降全部解释为轴被纠正。45194的部分单关节路径在B中仍有相对自身起始状态的proxy上升，应看逐关节曲线，不能只看同步平均。

留出图像中，8961的GT百叶细节在原版及A/B渲染中仍明显缺失或不清晰，加入碰撞项没有恢复这一差异。45194保留部件类型/几何错误，单看Whole Chamfer会掩盖更大的逐部件误差。

本轮支持的结论是：这一碰撞项有效约束了运动重叠代理，但在当前权重、可训练表示与预算下，额外项没有提升GT几何或留出图像精度。8961的pivot有局部收益；45194的B减少了采样到的形变折叠。这些收益不应被抹去，也不能替代整体重建精度对照。

两次A复跑的留出 PSNR 相差 -0.000306 dB，Whole CD相差 -0.003128 个百分点。该波动不是跨种子统计置信区间，只是同一设置的重复性参照。

形变场额外检查（每单元中心的Jacobian，不是Mesh实体合法性认证）：

- 8961_native：A非正Jacobian中心 0 个，B 0 个。正样本也不能证明全域无折叠。

- 8961_sweep90：A非正Jacobian中心 0 个，B 0 个。正样本也不能证明全域无折叠。

- 45194_native：A非正Jacobian中心 5 个，B 0 个。正样本也不能证明全域无折叠。

实际中断恢复检查：第10步终止，从checkpoint恢复到30步，历史连续，和不中断版本最大参数差异 7.99e-07，passed=True。
