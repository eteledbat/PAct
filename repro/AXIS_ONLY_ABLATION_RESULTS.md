# 仅分析加入运动碰撞 Loss 对轴的影响

“仅分析”指只统计轴指标，不代表优化时冻结了几何。本报告使用的是此前几何与轴同时可变的 A/B 结果，不能据此分离轴变化与几何变形的作用，也没有优化运动起止值。后来完成的冻结部件实验另见 [FROZEN_JOINT_ROUTES_RESULTS.md](FROZEN_JOINT_ROUTES_RESULTS.md)，不可将两套结果混用。

A是不加碰撞Loss的同预算细化，B是加入碰撞Loss；两组初始化、图像注册、可训练参数和最终步数相同。位置是预测pivot到GT无限轴线的垂距，单位为GT静态部件对角线百分比；方向误差忽略轴正负。此处只评价轴，不将图像质量作为结论依据。

| 实验 / GT部件 | 位置误差 A → B（% GT D） | 方向误差 A → B（°） | 原点实际移动（% GT D） | 方向实际改变（°） |
| --- | --- | --- | --- | --- |
| 8961_native / 1 | 24.879 → 23.642 | 0.973 → 0.453 | 1.376 | 0.723 |
| 8961_native / 2 | 22.685 → 22.329 | 0.456 → 0.597 | 0.460 | 0.494 |
| 8961_sweep90 / 1 | 24.876 → 23.678 | 0.970 → 0.507 | 1.313 | 0.526 |
| 8961_sweep90 / 2 | 22.685 → 22.364 | 0.463 → 0.704 | 0.425 | 0.585 |
| 45194_native / 1 | 62.984 → 62.948 | 1.242 → 1.399 | 0.324 | 0.171 |
| 45194_native / 2 | 类型错误：fixed | 不计有效轴误差 | N/A | N/A |
| 45194_native / 3 | N/A → N/A | 4.697 → 4.689 | N/A | 0.045 |
| 45194_native / 4 | N/A → N/A | 0.738 → 1.623 | N/A | 0.978 |

图中位置投影到垂直GT轴的平面，以GT轴为原点；方向图画相对GT方向的两个倾角分量。箭头表示A到B。原点移动幅度和方向改变幅度不等于GT误差改善幅度。

8961两条pivot误差均小幅下降，第一条轴方向改善、第二条方向变差。45194的revolute pivot几乎没有改善；一个prismatic方向近乎不变，另一个明显变差，另一扇门被预测为fixed，不能计有效轴误差。

8961原始R0的导出轴经官方坐标轴吸附后方向误差为0°；A/B均偏离了这一方向。因此“B某条轴比A好”不等于“方向优于原始PAct”。本轮位置采用共同图像注册，与旧GT静态AABB对齐报告的绝对数值不可混用。

[8961_native：轴位置与方向变化图](/root/autodl-tmp/PAct/repro/outputs/reconstruction_ablation_20261025_v1/figures_8961_native/axis_only_changes.png)。

[8961_sweep90：轴位置与方向变化图](/root/autodl-tmp/PAct/repro/outputs/reconstruction_ablation_20261025_v1/figures_8961_sweep90/axis_only_changes.png)。

[45194_native：轴位置与方向变化图](/root/autodl-tmp/PAct/repro/outputs/reconstruction_ablation_20261025_v1/figures_45194_native/axis_only_changes.png)。
