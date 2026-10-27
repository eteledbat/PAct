# 冻结几何：直接关节优化与原文关节监督微调

本报告只包含本次真实完成的固定最终200步结果。所有局部部件形状、Gaussian 外观与尺寸、类型和层级冻结；没有 RGB 重建项、几何变形场或 GT 最优 checkpoint 选择。图像精度不作为主要指标。

两条路线需要分开解释。direct 为新增的逐物体参数后处理：A只保留原预测的二次先验，初始化处为驻点；B额外加入运动碰撞代理。head 为基于原文 masked articulation MSE 的冻结网络小规模微调：使用官方输出 MLP，A/B追加相同步数，B增加运动碰撞代理。它不是完整原版 flow-matching 续训，也不是“原版已有的逐场景细化”。

head 训练对象：10489, 31249, 45503, 9016；测试对象：8961, 45194。训练对象 GT 用于监督与坐标归一化，测试 GT 只用于封存后的指标。局部划分对象不重叠，但未排除与预训练数据重叠。原始单视图输入使用 oracle 2D masks。

head 缓存冻结投影后的四个单推理步特征（6/12/18/25），评估使用官方最后20步平均特征。没有伪称这些是随机 GT-noised latent 的完整训练分布。只训练官方 art_out_layers，输入投影、生成骨干和 decoder 不更新。masked MSE 额外限制到用户要求的连续关节通道，不修正错误类型。AdamW 为新建优化器，非作者 optimizer-state 恢复。

direct 开放轴方向、转动轴有效位置、q_start 和 q_end；方向限制5°、pivot限制初始资产对角线5%，端点相对初始值最多改变15°（转动）或0.1资产对角线（移动）。head 沿用官方表示的固定起点0，仅预测终点，没有新增起点输出头。

官方导出包含坐标轴吸附和 bbox pivot 投影。head 同时给出连续 pre-snap 诊断与官方后处理结果；碰撞 loss 作用在连续分支，没有使用 straight-through 梯度伪装成原版吸附可微。若连续分支改善但官方输出未改善，必须按此报告。

原始训练 encoder 用连续三个槽重复每个 range 标量，而官方 convert_data_range 使用交错槽平均，二者不完全互逆。本轮保留原版 encoder/decoder，不修复这一表示差异来改善指标。训练 GT 行程使用 b-a 的 signed span 并转为零起点 gauge；生成 rest-state 与 GT limit 零点未可靠对齐，绝对起止值 GT 误差 unavailable。预测起止值、实际行程变化全部保留。

位置误差为预测 pivot 到 GT 无限轴线的垂距，单位为 GT 静态部件对角线百分比；方向误差为 acos(abs(dot))。各方法复用同一新原版 baseline 的静态 AABB 相似变换，不对候选独立配准。移动关节的 pivot 不评分；错误类型不计有效轴误差。

## 轴和运动起止值

| 路线 / 场景 / 输出 | GT部件 | 方向误差 ° | Pivot误差 %GT D | 起点 | 终点 | 行程/原始行程 |
| --- | --- | --- | --- | --- | --- | --- |
| direct/8961/R0 | 1 | 0.000 | 18.246 | 0.000 | -2.605 | 1.000 |
| direct/8961/R0 | 2 | 0.000 | 17.517 | 0.000 | -1.505 | 1.000 |
| direct/8961/A | 1 | 0.000 | 18.246 | 0.000 | -2.605 | 1.000 |
| direct/8961/A | 2 | 0.000 | 17.517 | 0.000 | -1.505 | 1.000 |
| direct/8961/B | 1 | 0.206 | 20.421 | -3.074 | -4.922 | 0.709 |
| direct/8961/B | 2 | 0.864 | 22.591 | 4.142 | 2.640 | 0.998 |
| direct/45194/R0 | 1 | 0.000 | 56.019 | 0.000 | -43.302 | 1.000 |
| direct/45194/R0 | 2 | N/A | N/A | N/A | N/A | N/A |
| direct/45194/R0 | 3 | 1.000 | N/A | 0.000 | -0.441 | 1.000 |
| direct/45194/R0 | 4 | 0.644 | N/A | 0.000 | -0.494 | 1.000 |
| direct/45194/A | 1 | 0.000 | 56.019 | 0.000 | -43.302 | 1.000 |
| direct/45194/A | 2 | N/A | N/A | N/A | N/A | N/A |
| direct/45194/A | 3 | 1.000 | N/A | 0.000 | -0.441 | 1.000 |
| direct/45194/A | 4 | 0.644 | N/A | 0.000 | -0.494 | 1.000 |
| direct/45194/B | 1 | 0.025 | 55.972 | 0.029 | -43.299 | 1.001 |
| direct/45194/B | 2 | N/A | N/A | N/A | N/A | N/A |
| direct/45194/B | 3 | 0.951 | N/A | -0.009 | -0.440 | 0.978 |
| direct/45194/B | 4 | 4.223 | N/A | 0.003 | -0.501 | 1.019 |
| head/8961/R0 | 1 | 0.000 | 18.246 | 0.000 | -2.605 | 1.000 |
| head/8961/R0 | 2 | 0.000 | 17.517 | 0.000 | -1.505 | 1.000 |
| head/8961/R0_raw | 1 | 11.210 | 23.517 | -0.000 | -2.605 | 1.000 |
| head/8961/R0_raw | 2 | 12.059 | 21.781 | -0.000 | -1.505 | 1.000 |
| head/8961/A_raw | 1 | 7.475 | 21.312 | 0.000 | 11.995 | 4.605 |
| head/8961/A_raw | 2 | 9.312 | 24.789 | 0.000 | 30.568 | 20.306 |
| head/8961/A_official | 1 | 0.000 | 18.246 | 0.000 | 11.995 | 4.605 |
| head/8961/A_official | 2 | 0.000 | 17.517 | 0.000 | 30.568 | 20.306 |
| head/8961/B_raw | 1 | 10.751 | 22.392 | -0.000 | -43.891 | 16.849 |
| head/8961/B_raw | 2 | 10.028 | 37.250 | -0.000 | -38.861 | 25.815 |
| head/8961/B_official | 1 | 0.000 | 14.981 | 0.000 | -43.891 | 16.849 |
| head/8961/B_official | 2 | 0.000 | 10.147 | 0.000 | -38.861 | 25.815 |
| head/45194/R0 | 1 | 0.000 | 56.019 | 0.000 | -43.302 | 1.000 |
| head/45194/R0 | 2 | N/A | N/A | N/A | N/A | N/A |
| head/45194/R0 | 3 | 1.000 | N/A | 0.000 | -0.441 | 1.000 |
| head/45194/R0 | 4 | 0.644 | N/A | 0.000 | -0.494 | 1.000 |
| head/45194/R0_raw | 1 | 7.204 | 25.089 | -0.000 | -43.302 | 1.000 |
| head/45194/R0_raw | 2 | N/A | N/A | N/A | N/A | N/A |
| head/45194/R0_raw | 3 | 1.000 | N/A | -0.000 | -0.441 | 1.000 |
| head/45194/R0_raw | 4 | 0.644 | N/A | -0.000 | -0.494 | 1.000 |
| head/45194/A_raw | 1 | 42.305 | 36.737 | -0.000 | -27.050 | 0.625 |
| head/45194/A_raw | 2 | N/A | N/A | N/A | N/A | N/A |
| head/45194/A_raw | 3 | 2.138 | N/A | -0.000 | -0.235 | 0.532 |
| head/45194/A_raw | 4 | 2.958 | N/A | -0.000 | -0.302 | 0.611 |
| head/45194/A_official | 1 | 0.000 | 56.019 | 0.000 | -27.050 | 0.625 |
| head/45194/A_official | 2 | N/A | N/A | N/A | N/A | N/A |
| head/45194/A_official | 3 | 2.138 | N/A | 0.000 | -0.235 | 0.532 |
| head/45194/A_official | 4 | 2.958 | N/A | 0.000 | -0.302 | 0.611 |
| head/45194/B_raw | 1 | 5.664 | 43.221 | -0.000 | -64.078 | 1.480 |
| head/45194/B_raw | 2 | N/A | N/A | N/A | N/A | N/A |
| head/45194/B_raw | 3 | 0.624 | N/A | -0.000 | -0.234 | 0.530 |
| head/45194/B_raw | 4 | 4.678 | N/A | -0.000 | -0.288 | 0.583 |
| head/45194/B_official | 1 | 0.000 | 58.744 | 0.000 | -64.078 | 1.480 |
| head/45194/B_official | 2 | N/A | N/A | N/A | N/A | N/A |
| head/45194/B_official | 3 | 0.624 | N/A | 0.000 | -0.234 | 0.530 |
| head/45194/B_official | 4 | 4.678 | N/A | 0.000 | -0.288 | 0.583 |

转动起止值单位为度，移动关节为导出坐标长度；方向符号相对原始轴统一。行程比小于1表示缩短，不能单凭自己的运动范围上碰撞降低就认定轴修正成功。

## 运动过程重叠

| 路线 / 场景 / 输出 | 自身范围平均代理 | 共同原始范围平均代理 | 自身范围最坏代理 |
| --- | --- | --- | --- |
| direct/8961/R0 | 0.025677 | 0.025677 | 0.031244 |
| direct/8961/A | 0.025677 | 0.025677 | 0.031244 |
| direct/8961/B | 0.002124 | 0.026129 | 0.004135 |
| direct/45194/R0 | 0.046732 | 0.046732 | 0.063836 |
| direct/45194/A | 0.046732 | 0.046732 | 0.063836 |
| direct/45194/B | 0.027164 | 0.034372 | 0.055185 |
| head/8961/R0 | 0.025677 | 0.025677 | 0.031244 |
| head/8961/R0_raw | 0.025575 | 0.025575 | 0.032251 |
| head/8961/A_raw | 0.018675 | 0.025970 | 0.047220 |
| head/8961/A_official | 0.021301 | 0.025677 | 0.048237 |
| head/8961/B_raw | 0.011038 | 0.026603 | 0.039006 |
| head/8961/B_official | 0.017812 | 0.024403 | 0.025930 |
| head/45194/R0 | 0.046732 | 0.046732 | 0.063836 |
| head/45194/R0_raw | 0.046852 | 0.046852 | 0.064525 |
| head/45194/A_raw | 0.046770 | 0.042938 | 0.063952 |
| head/45194/A_official | 0.046783 | 0.042926 | 0.063952 |
| head/45194/B_raw | 0.045403 | 0.040105 | 0.063974 |
| head/45194/B_official | 0.045344 | 0.040054 | 0.063796 |

上述为全部单关节独立路径与同步路径的汇总。每条41状态，含训练21状态之间的插值状态；评价源中心与训练源中心不重叠。最近 Gaussian 中心距离和1-sigma间隙曲线见各 JSON。代理及其负间隙不是实体穿透深度。未证实原始 Mesh 为合法封闭实体，实体交叠体积/穿透深度均 unavailable，未写成0。

## A 与 B 的实测差异

| 路线 / 场景 / 分支 | 平均方向误差 A→B ° | 平均Pivot A→B %GT D | 共同范围代理相对变化 |
| --- | --- | --- | --- |
| direct/8961/direct | 0.000 → 0.535 | 17.882 → 21.506 | 1.76% |
| direct/45194/direct | 0.548 → 1.733 | 56.019 → 55.972 | -26.45% |
| head/8961/continuous | 8.394 → 10.389 | 23.051 → 29.821 | 2.44% |
| head/8961/official | 0.000 → 0.000 | 17.882 → 12.564 | -4.96% |
| head/45194/continuous | 15.800 → 3.655 | 36.737 → 43.221 | -6.60% |
| head/45194/official | 1.699 → 1.767 | 56.019 → 58.744 | -6.69% |

## A→B 的实际参数变化

| 路线 / 场景 / 分支 | GT部件 | 原点移动 %GT D | 轴实际旋转 ° | 起点变化 | 终点变化 |
| --- | --- | --- | --- | --- | --- |
| direct/8961/direct | 1 | 2.258 | 0.206 | -3.074 | -2.317 |
| direct/8961/direct | 2 | 5.075 | 0.864 | 4.142 | 4.146 |
| direct/45194/direct | 1 | 0.084 | 0.025 | 0.029 | 0.003 |
| direct/45194/direct | 3 | N/A | 0.474 | -0.009 | 0.001 |
| direct/45194/direct | 4 | N/A | 3.594 | 0.003 | -0.007 |
| head/8961/continuous | 1 | 12.837 | 5.169 | -0.000 | -55.886 |
| head/8961/continuous | 2 | 17.300 | 3.007 | -0.000 | -69.430 |
| head/8961/official | 1 | 9.849 | 0.000 | 0.000 | -55.886 |
| head/8961/official | 2 | 13.859 | 0.000 | 0.000 | -69.430 |
| head/45194/continuous | 1 | 14.719 | 37.051 | 0.000 | -37.028 |
| head/45194/continuous | 3 | N/A | 1.719 | 0.000 | 0.001 |
| head/45194/continuous | 4 | N/A | 6.739 | 0.000 | 0.014 |
| head/45194/official | 1 | 22.007 | 0.000 | 0.000 | -37.028 |
| head/45194/official | 3 | N/A | 1.719 | 0.000 | 0.001 |
| head/45194/official | 4 | N/A | 6.739 | 0.000 | 0.014 |

原点沿轴的移动不改变轴线；原点移动幅度不能单独当作轴位置的改善，需结合到GT轴的垂距与方向误差。

## 如何解释本轮结果

冻结几何后，尚未得到一致的轴精度改善。direct的8961自身范围平均代理大幅降低，但共同原始范围上代理上升，pivot与方向误差都变差；45194共同范围代理降低，但pivot几乎不变、方向变差。不能把这两项直接优化写成已恢复更准确的轴。

head的8961官方输出pivot有局部改善，但连续raw pivot反而变差，官方方向误差同为0°。这项位置收益依赖官方bbox投影对不同raw预测的处理，不能解释为连续预测本身更接近GT，也不能把坐标轴吸附后的0°称为新Loss提高了方向精度。

head的45194连续方向相对A明显改善，但pivot变差；其B的连续平均方向误差仍高于原始R0_raw，因此这主要说明碰撞项缓解了本轮小样本继续训练造成的方向退化，不能说整体方向已优于原始PAct。官方输出的方向与pivot均未改善。连续分支与官方后处理后的分支必须同时看。

起止值的变化是重要混杂：direct的8961两个起点从0变为约-3.07°与4.14°；head的8961终点从A约12°/31°变成B约-44°/-39°（相对原始轴统一符号）。减少重叠可以伴随运动区间平移、缩短或运动方向变化，不一定来自更正确的机械轴。绝对GT运动零点尚未可靠对齐，不能据这些数值宣称开合方向已正确。

本轮支持继续区分“轴变准”和“找到更少重叠的运动装配”。它不能单独证明几何或轴哪一个是唯一瓶颈；几何已冻结，几何误差、坐标/零点约定、原版硬后处理和小样本微调的泛化误差仍会影响结果。

## 运行预算与耗时

direct每个对象各200步；head为四个训练对象共享200次更新，每次一个对象，不是每个对象200步。A/B预算一致，跨路线的监督和预算不同，不直接当作同一训练设置排名。

| 正式任务 | 步数 | 耗时 秒 | Torch峰值allocated MiB |
| --- | --- | --- | --- |
| direct_8961_A | 200 | 49.8 | 25.8 |
| direct_8961_B | 200 | 51.0 | 25.8 |
| direct_45194_A | 200 | 145.1 | 41.5 |
| direct_45194_B | 200 | 147.6 | 41.5 |
| head_shared_A | 200 | 90.7 | 159.9 |
| head_shared_B | 200 | 90.5 | 159.9 |

这是4个训练对象、2个测试对象、单seed、单碰撞权重的小规模实验，不构成泛化有效性的充分证据。几何冻结消除了本轮的形状变形自由度，但错误几何和错误类型仍会限制轴优化；不能将“存在可行碰撞下降”等同于找到GT机械轴。

## 核验与产物

- 实验根目录：[frozen_joint_routes_20261027T013904Z](/root/autodl-tmp/PAct/repro/outputs/frozen_joint_routes_20261027T013904Z)。
- 配置：[config.json](/root/autodl-tmp/PAct/repro/outputs/frozen_joint_routes_20261027T013904Z/config.json)。
- 真数据梯度/FK/刚体/官方后处理核验：[gradient_and_rigidity_checks.json](/root/autodl-tmp/PAct/repro/outputs/frozen_joint_routes_20261027T013904Z/gradient_and_rigidity_checks.json)。
- 零权重对照：[direct](/root/autodl-tmp/PAct/repro/outputs/frozen_joint_routes_20261027T013904Z/direct_zero_control.json)、[head](/root/autodl-tmp/PAct/repro/outputs/frozen_joint_routes_20261027T013904Z/head_zero_control.json)。
- 真实中断恢复：[resume_control.json](/root/autodl-tmp/PAct/repro/outputs/frozen_joint_routes_20261027T013904Z/resume_control.json)。
- 正式优化通过 tmux 顺序执行；每10步保存 optimizer、参数和历史，来源/配置 SHA 匹配后才能恢复。封存前重新核验全部冻结文件 SHA。各阶段日志、原版权重加载审计与所有失败尝试保留。

### direct / 8961

![轴位置与方向](/root/autodl-tmp/PAct/repro/outputs/frozen_joint_routes_20261027T013904Z/eval_direct_8961/axes_vs_GT.png)

![自身范围与共同范围碰撞曲线](/root/autodl-tmp/PAct/repro/outputs/frozen_joint_routes_20261027T013904Z/eval_direct_8961/motion_overlap.png)

[固定部件刚体运动视频](/root/autodl-tmp/PAct/repro/outputs/frozen_joint_routes_20261027T013904Z/eval_direct_8961/rigid_motion.mp4)。

### direct / 45194

![轴位置与方向](/root/autodl-tmp/PAct/repro/outputs/frozen_joint_routes_20261027T013904Z/eval_direct_45194/axes_vs_GT.png)

![自身范围与共同范围碰撞曲线](/root/autodl-tmp/PAct/repro/outputs/frozen_joint_routes_20261027T013904Z/eval_direct_45194/motion_overlap.png)

[固定部件刚体运动视频](/root/autodl-tmp/PAct/repro/outputs/frozen_joint_routes_20261027T013904Z/eval_direct_45194/rigid_motion.mp4)。

### head / 8961

![轴位置与方向](/root/autodl-tmp/PAct/repro/outputs/frozen_joint_routes_20261027T013904Z/eval_head_8961/axes_vs_GT.png)

![自身范围与共同范围碰撞曲线](/root/autodl-tmp/PAct/repro/outputs/frozen_joint_routes_20261027T013904Z/eval_head_8961/motion_overlap.png)

[固定部件刚体运动视频](/root/autodl-tmp/PAct/repro/outputs/frozen_joint_routes_20261027T013904Z/eval_head_8961/rigid_motion.mp4)。

### head / 45194

![轴位置与方向](/root/autodl-tmp/PAct/repro/outputs/frozen_joint_routes_20261027T013904Z/eval_head_45194/axes_vs_GT.png)

![自身范围与共同范围碰撞曲线](/root/autodl-tmp/PAct/repro/outputs/frozen_joint_routes_20261027T013904Z/eval_head_45194/motion_overlap.png)

[固定部件刚体运动视频](/root/autodl-tmp/PAct/repro/outputs/frozen_joint_routes_20261027T013904Z/eval_head_45194/rigid_motion.mp4)。
