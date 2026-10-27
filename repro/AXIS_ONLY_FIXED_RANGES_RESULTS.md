# 只优化轴：冻结部件和原版起止数值

本轮直接检验：当部件几何与每个关节的原版预测起点、终点均固定时，加入运动碰撞Loss是否使轴位置和方向更接近GT。以下只包含完成的固定最终200步结果，原版和上一轮起止值可变结果保留。

两条路线各保留A不加/B加碰撞Loss。direct为逐物体轴后处理，A仅原预测二次先验（初始化为驻点）；B加同一定义的运动Gaussian代理。head为官方关节输出MLP的小规模GT轴监督微调，A/B只相差碰撞权重，不是完整flow-matching续训。

沿用上一轮同一次原版生成的部件、缓存特征和原版关节初始化，不重新生成、也不从上轮训练后的A/B checkpoint继续。训练对象10489, 31249, 45503, 9016，测试对象8961, 45194。原始输入使用oracle 2D部件mask；训练GT仅监督轴，测试GT只用于封存后评价。未排除预训练集对象重叠，且不是盲测。

direct仅开放方向切平面参数和转动轴有效pivot位置，沿用5°方向界限与0.05初始资产对角线pivot界限。range_delta无梯度且不在optimizer中。head仅使用方向与转动轴pivot输出，类型、语义、移动轴无意义的原点、所有range通道均回放该对象的原版预测。不能仅冻结输出层权重或detach范围：共享隐藏层更新仍可能改变数值，本轮在训练、推理和导出三处直接固定有效输出。

原版GT范围目标在打包时已清零且mask为0，optimizer不读取GT起止值作为目标。head仍沿用官方masked MSE表达式，但只在有效轴通道训练，缓存推理特征且仅训练输出MLP，须与完整原文训练区分。步数、种子、训练顺序、学习率、碰撞权重、点源采样和物理代理与上一轮一致。

起止数值始终使用原版JSON中的原始有符号值。轴方向仍可被优化甚至反向；数值范围冻结不等于物理运动方向一定不变。额外记录有向轴旋转和轴符号反向，不能让忽略正负的GT轴线角度掩盖运动方向反转。

## A/B汇总：GT轴误差与固定范围运动代理

位置误差为pivot到GT无限轴线的垂距，占GT静态部件对角线百分比；方向为无符号轴线角度（度）。两者越低越好。移动关节位置为N/A；错误类型保留并排除有效轴均值。所有候选使用同一原版baseline的静态AABB相似变换，无候选独立配准。

| 路线 / 场景 / 分支 | 方向误差 A→B ° | 位置误差 A→B %GT D | 固定范围代理变化 |
| --- | --- | --- | --- |
| direct/8961/direct | 0.000 → 0.194 | 17.882 → 17.316 | -3.63% |
| direct/45194/direct | 0.548 → 1.893 | 56.019 → 55.977 | -25.60% |
| head/8961/continuous | 10.847 → 14.768 | 24.851 → 26.689 | 4.08% |
| head/8961/official | 0.000 → 0.000 | 17.882 → 12.564 | -4.96% |
| head/45194/continuous | 13.385 → 11.621 | 37.101 → 59.321 | -5.68% |
| head/45194/official | 1.010 → 3.318 | 56.019 → 56.019 | -5.91% |

## 实测结果解释

冻结确实生效，但本轮仍没有得到两个场景、两条路线都能改善轴精度的证据。下面的解释针对本轮固定最终步结果，不用于反选超参数或checkpoint。

- direct / 8961：平均位置误差17.882→17.316% GT D，方向0→0.194°，运动代理下降3.63%。逐轴看，节点0位置18.246→20.450%变差，节点1位置17.517→14.182%改善；不能只用平均值称两根轴都被修正。原版范围仅约2.60°/1.51°，提供的运动约束很弱。
- direct / 45194：代理下降25.60%，但平均方向0.548→1.893°退化，转动轴位置56.019→55.977%几乎不变。该转动关节单独运动路径的全物体代理约+0.0024%，并没有随总均值大幅改善。明显下降出现在节点2、节点4单独运动以及同步路径；节点2没有GT匹配，节点4方向误差0.644→4.897°。这是减少运动重叠却偏离GT方向的实例，不能把总代理下降归为旋转轴位置修复成功。
- head / 8961：官方导出位置17.882→12.564%改善，方向两组均为硬吸附后的0°。然而连续输出的方向10.847→14.768°、位置24.851→26.689%均退化，代理也增加4.08%。因此正向结果依赖原版后处理，不能据此说网络学到了更准确的连续轴参数。
- head / 45194：连续输出方向13.385→11.621°略好，但位置37.101→59.321%明显退化；官方导出方向1.010→3.318°变差，位置56.019%不变。连续B也远差于原版连续输出的2.949°/25.089%，所以B优于A的局部读数不能当成优于原版。

本轮排除了几何形变和数值行程收缩作为降低Loss的自由度，但没有排除原始几何错误、部件/类型错误、代理与实体碰撞不一致，以及缺少能唯一确定轴的运动信号。45194仍有一条GT转动关节被预测为fixed，另有未匹配活动部件；这些不在本轮允许修正的范围内。只能据此说当前运动Gaussian代理不足以稳定改善GT轴，尚不能把失败单独归因于几何质量，也不能认为冻结后问题就已解决。


### 原版参照

A/B比较之外必须检查原版：head的A若因小样本微调退化，B优于A也不自动表示优于原版。R0为官方后处理结果，R0_raw为原始连续关节输出，二者使用相同生成部件。

| 场景 / 原版输出 | 方向误差 ° | 位置误差 %GT D |
| --- | --- | --- |
| 8961/R0 | 0.000 | 17.882 |
| 8961/R0_raw | 11.635 | 22.649 |
| 45194/R0 | 0.548 | 56.019 |
| 45194/R0_raw | 2.949 | 25.089 |

### 类型与GT匹配覆盖

GT条件轴均值仅覆盖匹配且类型正确的活动关节，不能代表所有生成部件。未匹配的活动部件仍参与运动代理，但其轴精度不可评估。

| 场景 | GT匹配关节类型正确数 | 错误类型节点 | 未匹配的预测活动节点 |
| --- | --- | --- | --- |
| 8961 | 2/2 | none | none |
| 45194 | 3/4 | 1: fixed vs GT revolute | 2 |

## 逐轴误差和实际参数变化

| 路线 / 场景 / 分支 | GT部件 | 方向误差 A→B ° | 位置误差 A→B %GT D | 原点实际移动 %GT D | 轴实际旋转 ° | 有向旋转 ° |
| --- | --- | --- | --- | --- | --- | --- |
| direct/8961/direct | 1 | 0.000 → 0.285 | 18.246 → 20.450 | 2.205 | 0.285 | 0.285 |
| direct/8961/direct | 2 | 0.000 → 0.102 | 17.517 → 14.182 | 3.499 | 0.102 | 0.102 |
| direct/45194/direct | 1 | 0.000 → 0.014 | 56.019 → 55.977 | 0.106 | 0.014 | 0.014 |
| direct/45194/direct | 2 | N/A: fixed / GT revolute | N/A | N/A | N/A | N/A |
| direct/45194/direct | 3 | 1.000 → 0.770 | N/A → N/A | N/A | 0.508 | 0.508 |
| direct/45194/direct | 4 | 0.644 → 4.897 | N/A → N/A | N/A | 4.494 | 4.494 |
| head/8961/continuous | 1 | 11.841 → 14.565 | 17.670 → 15.951 | 16.337 | 16.237 | 16.237 |
| head/8961/continuous | 2 | 9.853 → 14.971 | 32.033 → 37.426 | 12.275 | 11.535 | 11.535 |
| head/8961/official | 1 | 0.000 → 0.000 | 18.246 → 14.981 | 13.121 | 0.000 | 0.000 |
| head/8961/official | 2 | 0.000 → 0.000 | 17.517 → 10.147 | 15.467 | 0.000 | 0.000 |
| head/45194/continuous | 1 | 37.125 → 24.909 | 37.101 → 59.321 | 24.145 | 13.261 | 13.261 |
| head/45194/continuous | 2 | N/A: fixed / GT revolute | N/A | N/A | N/A | N/A |
| head/45194/continuous | 3 | 1.389 → 4.560 | N/A → N/A | N/A | 3.701 | 3.701 |
| head/45194/continuous | 4 | 1.641 → 5.393 | N/A → N/A | N/A | 5.869 | 5.869 |
| head/45194/official | 1 | 0.000 → 0.000 | 56.019 → 56.019 | 1.811 | 0.000 | 0.000 |
| head/45194/official | 2 | N/A: fixed / GT revolute | N/A | N/A | N/A | N/A |
| head/45194/official | 3 | 1.389 → 4.560 | N/A → N/A | N/A | 3.701 | 3.701 |
| head/45194/official | 4 | 1.641 → 5.393 | N/A → N/A | N/A | 5.869 | 5.869 |

原点沿轴平移不改变轴线；实际原点移动量不等于轴位置改善量，必须结合GT垂距和方向看。

## 固定起止值核验

| 场景 / 节点 | 原版固定起点 | 原版固定终点 | 单位 |
| --- | --- | --- | --- |
| 8961/0 | 0 | -2.60494351387 | degrees |
| 8961/1 | 0 | -1.5053678751 | degrees |
| 45194/0 | 0 | -43.3021583557 | degrees |
| 45194/2 | 0 | -0.465298265219 | export length units |
| 45194/3 | 0 | -0.441375285387 | export length units |
| 45194/4 | 0 | -0.494420170784 | export length units |

六项正式训练的每步检查、所有预测/导出文件和全部评价的最大起止数值变化均为0。若有轴方向翻转，另列如下；它不表示range数值更新。

| 路线 / 场景 / 输出 | 相对原版方向反向的节点 |
| --- | --- |
| direct/8961/A | none |
| direct/8961/B | none |
| direct/45194/A | none |
| direct/45194/B | none |
| head/8961/A_raw | none |
| head/8961/A_official | none |
| head/8961/B_raw | none |
| head/8961/B_official | none |
| head/45194/A_raw | none |
| head/45194/A_official | none |
| head/45194/B_raw | none |
| head/45194/B_official | none |

此表检查全部正式A/B预测活动节点，包含没有GT匹配的预测部件；逐节点有向角见all_predicted_axis_changes.json。

## 每条运动路径的重叠代理

下列均为固定数值范围上的41状态均值。路径编号按原版活动节点顺序，非GT部件编号。单关节路径仍统计全物体部件对重叠，其他关节保持原版起点，不能把整条路径下降全部归给该关节局部接触。

| 路线 / 场景 / 分支 | 路径（预测节点） | A→B 代理变化 |
| --- | --- | --- |
| direct/8961/direct | only_index_0 (node 0) | -2.05% |
| direct/8961/direct | only_index_1 (node 1) | -3.23% |
| direct/8961/direct | synchronized (node all) | -5.47% |
| direct/45194/direct | only_index_0 (node 0) | 0.00% |
| direct/45194/direct | only_index_1 (node 2) | -53.59% |
| direct/45194/direct | only_index_2 (node 3) | -0.03% |
| direct/45194/direct | only_index_3 (node 4) | -37.47% |
| direct/45194/direct | synchronized (node all) | -69.42% |
| head/8961/continuous | only_index_0 (node 0) | 7.29% |
| head/8961/continuous | only_index_1 (node 1) | 0.00% |
| head/8961/continuous | synchronized (node all) | 5.98% |
| head/8961/official | only_index_0 (node 0) | 6.44% |
| head/8961/official | only_index_1 (node 1) | -11.70% |
| head/8961/official | synchronized (node all) | -7.48% |
| head/45194/continuous | only_index_0 (node 0) | 0.17% |
| head/45194/continuous | only_index_1 (node 2) | -38.27% |
| head/45194/continuous | only_index_2 (node 3) | 0.03% |
| head/45194/continuous | only_index_3 (node 4) | -30.04% |
| head/45194/continuous | synchronized (node all) | 12.50% |
| head/45194/official | only_index_0 (node 0) | 0.00% |
| head/45194/official | only_index_1 (node 2) | -38.27% |
| head/45194/official | only_index_2 (node 3) | 0.03% |
| head/45194/official | only_index_3 (node 4) | -30.04% |
| head/45194/official | synchronized (node all) | 11.56% |

## 与上一轮起止可变实验对照

两轮使用同一批原版部件与特征，坐标/GT评价一致。新一轮移除运动区间自由度；head的范围监督也随之移除。下表是协议消融，不能将跨协议差异全归因于碰撞Loss。每轮内部的A/B才是加入该Loss的直接对照。

| 路线 / 场景 / 分支 | 上轮起止可变：方向 A→B ° | 本轮固定：方向 A→B ° | 上轮位置 A→B %GT D | 本轮位置 A→B %GT D |
| --- | --- | --- | --- | --- |
| direct/8961/direct | 0.000 → 0.535 | 0.000 → 0.194 | 17.882 → 21.506 | 17.882 → 17.316 |
| direct/45194/direct | 0.548 → 1.733 | 0.548 → 1.893 | 56.019 → 55.972 | 56.019 → 55.977 |
| head/8961/continuous | 8.394 → 10.389 | 10.847 → 14.768 | 23.051 → 29.821 | 24.851 → 26.689 |
| head/8961/official | 0.000 → 0.000 | 0.000 → 0.000 | 17.882 → 12.564 | 17.882 → 12.564 |
| head/45194/continuous | 15.800 → 3.655 | 13.385 → 11.621 | 36.737 → 43.221 | 37.101 → 59.321 |
| head/45194/official | 1.699 → 1.767 | 1.010 → 3.318 | 56.019 → 58.744 | 56.019 → 56.019 |

## 读数限制与原版后处理

同时保留R0官方输出与R0_raw连续输出。官方导出仍采用原函数的坐标轴吸附及Gaussian动画→Mesh导出的bbox投影顺序；无straight-through伪梯度。连续分支和官方分支的改善可能不同，尤其0°可能来自硬吸附，不能自动称为方向学习更准。

所有运动路径固定为原版预测范围，每条训练21状态（独立关节及同步）、每步循环3状态；评价41状态和独立最多2048源Gaussian/部件，排除训练的512源中心，对全部保留目标做精确最近邻。不用GT关节角扩大或修正范围。8961原版范围很小，不能把t=1称为真实机械上的完全打开。

同一几何、轴、范围输入完全相同时，可复用上轮已封存的运动代理数组，仅以float32精确参数字节签名匹配，绝不按近似GT误差复用。新增候选均重新计算GT轴误差，复用来源见motion_cache_provenance.json。几何和范围均固定，因此本轮own/common两套数组应完全一致。

Gaussian软重叠、最近中心距离与1-sigma间隙是代理，不是实体交叠体积或穿透深度。未验证Mesh为合法封闭实体，实体指标为unavailable，未当作0。此处也没有以表面接触判定真实穿透。

direct受5°/0.05D范围限制；head只用四个训练对象、单seed、单权重，输入特征是冻结推理缓存，输出MLP微调不等价完整原版训练。原始几何与错误类型无法在本轮修正，不预设冻结后必然能找到GT轴。测试GT不用于选择checkpoint、seed或权重。

## 运行与核验

direct每场景每组200步；head每组共享200步，每步轮流一个训练对象（约50次/对象）。两条路线的监督数据不同，不能直接作同一训练协议排名。

| 正式任务 | 步数 | 耗时 秒 | Torch峰值allocated MiB |
| --- | --- | --- | --- |
| direct_8961_A | 200 | 46.5 | 25.8 |
| direct_8961_B | 200 | 47.2 | 25.8 |
| direct_45194_A | 200 | 143.4 | 41.5 |
| direct_45194_B | 200 | 143.2 | 41.5 |
| head_shared_A | 200 | 85.0 | 159.9 |
| head_shared_B | 200 | 84.8 | 159.9 |

根目录：[axis_only_fixed_ranges_20261027T075123Z](/root/autodl-tmp/PAct/repro/outputs/axis_only_fixed_ranges_20261027T075123Z)。

[范围/梯度/刚体核验](/root/autodl-tmp/PAct/repro/outputs/axis_only_fixed_ranges_20261027T075123Z/gradient_and_rigidity_checks.json)；[零权重direct](/root/autodl-tmp/PAct/repro/outputs/axis_only_fixed_ranges_20261027T075123Z/direct_zero_control.json)；[零权重head](/root/autodl-tmp/PAct/repro/outputs/axis_only_fixed_ranges_20261027T075123Z/head_zero_control.json)；[真实中断恢复](/root/autodl-tmp/PAct/repro/outputs/axis_only_fixed_ranges_20261027T075123Z/resume_control.json)。

正式任务经tmux顺序执行，每10步保存参数、Adam、历史和状态；来源及配置SHA匹配后恢复。旧实验、旧源码和旧文件未覆盖。

### direct / 8961

![GT轴位置与方向](/root/autodl-tmp/PAct/repro/outputs/axis_only_fixed_ranges_20261027T075123Z/eval_direct_8961/axes_vs_GT.png)

![固定范围运动曲线](/root/autodl-tmp/PAct/repro/outputs/axis_only_fixed_ranges_20261027T075123Z/eval_direct_8961/motion_overlap.png)

[21状态固定部件运动视频](/root/autodl-tmp/PAct/repro/outputs/axis_only_fixed_ranges_20261027T075123Z/eval_direct_8961/rigid_motion.mp4)。

### direct / 45194

![GT轴位置与方向](/root/autodl-tmp/PAct/repro/outputs/axis_only_fixed_ranges_20261027T075123Z/eval_direct_45194/axes_vs_GT.png)

![固定范围运动曲线](/root/autodl-tmp/PAct/repro/outputs/axis_only_fixed_ranges_20261027T075123Z/eval_direct_45194/motion_overlap.png)

[21状态固定部件运动视频](/root/autodl-tmp/PAct/repro/outputs/axis_only_fixed_ranges_20261027T075123Z/eval_direct_45194/rigid_motion.mp4)。

### head / 8961

![GT轴位置与方向](/root/autodl-tmp/PAct/repro/outputs/axis_only_fixed_ranges_20261027T075123Z/eval_head_8961/axes_vs_GT.png)

![固定范围运动曲线](/root/autodl-tmp/PAct/repro/outputs/axis_only_fixed_ranges_20261027T075123Z/eval_head_8961/motion_overlap.png)

[21状态固定部件运动视频](/root/autodl-tmp/PAct/repro/outputs/axis_only_fixed_ranges_20261027T075123Z/eval_head_8961/rigid_motion.mp4)。

### head / 45194

![GT轴位置与方向](/root/autodl-tmp/PAct/repro/outputs/axis_only_fixed_ranges_20261027T075123Z/eval_head_45194/axes_vs_GT.png)

![固定范围运动曲线](/root/autodl-tmp/PAct/repro/outputs/axis_only_fixed_ranges_20261027T075123Z/eval_head_45194/motion_overlap.png)

[21状态固定部件运动视频](/root/autodl-tmp/PAct/repro/outputs/axis_only_fixed_ranges_20261027T075123Z/eval_head_45194/rigid_motion.mp4)。
