# 8961 / 45194：只改方向与只改轴位置的A/B实测

本轮仅评价原来的8961、45194，分别做direction-only和position-only，两条路线均保留A不加/B加入相同运动Gaussian代理。12项正式任务全部固定最后200步，原始部件与起止值不变；没有重新生成、没有从此前训练后的A/B继续。

direct为逐物体参数后处理，A仅原预测二次先验、初始为驻点。head每模式从官方原始输出MLP重新开始，用原来四个对象的GT所选轴分量监督，共享200步，测试只看8961/45194。head使用冻结推理缓存，仅更新输出MLP，不能称为完整PAct/flow-matching续训。

direction-only把原版导出原点固定，在FK、连续导出和后处理后均只放开方向。position-only把原版导出方向固定，仅放开转动轴有效位置，平移origin保持固定。共享MLP与官方bbox投影均可能带动未选字段，本轮在实际前向及后处理之后明确回放，不能只冻结末层权重或detach。冻结的是原版导出坐标中的关节参数，父部件运动仍按原树传播。head checkpoint的配套推理入口为本轮封装及该对象的原始缓存，导出的JSON已包含冻结后的实际参数。

## 所选参数的GT精度

方向为无符号轴线误差（°），位置为预测pivot到GT无限轴的垂距 / GT静态部件对角线D（%）。只对匹配且类型正确的关节取均值，错误类型与无GT匹配的活动部件保留，不重排或强制修正。

| 路线 / 场景 / 输出 | 只改方向：方向 A→B ° | 只改位置：位置 A→B %GT D |
| --- | --- | --- |
| direct/8961/direct | 0.000 → 0.201 | 17.882 → 17.316 |
| direct/45194/direct | 0.548 → 1.892 | 56.019 → 55.977 |
| head/8961/continuous | 8.367 → 18.012 | 27.502 → 40.847 |
| head/8961/official | 0.000 → 0.000 | 17.882 → 12.564 |
| head/45194/continuous | 15.861 → 11.572 | 36.323 → 56.468 |
| head/45194/official | 30.623 → 3.000 | 56.019 → 56.019 |

![所选轴分量误差，黑点为各自初始参照](/root/autodl-tmp/PAct/repro/outputs/axis_mode_ablation_20261027T100655Z/selected_axis_errors.png)

## 实测解释

本轮最清楚的有限正向结果是8961的direct位置单独优化：保留了联合优化的平均位置收益，同时避免方向漂移；但逐轴并非一致改善。其余结果须区分原版、A/B、连续输出与后处理。

- **direct / 8961**：只改方向时，方向误差0→0.201°，代理仅下降0.0082%，位置保持17.882% GT D。只改位置时，平均位置17.882→17.316% GT D，代理下降3.64%，方向保持0°；与此前both位置17.316%几乎相同。逐轴看，节点0位置18.246→20.454%变差，节点1位置17.517→14.178%改善，不能把平均改善称为两根轴均修正。
- **direct / 45194**：只改方向时代理下降25.60%，方向误差0.548→1.892°退化，位置固定56.019%。只改位置时位置56.019→55.977%仅微小改善，留出代理约+0.0012%、基本不变。在当前预算/界限下，此前both约25.60%的重叠收益主要由方向自由度提供，而非pivot修正；该收益仍伴随GT方向变差。
- **head方向 / 8961**：连续方向8.367→18.012°、代理+0.76%，均退化；官方后处理+冻结后，A/B均回到原版0°和相同轴位置/运动，没有实际导出收益。
- **head方向 / 45194**：连续方向15.861→11.572°、代理−13.52%，B优于A；官方平均方向30.623→3.000°。其中转动节点0的A连续误差45.714°被硬吸附成90°，B连续25.716°被吸附成0°，所以很大的官方差异包含吸附边界效应。B仍差于原版连续2.949°/官方0.548°，不能把抑制A退化说成超越原版。
- **head位置 / 8961**：连续位置27.502→40.847%明显退化，留出代理+27.27%；官方后处理+冻结后位置17.882→12.564%、代理−4.96%改善。官方正向输出如实保留，但它依赖bbox投影，不代表连续pivot学习变准。方向在所有分支均固定0°，故这里的连续位置退化不能再归咎于方向变化或局部几何形变。
- **head位置 / 45194**：连续位置36.323→56.468%退化，代理−0.10%；官方位置56.019%及代理不变。节点0的官方origin虽沿轴平移约1.180% GT D，其垂轴分量为0，轴线并没有移动，这不是轴位置改善。

所有未选物理字段和原版数值起止值均严格冻结，没有轴符号反向。所谓“只改位置”须看轴线垂直偏移，而非把任意三维origin位移算成有效轴改动。运动曲线是Gaussian代理，不是合法实体碰撞量。

这些结果支持区分自由度、保留连续输出及识别后处理作用；尚不支持当前Loss能稳定校正GT轴的结论。8961本来转动轴方向已准确而范围小，45194含类型/部件角色错误，head又只用4对象/单seed/固定权重。预检可见物理/监督梯度尺度差异，但未测梯度夹角，不能据此唯一解释失败，也不能把失败唯一归因于几何。


## 完整A/B与运动代理

| 模式 / 路线 / 场景 / 输出 | 方向误差 A→B ° | 位置误差 A→B %GT D | 运动代理变化 |
| --- | --- | --- | --- |
| direction/direct/8961/direct | 0.000 → 0.201 | 17.882 → 17.882 | -0.01% |
| direction/direct/45194/direct | 0.548 → 1.892 | 56.019 → 56.019 | -25.60% |
| direction/head/8961/continuous | 8.367 → 18.012 | 17.882 → 17.882 | 0.76% |
| direction/head/8961/official | 0.000 → 0.000 | 17.882 → 17.882 | 0.00% |
| direction/head/45194/continuous | 15.861 → 11.572 | 56.019 → 56.019 | -13.52% |
| direction/head/45194/official | 30.623 → 3.000 | 56.019 → 56.019 | -13.54% |
| position/direct/8961/direct | 0.000 → 0.000 | 17.882 → 17.316 | -3.64% |
| position/direct/45194/direct | 0.548 → 0.548 | 56.019 → 55.977 | 0.00% |
| position/head/8961/continuous | 0.000 → 0.000 | 27.502 → 40.847 | 27.27% |
| position/head/8961/official | 0.000 → 0.000 | 17.882 → 12.564 | -4.96% |
| position/head/45194/continuous | 0.548 → 0.548 | 36.323 → 56.468 | -0.10% |
| position/head/45194/official | 0.548 → 0.548 | 56.019 → 56.019 | 0.00% |

## 原版与本轮连续初始参照

head的H0把所选原始连续输出与原版导出的冻结分量组合，因而不同于此前全连续R0_raw；不能把H0与原版官方R0混用。R0保持原版完整导出，head的官方初始结果经冻结回放复现R0。原版固定坐标/静态AABB对齐在每个候选间复用，不重新拟合。

| 模式 / 场景 / 初始参照 | 方向误差 ° | 位置误差 %GT D |
| --- | --- | --- |
| direction/8961/R0 | 0.000 | 17.882 |
| direction/8961/H0 | 11.635 | 17.882 |
| direction/45194/R0 | 0.548 | 56.019 |
| direction/45194/H0 | 2.949 | 56.019 |
| position/8961/R0 | 0.000 | 17.882 |
| position/8961/H0 | 0.000 | 22.649 |
| position/45194/R0 | 0.548 | 56.019 |
| position/45194/H0 | 0.548 | 25.089 |

## 逐个GT匹配关节

| 模式 / 路线 / 场景 / 输出 / 节点 | 预测类型 / GT类型 | 方向 A→B ° | 位置 A→B %GT D |
| --- | --- | --- | --- |
| direction/direct/8961/direct/0 | revolute / revolute | 0.000 → 0.373 | 18.246 → 18.246 |
| direction/direct/8961/direct/1 | revolute / revolute | 0.000 → 0.028 | 17.517 → 17.517 |
| direction/direct/45194/direct/0 | revolute / revolute | 0.000 → 0.010 | 56.019 → 56.019 |
| direction/direct/45194/direct/1 | fixed / revolute | N/A → N/A | N/A → N/A |
| direction/direct/45194/direct/3 | prismatic / prismatic | 1.000 → 0.770 | N/A → N/A |
| direction/direct/45194/direct/4 | prismatic / prismatic | 0.644 → 4.897 | N/A → N/A |
| direction/head/8961/continuous/0 | revolute / revolute | 8.659 → 18.281 | 18.246 → 18.246 |
| direction/head/8961/continuous/1 | revolute / revolute | 8.075 → 17.743 | 17.517 → 17.517 |
| direction/head/8961/official + freeze/0 | revolute / revolute | 0.000 → 0.000 | 18.246 → 18.246 |
| direction/head/8961/official + freeze/1 | revolute / revolute | 0.000 → 0.000 | 17.517 → 17.517 |
| direction/head/45194/continuous/0 | revolute / revolute | 45.714 → 25.716 | 56.019 → 56.019 |
| direction/head/45194/continuous/1 | fixed / revolute | N/A → N/A | N/A → N/A |
| direction/head/45194/continuous/3 | prismatic / prismatic | 0.649 → 3.228 | N/A → N/A |
| direction/head/45194/continuous/4 | prismatic / prismatic | 1.219 → 5.773 | N/A → N/A |
| direction/head/45194/official + freeze/0 | revolute / revolute | 90.000 → 0.000 | 56.019 → 56.019 |
| direction/head/45194/official + freeze/1 | fixed / revolute | N/A → N/A | N/A → N/A |
| direction/head/45194/official + freeze/3 | prismatic / prismatic | 0.649 → 3.228 | N/A → N/A |
| direction/head/45194/official + freeze/4 | prismatic / prismatic | 1.219 → 5.773 | N/A → N/A |
| position/direct/8961/direct/0 | revolute / revolute | 0.000 → 0.000 | 18.246 → 20.454 |
| position/direct/8961/direct/1 | revolute / revolute | 0.000 → 0.000 | 17.517 → 14.178 |
| position/direct/45194/direct/0 | revolute / revolute | 0.000 → 0.000 | 56.019 → 55.977 |
| position/direct/45194/direct/1 | fixed / revolute | N/A → N/A | N/A → N/A |
| position/direct/45194/direct/3 | prismatic / prismatic | 1.000 → 1.000 | N/A → N/A |
| position/direct/45194/direct/4 | prismatic / prismatic | 0.644 → 0.644 | N/A → N/A |
| position/head/8961/continuous/0 | revolute / revolute | 0.000 → 0.000 | 18.945 → 20.248 |
| position/head/8961/continuous/1 | revolute / revolute | 0.000 → 0.000 | 36.060 → 61.445 |
| position/head/8961/official + freeze/0 | revolute / revolute | 0.000 → 0.000 | 18.246 → 14.981 |
| position/head/8961/official + freeze/1 | revolute / revolute | 0.000 → 0.000 | 17.517 → 10.147 |
| position/head/45194/continuous/0 | revolute / revolute | 0.000 → 0.000 | 36.323 → 56.468 |
| position/head/45194/continuous/1 | fixed / revolute | N/A → N/A | N/A → N/A |
| position/head/45194/continuous/3 | prismatic / prismatic | 1.000 → 1.000 | N/A → N/A |
| position/head/45194/continuous/4 | prismatic / prismatic | 0.644 → 0.644 | N/A → N/A |
| position/head/45194/official + freeze/0 | revolute / revolute | 0.000 → 0.000 | 56.019 → 56.019 |
| position/head/45194/official + freeze/1 | fixed / revolute | N/A → N/A | N/A → N/A |
| position/head/45194/official + freeze/3 | prismatic / prismatic | 1.000 → 1.000 | N/A → N/A |
| position/head/45194/official + freeze/4 | prismatic / prismatic | 0.644 → 0.644 | N/A → N/A |

## 实际参数变化

[全部活动预测节点的A/B方向与原点坐标CSV](/root/autodl-tmp/PAct/repro/outputs/axis_mode_ablation_20261027T100655Z/axis_parameter_changes.csv) · [完整JSON](/root/autodl-tmp/PAct/repro/outputs/axis_mode_ablation_20261027T100655Z/axis_parameter_changes.json)。坐标列明确区分export坐标与GT世界坐标下的位移百分比。包含未匹配活动部件，不将其GT误差伪造为0。

| 模式 / 路线 / 场景 / 输出 / 节点 | A→B有向旋转 ° | A→B原点位移 %GT D | 平行轴线实际偏移 %GT D |
| --- | --- | --- | --- |
| direction/direct/8961/direct/0 | 0.373 | 0.000 | N/A |
| direction/direct/8961/direct/1 | 0.028 | 0.000 | N/A |
| direction/direct/45194/direct/0 | 0.010 | 0.000 | N/A |
| direction/direct/45194/direct/2 | 5.000 | N/A | N/A |
| direction/direct/45194/direct/3 | 0.508 | N/A | N/A |
| direction/direct/45194/direct/4 | 4.494 | N/A | N/A |
| direction/head/8961/continuous/0 | 17.452 | 0.000 | N/A |
| direction/head/8961/continuous/1 | 11.677 | 0.000 | N/A |
| direction/head/8961/official + freeze/0 | 0.000 | 0.000 | N/A |
| direction/head/8961/official + freeze/1 | 0.000 | 0.000 | N/A |
| direction/head/45194/continuous/0 | 22.741 | 0.000 | N/A |
| direction/head/45194/continuous/2 | 6.280 | N/A | N/A |
| direction/head/45194/continuous/3 | 3.876 | N/A | N/A |
| direction/head/45194/continuous/4 | 6.876 | N/A | N/A |
| direction/head/45194/official + freeze/0 | 90.000 | 0.000 | N/A |
| direction/head/45194/official + freeze/2 | 6.280 | N/A | N/A |
| direction/head/45194/official + freeze/3 | 3.876 | N/A | N/A |
| direction/head/45194/official + freeze/4 | 6.876 | N/A | N/A |
| position/direct/8961/direct/0 | 0.000 | 2.209 | 2.209 |
| position/direct/8961/direct/1 | 0.000 | 3.503 | 3.503 |
| position/direct/45194/direct/0 | 0.000 | 0.099 | 0.099 |
| position/direct/45194/direct/2 | 0.000 | N/A | N/A |
| position/direct/45194/direct/3 | 0.000 | N/A | N/A |
| position/direct/45194/direct/4 | 0.000 | N/A | N/A |
| position/head/8961/continuous/0 | 0.000 | 28.073 | 27.586 |
| position/head/8961/continuous/1 | 0.000 | 29.725 | 29.362 |
| position/head/8961/official + freeze/0 | 0.000 | 10.989 | 9.679 |
| position/head/8961/official + freeze/1 | 0.000 | 14.613 | 13.858 |
| position/head/45194/continuous/0 | 0.000 | 20.451 | 20.417 |
| position/head/45194/continuous/2 | 0.000 | N/A | N/A |
| position/head/45194/continuous/3 | 0.000 | N/A | N/A |
| position/head/45194/continuous/4 | 0.000 | N/A | N/A |
| position/head/45194/official + freeze/0 | 0.000 | 1.180 | 0.000 |
| position/head/45194/official + freeze/2 | 0.000 | N/A | N/A |
| position/head/45194/official + freeze/3 | 0.000 | N/A | N/A |
| position/head/45194/official + freeze/4 | 0.000 | N/A | N/A |

direct的pivot仅开放垂轴的两个有效自由度，head仍沿用原版三维origin输出与所选GT MSE。位置模式额外分解垂轴偏移与沿轴原点平移：只有前者改变平行轴线，沿轴平移不改变轴线或运动。只改方向时固定的是原点坐标，整条轴线会随方向变化，不能使用平行轴偏移定义，记N/A。完全相同方向向量的A/B夹角明确为0，避免acos自比值的浮点微角度伪差。所有相对原版的轴反向另存all_predicted_axis_changes.json。

## 冻结检查

所有正式步记录起止值与未选物理分量的变化，必须均为0；head每次optimizer更新后重新前向断言，所有连续/后处理导出逐节点校验，仅允许指定轴字段改变。方向模式GT pivot点垂距应不变，位置模式GT方向误差应不变。

| 模式 | 正式任务数 | 最大起止变化 | 最大未选字段变化 |
| --- | --- | --- | --- |
| direction | 6 | 0 | 0 |
| position | 6 | 0 | 0 |

## 与此前方向+位置同时优化参照

以下是同一原版资产和GT对齐下的既有both实验，未重训或重选结果。单变量head同时改变所选监督分量与冻结的物理参照，连续初态也随H0定义变化，跨设置差异不能全部归因于碰撞Loss；每行内部A/B才是直接对照。

| 路线 / 场景 / 输出 | 此前both方向 A→B ° | 此前both位置 A→B %GT D | 此前both运动代理变化 |
| --- | --- | --- | --- |
| direct/8961/direct | 0.000 → 0.194 | 17.882 → 17.316 | -3.63% |
| direct/45194/direct | 0.548 → 1.893 | 56.019 → 55.977 | -25.60% |
| head/8961/continuous | 10.847 → 14.768 | 24.851 → 26.689 | 4.08% |
| head/8961/official | 0.000 → 0.000 | 17.882 → 12.564 | -4.96% |
| head/45194/continuous | 13.385 → 11.621 | 37.101 → 59.321 | -5.68% |
| head/45194/official | 1.010 → 3.318 | 56.019 → 56.019 | -5.91% |

## 已有预检中的梯度尺度

以下来自正式运行前已有预检：平均推理特征、同步路径固定三个状态[2,10,20]，输出MLP参数的梯度范数。它不是实际每步使用的单时刻特征/循环三状态梯度；未测梯度夹角，不能仅凭范数称为方向冲突。保留相同权重也不意味着方向与位置模式的物理/监督梯度比例相同，本轮没有用这些数值调参。

| 模式 / 训练对象 | 0.02×物理梯度范数 / GT MSE梯度范数 |
| --- | --- |
| direction/10489 | 52.38 |
| direction/31249 | 83.51 |
| direction/45503 | 11.15 |
| direction/9016 | 546.01 |
| position/10489 | 5.14 |
| position/31249 | 23.37 |
| position/45503 | 8.63 |
| position/9016 | 27.30 |

## 类型、数据与读数限制

仍使用同一原版RGB及oracle二维部件mask；GT几何经mask进入输入，不能称RGB-only。GT只监督四个原训练对象的所选轴分量，测试GT不进入更新或checkpoint选择。测试对象此前已被检查，不是盲测，未排除PAct预训练重叠。

8961原版范围约2.60°/1.51°，运动信号小。45194仍有GT门被预测fixed、GT静态部件被预测为活动的角色错误；本轮固定类型/结构不修复，条件轴均值不能代表全机构正确。单seed、单权重和两个对象不支持稳定泛化结论。

沿用direct先验0.05、碰撞权重1，head碰撞权重0.02、lr1e-4、原24通道masked MSE分母；每次head共享200步、轮流4对象，约50次/对象。12项任务不是12个独立对象。训练21状态/路径、每步3状态、512源/部件；评价41状态、最多2048不同源/部件并排除训练源，targets全部保留。

所有运动沿原版有符号数值范围，不用GT扩大行程。软重叠、最近采样中心距离和1-sigma间隙只是Gaussian代理。未证明Mesh合法封闭，真实交叠体积和穿透深度保持unavailable，未填0。

连续输出与官方后处理+冻结回放均保留。后处理仍可能吸附可变方向或投影可变pivot，不能把吸附后的0°归为学习改善。冻结字段在后处理之后恢复，这是本轮按要求限定自由度的协议，不冒充不加限制的原版导出。

完全相同的Gaussian几何、采样、轴和范围才可按float32实际FK输入字节签名复用已封存运动数组；不按近似轴或GT误差复用，来源见motion_cache_provenance.json，GT轴误差重新评分。

## 实际运行

| 模式 / 任务 | 步数 | 耗时 秒 | Torch峰值allocated MiB |
| --- | --- | --- | --- |
| direction/direct_8961_A | 200 | 46.1 | 25.8 |
| direction/direct_8961_B | 200 | 46.7 | 25.8 |
| direction/direct_45194_A | 200 | 143.0 | 41.5 |
| direction/direct_45194_B | 200 | 143.6 | 41.5 |
| direction/head_shared_A | 200 | 84.9 | 159.9 |
| direction/head_shared_B | 200 | 85.4 | 159.9 |
| position/direct_8961_A | 200 | 46.5 | 25.8 |
| position/direct_8961_B | 200 | 46.1 | 25.8 |
| position/direct_45194_A | 200 | 140.1 | 41.0 |
| position/direct_45194_B | 200 | 138.6 | 41.0 |
| position/head_shared_A | 200 | 84.0 | 159.9 |
| position/head_shared_B | 200 | 83.5 | 159.9 |

根目录：[axis_mode_ablation_20261027T100655Z](/root/autodl-tmp/PAct/repro/outputs/axis_mode_ablation_20261027T100655Z)。预检、smoke、零权重控制、真实10→30步恢复、配置、源码SHA、原始checkpoint和Adam均保存；tmux顺序执行，独立监控脚本不依赖持续调用Codex。旧实验与部件不覆盖。

### direction / direct / 8961

![GT轴位置与方向](/root/autodl-tmp/PAct/repro/outputs/axis_mode_ablation_20261027T100655Z/direction/eval_direct_8961/axes_vs_GT.png)

![41状态运动重叠曲线](/root/autodl-tmp/PAct/repro/outputs/axis_mode_ablation_20261027T100655Z/direction/eval_direct_8961/motion_overlap.png)

[21帧刚体运动视频](/root/autodl-tmp/PAct/repro/outputs/axis_mode_ablation_20261027T100655Z/direction/eval_direct_8961/rigid_motion.mp4)。

### direction / direct / 45194

![GT轴位置与方向](/root/autodl-tmp/PAct/repro/outputs/axis_mode_ablation_20261027T100655Z/direction/eval_direct_45194/axes_vs_GT.png)

![41状态运动重叠曲线](/root/autodl-tmp/PAct/repro/outputs/axis_mode_ablation_20261027T100655Z/direction/eval_direct_45194/motion_overlap.png)

[21帧刚体运动视频](/root/autodl-tmp/PAct/repro/outputs/axis_mode_ablation_20261027T100655Z/direction/eval_direct_45194/rigid_motion.mp4)。

### direction / head / 8961

![GT轴位置与方向](/root/autodl-tmp/PAct/repro/outputs/axis_mode_ablation_20261027T100655Z/direction/eval_head_8961/axes_vs_GT.png)

![41状态运动重叠曲线](/root/autodl-tmp/PAct/repro/outputs/axis_mode_ablation_20261027T100655Z/direction/eval_head_8961/motion_overlap.png)

[21帧刚体运动视频](/root/autodl-tmp/PAct/repro/outputs/axis_mode_ablation_20261027T100655Z/direction/eval_head_8961/rigid_motion.mp4)。

### direction / head / 45194

![GT轴位置与方向](/root/autodl-tmp/PAct/repro/outputs/axis_mode_ablation_20261027T100655Z/direction/eval_head_45194/axes_vs_GT.png)

![41状态运动重叠曲线](/root/autodl-tmp/PAct/repro/outputs/axis_mode_ablation_20261027T100655Z/direction/eval_head_45194/motion_overlap.png)

[21帧刚体运动视频](/root/autodl-tmp/PAct/repro/outputs/axis_mode_ablation_20261027T100655Z/direction/eval_head_45194/rigid_motion.mp4)。

### position / direct / 8961

![GT轴位置与方向](/root/autodl-tmp/PAct/repro/outputs/axis_mode_ablation_20261027T100655Z/position/eval_direct_8961/axes_vs_GT.png)

![41状态运动重叠曲线](/root/autodl-tmp/PAct/repro/outputs/axis_mode_ablation_20261027T100655Z/position/eval_direct_8961/motion_overlap.png)

[21帧刚体运动视频](/root/autodl-tmp/PAct/repro/outputs/axis_mode_ablation_20261027T100655Z/position/eval_direct_8961/rigid_motion.mp4)。

### position / direct / 45194

![GT轴位置与方向](/root/autodl-tmp/PAct/repro/outputs/axis_mode_ablation_20261027T100655Z/position/eval_direct_45194/axes_vs_GT.png)

![41状态运动重叠曲线](/root/autodl-tmp/PAct/repro/outputs/axis_mode_ablation_20261027T100655Z/position/eval_direct_45194/motion_overlap.png)

[21帧刚体运动视频](/root/autodl-tmp/PAct/repro/outputs/axis_mode_ablation_20261027T100655Z/position/eval_direct_45194/rigid_motion.mp4)。

### position / head / 8961

![GT轴位置与方向](/root/autodl-tmp/PAct/repro/outputs/axis_mode_ablation_20261027T100655Z/position/eval_head_8961/axes_vs_GT.png)

![41状态运动重叠曲线](/root/autodl-tmp/PAct/repro/outputs/axis_mode_ablation_20261027T100655Z/position/eval_head_8961/motion_overlap.png)

[21帧刚体运动视频](/root/autodl-tmp/PAct/repro/outputs/axis_mode_ablation_20261027T100655Z/position/eval_head_8961/rigid_motion.mp4)。

### position / head / 45194

![GT轴位置与方向](/root/autodl-tmp/PAct/repro/outputs/axis_mode_ablation_20261027T100655Z/position/eval_head_45194/axes_vs_GT.png)

![41状态运动重叠曲线](/root/autodl-tmp/PAct/repro/outputs/axis_mode_ablation_20261027T100655Z/position/eval_head_45194/motion_overlap.png)

[21帧刚体运动视频](/root/autodl-tmp/PAct/repro/outputs/axis_mode_ablation_20261027T100655Z/position/eval_head_45194/rigid_motion.mp4)。
