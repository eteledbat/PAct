# 追加场景：冻结部件和数值起止值，只优化轴

正式测试对象为20411, 10655。direct各A/B固定200步；head直接复用上一轮封存的A/B输出MLP checkpoint，在新对象上测试，没有新增训练步或调参。A不加、B加运动Gaussian代理。全部局部部件、类型、层级及原版native起止值冻结。

20411为用户指定的平移关节案例，origin不影响运动，因此轴位置/pivot精度为N/A，只评价方向。10655含转动关节，可评价轴位置；原版已有部件/类型失败，不能称为纯净的pivot缺陷样本。

第二个对象最初固定为101284（frame0、seed42），原版推理成功生成三个部件，但全部为fixed。在冻结类型的协议下没有可优化轴，失败原样保留，未改成GT类型。随后按已有原版输出存在可评估活动关节的条件选10655，没有按优化效果挑样本。该选择及历史GT访问意味着不是盲测。

## A/B轴精度与运动重叠

方向为无符号轴线角度，位置为预测pivot到GT无限轴线的垂距 / GT静态部件对角线D。均值仅包含匹配且类型正确的关节；类型错判不能通过忽略该关节美化为全模型准确率。Gaussian代理不是实体穿透体积。

| 路线 / 场景 / 输出 | 方向 A→B ° | 位置 A→B %GT D | 运动代理变化 |
| --- | --- | --- | --- |
| direct/20411/direct | 0.633 → 4.937 | N/A → N/A | -80.95% |
| direct/10655/direct | 0.000 → 0.438 | 51.165 → 53.152 | -6.84% |
| head/20411/continuous | 0.457 → 3.261 | N/A → N/A | 156.62% |
| head/20411/official | 0.457 → 3.261 | N/A → N/A | 156.62% |
| head/10655/continuous | 2.319 → 13.160 | 38.460 → 75.915 | -36.20% |
| head/10655/official | 0.000 → 0.000 | 51.165 → 65.860 | -13.90% |

## 实测解释

这两个正式案例均未提供稳定改善GT轴的证据；这里没有按GT挑选seed、权重或中间步。局部部件及数值起止值冻结通过，测试候选也没有相对原版轴方向反向。

- **20411直接优化**：方向误差0.633→4.937°，运动Gaussian代理下降80.95%。原版方向本来已准确；降低重叠仍然把平移轨迹推偏。数值行程固定为[0, −0.4178074896]，不是靠缩短行程获得下降。平移origin不可识别，没有所谓位置改善。
- **20411关节头迁移**：A→B方向0.457→3.261°，运动代理反而增加156.62%。该平移关节的连续/官方分支数值相同；不能用坐标轴吸附解释为方向改善。A的方向虽略好于原版0.633°，其代理已高于原版；B在该新对象上未表现出减少重叠的迁移能力。
- **10655直接优化**：可评分转动关节的方向0→0.438°，位置51.165→53.152% GT D，均退化；运动代理下降6.84%（同步路径下降22.40%）。仍是代理改善而真实轴变差。
- **10655关节头迁移**：连续轴方向2.319→13.160°、位置38.460→75.915% GT D，显著退化，代理却下降36.20%。官方导出将方向都吸附为0°，但位置51.165→65.860%仍变差，代理下降13.90%。只读官方0°会掩盖连续方向预测退化。

10655的活动GT类型仅1/2正确：一根GT转动关节对应的节点0被预测为fixed，而GT静态部件对应的节点2却被预测为revolute；本轮冻结这些类型。其较大的pivot误差混有部件角色/几何/装配错误，不能称为只差一个轴位置的干净例子，也不能将此类运动代理的下降推广为正确机构上的收益。101284的全fixed失败同样保留，未事后制造活动轴。

合并上一轮与本轮，只能判断：当前运动Gaussian代理即使排除了局部几何形变和起止数值变化，仍不足以稳定校正GT轴；对于head还存在小样本训练与新对象迁移失败。结果没有证明几何是唯一瓶颈，也没有证明纯碰撞目标能够唯一确定真实轴。这里没有新加其他Loss或改动原始物理项来追逐正向指标。


## 原版与类型覆盖

| 场景 / 原版输出 | 方向误差 ° | 位置误差 %GT D | GT匹配关节类型正确数 |
| --- | --- | --- | --- |
| 20411/R0 | 0.633 | N/A | 1/1 |
| 20411/R0_raw | 0.633 | N/A | 1/1 |
| 10655/R0 | 0.000 | 51.165 | 1/2 |
| 10655/R0_raw | 0.249 | 50.358 | 1/2 |

| 场景 | GT静态部件对应预测节点 / 类型 | 错误活动GT类型节点 | 无活动GT匹配的预测活动节点 |
| --- | --- | --- | --- |
| 20411 | 0 / fixed | none | none |
| 10655 | 2 / revolute | 0 | 2 |

GT静态部件若被预测为活动，它仍按原版错误结构参与运动代理，不能把代理下降解释为完整正确机构的碰撞修复。

## 每个GT匹配关节的轴变化

| 路线 / 场景 / 输出 / 预测节点 | GT类型 / 预测类型 | 方向误差 A→B ° | 位置误差 A→B %GT D | 轴有向旋转 A→B ° | 原点位移 A→B %GT D |
| --- | --- | --- | --- | --- | --- |
| direct/20411/direct/node1 | prismatic / prismatic | 0.633 → 4.937 | N/A → N/A | 4.544 | N/A |
| direct/10655/direct/node0 | revolute / fixed | N/A → N/A | N/A → N/A | N/A | N/A |
| direct/10655/direct/node1 | revolute / revolute | 0.000 → 0.438 | 51.165 → 53.152 | 0.438 | 2.555 |
| head/20411/continuous/node1 | prismatic / prismatic | 0.457 → 3.261 | N/A → N/A | 3.045 | N/A |
| head/20411/official/node1 | prismatic / prismatic | 0.457 → 3.261 | N/A → N/A | 3.045 | N/A |
| head/10655/continuous/node0 | revolute / fixed | N/A → N/A | N/A → N/A | N/A | N/A |
| head/10655/continuous/node1 | revolute / revolute | 2.319 → 13.160 | 38.460 → 75.915 | 12.748 | 37.962 |
| head/10655/official/node0 | revolute / fixed | N/A → N/A | N/A → N/A | N/A | N/A |
| head/10655/official/node1 | revolute / revolute | 0.000 → 0.000 | 51.165 → 65.860 | 0.000 | 15.777 |

原点沿轴移动不改变轴线，位移不等于精度改善。平移关节位置不可识别，不把N/A当0。

## 固定起止值与方向符号

| 场景 / 节点 | 预测类型 | 固定起点 | 固定终点 |
| --- | --- | --- | --- |
| 20411/0 | fixed | 0 | 0 |
| 20411/1 | prismatic | 0 | -0.417807489634 |
| 10655/0 | fixed | 0 | 0 |
| 10655/1 | revolute | 0 | -114.688743591 |
| 10655/2 | revolute | 0 | 96.5553970337 |

转动范围为degrees，平移范围为导出模型长度单位。所有正式direct步、迁移head的实际FK输出及导出均断言数值完全不变。轴方向反向仍可能改变物理运动，因此另查所有活动预测节点（包含无GT匹配者）：

| 路线 / 场景 / 输出 | 相对原版方向反向的节点 |
| --- | --- |
| direct/20411/A | none |
| direct/20411/B | none |
| direct/10655/A | none |
| direct/10655/B | none |
| head/20411/A_raw | none |
| head/20411/A_official | none |
| head/20411/B_raw | none |
| head/20411/B_official | none |
| head/10655/A_raw | none |
| head/10655/A_official | none |
| head/10655/B_raw | none |
| head/10655/B_official | none |

## 每条运动路径的留出代理

路径按原版活动节点顺序；单关节路径仍统计全物体部件对，其余关节保持原版起点。

| 路线 / 场景 / 输出 | 路径（预测节点） | A→B均值变化 |
| --- | --- | --- |
| direct/20411/direct | only_index_0 (node 1) | -80.95% |
| direct/20411/direct | synchronized (node all) | -80.95% |
| direct/10655/direct | only_index_0 (node 1) | -4.63% |
| direct/10655/direct | only_index_1 (node 2) | -2.90% |
| direct/10655/direct | synchronized (node all) | -22.40% |
| head/20411/continuous | only_index_0 (node 1) | 156.62% |
| head/20411/continuous | synchronized (node all) | 156.62% |
| head/20411/official | only_index_0 (node 1) | 156.62% |
| head/20411/official | synchronized (node all) | 156.62% |
| head/10655/continuous | only_index_0 (node 1) | -22.75% |
| head/10655/continuous | only_index_1 (node 2) | -24.70% |
| head/10655/continuous | synchronized (node all) | -80.58% |
| head/10655/official | only_index_0 (node 1) | -5.14% |
| head/10655/official | only_index_1 (node 2) | -14.03% |
| head/10655/official | synchronized (node all) | -45.54% |

## 设置、GT与读数限制

沿用上轮参数：direct方向/位置lr 0.002/0.0005D、5°/0.05D界限、先验0.05、B代理权重1；head复用上轮lr=1e-4、权重0.02、训练200步的A/B结果，训练ID为10489/31249/45503/9016。没有新head优化，也没有新GT监督。它是冻结推理特征上的官方输出MLP微调权重迁移，不等价完整PAct续训。direct A是原预测驻点先验，不冒充原版生成训练续跑。

训练运动路径21状态，每步3状态、512 Gaussian源/部件；评价每路径41状态，最多2048个源/部件，排除训练源，全部保留targets。部件几何与外观只读，FK刚体运动。head仅取有效方向/转动轴pivot，非轴通道回放该对象原版预测，FK/导出均固定native limits。

原版输入含GT Mesh投影的oracle二维部件mask，并非RGB-only。GT轴不进入direct优化或新增head更新（本次没有head更新）；测试GT轴在所有候选封存后评分。20411的GT此前已被研究，10655也有旧评估，故不声称首次访问或盲测。未排除PAct预训练重叠。

20411沿用实际六状态GLB验证的平移方向与固定identity坐标，10655沿用VideoArtGS固定坐标约定。每个场景只由原版静态部件AABB确定一次尺度和平移，候选不另行配准，不用GT选最终步数。

R0与R0_raw同时保留；官方输出含原版坐标轴硬吸附和GS动画→Mesh导出bbox投影，连续分支的轴学习效果须单看，不能把后处理的0°都当成优化收益。

Gaussian重叠、采样最近中心距离与1-sigma间隙均为代理，未证明Mesh为合法封闭实体，真实交叠体积与穿透深度unavailable，不填0。所有曲线使用原版预测范围，范围很小时不称真实完全打开。

首次预检对20411请求无效pivot梯度触发unused-tensor拒绝；只修预检以接受平移pivot不参与计算图，未改优化Loss或轴结果，失败日志保留。原版新推理与旧缓存存在运行漂移，记录在feature_capture.json；本轮A/B始终使用同一次新推理配对，不混用旧部件与新特征。

## 实际运行

| direct正式任务 | 步数 | 耗时 秒 | Torch峰值allocated MiB |
| --- | --- | --- | --- |
| direct_20411_A | 200 | 25.9 | 31.6 |
| direct_20411_B | 200 | 26.4 | 31.6 |
| direct_10655_A | 200 | 36.4 | 31.0 |
| direct_10655_B | 200 | 38.2 | 31.0 |

head两组只做新对象前向推理，各新增优化步数0，原训练checkpoint保持SHA不变。源配置、来源、命令、梯度检查、smoke/零权重/实际恢复对照均保存；独立监控见协议。

根目录：[axis_extra_cases_20261027T082449Z](/root/autodl-tmp/PAct/repro/outputs/axis_extra_cases_20261027T082449Z)。

[预检](/root/autodl-tmp/PAct/repro/outputs/axis_extra_cases_20261027T082449Z/gradient_and_rigidity_checks.json) · [恢复控制](/root/autodl-tmp/PAct/repro/outputs/axis_extra_cases_20261027T082449Z/resume_control.json) · [101284失败](/root/autodl-tmp/PAct/repro/outputs/axis_extra_cases_20261027T082449Z/ineligible_cases.json)。

### direct/20411

![GT轴位置和方向](/root/autodl-tmp/PAct/repro/outputs/axis_extra_cases_20261027T082449Z/eval_direct_20411/axes_vs_GT.png)

![固定范围运动重叠](/root/autodl-tmp/PAct/repro/outputs/axis_extra_cases_20261027T082449Z/eval_direct_20411/motion_overlap.png)

[21帧刚体运动视频](/root/autodl-tmp/PAct/repro/outputs/axis_extra_cases_20261027T082449Z/eval_direct_20411/rigid_motion.mp4)。

### direct/10655

![GT轴位置和方向](/root/autodl-tmp/PAct/repro/outputs/axis_extra_cases_20261027T082449Z/eval_direct_10655/axes_vs_GT.png)

![固定范围运动重叠](/root/autodl-tmp/PAct/repro/outputs/axis_extra_cases_20261027T082449Z/eval_direct_10655/motion_overlap.png)

[21帧刚体运动视频](/root/autodl-tmp/PAct/repro/outputs/axis_extra_cases_20261027T082449Z/eval_direct_10655/rigid_motion.mp4)。

### head/20411

![GT轴位置和方向](/root/autodl-tmp/PAct/repro/outputs/axis_extra_cases_20261027T082449Z/eval_head_20411/axes_vs_GT.png)

![固定范围运动重叠](/root/autodl-tmp/PAct/repro/outputs/axis_extra_cases_20261027T082449Z/eval_head_20411/motion_overlap.png)

[21帧刚体运动视频](/root/autodl-tmp/PAct/repro/outputs/axis_extra_cases_20261027T082449Z/eval_head_20411/rigid_motion.mp4)。

### head/10655

![GT轴位置和方向](/root/autodl-tmp/PAct/repro/outputs/axis_extra_cases_20261027T082449Z/eval_head_10655/axes_vs_GT.png)

![固定范围运动重叠](/root/autodl-tmp/PAct/repro/outputs/axis_extra_cases_20261027T082449Z/eval_head_10655/motion_overlap.png)

[21帧刚体运动视频](/root/autodl-tmp/PAct/repro/outputs/axis_extra_cases_20261027T082449Z/eval_head_10655/rigid_motion.mp4)。
