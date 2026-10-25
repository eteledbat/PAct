# PAct 固定几何后的碰撞引导轴优化

完成记录更新：2026-10-24T12:00:48.492189+00:00。

[完整汇总数值、环境、配置及封存SHA](evidence/collision-refinement-results.json)。

本轮新运行2个已有PAct案例、每例3项200步后处理消融，另保留原版对照。不从头训练PAct，不更新Gaussian/Mesh，不改关节类型、行程或初始姿态。所有候选保留并在固定最终步封存，GT只作事后统计，不按新GT成绩挑参数、checkpoint或案例。

## 之前哪些案例有明显冲突

既有FreeArt3D的26657确有严重封闭几何重叠，Original的21状态平均重叠约76.73%，详见[旧FreeArt3D报告](/root/autodl-tmp/FreeArt3D/GT_NONPENETRATION_COMPARISON.md)。那是不同方法和场景，旧冻结joint修改几何的实验不能作为本轮轴优化成绩。

PAct官方Dishwasher_001在运动后半程出现明显增多的完整三角面相交；31249起始状态已经有较多相交。这些是重建表面的几何冲突证据，但表面交线也可能含合法接触。Dishwasher主体开放；31249至少两个部件自交，其余实体合法性未全部验证，所以真实实体交叠体积及精确穿透深度均unavailable。没有补洞、做凸包或将undefined填0。10489运动仅约6–10°，只保留筛查，不据此证明全运动修复；101808类型/部件语义失败不由局部轴优化修复。

## 优化与独立验证

[运行前协议及调度补充](COLLISION_REFINEMENT_DESIGN.md)。每部件2048个训练源中心、seed0；最近邻目标使用全部opacity≥0.05中心。21个均匀运动状态，对所有跨部件对求opacity加权Gaussian核均值，除以原版核均值，再加0.05倍参数位移正则。方向最多改变5°，旋转轴origin在初始轴垂直平面内最多移动资产对角线5%。Prismatic只调方向，origin不调。两例使用同一组设置，Adam固定200步；不吸附新轴、不按碰撞最低步选结果。

优化核是重叠代理，可能处罚正常接触，不能称为严格不穿模约束。独立评估41个状态，含20个训练状态之间的插值状态；每部件至多12000个源中心与训练源中心不重叠，共享完整目标密度。训练用CPU KDTree，验证用独立Kaolin CUDA最近邻并在真实点子集与CPU核验，再用float64算核。另用VTK全体原始三角面独立测21状态的相交线长度。横切线排除两面法向夹角≤5°，仍不是真实体积。

t=0的几何与轴无关，初始重叠无法靠轴优化消除。所有曲线包含该状态，不能把它删掉再报告“零穿模”。

## 定量对照

Δ为相对同案例原版的降低百分比，负值表示恶化。Gaussian核无量纲；交线长度除以资产对角线D。GT方向均值只含类型正确关节，31249始终为3/4；pivot为预测origin到GT无限轴的垂距，占GT静态主体对角线的百分比。

| Case / 方法 | 41状态Gaussian核均值 | 降低% | 21状态表面交线/D均值 | 降低% | GT有效轴均误差° | 旋转关节Pivot% |
|---|---:|---:|---:|---:|---:|---:|
| 31249 / 原版 | 0.0084428 | 0.00 | 6.88717 | 0.00 | 0.404 | 0.596 |
| 31249 / 只调位置 | 0.0064008 | 24.19 | 6.18863 | 10.14 | 0.404 | 4.589 |
| 31249 / 只调方向 | 0.0069620 | 17.54 | 6.65188 | 3.42 | 3.595 | 0.596 |
| 31249 / 位置+方向 | 0.0056748 | 32.79 | 6.05106 | 12.14 | 3.171 | 4.129 |
| dishwasher / 原版 | 0.0014464 | 0.00 | 0.98793 | 0.00 | unavailable | unavailable |
| dishwasher / 只调位置 | 0.0000512 | 96.46 | 0.00916 | 99.07 | unavailable | unavailable |
| dishwasher / 只调方向 | 0.0002078 | 85.64 | 0.18551 | 81.22 | unavailable | unavailable |
| dishwasher / 位置+方向 | 0.0000473 | 96.73 | 0.01009 | 98.98 | unavailable | unavailable |

实测联合优化的效果：

- 31249：Gaussian核均值降低32.79%，完整表面交线均值降低12.14%。 主协议GT有效轴误差0.404° → 3.171°（变化+2.767°，正值为恶化）；Pivot 0.596% → 4.129%。
- dishwasher：Gaussian核均值降低96.73%，完整表面交线均值降低98.98%。 无可靠GT，不能报告轴或pivot精度提升。

完整几何和外观在rest-state保持原值，不能将本轮结果写成重建几何/Chamfer提升。关节类型和range也原样保留，误分类没有通过轴优化被改正。

## 原版与优化后的运动曲线

![31249 运动曲线](outputs/collision_refinement_20261024T105559Z/collision_curves_31249.png)

31249：原版 → 联合优化，插值留出状态均值 0.0080280 → 0.0051805；最坏状态的跨部件核均值 0.0424417 → 0.0424417。

[31249 原版/位置/方向/联合，两视角运动视频](outputs/collision_refinement_20261024T105559Z/previews_31249/original_position_direction_both.mp4)

![31249 t=0.0 同一实际Mesh对照](outputs/collision_refinement_20261024T105559Z/previews_31249/mesh_state00_comparison.png)

![31249 t=0.5 同一实际Mesh对照](outputs/collision_refinement_20261024T105559Z/previews_31249/mesh_state10_comparison.png)

![31249 t=1.0 同一实际Mesh对照](outputs/collision_refinement_20261024T105559Z/previews_31249/mesh_state20_comparison.png)

![31249 实际表面横切交线X-ray](outputs/collision_refinement_20261024T105559Z/hotspots_31249/intersections_state00.png)

![31249 实际表面横切交线X-ray](outputs/collision_refinement_20261024T105559Z/hotspots_31249/intersections_state10.png)

![31249 实际表面横切交线X-ray](outputs/collision_refinement_20261024T105559Z/hotspots_31249/intersections_state20.png)

[相交线区域坐标与显示采样说明](outputs/collision_refinement_20261024T105559Z/hotspots_31249/regions.json)

![dishwasher 运动曲线](outputs/collision_refinement_20261024T105559Z/collision_curves_dishwasher.png)

dishwasher：原版 → 联合优化，插值留出状态均值 0.0013962 → 0.0000347；最坏状态的跨部件核均值 0.0064628 → 0.0009242。

[dishwasher 原版/位置/方向/联合，两视角运动视频](outputs/collision_refinement_20261024T105559Z/previews_dishwasher/original_position_direction_both.mp4)

![dishwasher t=0.0 同一实际Mesh对照](outputs/collision_refinement_20261024T105559Z/previews_dishwasher/mesh_state00_comparison.png)

![dishwasher t=0.5 同一实际Mesh对照](outputs/collision_refinement_20261024T105559Z/previews_dishwasher/mesh_state10_comparison.png)

![dishwasher t=1.0 同一实际Mesh对照](outputs/collision_refinement_20261024T105559Z/previews_dishwasher/mesh_state20_comparison.png)

![dishwasher 实际表面横切交线X-ray](outputs/collision_refinement_20261024T105559Z/hotspots_dishwasher/intersections_state00.png)

![dishwasher 实际表面横切交线X-ray](outputs/collision_refinement_20261024T105559Z/hotspots_dishwasher/intersections_state10.png)

![dishwasher 实际表面横切交线X-ray](outputs/collision_refinement_20261024T105559Z/hotspots_dishwasher/intersections_state20.png)

[相交线区域坐标与显示采样说明](outputs/collision_refinement_20261024T105559Z/hotspots_dishwasher/regions.json)

## 相交发生在哪些部件

下表保留全部部件对，包括零交线的对。数值是VTK表面交线长度/D；零交线只是此指标的零测量，不是实体碰撞体积为零的证明。

| Case / node对 | 原版均值 | 联合优化均值 | 原版末端 | 联合优化末端 |
|---|---:|---:|---:|---:|
| 31249/[0, 1] | 0.58317 | 0.34204 | 0.05304 | 0.00000 |
| 31249/[0, 2] | 0.63593 | 0.17248 | 0.57583 | 0.00000 |
| 31249/[0, 3] | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| 31249/[0, 4] | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| 31249/[1, 2] | 1.16207 | 1.20440 | 0.65570 | 0.75531 |
| 31249/[1, 3] | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| 31249/[1, 4] | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| 31249/[2, 3] | 2.33105 | 2.19723 | 1.90391 | 1.57886 |
| 31249/[2, 4] | 2.17495 | 2.13490 | 1.58112 | 1.31823 |
| 31249/[3, 4] | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| dishwasher/[0, 1] | 0.98793 | 0.01009 | 2.60807 | 0.00000 |

Dishwasher的交线主要位于门与柜体前下沿连接区域；优化后末端非零长度横切线消失，仍不能据此认证实体无碰撞。31249中node0为正确类型的旋转门、node1为被错判成平移关节的另一门、node2为柜体、node3/4为抽屉。联合优化减轻node0/柜体的相交，但node1/柜体的交线均值反而增加（约1.16207→1.20440），抽屉相关相交也大量残留。这是全体原始Mesh的独立检测，并非只展示训练代理最有利的区域。

## GT轴与位置：逐关节和对齐敏感性

31249沿用旧评估的固定坐标、尺度和平移；额外沿用原版静态几何ICP变换，全部候选共享，不重新按GT配准。原数据使用GT投影mask，且公开划分中有增强对象家族重叠；这不是无GT mask或严格未见对象泛化测试。

| GT mobility id / 类型 | 方法 | 类型正确 | 主协议轴误差° | 主协议Pivot% | 同一旧静态配准的轴误差° | 同一配准Pivot% |
|---|---|---|---:|---:|---:|---:|
| 0 / prismatic | 原版 | True | 0.537 | unavailable | 3.518 | unavailable |
| 1 / prismatic | 原版 | True | 0.676 | unavailable | 3.456 | unavailable |
| 3 / revolute | 原版 | True | 0.000 | 0.596 | 2.986 | 0.426 |
| 4 / revolute | 原版 | False | unavailable | unavailable | unavailable | unavailable |
| 0 / prismatic | 只调位置 | True | 0.537 | unavailable | 3.518 | unavailable |
| 1 / prismatic | 只调位置 | True | 0.676 | unavailable | 3.456 | unavailable |
| 3 / revolute | 只调位置 | True | 0.000 | 4.589 | 2.986 | 4.642 |
| 4 / revolute | 只调位置 | False | unavailable | unavailable | unavailable | unavailable |
| 0 / prismatic | 只调方向 | True | 4.659 | unavailable | 4.562 | unavailable |
| 1 / prismatic | 只调方向 | True | 3.124 | unavailable | 2.239 | unavailable |
| 3 / revolute | 只调方向 | True | 3.003 | 0.596 | 3.157 | 0.426 |
| 4 / revolute | 只调方向 | False | unavailable | unavailable | unavailable | unavailable |
| 0 / prismatic | 位置+方向 | True | 4.663 | unavailable | 4.591 | unavailable |
| 1 / prismatic | 位置+方向 | True | 3.123 | unavailable | 2.238 | unavailable |
| 3 / revolute | 位置+方向 | True | 1.729 | 4.129 | 2.248 | 4.200 |
| 4 / revolute | 位置+方向 | False | unavailable | unavailable | unavailable | unavailable |

同一旧静态配准下，有效轴均误差原版3.320° → 联合优化3.026°，与主协议的变化方向不同；这是必须保留的对齐敏感性。该配准下旋转pivot也由约0.426%恶化到4.200%。因此不能笼统宣称所有GT指标都更好或都更差，更不能选择有利配准宣称整体关节恢复成功。

主协议原版的旋转轴方向已经被启发式吸附到0°误差，因此旋转轴方向不能在该主指标上进一步下降。错误类型的残留方向仍保存在完整指标里，只作诊断，不当作正确旋转轴成绩。固定几何配准存在歧义，主/补充协议结论可能不同，不能挑选更有利的一组宣称恢复GT。

## 参数偏移及连接风险

原pivot锚点随新运动产生的位移是连接漂移诊断，不能直接当作实际接触间隙或穿透深度。

| Case / 方法 / node | 类型 | 方向偏离原版° | origin偏移%资产D | 原pivot锚点最大运动%资产D |
|---|---|---:|---:|---:|
| 31249/只调位置/0 | revolute | 0.000 | 5.000 | 3.018 |
| 31249/只调位置/1 | prismatic | 0.000 | unavailable | unavailable |
| 31249/只调位置/3 | prismatic | 0.000 | unavailable | unavailable |
| 31249/只调位置/4 | prismatic | 0.000 | unavailable | unavailable |
| 31249/只调方向/0 | revolute | 3.003 | 0.000 | 0.000 |
| 31249/只调方向/1 | prismatic | 5.000 | unavailable | unavailable |
| 31249/只调方向/3 | prismatic | 4.871 | unavailable | unavailable |
| 31249/只调方向/4 | prismatic | 3.164 | unavailable | unavailable |
| 31249/位置+方向/0 | revolute | 1.729 | 4.562 | 2.753 |
| 31249/位置+方向/1 | prismatic | 3.895 | unavailable | unavailable |
| 31249/位置+方向/3 | prismatic | 4.871 | unavailable | unavailable |
| 31249/位置+方向/4 | prismatic | 3.163 | unavailable | unavailable |
| dishwasher/只调位置/1 | revolute | 0.000 | 2.271 | 2.163 |
| dishwasher/只调方向/1 | revolute | 3.395 | 0.000 | 0.000 |
| dishwasher/位置+方向/1 | revolute | 1.678 | 1.548 | 1.473 |

三根类型正确关节的GT/原版/全部消融轴叠加（同一旧对齐和固定几何）：

![GT_mobility_0](outputs/collision_refinement_20261024T105559Z/GT_axis_previews/GT_mobility_0.png)

![GT_mobility_1](outputs/collision_refinement_20261024T105559Z/GT_axis_previews/GT_mobility_1.png)

![GT_mobility_3](outputs/collision_refinement_20261024T105559Z/GT_axis_previews/GT_mobility_3.png)


本轮新增loss只对轴方向切平面和pivot参数求导；Gaussian/Mesh不是优化参数，实际梯度检查和文件SHA共同验证了这一点。它是新的后处理实验，不是原版PAct已经包含的物理Joint优化。联合优化中原pivot锚点最大位移约为31249资产D的2.75%、Dishwasher资产D的1.47%；它提示连接可能偏离原估计，但不是已测真实间隙。只调位置的31249还触及5%偏移上限。避碰可能把正确轴推向错误的空闲区域，以补偿不准确几何或行程；本实验未分离这些潜在原因。

## 可追溯性与限制

独立分支 `research/collision-axis-refinement-20261024`；初始实现commit `fc7fe30`。原baseline分支及输出未覆盖。正式六项固定200步，输出原candidate、完整Adam checkpoint、history、冻结文件SHA和封存SHA。优化程序audit hook禁止打开GT、旧GT评估和其他项目数据；源码/配置保留。旧GT基线分数此前已经看过，因此不声称盲测。

初次不冻结NN分支的有限差分检查遇到邻居切换未通过；之后明确检查所选邻居分支的解析梯度，生产优化仍每次刷新NN。实际导出数据上的位置/方向梯度与NumPy运动一致性检查通过。5步GPU smoke和中断恢复检查证据见汇总JSON；调度器切换时第一项从50步恢复，未选择其他checkpoint。并发运行的wall耗时不能解释为单项独占速度；Torch峰值分配不包含CUDA context。

本轮只有两个案例、一个有GT、单一固定初始化；不能证明方法普适，也不能保证连续运动中严格无碰撞。Gaussian重叠、完整三角面表面相交、真实实体碰撞是三个不同量。降低其中一个不能自动证明其他量改善，也不能自动证明轴恢复更准确。

独立监控（无需Codex持续调用）：

```bash
/root/autodl-tmp/conda_envs/pact/bin/python /root/autodl-tmp/PAct/repro/tools/monitor_collision_refinement.py --watch
```

全部工具和小型证据提交本分支；大型模型和实验输出留在本机，未推送官方上游。

耗时为从protocol写入至完成的wall时间，包含并发竞争和中断恢复；不作为独占性能benchmark。

| Case / 方法 | 最终步数 | wall分钟 | Torch峰值分配MiB（不含context） |
|---|---:|---:|---:|
| 31249/只调位置 | 200 | 22.67 | 53.60 |
| 31249/只调方向 | 200 | 21.95 | 70.50 |
| 31249/位置+方向 | 200 | 22.52 | 70.50 |
| dishwasher/只调位置 | 200 | 3.19 | 36.81 |
| dishwasher/只调方向 | 200 | 3.21 | 36.82 |
| dishwasher/位置+方向 | 200 | 3.23 | 36.82 |

新增候选用本地ply/gs符号链接引用原始几何，candidate JSON及原始几何SHA均未变化；可直接按相对路径检查，未复制或上传大模型。相机首版空白较多，保留在previews_camera_v1_*；最终使用更近的共同相机，全部方法和状态共享，不改变任何预测。

数值后端复核：真实运动子集的CPU/GPU最近邻距离通过容差检查，所检子集Gaussian核逐项一致；这是数值验证，不是新改善样本。环境、具体版本和所有原始指标的SHA见汇总证据。


## 20411 单独实测

新增 [PAct 20411 原始重建与减少重叠轴优化](PACT_20411_RESULTS.md)。该案例来自 FreeArt3D 已有 PartNet 数据，使用单 RGB + GT-mask，GT 轴未参与优化或候选选择；原始方向误差 0.636°，优化后 5.092°，平均 Gaussian 代理下降 78.64%，全三角形交线下降 85.38%。与本报告前述案例分开统计，保留其数据和对齐协议差异。


## 轴位置案例筛查与10489补充

新增 [四场景筛查与10489原始行程下的pivot-only实测](PIVOT_CASE_SCREEN_RESULTS.md)。10489形状相对较好但运动幅度不足；固定几何及方向的200步优化降低重叠代理，却使GT pivot误差从3.49%/3.52%增至8.88%/6.06%。四个新场景因部件类型/角色及几何问题未通过筛查。该诊断按GT筛选并保留全部失败，不作为无偏泛化或成功证明。
