# Pivot 缺陷案例筛查与10489仅轴位置优化实测

目前最接近“几何相对较好、轴方向准确但 pivot 有误差”的实际 PAct 案例是 **10489**。四个新增场景未找到严格满足条件的干净案例。对10489的200步只改pivot实测中，重叠代理降低，**GT pivot反而变差**，不能宣称该方法已验证有效。

## 全部筛查结果

| 场景 | 固定部件 RMS / GT D | 活动部件 shape CD / GT D | 类型正确 | ≥20°运动信号 | 筛查结论 |
|---|---:|---|---:|---|---|
| 10489 | 2.547% | 1.829%, 2.182% | 2/2 | False | 低运动幅度的有限诊断 |
| 45503 | 7.788% | 15.268%, 2.362%, 3.798% | 2/3 | False | 部件类型/角色、几何不合格 |
| 10655 | 9.854% | 12.936%, 0.925% | 1/2 | False | 部件类型/角色、几何不合格 |
| 9016 | 21.651% | 19.638%, 3.203% | 1/2 | False | 部件类型/角色、几何不合格 |
| 100481 | 14.574% | 12.297%, 5.174% | 1/2 | False | 部件类型/角色、几何不合格 |


static RMS 为同一固定 canonical/static-AABB 对齐下、以GT静态部件对角线D归一的双向表面RMS。活动部件的 shape CD 是**独立刚性姿态配准后的形状误差下界**，4000点拟合、8000个新面积采样点评分，PCA24个proper rotations + 最多35步90% trimmed ICP，不允许额外scale。它仅排除输入打开姿态与生成rest不同的影响，**不是**装配结果更正，也未用于轴评分或优化。未配准的global误差也完整保留在 `geometry_CASE.json`。形状分数较小不能单独证明关节附近几何足够准确。

10489的两扇门原始类型正确，方向0°，pivot约3.5%D，shape CD约1.83%/2.18%D，故适合有限诊断；但原始range仅−6.525°/+9.948°，不满足预先规定≥20°条件。未将range改成GT或人为大角度来制造优化成功。其GT静态ICP也有配准敏感性，两个对齐协议都保留。

45503、10655、9016、100481均出现GT活动部件被预测为fixed，并伴随部件角色/形状不匹配。它们的大pivot数字混有语义/对应与重建错误，不能当作纯轴位置缺陷。未强制改为GT类型、重排部件以提高GT指标或换seed。旧31249有一扇门类型错误且正确门pivot仅0.596%D；官方Dishwasher没有可靠GT；20411是prismatic，origin不影响运动，均不作为这次主pivot案例。

## 10489：固定几何和方向，只改pivot

| GT 部件 | 原始 pivot 误差 | 优化后 | 同一静态 ICP：原始→优化 | 轴方向 |
|---|---:|---:|---|---|
| 1 | 3.489% | 8.877% | 6.664% → 12.066% | 0° → 0° |
| 2 | 3.522% | 6.063% | 6.611% → 9.197% | 0° → 0° |


误差为预测pivot到GT无限轴线的垂直距离 / GT static diagonal，并非任意沿轴origin点的欧氏距离。方向始终不变，表中主要固定对齐与复用同一静态ICP均显示pivot恶化。未针对候选重新配准。

41状态、训练source不重合的独立Gaussian均值 0.000177751521 → 5.91912562e-05，下降 **66.70%**。21状态全部原始三角形的平均交线/D 0.140591 → 0.099544，下降 **29.20%**。它们是重叠/表面交线代理，不是真实实体体积。两个pivot均达到5% asset diagonal的位移上限；原始pivot锚点在运动中的位移变化上限分别约0.867%/0.569% asset diagonal，反映小角度下较大的轴位置变化也可能只带来较小的表面运动差异。

![独立碰撞曲线及GT pivot误差](outputs/pivot_case_screen_20261025T100108Z/10489_native_position/diagnostics/pivot_and_collision.png)

![GT铰链垂直平面截面与原始/优化轴位置](outputs/pivot_case_screen_20261025T100108Z/10489_native_position/diagnostics/GT_pivot_sections.png)

红叉是GT轴穿过该切面的位置，蓝点是原始轴，紫色加号是优化轴。曲线为GT/原始预测Mesh与同一切面的实际交线；两套PAct几何共享且未改变，不是把几何拟合到GT轴。保留GT开口，不填充实体。平面与坐标基记录在 [figure_info.json](outputs/pivot_case_screen_20261025T100108Z/10489_native_position/diagnostics/figure_info.json)。

[原始/优化双视角21状态动画](outputs/pivot_case_screen_20261025T100108Z/10489_native_position/previews_10489/original_position.mp4) · [运动中点交线位置](outputs/pivot_case_screen_20261025T100108Z/10489_native_position/hotspots_10489/intersections_state10.png) · [打开终点对照](outputs/pivot_case_screen_20261025T100108Z/10489_native_position/previews_10489/mesh_state20_comparison.png)

这个例子说明：即使形状相对较好且方向固定，当前纯排斥目标也可通过把pivot推离GT来减少代理。小角度、未施加铰链接触/装配锚定以及残余几何误差都可能影响该结果，不能仅凭本例将失败唯一归因于其中某项；但也不能把所有效果不佳都归因于20411几何差。**本轮没有找到或制造一个干净的成功案例。**

## 运行与证据

沿用原来的position-only目标和超参数：冻结所有Mesh/GS、方向、type和range；21个训练状态，2048 source/part、全部opacity≥.05 Gaussian targets；pivot≤5% assetD，正则.05，lr=.0005D；固定最后200步，GT不参与梯度或checkpoint选择。实际资产梯度和5步GPU smoke先通过，再用tmux正式运行。严格校验200步loss与pivot梯度finite、方向梯度未启用、Mesh/GS SHA和全部非origin字段不变，t=0变换/指标/去文字图像像素一致，41/21状态和21帧视频完整。所有验证通过。

案例选择本身明确使用GT，是诊断筛选，不能当无偏测试集均值；四个新增场景和全部失败保留。输入为单RGB+GT Mesh投影得到的oracle 2D part mask，silhouette IoU>.98；GT joint数值不进入模型或优化。10489、45503、10655在本地公开SINGAPO train清单中；实际PAct完整训练清单未核实，不作未见物体泛化声称。真实实体体积/穿透深度保持unavailable。

新增重建固定frame0/seed42、25+25步、CFG7、arti_mean_num20、同一官方权重，无重新训练网络。原始输出路径和严格模型load审计保留在各 `repro/current_gt_CASE.txt` 指向的目录。筛查控制日志、阶段命令、耗时、完整GT/形状数值、优化checkpoint/Adam/history/seal在 `/root/autodl-tmp/PAct/repro/outputs/pivot_case_screen_20261025T100108Z`；仅轴位置实验在 `/root/autodl-tmp/PAct/repro/outputs/pivot_case_screen_20261025T100108Z/10489_native_position`。旧实验未覆盖，Part²GS没有恢复。

[预先协议及扩展记录](PIVOT_CASE_SCREEN_PROTOCOL.md) · [完整筛查与实测JSON](evidence/pivot-case-screen-results.json) · [完整性检查](outputs/pivot_case_screen_20261025T100108Z/10489_native_position/integrity.json) · [固定shape诊断](outputs/pivot_case_screen_20261025T100108Z/geometry_10489.json)

复查或中断恢复：

```bash
PYTHONDONTWRITEBYTECODE=1 /root/autodl-tmp/conda_envs/pact/bin/python repro/tools/run_pivot_only_case.py --root "$(cat repro/current_pivot_screen.txt)/10489_native_position"
python repro/tools/monitor_collision_refinement.py --root "$(cat repro/current_pivot_screen.txt)/10489_native_position" --watch
```

仅本地独立分支提交代码、工具和文档，不上传权重、不推送官方上游。
