# 多 case：运动先验与闭合图形状优化

预先选定现有 10 个 case：100481, 101808, 10489, 10655, 20411, 31249, 45194, 45503, 8961, 9016。所有已完成和失败状态均列入覆盖表，不按指标排除案例。

三个阶段为原版 PAct、显式运动先验修正、单张闭合图部件形状优化。每个 case 使用相同的 100 步图像配准及 240 步几何优化，展示固定最终步；没有根据 GT 轴误差或未拟合视角挑选 checkpoint。

## 整体结果

| 指标 | 原版 | 运动先验 | 图像优化 |
| --- | ---: | ---: | ---: |
| 闭合拟合图 IoU，case 宏均值 % | 83.69 | 86.22 | 92.87 |
| 闭合拟合图 PSNR，dB | 16.49 | 16.85 | 20.86 |
| 未拟合开放状态视角 IoU，9 case 宏均值 % | 72.98 | 72.98 | 76.31 |
| 未拟合开放状态视角 PSNR，dB | 16.50 | 16.50 | 17.24 |
| 三维表面 CD，% GT 静态对角线，越低越好 | 4.02 | 4.02 | 3.90 |
| 独立且类型正确的 GT 关节覆盖率 % | 87.50 | 100.00 | 100.00 |
| 轴正确率：方向≤5°、铰链位置≤5% D % | 33.33 | 54.17 | 54.17 |
| 匹配且类型正确关节的轴角均值，度 | 8.92 | 7.82 | 7.82 |
| 有效铰链的轴位置均值，% D | 18.04 | 8.78 | 8.78 |

几何优化相对运动先验：拟合图 IoU 改善 10/10；未拟合视角 IoU 改善 7/9；三维表面 CD 改善 7/10。

独立且类型正确的关节数：原版 21/24，先验/图像优化 24/24。轴正确数：原版 8/24，先验/图像优化 13/24。

**图像优化阶段固定了关节，因此先验版与图像优化版的轴数值和正确率相同。轴改善应归因于运动结构/连接先验，不能归因于几何拟合。**

在两阶段共同匹配且类型正确的 21 个关节子集上，轴角均值 8.92→8.75°；有效铰链位置 18.04→8.96% D。条件均值不包含缺失/错误类型，必须与关节覆盖率同时看。

![整体指标](/root/autodl-tmp/PAct/repro/outputs/closed_multicase_20261028_v1/overall_metrics.png)

![拟合改善是否传递到其他视角](/root/autodl-tmp/PAct/repro/outputs/closed_multicase_20261028_v1/fit_vs_unfitted_change.png)

## 逐 case 重建指标与视频

视频均为左原版、中运动先验、右图像优化；上下为两个共同相机。主视频保留原版输出区间，先验/优化版从几何估计闭合姿态开始，保留原始绝对行程；原 fixed 片段被解锁时的 90° 行程单独标注为假设。小于 10° 的预测转轴另外给出明确标注的 90° 诊断视频，不计作真实行程改善。

| case | 拟合 IoU 先验→优化 % | 未拟合视角 IoU 先验→优化 % | 三维 CD 先验→优化 %D | 三路视频 |
| --- | ---: | ---: | ---: | --- |
| 100481 | 83.43→84.48 | 85.98→81.73 | 7.54→7.83 | [视频](/root/autodl-tmp/PAct/repro/outputs/closed_multicase_20261028_v1/100481/100481_three_versions.mp4) |
| 101808 | 94.66→95.33 | 86.10→86.41 | 2.60→2.69 | [视频](/root/autodl-tmp/PAct/repro/outputs/closed_multicase_20261028_v1/101808/101808_three_versions.mp4) |
| 10489 | 97.76→99.29 | 78.45→84.94 | 2.21→1.96 | [视频](/root/autodl-tmp/PAct/repro/outputs/closed_multicase_20261028_v1/10489/10489_three_versions.mp4) |
| 10655 | 66.88→83.56 | 62.92→71.80 | 3.98→3.63 | [视频](/root/autodl-tmp/PAct/repro/outputs/closed_multicase_20261028_v1/10655/10655_three_versions.mp4) |
| 20411 | 82.82→88.93 | N/A | 3.46→3.24 | [视频](/root/autodl-tmp/PAct/repro/outputs/closed_multicase_20261028_v1/20411/20411_three_versions.mp4) |
| 31249 | 97.40→97.68 | 86.56→87.41 | 1.77→1.79 | [视频](/root/autodl-tmp/PAct/repro/outputs/closed_multicase_20261028_v1/31249/31249_three_versions.mp4) |
| 45194 | 87.45→95.90 | 71.71→80.17 | 4.48→4.31 | [视频](/root/autodl-tmp/PAct/repro/outputs/closed_multicase_20261028_v1/45194/45194_three_versions.mp4) |
| 45503 | 95.77→99.38 | 77.18→79.88 | 3.77→3.45 | [视频](/root/autodl-tmp/PAct/repro/outputs/closed_multicase_20261028_v1/45503/45503_three_versions.mp4) |
| 8961 | 61.44→86.91 | 53.21→60.09 | 2.90→2.66 | [视频](/root/autodl-tmp/PAct/repro/outputs/closed_multicase_20261028_v1/8961/8961_three_versions.mp4) |
| 9016 | 94.62→97.25 | 54.71→54.32 | 7.43→7.41 | [视频](/root/autodl-tmp/PAct/repro/outputs/closed_multicase_20261028_v1/9016/9016_three_versions.mp4) |

## 逐 case 关节与轴

| case | GT 关节数 | 类型正确覆盖：原版→先验 | 轴正确：原版→先验 | 条件轴角：原版→先验 ° | 条件位置：原版→先验 %D |
| --- | ---: | ---: | ---: | ---: | ---: |
| 100481 | 2 | 2/2→2/2 | 0/2→0/2 | 90.00→90.00 | 31.66→18.66 |
| 101808 | 2 | 1/2→2/2 | 0/2→0/2 | 3.80→3.80 | 9.26→12.21 |
| 10489 | 2 | 2/2→2/2 | 2/2→2/2 | 0.00→0.00 | 3.51→3.39 |
| 10655 | 2 | 2/2→2/2 | 0/2→0/2 | 0.00→0.00 | 9.19→7.00 |
| 20411 | 1 | 1/1→1/1 | 1/1→1/1 | 0.64→0.00 | N/A→N/A |
| 31249 | 4 | 3/4→4/4 | 3/4→4/4 | 0.40→0.00 | 0.60→0.65 |
| 45194 | 4 | 3/4→4/4 | 2/4→2/4 | 0.55→0.00 | 61.63→6.86 |
| 45503 | 3 | 3/3→3/3 | 0/3→2/3 | 0.00→0.00 | 14.98→8.75 |
| 8961 | 2 | 2/2→2/2 | 0/2→2/2 | 0.00→0.00 | 17.85→1.00 |
| 9016 | 2 | 2/2→2/2 | 0/2→0/2 | 0.00→0.00 | 23.90→20.51 |

## 评价口径

- IoU 是预测前景与 RGB alpha 的二维轮廓 IoU，阈值均为 0.5，不是实体体积 IoU。闭合图也是拟合目标，必须与未拟合视角分开。三阶段使用同一个在先验版上得到的图像尺度/平移配准，原版不单独重新配准。
- 闭合图的原版指标是闭合姿态诊断：使用由部件几何估计的关节量，但通过原版轴线/FK 渲染；共享轴只能取第一实例的闭合量，不能同时满足两门。它不是原版自然生成参考姿态的分数。主视频的原版仍展示其原始预测区间，因此其起始姿态与闭合图诊断可能不同。
- 未拟合视角固定为 state0/time0 的帧 25、50、75、100、125、149；这些相机没有参与单图形状拟合，帧0是原 PAct 条件图，已排除。所有方法保持原生成参考姿态 q=0，不根据验证图重新调整姿态、尺度或相机。该指标同时反映形状和生成参考姿态偏差。20411 只有固定相机六状态输入，无法提供相同的新相机评价，因此其该项为 N/A，不填 0，也不混入 9-case 均值。
- 三维表面 CD 采用 16,000 个面积采样点，双向非平方欧氏距离的均值，除以 GT 静态部件对角线 D；没有把它称为实体穿透或体积误差。GT 参考为原生成条件状态；不做逐部件刚性对齐，因此包含姿态/装配误差。
- GT 仅在全部拟合候选封存后的独立进程中读取。位置评价使用真正生成柜体/固定体的原始 AABB 与 GT 静态体求一次相似变换；三阶段共用，不用候选各自重新配准，不做 ICP 或轴拟合。
- GT 运动部件与预测关节用固定坐标下的几何表面距离作类型/轴无关的一对一匹配。原版按它实际输出的每个源刚体完整几何匹配；先验/优化版共同使用几何优化前的物理分组匹配，并冻结该对应关系。一个原始共享关节只能为一个 GT 关节计分。这样也保留了 31249 中原始错误类型但独立运动的第二个门片，避免按修正后的主门片归属遗漏这条原始预测。
- 轴角使用无符号 acos(abs(dot))，不评价轴正负/机械运动范围。铰链位置为预测轴原点到 GT 无限轴线的垂距，占 D 百分比。移动关节的原点不可辨识，位置为 N/A。
- 轴正确率按 GT 全部关节为分母：类型正确、独立关节存在、方向≤5°，且铰链位置≤5% D。缺失、错误类型、共用轴造成的独立关节缺失都算失败。条件方向/位置均值另列有效数量。
- 附加的 geometry_supported_axis_correct_rate 要求上述轴正确且至少 30% GT 部件表面在预测对应部件的 5% D 内，仅作几何支撑诊断；不能用一个很小的活动碎片代表完整门。
- 原始 PAct 输入使用此前的二维 oracle 部件 mask；本轮闭合图前板矩形与运动结构均为显式人工先验，不是无需分割的自动重建，也不是新训练的通用模型。未排除原预训练数据重叠。
- 不对单图不可见厚度/背面作自由重建，不改变颜色/透明度，不根据 GT 调轴或换 checkpoint；实体碰撞、运动范围精度和交互后的稳定性未评价。

45194 此处按统一批次重新运行并重新评价。旧试验的输入节点对应不能用来评价新的物理门实例，旧对齐也可能把抽屉当成静态体；因此本报告的轴数值不应与旧报告直接拼成趋势。统一批次还为所有 case 明确设置了几何闭合参考、按连接距离选轴，以及共享的图像平面配准，和旧的个案探索有实现差异。

评价对应关系经过一次明确复核：初次版本由修正后主部件反推原始关节归属，会漏掉 31249 中那条错误类型但独立存在的原始活动部件。最终原版改按实际源刚体完整几何独立匹配；该修正不读轴误差、不改变任何训练结果或阈值。初次结果保存在 evaluation_v1_inherited_ownership，逐 case 评价另保留该口径作为敏感性记录。

## 主要正负结果

- 10489：两根轴均通过严格标准，闭合图和未拟合视角均改善，是本批较稳定的正例。
- 31249：分离共享门轴、把错误标为抽屉的重复门片绑定到右门后，轴通过数由 3/4 到 4/4；但形状拟合后的三维 CD 略有退化，不能称所有指标都改善。
- 45194：类型正确的独立关节覆盖由 3/4 到 4/4，闭合图与未拟合视角改善；两根门轴方向正确，但平均位置仍约 6.86% D，所以严格轴通过数仍是 2/4，仅两条滑轨通过。
- 8961：轴位置修正有效，但原预测行程不足 1°。主视频几乎不动是原始行程问题；额外 90° 诊断仅用于观察轴，不能称为预测行程修复。
- 100481：篮筐手柄不符合当前设定的 X 向手柄轴先验。两根轴仍有 90° 方向误差，未拟合视角 IoU 和三维 CD 退化；这是该先验不能直接泛化到其他运动族的反例。该错误没有在看到 GT 后重新调参掩盖。
- 9016：闭合图 IoU 很高，但未拟合视角略退化，轴位置误差仍约 20.5% D，说明轮廓拟合无法保证正确的三维连接位置。
- 101808：原版上门只有小片段/融合几何，解锁该片段虽让关节类型覆盖完整，但没有重建出可靠完整门；两根轴均未通过严格标准，三维 CD 也退化。
- 10655：多视角轮廓改善，但轴位置约 7.0% D，仍未达到 5% D 门槛。20411 的滑轨方向通过，但没有同类新相机数据，不参与 9-case 未拟合视角均值。

## 文件与覆盖

[完整 CSV](/root/autodl-tmp/PAct/repro/outputs/closed_multicase_20261028_v1/metrics.csv) · [汇总 JSON](/root/autodl-tmp/PAct/repro/outputs/closed_multicase_20261028_v1/aggregate.json) · [预先固定协议](/root/autodl-tmp/PAct/repro/outputs/closed_multicase_20261028_v1/protocol.json) · [所有先验与标注](/root/autodl-tmp/PAct/repro/outputs/closed_multicase_20261028_v1/recipes.json)

| case | 拟合 | 评价 | 视频 | 文件核验 |
| --- | --- | --- | --- | --- |
| 100481 | 完成 | 完成 | 完成 | 完成 |
| 101808 | 完成 | 完成 | 完成 | 完成 |
| 10489 | 完成 | 完成 | 完成 | 完成 |
| 10655 | 完成 | 完成 | 完成 | 完成 |
| 20411 | 完成 | 完成 | 完成 | 完成 |
| 31249 | 完成 | 完成 | 完成 | 完成 |
| 45194 | 完成 | 完成 | 完成 | 完成 |
| 45503 | 完成 | 完成 | 完成 | 完成 |
| 8961 | 完成 | 完成 | 完成 | 完成 |
| 9016 | 完成 | 完成 | 完成 | 完成 |

### 100481

[闭合拟合图](/root/autodl-tmp/PAct/repro/outputs/closed_multicase_20261028_v1/100481/closed_target_and_three_versions.png) · [未拟合视角对比](/root/autodl-tmp/PAct/repro/outputs/closed_multicase_20261028_v1/100481/unfitted_open_view_comparison.png) · [原生优化资产](/root/autodl-tmp/PAct/repro/outputs/closed_multicase_20261028_v1/100481/optimized_asset/object.json) · [逐关节评价](/root/autodl-tmp/PAct/repro/outputs/closed_multicase_20261028_v1/100481/evaluation.json)

### 101808

[闭合拟合图](/root/autodl-tmp/PAct/repro/outputs/closed_multicase_20261028_v1/101808/closed_target_and_three_versions.png) · [未拟合视角对比](/root/autodl-tmp/PAct/repro/outputs/closed_multicase_20261028_v1/101808/unfitted_open_view_comparison.png) · [原生优化资产](/root/autodl-tmp/PAct/repro/outputs/closed_multicase_20261028_v1/101808/optimized_asset/object.json) · [逐关节评价](/root/autodl-tmp/PAct/repro/outputs/closed_multicase_20261028_v1/101808/evaluation.json)

### 10489

[闭合拟合图](/root/autodl-tmp/PAct/repro/outputs/closed_multicase_20261028_v1/10489/closed_target_and_three_versions.png) · [未拟合视角对比](/root/autodl-tmp/PAct/repro/outputs/closed_multicase_20261028_v1/10489/unfitted_open_view_comparison.png) · [原生优化资产](/root/autodl-tmp/PAct/repro/outputs/closed_multicase_20261028_v1/10489/optimized_asset/object.json) · [逐关节评价](/root/autodl-tmp/PAct/repro/outputs/closed_multicase_20261028_v1/10489/evaluation.json)

[假设 90° 诊断视频，非原预测范围](/root/autodl-tmp/PAct/repro/outputs/closed_multicase_20261028_v1/10489/10489_assumed90_diagnostic.mp4)

### 10655

[闭合拟合图](/root/autodl-tmp/PAct/repro/outputs/closed_multicase_20261028_v1/10655/closed_target_and_three_versions.png) · [未拟合视角对比](/root/autodl-tmp/PAct/repro/outputs/closed_multicase_20261028_v1/10655/unfitted_open_view_comparison.png) · [原生优化资产](/root/autodl-tmp/PAct/repro/outputs/closed_multicase_20261028_v1/10655/optimized_asset/object.json) · [逐关节评价](/root/autodl-tmp/PAct/repro/outputs/closed_multicase_20261028_v1/10655/evaluation.json)

### 20411

[闭合拟合图](/root/autodl-tmp/PAct/repro/outputs/closed_multicase_20261028_v1/20411/closed_target_and_three_versions.png) · [原生优化资产](/root/autodl-tmp/PAct/repro/outputs/closed_multicase_20261028_v1/20411/optimized_asset/object.json) · [逐关节评价](/root/autodl-tmp/PAct/repro/outputs/closed_multicase_20261028_v1/20411/evaluation.json)

### 31249

[闭合拟合图](/root/autodl-tmp/PAct/repro/outputs/closed_multicase_20261028_v1/31249/closed_target_and_three_versions.png) · [未拟合视角对比](/root/autodl-tmp/PAct/repro/outputs/closed_multicase_20261028_v1/31249/unfitted_open_view_comparison.png) · [原生优化资产](/root/autodl-tmp/PAct/repro/outputs/closed_multicase_20261028_v1/31249/optimized_asset/object.json) · [逐关节评价](/root/autodl-tmp/PAct/repro/outputs/closed_multicase_20261028_v1/31249/evaluation.json)

### 45194

[闭合拟合图](/root/autodl-tmp/PAct/repro/outputs/closed_multicase_20261028_v1/45194/closed_target_and_three_versions.png) · [未拟合视角对比](/root/autodl-tmp/PAct/repro/outputs/closed_multicase_20261028_v1/45194/unfitted_open_view_comparison.png) · [原生优化资产](/root/autodl-tmp/PAct/repro/outputs/closed_multicase_20261028_v1/45194/optimized_asset/object.json) · [逐关节评价](/root/autodl-tmp/PAct/repro/outputs/closed_multicase_20261028_v1/45194/evaluation.json)

### 45503

[闭合拟合图](/root/autodl-tmp/PAct/repro/outputs/closed_multicase_20261028_v1/45503/closed_target_and_three_versions.png) · [未拟合视角对比](/root/autodl-tmp/PAct/repro/outputs/closed_multicase_20261028_v1/45503/unfitted_open_view_comparison.png) · [原生优化资产](/root/autodl-tmp/PAct/repro/outputs/closed_multicase_20261028_v1/45503/optimized_asset/object.json) · [逐关节评价](/root/autodl-tmp/PAct/repro/outputs/closed_multicase_20261028_v1/45503/evaluation.json)

### 8961

[闭合拟合图](/root/autodl-tmp/PAct/repro/outputs/closed_multicase_20261028_v1/8961/closed_target_and_three_versions.png) · [未拟合视角对比](/root/autodl-tmp/PAct/repro/outputs/closed_multicase_20261028_v1/8961/unfitted_open_view_comparison.png) · [原生优化资产](/root/autodl-tmp/PAct/repro/outputs/closed_multicase_20261028_v1/8961/optimized_asset/object.json) · [逐关节评价](/root/autodl-tmp/PAct/repro/outputs/closed_multicase_20261028_v1/8961/evaluation.json)

[假设 90° 诊断视频，非原预测范围](/root/autodl-tmp/PAct/repro/outputs/closed_multicase_20261028_v1/8961/8961_assumed90_diagnostic.mp4)

### 9016

[闭合拟合图](/root/autodl-tmp/PAct/repro/outputs/closed_multicase_20261028_v1/9016/closed_target_and_three_versions.png) · [未拟合视角对比](/root/autodl-tmp/PAct/repro/outputs/closed_multicase_20261028_v1/9016/unfitted_open_view_comparison.png) · [原生优化资产](/root/autodl-tmp/PAct/repro/outputs/closed_multicase_20261028_v1/9016/optimized_asset/object.json) · [逐关节评价](/root/autodl-tmp/PAct/repro/outputs/closed_multicase_20261028_v1/9016/evaluation.json)
