# PAct：已有数据集的关节 GT 对照

三个案例均已完成实际 GPU 推理和评估；没有从头训练、关节 refinement、GT 误差选 seed 或重新配对部件。Part²GS 保持暂停。

## 数据和协议

使用本机 VideoArtGS/SAPIEN 的 10489、31249、101808，每例固定第 0 帧。RGB 来自原数据；由于没有现成二维部件标签，将已有 GT 部件 Mesh 按标定相机投影为 EXR mask。**这是 GT-mask / oracle segmentation 条件**：提供了真实部件划分及部件数，不能称为无 GT 的自动单图结果。原图不做内容编辑。GT 轴、GT 类型、GT origin 未送入 PAct；GT Mesh 不作为三维生成输入，但用于 mask 和事后评估。

GT 来自各场景 `gt/mobility_v2.json`；不使用根目录估计的 joint_infos 文件。过滤规则及固定坐标旋转来自已有 VideoArtGS `utils/metrics.py:read_gt`。部件对应由原版输入 mask 的排序确定，不按 GT 轴误差找最优排列。101808 原 mobility 含旋钮关节，按此数据协议只评估两个门，旋钮并入静态几何，不把它写成全物体九关节评估。

官方代码 SHA `a5ccec5056e7b27f39ddb5b83613c0efba9b2861`，模型 revision `8c6b56c7011454bbfb45a83e2dbfa6bbbfde63e9`。三例都用 seed42、SS25/SLAT25、CFG7、最后20步 feature 聚合；每例只运行一个预测。独立环境和运行兼容修复见 [CHANGES.md](CHANGES.md)；没有修改原版轴吸附、bbox origin 后处理或 Loss。

## 主评估：固定坐标规范

方向只使用源码规定的固定旋转；没有 ICP 或 GT 轴拟合。位置使用静态部件 AABB 对角线的单一尺度与中心平移，GT 静态几何仅在评估中使用。角度为 `acos(abs(dot(unit(pred), unit(GT))))`，单位度，不检验轴正负或开合方向。Pivot 指预测 origin 到 GT 无限轴的垂距，以 GT 静态主体对角线百分比表示；不是厘米，也不惩罚不可辨识的沿轴位移。另存无限轴最短线距；非平行相交线可得到零线距，不能单靠该值判断 pivot。prismatic 和类型错误的 pivot 均为 unavailable。

| Case | 活动关节类型正确 | 类型正确子集的 raw 平均角误差 | 同一子集原版导出平均角误差 | 有效角度数 / 全部活动关节 |
|---|---:|---:|---:|---:|
| 10489 | 2/2 | 2.964° | 0.000° | 2/2 |
| 31249 | 3/4 | 0.477° | 0.404° | 3/4 |
| 101808 | 0/2 | unavailable | unavailable | 0/2 |

**平均角度只统计类型正确的子集，必须与覆盖数一起阅读。**总共5/8个活动关节类型正确；这不是5根轴代表8根关节都成功。原始 joint_errors.json 中原有 mean 字段统计所有非零向量，包含错误类型；它们保留用于审计，不能当作类型正确的轴精度。上表由完整记录另算，未删除失败样本。

| Case / GT mobility id（Mesh index） | GT | raw 类型 / 方向诊断角 | 原版导出类型 / 方向诊断角 | raw → 导出 Pivot（%主体对角线） |
|---|---|---|---|---|
| 10489 / 1（1） | revolute | revolute / 2.718° | revolute / 0.000° | 8.602 → 3.489 |
| 10489 / 2（2） | revolute | revolute / 3.211° | revolute / 0.000° | 4.082 → 3.522 |
| 31249 / 0（1） | prismatic | prismatic / 0.537° | prismatic / 0.537° | unavailable → unavailable |
| 31249 / 1（2） | prismatic | prismatic / 0.676° | prismatic / 0.676° | unavailable → unavailable |
| 31249 / 3（3） | revolute | revolute / 0.217° | revolute / 0.000° | 15.302 → 0.596 |
| 31249 / 4（4） | revolute | prismatic / 80.810° | prismatic / 80.810° **类型错误；非有效轴分数** | unavailable → unavailable |
| 101808 / 0（1） | revolute | fixed / 74.050° | fixed / unavailable **类型错误；非有效轴分数** | unavailable → unavailable |
| 101808 / 1（2） | revolute | prismatic / 76.967° | fixed / 76.967° **类型错误；非有效轴分数** | unavailable → unavailable |

10489：两个连续 raw 方向误差为2.718°、3.211°；原版 `rescale_axis` 将 revolute 吸附到 X/Y/Z 后均为0°。因此0°不是连续回归恰好精确；bbox origin 后处理后仍有3.489%、3.522%的轴线位置偏差。

31249：两抽屉方向误差约0.537°、0.676°；一个门 raw 为0.217°，吸附后0°，其 pivot 从15.302%降到0.596%。另一个真实旋转门被预测成 prismatic，方向向量相差80.810°；此数仅用于定位失败，不是有效 revolute 轴误差。

101808：两门最终均为 fixed，活动关节类型0/2正确，没有有效轴精度或 pivot。GT 静态主体却被标为 door/revolute，两门被标为 base。第二门 raw 类型为 prismatic，被原版 `parse_tree` 的 base 语义强制设成 fixed，却仍留下非零方向字段；其76.967°是残留向量诊断，不能当作 fixed 关节存在可用轴。该例几何/整体姿态也失败，不能将大方向偏差全部归因于局部轴回归。

## 只用静态几何的配准敏感性

额外做一个固定初始化的局部刚体 ICP：静态表面12000点、80%截尾、最多50次迭代；不使用 GT 轴、类型、origin，不多起点挑分数。用独立seed1表面点衡量几何残差。只在评估中变换预测，不覆盖模型输出；主评估仍保留固定坐标结果。静态形状不完整或对称时局部配准也不唯一，因此它只是敏感性诊断。

| Case | 配准旋转变化 | 独立静态表面 RMS（%对角线）前 → 后 | 类型正确子集 raw 角均值前 → 后 | 导出角均值前 → 后 |
|---|---:|---:|---:|---:|
| 10489 | 0.382° | 2.531 → 2.526 | 2.964 → 2.829 | 0.000 → 0.382 |
| 31249 | 2.986° | 1.979 → 1.979 | 0.477 → 3.248 | 0.404 → 3.320 |
| 101808 | 45.345° | 12.568 → 11.996 | unavailable → unavailable | unavailable → unavailable |

10489 配准后的导出 pivot 为6.664%、6.611%，比固定 AABB 归一化下更大，说明位置量对整体平移有敏感性。31249 几何残差略变差，却可旋转约2.99°，说明不能通过选一个配准来宣称亚度级精度。101808 配准旋转约45.34°后残差仍约12%，且类型仍全错；未把配准当成修复成功。

## 输入、几何及轴叠加

下面左侧为实际原图、中间为投影标签、右侧为原版 Gaussian 动画第一帧。右侧相机与输入不同，不做像素指标。轴图是完整 Mesh 顶点的抽样多视角叠加：GT 红、raw 橙、导出蓝。错误类型的非零字段只表示诊断向量。

### 10489

![10489 实际输入、标签和生成结果](outputs/gt_10489_20261024T070130Z/joint_evaluation/input_mask_generated.jpg)

[原版关节动画](outputs/gt_10489_20261024T070130Z/seed42_slatcfg7.0_sscfg7.0_sssteps25_slatsteps25_artioutmean_feature_regression_steps/00__articulation_animation.mp4)；[逐关节原始指标](outputs/gt_10489_20261024T070130Z/joint_evaluation/joint_errors.json)；[配准敏感性](outputs/gt_10489_20261024T070130Z/joint_evaluation/geometry_registration_sensitivity.json)

![10489 类型正确的旋转关节与 GT 叠加](outputs/gt_10489_20261024T070130Z/joint_evaluation/axes_gtpart1.png)

### 31249

![31249 实际输入、标签和生成结果](outputs/gt_31249_20261024T070509Z/joint_evaluation/input_mask_generated.jpg)

[原版关节动画](outputs/gt_31249_20261024T070509Z/seed42_slatcfg7.0_sscfg7.0_sssteps25_slatsteps25_artioutmean_feature_regression_steps/00__articulation_animation.mp4)；[逐关节原始指标](outputs/gt_31249_20261024T070509Z/joint_evaluation/joint_errors.json)；[配准敏感性](outputs/gt_31249_20261024T070509Z/joint_evaluation/geometry_registration_sensitivity.json)

![31249 类型正确的旋转关节与 GT 叠加](outputs/gt_31249_20261024T070509Z/joint_evaluation/axes_gtpart3.png)

### 101808

![101808 实际输入、标签和生成结果](outputs/gt_101808_20261024T071040Z/joint_evaluation/input_mask_generated.jpg)

[原版关节动画](outputs/gt_101808_20261024T071040Z/seed42_slatcfg7.0_sscfg7.0_sssteps25_slatsteps25_artioutmean_feature_regression_steps/00__articulation_animation.mp4)；[逐关节原始指标](outputs/gt_101808_20261024T071040Z/joint_evaluation/joint_errors.json)；[配准敏感性](outputs/gt_101808_20261024T071040Z/joint_evaluation/geometry_registration_sensitivity.json)

本例输出将原静态主体识别成活动门，两门识别成 base；不将残留 fixed 方向字段画成有效关节轴。完整输入部件与输出 node/语义对应保存在汇总 JSON。

## 划分、耗时与可复查性

固定版本 SINGAPO split：10489 和101808在 train；31249 基础对象在 test，但 train 包含3个31249-*增强实例。因此本报告不是严格未见对象泛化评估。PAct 实际预训练数据 manifest 未独立核实，这份公开划分只能提示潜在重叠。[固定 split 文件](evidence/singapo-data-split.json)；[匹配条目](evidence/gt-case-split-audit.json)。

| Case | RGBA前景/投影mask IoU | 推理+导出+校验秒数 | 采样观察到的显存 MiB |
|---|---:|---:|---:|
| 10489 | 0.999860 | 60.74 | 17776 |
| 31249 | 0.999896 | 80.93 | 22752 |
| 101808 | 0.999537 | 70.77 | 21554 |

显存每10秒采样，不是精确峰值；耗时不含准备mask和后续评估。所有模型 state_dict 加载均无缺失或多余 key。

生成结果目录各自保存 command、源码差异、日志、raw head、后处理前后 joint JSON、Mesh、GS、视频和评估。GT annotation、原始图像、标定、Mesh 和相关评估证据均记录 SHA256。指标公式通过解析构造的符号反转、30°、平行轴偏移、相交轴、零向量和刚体/尺度协变检查；这些是公式测试，不是新实验结果。[公式测试证据](evidence/joint-metric-math.json)。

按阶段复查或运行新案例（新目录拒绝覆盖）：

```bash
cd /root/autodl-tmp/PAct
PYTHONDONTWRITEBYTECODE=1 /root/autodl-tmp/conda_envs/pact/bin/python repro/tools/prepare_video_gt_case.py --scene SCENE_ID
PYTHONDONTWRITEBYTECODE=1 /root/autodl-tmp/conda_envs/pact/bin/python repro/tools/run_gt_case.py --scene SCENE_ID
# 将 run 替换为该次输出目录
/root/autodl-tmp/conda_envs/pact/bin/python repro/tools/evaluate_gt_joints.py --run /absolute/run
/root/autodl-tmp/conda_envs/pact/bin/python repro/tools/evaluate_registration_sensitivity.py --run /absolute/run
```

运行中可直接读取 `repro/current_gt_SCENE_ID.txt` 指向目录下的 `status.json` 和 `inference.log`；也可独立监控（其他case替换数字）：

```bash
/root/autodl-tmp/conda_envs/pact/bin/python repro/tools/run_status.py --gt-scene 31249 --watch
```

已完成三例均无需继续训练，进程已退出。汇总报告可从保存的输出重建：

```bash
/root/autodl-tmp/conda_envs/pact/bin/python repro/tools/build_gt_joint_report.py
```

## 解释边界

当前结果支持：部分正确类型关节 raw 轴已较准，原版坐标轴吸附进一步压低方向误差；轴位置、类型与部件语义仍会失败。不能由0°方向误差推出 pivot 正确或全运动无穿模。本轮没有对三例计算新碰撞曲线、标准 Chamfer 或图像重建指标，也没有关节范围对齐后的误差，均不冒充已测。原始 GT limit 与预测 range 保留在 JSON；输入处于非零开合状态，未强行比较不同 rest-state 的范围。10489原版动画看似不动的原因见 [运动幅度检查](ANIMATION_10489_DIAGNOSIS.md)：两门仅预测约6.53°和9.95°的运动，不能用方向0°误差代表完整运动成功。原版官方 Dishwasher_001 的21状态代理分析仍单独见 [原 smoke 报告](REPRODUCTION_REPORT.md)，不能移植为三例的碰撞结果。

所有工具和小型证据提交独立分支；大型模型、输入/输出Mesh和视频仅保留本机，不向官方上游提交或推送。


## 20411 补充案例

新增 [PAct 20411 原始重建与减少重叠轴优化](PACT_20411_RESULTS.md)。该案例来自 FreeArt3D 已有 PartNet 数据，使用单 RGB + GT-mask，GT 轴未参与优化或候选选择；原始方向误差 0.636°，优化后 5.092°，平均 Gaussian 代理下降 78.64%，全三角形交线下降 85.38%。与本报告前述案例分开统计，保留其数据和对齐协议差异。


## 轴位置案例筛查与10489补充

新增 [四场景筛查与10489原始行程下的pivot-only实测](PIVOT_CASE_SCREEN_RESULTS.md)。10489形状相对较好但运动幅度不足；固定几何及方向的200步优化降低重叠代理，却使GT pivot误差从3.49%/3.52%增至8.88%/6.06%。四个新场景因部件类型/角色及几何问题未通过筛查。该诊断按GT筛选并保留全部失败，不作为无偏泛化或成功证明。
