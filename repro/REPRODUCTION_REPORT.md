# PAct 复现与物理诊断：首个官方示例

更新：2026-10-24T03:24:48.825918+00:00。

**范围：官方 Dishwasher_001 的真实推理 smoke 与运动诊断。不是 PARIS Storage45135 完整 benchmark 复现；该场景尚未运行。**

新增已有数据案例的 GT 对照（10489、31249、101808），单独见 [GT 关节评估报告](GT_JOINT_EVALUATION.md)。下文仅描述原 Dishwasher smoke；其 unavailable GT 项不代表新增案例没有 GT。

## 1. 来源与环境

官方源码 `Mobiuslqm/PAct`，commit `a5ccec5056e7b27f39ddb5b83613c0efba9b2861`；官方模型 `PAct000/PAct`，revision `8c6b56c7011454bbfb45a83e2dbfa6bbbfde63e9`，3,972,142,331 字节、14 文件，已逐一核对大小和官方 LFS SHA256。

工作分支 `repro/pact-baseline-20261024`；独立环境 `/root/autodl-tmp/conda_envs/pact`。Python 3.10.22 (main, Oct  1 2026, 20:14:04) [GCC 14.3.0]; torch 2.4.0+cu121; CUDA runtime 12.1。GPU RTX4080 SUPER，报告容量约32GiB。

官方 YAML 为 torch2.4.0/CUDA11.8，本机独立克隆为2.4.0+cu121/CUDA12.1，部分编译依赖版本有偏差；实际导入和推理已验证，不声称完整 YAML 逐项相同。[全部改动和偏差](CHANGES.md)；[环境验证](evidence/environment-smoke.json)。原 FreeArt3D、VideoArtGS、Part²GS 环境未修改。

DINOv2 从已有缓存复制，记录本地 SHA256，使用新克隆的官方固定源码，严格 state_dict 加载通过；本地计算的 SHA 并非独立发布的官方校验值。[DINOv2 来源](evidence/dinov2-provenance.json)。

## 2. 输入、采样与运行

输入为官方仓库 Dishwasher_001 RGBA 和 EXR 二维部件 mask，识别为2部件。mask 是作者提供，自动/GT 来源未独立验证，不能描述成完全无 mask 的单图输入。没有 GT mesh、axis 或 joint type 作为本次推理输入。

原采样：SS25步、SLAT25步，CFG均为7，最后20步 feature 聚合关节，seed42。加载、推理、写出和校验总计约 62.17 秒；不是纯采样速度，也不直接与论文A800的耗时对比。

共记录 6 次模型 state_dict 加载，所有缺失/多余 key 列表为空：[加载审计](outputs/official_dishwasher001_20261024T030718Z/checkpoint_loads.jsonl)。

首次实际推理在 exploded video 写出时因原版 imageio.imwrite 接收4D数组失败；修成视频接口 mimsave 后，用相同配置在新目录重跑通过。失败目录保留；没有选择更好关节/碰撞结果作为成功依据。

输出包括原版60帧关节动画、21帧 orbit 分离预览、raw mesh PLY、Gaussian PLY 与 object.json；未做可选 GLB 贴图烘焙。[命令和模型来源](outputs/official_dishwasher001_20261024T030718Z/command.json)；[输入哈希](outputs/official_dishwasher001_20261024T030718Z/input_manifest.json)。

## 3. 预测和几何

| 部件 | 预测类型 | 预测运动范围 | 顶点 | 面 | Mesh封闭 |
|---|---|---|---:|---:|---|
| base | fixed | [0.0, 0.0] | 588454 | 1176888 | False |
| door | revolute | [0.0, 56.8550910949707] | 201092 | 402180 | True |

范围为预测值，revolute单位度；几何距离为模型归一化坐标，不能称为米。原始 mesh 未补洞、重建或优化。base 不封闭，因此真实实体交集体积/实体穿透深度 unavailable，不填0。

原版 rescale_axis 会将 revolute 轴吸附到最近 X/Y/Z 方向，当前导出 door 轴为(1,0,0)。这是经过启发式处理的轴，不能代表原始连续回归方向误差为0。原版 animate/export 还分别依据 Gaussian/Mesh bbox 投影 origin；保留这些处理，没有进行新的 refinement。

## 4. 21状态运动诊断

均匀采样导出预测范围内21状态，使用原版 _local_transform 和 object.json。无GT拟合。分析耗时 413.29 秒。

Gaussian PLY 保存时比 Mesh/JSON 多一次固定旋转，评估依据代码撤销该序列化旋转，对齐到 Mesh/JSON；不是ICP/GT对齐。原始文件未改。

opacity≥0.05 后使用全部保留点，计算双向、opacity加权的最近中心 Gaussian 核；sigma取三轴activated scales均值。该定义不同于全点对核平均，也不是真实体积。最小中心距离在全部保留跨部件点之间求得，1σ球间隙仅针对最近中心点对。

| 统计 | 实测 |
|---|---:|
| 最大最近中心 Gaussian overlap proxy | 0.0056945267 |
| 最小中心距离 | 4.0850593e-05 |
| 最小最近对1σ球间隙 proxy | -0.007000973 |
| 真实实体交叠体积 | unavailable：base Mesh不封闭 |
| 真实实体穿透深度 | unavailable |

高 Gaussian 核或负球间隙也可能来自正常接触、Gaussian尺度和表面表示；不能仅据此断言严重实体穿模。当前没有对照实验，因此不能归因某个物理项有效或无效。

![实际输入与生成结果，视角未配准](outputs/official_dishwasher001_20261024T030718Z/input_vs_generated.png)

![21状态碰撞代理曲线](outputs/official_dishwasher001_20261024T030718Z/motion_analysis/collision_curve.png)

![预测起始状态及轴](outputs/official_dishwasher001_20261024T030718Z/motion_analysis/gaussians_state00.png)

![预测中间状态及轴](outputs/official_dishwasher001_20261024T030718Z/motion_analysis/gaussians_state10.png)

![预测终止状态及轴](outputs/official_dishwasher001_20261024T030718Z/motion_analysis/gaussians_state20.png)

高代理值区域仅定位最近核候选，不是实际穿透标签；终止状态热点的空间范围见 [热点边界与阈值](outputs/official_dishwasher001_20261024T030718Z/motion_analysis/hotspot_summary.json)。

![高代理值区域，状态0](outputs/official_dishwasher001_20261024T030718Z/motion_analysis/hotspot_state00.png)

![高代理值区域，状态10](outputs/official_dishwasher001_20261024T030718Z/motion_analysis/hotspot_state10.png)

![高代理值区域，状态20](outputs/official_dishwasher001_20261024T030718Z/motion_analysis/hotspot_state20.png)

base为一个连通表面，28条边界边；door为两个封闭表面分量，最大分量占面数约53.2%。两个表面分量也可能对应内外壳，不能直接称为碎片；未做自交完整验证。audit中的signed_volume只是有向四面体积分诊断，开放base的该值不是真实实体体积。

代理曲线最低在状态8/20；终止/起始值比约 5.16。该比值只描述本代理，不是实体穿透变化倍率。

![实际三角 Mesh，状态0](outputs/official_dishwasher001_20261024T030718Z/mesh_previews/mesh_state00_3views.png)

![实际三角 Mesh，状态10](outputs/official_dishwasher001_20261024T030718Z/mesh_previews/mesh_state10_3views.png)

![实际三角 Mesh，状态20](outputs/official_dishwasher001_20261024T030718Z/mesh_previews/mesh_state20_3views.png)

[21状态原始Mesh运动视频](outputs/official_dishwasher001_20261024T030718Z/mesh_previews/raw_mesh_motion_21.mp4)

[逐状态数值与变换](outputs/official_dishwasher001_20261024T030718Z/motion_analysis/collision_states.json)

[Mesh边界、碎片和合法性诊断](outputs/official_dishwasher001_20261024T030718Z/motion_analysis/mesh_audit.json)

[原版Gaussian关节运动视频](outputs/official_dishwasher001_20261024T030718Z/seed42_slatcfg7.0_sscfg7.0_sssteps25_slatsteps25_artioutmean_feature_regression_steps/00__articulation_animation.mp4)

## 5. GT指标与协议限制

| 指标 | 状态 |
|---|---|
| PSNR / SSIM / LPIPS | unavailable：无标定多视角目标，不将不同视角图像直接比较 |
| Fixed / Moving / Whole CD | unavailable：无已核验对应GT几何与对齐 |
| Joint Type Accuracy / Axis Error / Pivot Error / Range Error | unavailable：没有可靠GT关节 |

这是官方释放预训练模型的推理验证，没有本机从头训练 PAct、按对象优化或按GT误差选择 checkpoint。GT-independent Selection 对本次对象级推理没有Part²GS那样的best/final区别；不能声称预训练本身无GT监督，也未验证该示例与训练集是否重叠。

Storage45135正式实验需补齐可追溯二维部件mask、数据预处理/坐标对齐及训练测试集重叠审计；官方示例没有冒充该场景。后续阶段在这些条件核实后再启动。

## 6. 保存与监控

成功输出：`/root/autodl-tmp/PAct/repro/outputs/official_dishwasher001_20261024T030718Z`。大型模型/视频/点云只在本机，Git忽略；工具、报告、compact evidence提交独立分支，未向上游推送。

```bash
/root/autodl-tmp/conda_envs/pact/bin/python /root/autodl-tmp/PAct/repro/tools/run_status.py --watch
```

Part²GS依用户要求保持暂停，完整checkpoint在iteration5020后，可从5021恢复；本轮没有重启。

## 7. Storage45135 划分预审

依据 [SINGAPO 官方固定版本 data_split.json](https://raw.githubusercontent.com/3dlg-hcvc/singapo/de4b46616a7ecb1290bb84e215da0facb765bda5/data/data_split.json)，StorageFurniture/45135 在 test；train 同时含6条45135-*增强实例。本记录不能证明PAct实际训练用了同一几何，但增强来源与实际训练manifest尚未核验，不能先称为严格未见对象测试。[固定版本与匹配记录](evidence/storage45135-split-audit.json)。

本轮PAct执行的是预训练推理，没有Part²GS的逐对象Repel/Repulsion/Contact等训练步骤或优化器。本轮未做这些Loss的消融或Joint梯度优化；运动由原版回归关节和几何后处理生成。
