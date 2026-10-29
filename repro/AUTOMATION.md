# PAct 自动语义、轴修复与单图几何优化

当前代码版本：**`v0.1.1-handle-attachment-axis`（2026-10-30）**。

本补丁把可转动提手的后备轴从支杆中心移到主体连接端证据处；45项测试通过，100481已完成新轴几何拟合和四列视频。见[补丁说明](versions/v0.1.1-handle-attachment-axis.md)及[文件清单](versions/v0.1.1-handle-attachment-axis.manifest.json)。

[版本流程、运行方法、实验边界与结果](versions/v0.1.0-auto-semantics-pose.md) · [版本源码清单](versions/v0.1.0-auto-semantics-pose.manifest.json)

当前流程是：**原始 RGB＋已有实例 mask → PAct Origin → VLM 部件语义/连接证据 → 独立物理部件与规则轴修复 → 从同一输入估计抽屉观测位移 → 固定该姿态优化几何 → 封存后评价和渲染**。

- Origin 接收 RGB 和部件实例 ID mask；人工/VLM 的门、抽屉等类别文字不传入 Origin。本地实验沿用 oracle/GT 投影实例 mask，不能称为纯 RGB、无需 mask 的重建。
- 基础版VLM轴修复和固定q0几何实验覆盖11个case；“观测位移→几何”的人工/VLM对照验证了20411，发布后VLM流程完成旧目录10-case本地批量验证。本补丁的新连接端规则仅复核100481，旧批量指标不能作为新规则成绩。
- 新增观测位移只优化移动关节在输入照片里的状态，不学习机械最大行程，也不优化转动关节观测角度。
- 当前单图拟合不使用额外关闭图片。历史关闭图实验只作为明确声明额外监督的对照。
- 这是保留现有研究依赖的代码快照：运行需要本地输入、原生 PAct 导出、相机/配准及先验缓存；这些大文件不包含在 Git 中。迁移机器时须准备或迁移这些依赖，并核对配置中的路径。

主要入口：

| 阶段 | 工具 |
|---|---|
| 原实例 mask 上的两阶段语义识别 | `tools/recognize_semantics_two_stage.py` |
| 门把手可见位置识别 | `tools/recognize_door_pulls.py` |
| 自动物理部件分组与轴先验 | `tools/auto_component_axis_prior.py`、`tools/build_pull_guided_axes.py` |
| 固定 q0 的自动实例几何基线 | `tools/fit_vlm_component_geometry.py` |
| 抽屉观测位移与几何对照 | `tools/fit_observed_drawer_pose.py` |
| 独立评价、五列视频、核验 | `tools/evaluate_observed_drawer_pose.py`、`tools/render_observed_drawer_pose.py`、`tools/finalize_observed_drawer_pose.py` |

当前版本没有重新训练 PAct，没有从单张图估计真实机械限位，也没有实现重力/动力学稳定性求解。不得把规则先验或单张训练图 IoU 当成完整物理正确性证明。
