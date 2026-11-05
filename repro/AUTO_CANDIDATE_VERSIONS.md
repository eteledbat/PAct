# “自动候选”是哪一个版本

本文固定一个名称，避免后续继续从历史目录里按 case 取结果。

## 结论

用户所说的**自动候选**是：

```
repro/outputs/closed_auto_geometry_selected_20261104/
```

它是 2026-11-04 的**选定集合**，在 `closed_auto_selected_vs_manual_style_videos_20261105/INDEX.md` 中对应 **“上次选定版 / Selected automatic”** 一列，**不是**同一批视频里的 “New automatic | manual-style fit”。

它不是单一模型，而是由 `select_auto_closed_geometry.py` 从以下候选池逐 case 选出的集合：

| 标签 | 目录 | 被选中 case |
|---|---|---|
| `geometric_pose`（主候选，v8） | `closed_auto_geometry_v8_geometric_pose_20261104` | 8 例 |
| `closed_drawer_image_pose`（v6） | `closed_auto_geometry_v6_closed_drawers_20261104` | 45194 |
| `previous_auto`（旧 v2） | `closed_travel_reference_geometry_v011_20261031_v2` | 8961 |
| `image_pose`（v5，未胜出） | `closed_auto_geometry_v5_20261104` | 无 |

选择规则（`SELECTION.json` / `select_auto_closed_geometry.py`）：

- 默认保留 `geometric_pose`；
- 其他自动候选只有在**同一张关闭图**上的二值轮廓 IoU 高出 **0.02 以上**才替换；
- 不读取 GT 三维网格、GT 关节或人工版指标；
- 各候选并非每个 case 都齐全，所以这个选定集合是**逐 case 的实验候选选择结果**，不是一个统一实现的端到端模型。

`REPRODUCTION.md` 明确记录：八个 case 用 v8 主批，45194 用 v6，8961 用此前发布的自动拟合；选择由 `select_auto_closed_geometry.py` 复现。

## 不要混淆的三个东西

| 名称 | 目录 | 作用 |
|---|---|---|
| **自动候选（本次要找的）** | `closed_auto_geometry_selected_20261104` | 历史多候选逐 case 选择的自动结果 |
| 新自动实验版 | `closed_auto_manual_style_geometric_pose_20261104_v1` 等 | 2026-11-04/05 的 manual-style 组合实验，不是历史自动候选 |
| Origin | PAct 单图原生生成 | 所有候选的共同初始几何 |

## 共同参考姿态

历史流程中，拟合、CD 评价使用拟合出的关闭姿态 `q_closed`，而视频按各资产的**原生区间**播放；多数 case 的 `q_closed` 落在原生区间之外，视频因此没有经过被评价的关闭状态。`pose_convention.py` 把这一点显式化成 sidecar 记录：

- `q_closed`：拟合关闭姿态，用于几何拟合与 CD 评价；
- `playback`：`[q_closed, q_open_native]`，视频必须包含拟合关闭状态；
- 当 `q_closed` 超出原生区间，或原生行程近似为零时，标记 `open_direction_ambiguous`，交由第二个观测状态判定，而不是用规则强行选一个方向。

当前十例审计：27 个可动关节，24 个有拟合姿态，14 个的 `q_closed` 在原生区间之外，2 个（8961 两扇门）原生行程小于 1°，17 个需要第二状态证据。

## 论文表述边界

当前输入仍使用数据集 GT 投影得到的实例 mask，以及事先确定的关闭参考帧；因此不能写成“仅两张 RGB 的全自动重建”。本文件只固定候选定义与姿态约定，不改变任何已封存的拟合结果或评价。
