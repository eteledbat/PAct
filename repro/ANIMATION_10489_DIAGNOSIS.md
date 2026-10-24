# 10489 动画运动幅度检查

用户反馈动画看起来不动后，检查了实际保存的60帧视频、`animation_after.json` 和原版 `animate_and_render` / `_local_transform`。

| PAct node | 输入对应部件 | 原版动画类型 | 预测角度范围 |
|---|---|---|---|
| 1 | GT Mesh0，柜体主体 | fixed | 0 → 0° |
| 2 | GT Mesh1，上门 | revolute | 0 → −6.525° |
| 0 | GT Mesh2，下门 | revolute | 0 → +9.948° |

主体保持静止符合定义。门不是没有被动画程序驱动，而是预测转动幅度很小；原版动画将60帧的 t 从0线性插值到1，使用上述范围，没有人为限制到这些小角度。输入几何本来就处于开门状态，因此动画只在这个状态附近再转少量角度，没有完成闭合到全开的展示。

视频首尾帧的平均RGB绝对变化为0.458833（0–255）；任一通道变化超过5的像素占1.05667%，主要位于门上。差异图包含编码影响，仅用于检查动画是否静止，不是几何或关节精度指标。

![原版首帧、尾帧与8倍绝对差异](outputs/gt_10489_20261024T070130Z/joint_evaluation/animation_check/original_first_last_difference.png)

[原版动画](outputs/gt_10489_20261024T070130Z/seed42_slatcfg7.0_sscfg7.0_sssteps25_slatsteps25_artioutmean_feature_regression_steps/00__articulation_animation.mp4)；[视频及参数检查证据](evidence/10489-animation-probe.json)。

GT标注中两门limit均为0–180°，但输入开合姿态与生成rest-state的范围约定尚未对齐，因此没有把预测区间与GT直接相减并称为正式range error。能够确定的是：当前原版预测与动画只覆盖约6.53°、9.95°，不能展示正常完整开合。

此前报告中的“导出轴方向误差0°”仅涉及方向（且经过坐标轴吸附）；它不说明角度范围正确、轴线位置正确或运动重建成功。此次没有放大预测角度，没有使用GT范围替换原始输出，也没有重跑模型。轴GT对照仍见 [关节评估报告](GT_JOINT_EVALUATION.md)。

复查命令：

```bash
cd /root/autodl-tmp/PAct
PYTHONDONTWRITEBYTECODE=1 /root/autodl-tmp/conda_envs/pact/bin/python repro/tools/inspect_animation_motion.py --run /root/autodl-tmp/PAct/repro/outputs/gt_10489_20261024T070130Z
```
