# 自动识别语义，只用于轴选择

本流程不修改 PAct 几何生成、模型权重或现有几何。已有 RGB 与实例 mask 输入本地视觉语言模型，输出部件语义和关系，再由独立轴规则适配器处理；不是向 PAct 添加新的几何条件。

```
原始 RGB + 原实例 mask
        ↓ 本地 Qwen3-VL-4B-Instruct
先识别整个mask的部件类别，并封存结果
        ↓ 单独询问运动方式
运动候选 / 父部件 / 可见连接证据 / 不确定性
        ↓ 原始配准下的生成表面投影匹配
确认语义对应的实际生成部件
        ↓ 语义轴规则 + 几何支撑检查
只导出关节修改；原始几何文件保持不变
```

模型只需在每批识别开始时加载一次；识别结果缓存为 JSON，重复轴实验不需要再运行视觉模型。无需图像 API。模型权重从官方 Qwen 镜像下载，按固定 Hugging Face revision 的官方 LFS SHA256 校验。PAct 原环境保持不变，新版 Transformers 使用独立环境 `repro/envs/auto_semantics`。

语义包括 base、door、drawer、handle、wheel、knob、shelf、tray、unknown。类别针对整个mask覆盖的部件：门板加附属拉手归为门，独立提手区域才归为handle。部件 ID 不表示语义。同类不同实例保留独立 ID，父关系必须引用实际存在的实例；自指、成环或结构化输出错误会被拒绝。所有类别在 `CATEGORY_SEAL.json` 封存后才进入运动询问，后者不能改写类别。

门的类别与上翻/下翻/侧开/推拉分开处理。单帧没有可见铰链或运动线索时允许保留未知，不能把模型自报的 high 当作校准概率。抽屉采用规范 +Z 抽拉先验；固定拉手随门/抽屉；独立提手还需两个分离的几何连接区。轮子、旋钮、架子和托盘暂时只生成候选运动族。

参考 mask 与生成节点不保证一一正确对应。类型、轴无关的表面投影匹配可纠正节点错序；对合并门、几何归属错误、严重遮挡等没有明确对应的情况拒绝套用标签。目前不自动拆分/重组这些混合源节点，未通过的部件保持原生 PAct 运动。

运行示例（在 PAct 根目录）：

```bash
/root/autodl-tmp/conda_envs/pact/bin/python repro/tools/download_semantic_vlm.py --repo Qwen/Qwen3-VL-4B-Instruct

repro/envs/auto_semantics/bin/python repro/tools/recognize_semantics_two_stage.py \
  --root repro/outputs/auto_semantic_axes_new \
  --inventory repro/outputs/general_semantic_priors_20261029_v2/input_inventory.json \
  --stage prepare

repro/envs/auto_semantics/bin/python repro/tools/recognize_semantics_two_stage.py \
  --root repro/outputs/auto_semantic_axes_new --stage recognize

/root/autodl-tmp/conda_envs/pact/bin/python repro/tools/semantic_axis_router.py \
  --root repro/outputs/auto_semantic_axes_new \
  --registration-root repro/outputs/general_semantic_priors_20261029_v2
```

轴适配器复用原RGB/轮廓配准，仅使用尺度、屏幕平移、相机及坐标变换。既有人工语义、case 配方、旧几何拟合和 GT 关节不参与新语义与运动规则选择。输入 mask 仍是之前提供的 oracle 实例分割；自动分割本身不在本次范围。

每个 case 的 `semantics.json` 是机器识别结果；`axis_requests.json` 记录接受/拒绝及原因；`axis_only_export/object.json` 通过绝对路径引用原几何，仅包含被接受的关节修改。旧实验不覆盖。最近完成的结果由 `current_auto_semantic_axes.txt` 指向。

最新开发验证为[11-case报告](outputs/auto_semantic_axes_20261029_v5/REPORT.md)：38个参考实例中37个与既有人工语义注释一致，1个未知；12项测试通过。该一致率来自同一批开发case，不能视为独立测试准确率，也不是轴正确率。投影对应通过27/38，接受17个轴/固定关系请求，实际改变11个关节。源几何哈希、初始姿态和旧实验文件均已核验不变。

103015的三扇上窗自动归为door，但单帧中开合方式缺乏证据，运动子类型保留unknown；本流程没有补回旧实验的人工upward_door标签。100481的主体与两个提手语义也由模型识别，独立活动性和几何对应仍需另行通过证据检查。

后续已对封存的自动轴资产独立补做 GT 轴评价，并在冻结关节后运行11例原始条件帧几何拟合。详见[自动语义改轴与初始帧几何拟合报告](outputs/auto_semantic_initial_frame_20261029_v1/REPORT.md)。严格轴通过数 Origin→自动为8/27→9/27，但类型正确关节数24/27→21/27；几何拟合的训练 mask mIoU改善，未用视角IoU和三维CD宏均值退化。新增拟合实验单独保存，不改变上面的轴导出资产。

11例四路动态可视化见[GT / Origin / 自动改轴 / 初始帧拟合视频总览](outputs/auto_semantic_initial_frame_20261029_v1/AUTO_FOUR_VIDEOS.md)。
