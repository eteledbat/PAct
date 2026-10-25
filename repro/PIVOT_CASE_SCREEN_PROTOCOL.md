# 查找几何较好、轴位置有缺陷的实际案例

本轮是有目的的诊断筛选，使用 GT 评价来找到 pivot 缺陷，不是无偏 test-set benchmark。所有尝试的案例、失败及筛选依据均保留；不只展示成功优化。独立分支 `research/pivot-case-screen-20261025`。原始 20411 结果和全部模型文件保留。

已有候选：10489 两个 revolute 方向误差0°，pivot约3.5% GT static diagonal，但原始 range只有6.5°/9.9°；31249 正确门轴位置误差只有0.596%，且另一扇门类型错误；官方 Dishwasher 无可靠 GT。它们不能直接作为已经证实“几何好且主要是 pivot 问题”的成功案例。

新增小范围筛查固定 45503（3扇旋转门）和10655（2个旋转部件），来源是本地 VideoArtGS/SAPIEN 实际数据。先核对相机/GT-mask silhouette IoU>.98、原版 reader 中各部件可见；再沿用官方权重、frame0、seed42、25+25步、CFG7、arti_mean_num20，逐场景单 GPU 推理，不做网络训练、不搜索 seed。GT Mesh 投影的2D part mask 是明确的 oracle input；GT axis/type/pivot 不传入模型。GT 指标允许用于这次诊断案例选择，不能据此声明泛化效果或在优化中使用 GT。

筛选扩展记录：首批45503/10655均出现活动部件被预测为fixed，静态RMS分别7.79%/9.85%，故不合格。保留首批协议和status后，增加9016（几何简单的双开门）与100481（提手旋转的篮筐），仍固定相同frame/seed/模型，最多四个新增推理，不基于最终优化成功与否选择/隐去结果。未调整 loss 超参数。若无严格合格案例，10489可作低运动幅度的有限诊断，不能把它包装成满足≥20°条件的主成功案例。

筛选关注：revolute 类型正确；方向误差≤3°；pivot位置误差≥1% GT static diagonal；原始预测角度至少20°以提供运动信号。几何需额外评估固定部件和移动部件，不能只凭 RGB 外观或静态部件 RMS 判定。移动部件应将生成 rest 与输入打开姿态区分：保留共同全局对齐的误差，另给不改变尺度的 per-part rigid shape-fit 诊断，后者只用于判断形状而不用于轴评分或优化。静态 ICP 的已有全局变换作为敏感性分析，任何候选保持同一对齐。目标几何误差为静态 RMS≤3%、shape-fit moving Chamfer≤3% static diagonal；未达到就如实标记，不强称好几何。

若实际案例符合条件，固定几何、方向、类型和 range，仅对 revolute pivot 的垂直轴平面分量做上一轮相同的 position-only 优化：21训练状态，2048 sources/part、全部 retained Gaussian targets、pivot≤5% asset diagonal、正则.05、Adam position lr=.0005D、固定最终200步。先实际资产梯度检查和5步 smoke，再正式 tmux 运行。GT文件读取阻断继续生效；GT仅用于选案例与封存后的指标计算，不参与梯度或最终 checkpoint 选择。

评估原始和优化后的 pivot（固定全局帧及同一静态ICP）、方向不变、41状态 held-out-source Gaussian proxy、21状态全原始三角形交线、rest几何SHA不变及共享相机动画。实体合法性未验证时真实体积/穿透深度为 unavailable。必要的 GT-pivot 替换仅作明确标注的 oracle 诊断，不冒充方法预测；不人为扰动轴后把恢复实验当真实缺陷。
