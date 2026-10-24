# 固定 PAct 几何的碰撞引导轴优化：运行前协议

本实验由用户新要求授权，是独立后处理研究，不覆盖或改称原版 PAct baseline，也不改 Part²GS 物理 Loss。已有GT基线分数已知，因此不是盲测；新优化程序禁止读取GT文件和旧GT评估结果。

选用已运行的31249（有GT、存在类型错误）和官方Dishwasher_001（无GT、运动重叠代理升高）；10489只保留21状态筛查，因预测活动范围很小而不作为全程物理改善的主要样本。所有选择与限制披露，不另挑新seed/模型。

固定项：全部Gaussian的位置、尺度、opacity、外观；全部Mesh；部件对应、关节类型、关节行程、rest-state、相机。只调轴方向和revolute轴线位置。Prismatic的origin无意义，不调。原始文件SHA前后检查。

四组：A原版；B只调位置；C只调方向；D同时调位置和方向。B/C/D均固定200步Adam、训练每部件2048个opacity≥0.05的均匀Gaussian子集（seed0），21个等距运动状态。方向切平面学习率0.002，pivot学习率0.0005×资产对角线。方向偏离上限5°；pivot在初始轴的垂直平面内偏移不超过资产对角线5%，去除沿轴平移的不可辨识自由度。

目标为训练所有跨部件对和状态的均值：`nearest-center Gaussian overlap / original mean + 0.05 * normalized quadratic parameter displacement`。核与此前诊断相同，opacity和sigma固定。只采样源中心作为积分点，最近邻目标包含全部opacity≥0.05中心，并按源点抽样概率反加权，避免稀疏目标云改变密度定义。最近邻选择离散，在选定邻域内对双方位置、轴和pivot求导；每次生产loss刷新最近邻。只使用固定最终步checkpoint，不按GT或最低碰撞checkpoint择优。全部消融均报告。

最初两个5步开发smoke使用稀疏源/稀疏目标，梯度与输出正常，但目标稀疏会漏掉大量重叠。它们保留为开发记录，不作为正式方法或改进结果；正式试验前改为全密度目标并重新验证。该调整未读取新GT分数。

该目标是Gaussian重叠代理，可能惩罚正常接触，也不能保证真正不穿模。它可能通过不合理间隙降低代理。设置小幅偏移界限和原始参数正则，并独立统计轴偏移、原铰链锚点运动以及几何相交；不能将正则解释为物理可行性证书。

先完成实际导出数据的变换一致性、有限差分梯度和5步GPU smoke，再用tmux顺序运行6项正式实验。保留10步checkpoint、优化器与日志，支持显式resume。NaN/非有限梯度/文件变化立即停止。

主评估使用41个运动状态（含训练状态间的20个插值留出状态）以及训练未使用的Gaussian中心；固定seed2026，所有方法共享。另用VTK完整三角面相交检查至少21状态，和训练核不同。表面交线含接触，不等于实体穿透体积。各方法均保留t=0：该状态与轴无关，初始几何重叠不可能仅靠轴优化消除。

31249虽通过封闭/绕序初查，但两个部件已检测到自交；主体自交检测在超过180秒后停止，未知保持未知。Dishwasher主体开放。因此当前实体体积和精确穿透深度均unavailable，不补洞、不用凸包填空腔、不将undefined填0。[Open3D内外/距离前提](https://www.open3d.org/docs/release/tutorial/geometry/distance_queries.html)；[VTK表面相交接口](https://vtk.org/doc/nightly/html/classvtkCollisionDetectionFilter.html)。

新候选预测封存后，31249沿用此前固定坐标/尺度/平移做GT轴与pivot评估，另保留原静态配准敏感性。GT不进入优化或选结果。错判类型保留，不能用方向优化把prismatic称为恢复了revolute。Dishwasher无可靠GT，不提供GT提升百分比。统计两例观察和各消融，不能从两个样本宣称普适有效。
