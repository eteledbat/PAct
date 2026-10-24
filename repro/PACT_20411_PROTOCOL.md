# PAct 20411：预先固定的单场景对照

本轮在独立分支运行 PAct，读取 FreeArt3D 已有的 PartNet Table/20411 原始数据；不修改 FreeArt3D、VideoArtGS，也不恢复已暂停的 Part²GS。独立输出指针为 `repro/current_20411_experiment.txt`，上一轮结果指针保持不变。

输入预先选定第 05 张 RGB，单次 seed42，沿用官方预训练模型及此前的 25+25 steps、CFG7、arti_mean_num20。该 RGB 对应打开抽屉，PAct 生成的是模型预测的 canonical rest geometry 和关节。模型需要二维部件标签；用同拓扑 GT 端点 Mesh 区分固定/移动三角形，并按 FreeArt3D 原始 rendering camera 投影可见标签。原图 alpha 与投影轮廓 IoU 必须大于 .98，两个标签必须通过原版 37×37 读取器。此为 **GT-mask / oracle segmentation** 协议，GT 几何确实经二维 mask 进入输入；GT type、axis、origin、range 不进入推理。绝不将 `*_seg.png`（蓝色地面）误认成部件 mask。

原始导出确认为 fixed base + prismatic drawer。复用上一轮重叠代理和超参数：冻结所有 Mesh/GS、外观、关节类型、行程，21 个均匀训练状态，每部件 2048 个 source centers，opacity≥.05 的所有 Gaussian centers 为 NN targets，方向改变量≤5°，位移正则 .05，固定最终 200 步，不按 GT 或最优 collision 分数挑选结果。先检查真实资产梯度及 5 步 GPU smoke，再运行正式优化；自动检查 finite loss/gradient，保存 Adam、参数和 history，可从每 10 步 checkpoint 恢复。

**平移关节的 origin 不影响刚体运动。** 因而 position-only 严格等于原始运动，both 与 direction-only 等价。本场景仅运行一次方向优化，位置误差、位置改善和位置有效梯度均标记 N/A。复用优化器保留了上一轮包含 “all three ablations reported” 的通用 metadata 文本；本协议明确本轮的数学等价关系，不把两个未重复运行的分支伪装成独立实测。

优化器原有 GT 文件读取阻断、文件 SHA 检查继续生效。推理原始结果和最终候选均保留。GT 轴评估只在 prediction seal 后进行，GT 平移方向和行程由实际六状态 Mesh 位移核实，再与 GT joint label 交叉检查。固定坐标主结果使用两者 Y-up、x lateral、z front 的 canonical frame；位置尺度仅由固定部件 AABB 对齐。额外保留静态几何 ICP 的误差敏感性，不按候选重新配准。GT 00→05 位移用于明确标注 rest convention 的运动诊断；分别保存重建 rest 对 GT closed/open 的几何误差，避免把输入打开状态与生成闭合 rest 混淆。

GT 访问边界补充：上述 seal 后评估指正式对照评分阶段；准备数据时已查看 GT joint 元信息、六状态位移，并做原始方向 sanity check，因此不能声称首次读取 GT 在 seal 后。GT-mask 输入本身也依赖 GT。实际未用 GT 更新轴、调整本轮沿用的 loss 超参数或选择最终候选，优化子进程的 GT 读取阻断记录为空。

评估用独立 source samples（最多每部件 12000，与训练 disjoint），全部 targets，41 个时间状态，其中 20 个为训练时间间隔中点；独立 CUDA NN 对 CPU NN 做实际点核查。另用全量原始三角形、21 个状态计算 surface intersection curves，并输出静态/中间/终点、双视角动画与红色交线位置。所有方法共享几何、运动范围和相机。Gaussian overlap、中心间距和三角形交线均是代理，不能宣称真实实体碰撞体积。Mesh 未验证为合法封闭实体时 volume/depth 为 unavailable，不能写 0。

实际运行补充（仅评估加速，不改 loss/模型）：百万级 Mesh 的旧版整网格 VTK 空间查询每姿态约两分钟，保留了原始/优化各前六状态。新增三角形质心递归分块（每块≤12000 triangles），每个原始三角形严格属于一块；保守 AABB 粗检只排除不相交块对，其余仍用原版 VTK AllContacts、零最终判定容差。两方法各六个既有状态与旧实现交叉验证，接触计数、非零交线计数完全一致，归一长度差≤1.52e-7。通过后改用分块评估完成 21 状态；没有 decimation、补洞、凸化或近似几何，旧 partial outputs/logs 留在实验目录。该效率修复不影响优化参数或候选选择。

在本地保存的 SINGAPO data split 中，Table/20411 及其三个增强版本出现在 train。这不能支持未见物体泛化结论；实际 PAct 预训练样本完整清单仍未核实。与 FreeArt3D 的历史数字只作带协议说明的参考，不能当成相同输入、相同训练和相同对齐下的公平排名。

运行与监控：

```bash
PYTHONDONTWRITEBYTECODE=1 /root/autodl-tmp/conda_envs/pact/bin/python repro/tools/run_single_case_refinement.py --root "$(cat repro/current_20411_experiment.txt)"
PYTHONDONTWRITEBYTECODE=1 /root/autodl-tmp/conda_envs/pact/bin/python repro/tools/finish_single_case_artifacts.py --root "$(cat repro/current_20411_experiment.txt)"
python repro/tools/monitor_collision_refinement.py --root "$(cat repro/current_20411_experiment.txt)" --watch
```

正式任务在 tmux 中运行；上述第一条可以在中断后重启，完整阶段沿用已有输出，优化阶段按原 protocol/source SHA 核对后恢复。禁止覆盖已有完整实验。源脚本、运行命令、日志、环境和每阶段耗时保留在实验目录。结果见后续单独报告，不在此预写。
