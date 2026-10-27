# PAct 复现改动与环境偏差

2026-10-27追加对象：新增独立axis_extra工具与research/axis-only-extra-cases-20261027分支。20411/10655各200步direct A/B；不重复训练关节头，逐张量核实新特征的原始MLP后复用上轮封存A/B权重测试。新缓存入口从真实输入provenance选择FreeArt/VideoArtGS路径，支持未曾推理对象，不改原版采样。101284单次原版输出全fixed，失败保留、不强制GT类型。修预检允许平移pivot不参与计算图，第一次失败日志保留；原训练源码、Loss、旧资产和旧实验不改。新增后封存GT轴评估、41状态代理、21帧视频、类型/角色失败说明及完整性工具；结果见AXIS_EXTRA_CASES_RESULTS.md。

2026-10-27追加：research/axis-only-fixed-ranges-20261027进一步冻结原版native起止值，两条路线各A/B重新初始化，六项200步及评价已完成。新增独立axis_fixed工具，不修改上轮训练源码或已封存结果。direct仅theta/phi进入Adam；head只消费有效轴输出，其余通道回放原版，FK/导出显式固定原范围；训练GT目标剔除所有非轴通道。增加每步范围断言、共享MLP范围泄漏检查、真实中断恢复/零权重对照、全部活动节点有向旋转检查和GT匹配覆盖说明。保留相同运动Gaussian代理定义，不改原版Loss。完整参数、连续/官方导出两套GT轴读数、固定范围运动曲线与上一轮协议对照见AXIS_ONLY_FIXED_RANGES_RESULTS.md。未生成新几何、未恢复Part²GS、未执行完整flow-matching训练。

2026-10-27新增：在独立research/frozen-geometry-protocol-20261027分支完成冻结部件的direct/head两条A/B路线，共六次200步。新增GT训练对象打包、官方冻结特征/输出MLP缓存、纯刚体参数优化、梯度/刚体/后处理顺序核验、恢复与零权重控制、41状态只读评价和文档。head复用原版关节MSE及编码函数，但训练分布、冻结范围、连续通道mask与小样本划分有明确偏差；不是完整原版续训。未修改官方train.py、Stage2配置、原版训练Loss和模型；未恢复Part²GS，未修改旧部件。全部局部几何/外观文件SHA封存，细化只改关节参数或输出MLP；无自写RGB重建项。详情见FROZEN_JOINT_ROUTES_PROTOCOL.md和FROZEN_JOINT_ROUTES_RESULTS.md。

原版 commit `a5ccec5056e7b27f39ddb5b83613c0efba9b2861`，tag `upstream-original-a5ccec5`。工作分支 `repro/pact-baseline-20261024`，不推送 upstream。

| 改动 | 原因 | 算法影响 |
|---|---|---|
| infer_imgs.py 使用 args.data_dir | 原版无视 CLI，始终读取全部 22 个示例 | 仅选择输入，不改变生成计算 |
| 可选 PACT_DINOV2_REPO 本地官方源码 | 固定官方 DINOv2 commit，复用已核验缓存，避免在线 main 漂移 | 同一官方模型与权重，严格加载通过；未设变量时保持原入口 |
| guarded_infer.py 检查 checkpoint 参数兼容 | 官方 load_state_dict(strict=False) 可能容忍缺参数 | 不改变模型；缺少必需参数/多余 keys 时停止并记录，不用随机参数冒充完整模型 |
| 单样本 launcher、状态、输入 SHA、命令和 patch | 防止覆盖、保存可追溯证据、tmux 运行 | 不改 seed42、25+25 steps、CFG7、last20 feature 聚合 |
| HTTP Range 下载与官方 SHA256 验证 | SDK 长连接反复超时 | 只改变传输方式；所有权重逐字节校验 |
| 可选 PACT_JOINT_TRACE | 区分 raw head、坐标轴吸附与 Gaussian/Mesh bbox 后处理 | 只复制保存原函数输入输出；不改变预测、随机数或计算图 |
| GT-case 准备/启动/评估/汇总工具和监控 --gt-scene | 用已有标定数据做可追溯的轴 GT 对照 | GT Mesh 只投影输入 mask 并事后评估；GT 轴/类型不进入推理，未增加或改动物理 Loss |

独立环境 `/root/autodl-tmp/conda_envs/pact` 由 FreeArt3D 环境克隆，未修改源环境。官方导出的完整 YAML 包含许多训练/GUI 依赖，本轮先验证推理所用依赖；没有宣称完整 YAML 逐项一致。

PyTorch 2.4.0+cu121/CUDA12.1，与官方 YAML 的 2.4.0/CUDA11.8 存在偏差。pandas、transformers、huggingface-hub、safetensors、diffusers、Pillow、imageio、FFmpeg、OpenCV、SciPy、pymeshfix、trimesh、OpenEXR、timm 已按官方指定版本安装；编译依赖 FlashAttention 2.7.4.post1、spconv-cu120 2.3.6、nvdiffrast0.4.0、kaolin0.17.0 和 NumPy1.26.4 等暂用独立克隆中的版本，须通过实际 GPU 运行验证，具体版本见 evidence/environment-smoke.json。未使用的训练依赖不伪称已安装。

第一例是作者提供的 Dishwasher_001 RGB 与 EXR mask，仅为真实数据工具 smoke，不是 PARIS Storage45135 指标复现。21 个 orbit 预览帧只影响可视化；原版关节动画仍用默认60帧。原版生成、关节参数化、axis-origin bbox 后处理、运动动画及 raw mesh/GS 导出均保留。本轮不做 refinement，不增加物理 Loss，不调整预测值制造更好指标。

真实 GPU 首次运行已完成 25+25 采样、mesh/GS 导出和关节动画，但 exploded video 使用 imageio.imwrite 写 4D 视频触发 ValueError。修复为 imageio.mimsave(..., fps=video_fps)，仅修视频写出接口，不改帧、模型或预测。失败运行独立保留，不写成成功结果。


运动评估只读取输出，不回传梯度或改动预测。GS PLY 坐标撤销 save_ply 的默认固定旋转，以对齐 export 的 Mesh/JSON；没有GT/ICP拟合。21状态使用全部opacity≥0.05中心（base469723、door207728），最近中心核代理与真实mesh碰撞严格区分。分析使用原版 _local_transform。base28边界边，真实交叠体积/穿透深度均 unavailable。

Mesh预览使用原始完整三角面，没有补洞、平滑、修复。首版相机在运动终点裁切door，已保留在mesh_previews_camera_v1，用更远固定相机重新渲染到mesh_previews；只是观察视角调整，未重跑模型。官方模型源码中tracked __pycache__ 文件在本机导入后发生字节码变化，已仅恢复这些自动生成文件，不将其提交为源码修复。

已有数据的10489、31249、101808新增评估：oracle二维部件mask条件明确披露。主评估仅固定旋转，位置通过GT静态AABB尺度/中心归一化；单独静态几何ICP只作评估敏感性，不回写预测、不使用GT轴拟合。错误类型不计入条件轴均值，但保留全部类型失败、非零残留向量诊断和原始全向量汇总。原图、GT标注、部件排序、raw head、输出和工具SHA可追溯。划分含训练或增强家族重叠，未宣称严格未见泛化。详见GT_JOINT_EVALUATION.md。

用户随后明确授权探索碰撞引导轴优化：另建research/collision-axis-refinement-20261024分支，新增独立Gaussian核后处理目标及位置/方向/联合三项消融，未修改原版模型、生成Loss或原始预测。所有几何/外观、类型、range和rest-state冻结；GT不进入新优化/选择。6项均固定200步，保存完整Adam、SHA和日志。新增41状态留出源点GPU核评估与21状态VTK全三角面检查、GT同一对齐对照、共享相机视频与X-ray交线图。未用开放/自交网格计算实体穿透体积。原baseline和Part²GS checkpoint均不覆盖。独立环境不新增安装包；Open3D自交检查临时借用已有libEGL路径，不更改其他环境。详见COLLISION_REFINEMENT_DESIGN.md及COLLISION_REFINEMENT_RESULTS.md。
