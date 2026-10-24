# PAct 复现改动与环境偏差

原版 commit `a5ccec5056e7b27f39ddb5b83613c0efba9b2861`，tag `upstream-original-a5ccec5`。工作分支 `repro/pact-baseline-20261024`，不推送 upstream。

| 改动 | 原因 | 算法影响 |
|---|---|---|
| infer_imgs.py 使用 args.data_dir | 原版无视 CLI，始终读取全部 22 个示例 | 仅选择输入，不改变生成计算 |
| 可选 PACT_DINOV2_REPO 本地官方源码 | 固定官方 DINOv2 commit，复用已核验缓存，避免在线 main 漂移 | 同一官方模型与权重，严格加载通过；未设变量时保持原入口 |
| guarded_infer.py 检查 checkpoint 参数兼容 | 官方 load_state_dict(strict=False) 可能容忍缺参数 | 不改变模型；缺少必需参数/多余 keys 时停止并记录，不用随机参数冒充完整模型 |
| 单样本 launcher、状态、输入 SHA、命令和 patch | 防止覆盖、保存可追溯证据、tmux 运行 | 不改 seed42、25+25 steps、CFG7、last20 feature 聚合 |
| HTTP Range 下载与官方 SHA256 验证 | SDK 长连接反复超时 | 只改变传输方式；所有权重逐字节校验 |

独立环境 `/root/autodl-tmp/conda_envs/pact` 由 FreeArt3D 环境克隆，未修改源环境。官方导出的完整 YAML 包含许多训练/GUI 依赖，本轮先验证推理所用依赖；没有宣称完整 YAML 逐项一致。

PyTorch 2.4.0+cu121/CUDA12.1，与官方 YAML 的 2.4.0/CUDA11.8 存在偏差。pandas、transformers、huggingface-hub、safetensors、diffusers、Pillow、imageio、FFmpeg、OpenCV、SciPy、pymeshfix、trimesh、OpenEXR、timm 已按官方指定版本安装；编译依赖 FlashAttention 2.7.4.post1、spconv-cu120 2.3.6、nvdiffrast0.4.0、kaolin0.17.0 和 NumPy1.26.4 等暂用独立克隆中的版本，须通过实际 GPU 运行验证，具体版本见 evidence/environment-smoke.json。未使用的训练依赖不伪称已安装。

第一例是作者提供的 Dishwasher_001 RGB 与 EXR mask，仅为真实数据工具 smoke，不是 PARIS Storage45135 指标复现。21 个 orbit 预览帧只影响可视化；原版关节动画仍用默认60帧。原版生成、关节参数化、axis-origin bbox 后处理、运动动画及 raw mesh/GS 导出均保留。本轮不做 refinement，不增加物理 Loss，不调整预测值制造更好指标。

真实 GPU 首次运行已完成 25+25 采样、mesh/GS 导出和关节动画，但 exploded video 使用 imageio.imwrite 写 4D 视频触发 ValueError。修复为 imageio.mimsave(..., fps=video_fps)，仅修视频写出接口，不改帧、模型或预测。失败运行独立保留，不写成成功结果。
