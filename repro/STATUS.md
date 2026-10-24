# PAct baseline 准备记录

用户于 2026-10-24 指示暂停 Part²GS，改用 PAct。Part²GS 已保存到 iteration 5020，GPU 已释放；不自动恢复。

官方源码已克隆，commit 与官方模型 revision 见 evidence/provenance.json。独立分支 repro/pact-baseline-20261024，禁止直接 push upstream。

先以官方预训练模型和一个官方示例完成推理/导出 smoke，不从头训练生成模型，不将官方示例冒充 PARIS Storage 45135。后续研究目标仍为几何、关节和全运动范围的碰撞诊断；输入格式、mask 来源、测试集划分和 GT 依赖需单独核查。

已发现入口 infer_imgs.py 忽略 --data_dir，硬编码 assets/real_world_examples；--config 当前没有载入。首次单样本测试前需要修复数据入口，并明确只修运行配置、不改生成或关节预测算法。

PAct 输入除 RGB 外还使用 2D part mask。当前官方示例含作者提供的 EXR mask，不能将其描述为完全无 mask 的自动单图流程。暂无本地实验结果。

环境计划使用独立 conda_envs/pact；现有 FreeArt3D 可克隆作为独立安装起点；原环境不修改。实际导入版本为 PyTorch 2.4.0+cu121 / CUDA 12.1，官方环境为 PyTorch 2.4.0 / CUDA 11.8，因此不能声称完全一致。逐包审计见 evidence/cloned-base-package-audit.json；需要通过真实 GPU 推理验证，尚未宣称兼容。


2026-10-24 下载检查：独立 Conda 环境克隆已完成（offline 缺缓存失败后 online 成功），但 PAct 专用依赖尚未验证，不能据此声称可运行。原 SDK 下载长连接反复超时/EOF，约 30% 时改为 8 个 worker、4 MiB HTTP Range。已保留并复用约 702.5 MB 的 SDK 连续前缀，已完成的其他文件不重下。多块实际返回 206 且 Content-Range/字节数正确；全部下载后仍需逐文件官方 SHA-256 校验。没有开始 GPU 推理。

下载在 tmux 的 pact_model_download 中运行；无需持续调用 Codex。查看：

```bash
/root/autodl-tmp/conda_envs/pact/bin/python /root/autodl-tmp/PAct/repro/tools/model_status.py
```

断点信息保存在 repro/cache/model-range-progress.json；可用同一 fetch_model_ranges.py 继续，不用稀疏临时文件大小冒充完成进度。下载修复只涉及传输方式，不改变模型或数据。
