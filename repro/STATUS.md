# PAct baseline 准备记录

用户于 2026-10-24 指示暂停 Part²GS，改用 PAct。Part²GS 已保存到 iteration 5020，GPU 已释放；不自动恢复。

官方源码已克隆，commit 与官方模型 revision 见 evidence/provenance.json。独立分支 repro/pact-baseline-20261024，禁止直接 push upstream。

先以官方预训练模型和一个官方示例完成推理/导出 smoke，不从头训练生成模型，不将官方示例冒充 PARIS Storage 45135。后续研究目标仍为几何、关节和全运动范围的碰撞诊断；输入格式、mask 来源、测试集划分和 GT 依赖需单独核查。

已发现入口 infer_imgs.py 忽略 --data_dir，硬编码 assets/real_world_examples；--config 当前没有载入。首次单样本测试前需要修复数据入口，并明确只修运行配置、不改生成或关节预测算法。

PAct 输入除 RGB 外还使用 2D part mask。当前官方示例含作者提供的 EXR mask，不能将其描述为完全无 mask 的自动单图流程。暂无本地实验结果。

环境计划使用独立 conda_envs/pact；现有 FreeArt3D 可克隆作为独立安装起点；原环境不修改。实际导入版本为 PyTorch 2.4.0+cu121 / CUDA 12.1，官方环境为 PyTorch 2.4.0 / CUDA 11.8，因此不能声称完全一致。逐包审计见 evidence/cloned-base-package-audit.json；需要通过真实 GPU 推理验证，尚未宣称兼容。
