# 45194：闭门静止图引导的部件形状优化与三路视频

已完成。在上轮左右门独立铰链、左右抽屉独立滑轨的版本上，根据一张闭门正面图优化部件形状，导出了优化后的原生 Mesh/Gaussian 部件资产和视频。原版及上一轮资产没有被覆盖。

## 视频

[三路并排视频：左原版 PAct、中运动先验修正、右闭门图形状优化](/root/autodl-tmp/PAct/repro/outputs/closed_image_fit_45194_20261028_v5/45194_three_versions_comparison.mp4)。上下两行是共同的两个观察视角。

三个独立视频：

- [原版 PAct](/root/autodl-tmp/PAct/repro/outputs/closed_image_fit_45194_20261028_v5/45194_original.mp4)
- [两门/抽屉独立运动先验版](/root/autodl-tmp/PAct/repro/outputs/closed_image_fit_45194_20261028_v5/45194_motion_prior.mp4)
- [闭门图片优化版](/root/autodl-tmp/PAct/repro/outputs/closed_image_fit_45194_20261028_v5/45194_image_optimized.mp4)

另附 [优化版从拟合闭门姿态开始开门，再返回闭门](/root/autodl-tmp/PAct/repro/outputs/closed_image_fit_45194_20261028_v5/45194_optimized_closed_to_open.mp4)。

主三路对比沿用原始生成参考姿态到原预测终点的区间，先验版与形状优化版的关节轴、父子关系、数值范围及运动进度完全相同。生成参考姿态本来已有一定开门角度，因此主对比的起点不能叫作完全闭门。单独的闭门开合视频从拟合图片所用的闭门姿态出发，终点仍为同一原预测终点，故其角度区间与主对比不同。

所有视频使用固定部件几何，随时间只应用刚体关节变换，没有逐帧形变或逐帧图像拟合。

## 用了哪张图

使用已有 45194 RGB 数据中的 [000250.png](/root/autodl-tmp/VideoArtGS/data/videoartgs/sapien/45194/images/000250.png)，两扇柜门关闭、视角接近正面。该图片和其存储的相机标定是唯一拟合观测。

在同一张图上手工标记四个可见前板矩形，作为门和抽屉的二维区域与边界约束。没有把 GT Mesh 投影成此次训练标签，也没有读取 GT 关节、深度或其他帧进行拟合。查找闭门图片时检查了 RGB 状态预览；这不等于多帧拟合。

![左至右：闭门目标、原版共同铰链、独立门先验、图像形状优化](/root/autodl-tmp/PAct/repro/outputs/closed_image_fit_45194_20261028_v5/closed_target_and_three_versions.png)

原版只有一个门关节，不能同时表示左右门相反方向的闭合角度；在该静态图对比中，原版共同铰链使用左门的闭合角，两个独立门版本则分别使用自身闭合角。

## 优化了什么

先使用 100 步图像配准拟合共同的整体尺度和两个图像平面平移量，然后冻结配准；接着固定执行 240 步部件优化，使用最后一步，而非挑选最优 checkpoint。

优化变量为门片沿面板宽度的平滑展开、部件平面内宽高，以及受限的竖直装配偏移。门的宽度展开固定在铰链处，左右成对前板共享参数；抽屉内部件跟随所属抽屉。门宽约扩大到原来的 2.36 倍，使它们在闭合时覆盖更多柜体开口。原始门片过窄的问题因此得到实际形状修正，而非只改变关节角度。

门片展开使用沿面板宽度的平滑映射，近端固定、远端展开；Gaussian 协方差通过映射 Jacobian 同步更新。原 Mesh 顶点使用相同映射，拓扑不变。优化后的形状烘焙到新的部件资产中。

保留独立门/抽屉运动组、父子关系、竖直铰链轴线、滑轨方向、主对比的数值行程、Gaussian 颜色和透明度、顶点/面/点数量。柜体和抽屉的深度、门片局部厚度方向维持原形状；没有尝试从正面单图推断背面结构。

闭门观察姿态由门片水平主方向估计：左门约 +44.48°，右门约 -45.41°，是在原生成参考姿态基础上的旋转。这些只用于对齐闭门图片，没有回写替代原运动区间。

目标包含 RGB 前景加权 L1、轮廓 L1、可见部件颜色标签误差、前板投影边界误差和小幅形状正则。没有重新训练 PAct 生成网络或更新其官方权重。

## 单张拟合图的实测指标

以下比较的是同一独立门先验版本在几何优化前后、同一已配准相机下的结果。

| 指标 | 几何优化前 | 几何优化后 |
| --- | --- | --- |
| 全图 PSNR | 15.93 dB | 18.63 dB |
| 前景 PSNR | 11.86 dB | 16.30 dB |
| 前景平均绝对误差 | 0.17088 | 0.09451 |
| 前景轮廓 IoU | 86.28% | 93.95% |

这是**拟合所用的一张图**上的结果，不是留出视角、GT 三维或运动精度评价。正面仍有上沿/门框缝隙、底部残留形状及纹理差异；背面和遮挡区没有独立证据。没有测量这次形状改变后的实体碰撞或机械行程精度，不能从正面像素改善推断这些性质也已改善。

## 核验和交付

新资产为 [optimized_asset/object.json](/root/autodl-tmp/PAct/repro/outputs/closed_image_fit_45194_20261028_v5/optimized_asset/object.json)，其同目录包含六个部件的 Mesh 与 Gaussian PLY。Gaussian 文件保留 PAct 原有序列化坐标约定；元数据与 Mesh 使用原规范坐标。

实际写出文件重新读取核验，确认：

- 源文件 SHA、上轮运动组元数据及其关节字段不变。
- 新 Mesh 面拓扑、点数与对应原分组一致，顶点和 Gaussian 中心与优化数组一致。
- 导出的 Gaussian 尺度/旋转重建协方差与优化结果一致，颜色、透明度和其他非几何字段逐值不变。
- 优化后各部件在 41 个动画状态中的刚体距离保持一致。
- 五个视频全部解码检查了首尾帧；人工检查三路终点及闭门拟合关键图。

完整结果：[拟合数据与参数](/root/autodl-tmp/PAct/repro/outputs/closed_image_fit_45194_20261028_v5/fit_results.json)、[导出与视频核验](/root/autodl-tmp/PAct/repro/outputs/closed_image_fit_45194_20261028_v5/verification.json)、[优化部件数组](/root/autodl-tmp/PAct/repro/outputs/closed_image_fit_45194_20261028_v5/optimized_parts.npz)。

这是一次探索性方法修正：各轮按可见拟合残差依次加入竖直偏移、成对参数共享和前板边界约束，所有目录保留。v4 因内存/JSON 标注键类型差异在首个形状步骤失败，修复后 v5 运行完成；展示的是最终方法的固定末步，未按 GT 或留出指标选择。v3 视频初次核验曾把 float32 源距离与 float64 变换距离使用过严阈值比较；最终核验统一了计算精度，所有实际导出文件另行检查。

```bash
cd /root/autodl-tmp/PAct
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 /root/autodl-tmp/conda_envs/pact/bin/python \
  repro/tools/fit_closed_image_45194.py \
  --prior repro/outputs/semantic_motion_prior_45194_20261028_v1 \
  --root /tmp/45194_closed_image_fit_new
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 /root/autodl-tmp/conda_envs/pact/bin/python \
  repro/tools/render_closed_image_45194.py --root /tmp/45194_closed_image_fit_new
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 /root/autodl-tmp/conda_envs/pact/bin/python \
  repro/tools/verify_closed_image_45194.py --root /tmp/45194_closed_image_fit_new
```
