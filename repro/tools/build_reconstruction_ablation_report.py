"""Write measured paired comparisons; never infer success from collision alone."""
import argparse
import json
import pathlib
import subprocess

REPO=pathlib.Path(__file__).resolve().parents[2]


def mean_joint(result,key):
    values=[j[key] for j in result['joints'] if j[key] is not None];return sum(values)/len(values) if values else None


def summary(result):
    image=result['images']['heldout']['mean']
    return {'PSNR':image['PSNR'],'foreground_PSNR':image['foreground_PSNR'],'SSIM':image['SSIM'],'LPIPS_vgg':image['LPIPS_vgg'],
            **{f'{g}_CD_percent_GT_D':result['geometry']['groups'][g]['CD_percent_GT_static_D'] for g in ['fixed','moving','whole']},
            'mean_axis_degrees':mean_joint(result,'axis_degrees'),'mean_pivot_percent_GT_D':mean_joint(result,'pivot_percent_GT_static_D'),
            'Gaussian_motion_proxy':result['motion']['mean_across_paths'],'worst_Gaussian_motion_state':result['motion']['worst_state'],
            'mesh_surface_intersection_line_over_D':result['mesh_motion']['mean_line_over_fixed_D']}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=pathlib.Path(a.root).resolve();cfg=json.loads((root/'config.json').read_text());evidence={'root':str(root),'config':cfg,'experiments':{}}
    zero=json.loads((root/'zero_weight_control.json').read_text());replica=json.loads((root/'native_vs_sweep_A_control.json').read_text());assert zero['passed'] and replica['passed']
    lines=['# 运动碰撞 Loss 的重建 A/B 实测', '',
           '本轮完成三组成对对照、共六次200步优化：两组都进行相同的图像重建细化，只有运动碰撞 Loss 权重不同。R0保留共同图像注册后的原始几何和轴。它是多视角后处理消融，不是 PAct 原始单视图训练或整个网络微调。', '',
           '输入为每场景六个训练相机的 RGB 与 GT Mesh 投影的 oracle 2D 部件 mask，六个不同相机只用于最后评价。原始 PAct 推理仅首帧；细化使用额外视角，不能把额外观测带来的改进归因于碰撞 Loss。GT 三维几何、类型、轴和pivot不进入优化或checkpoint选择。GT已在此前基线诊断中被查看，因此不是盲测。', '',
           f'产物目录：[{root.name}]({root})。', '',
           '| 对照 | 留出 PSNR：A → B | Whole CD：A → B（% GT D） | 轴误差：A → B | Pivot：A → B（% GT D） | Gaussian 运动代理变化 | Mesh表面交线变化 |',
           '| --- | --- | --- | --- | --- | --- | --- |']
    details=[]
    def number(value):return 'unavailable' if value is None else f'{value:.4f}'
    def percent(value):return 'N/A (A=0)' if value is None else f'{value:+.2f}%'
    for experiment in cfg['experiments']:
        name=experiment['name'];results=json.loads((root/f'results_{name}.json').read_text());stats={m:summary(results[m]) for m in ['R0','A','B']};aa,bb=stats['A'],stats['B'];delta={k:None if aa[k] is None or bb[k] is None else bb[k]-aa[k] for k in aa};relative={k:None if aa[k] is None or bb[k] is None or aa[k]==0 else 100*(bb[k]/aa[k]-1) for k in aa}
        evidence['experiments'][name]={'summary':stats,'B_minus_A':delta,'B_relative_to_A_percent':relative,'individual_joints':{m:results[m]['joints'] for m in results}}
        lines.append(f"| {name} | {aa['PSNR']:.3f} → {bb['PSNR']:.3f} dB | {aa['whole_CD_percent_GT_D']:.3f} → {bb['whole_CD_percent_GT_D']:.3f} | {number(aa['mean_axis_degrees'])} → {number(bb['mean_axis_degrees'])}° | {number(aa['mean_pivot_percent_GT_D'])} → {number(bb['mean_pivot_percent_GT_D'])} | {percent(relative['Gaussian_motion_proxy'])} | {percent(relative['mesh_surface_intersection_line_over_D'])} |")
        details+=['',f'## {name}','','| 指标 | R0 | A：重建 | B：重建＋运动碰撞 | B−A |','| --- | --- | --- | --- | --- |']
        for key in aa:details.append(f"| {key} | {number(stats['R0'][key])} | {number(aa[key])} | {number(bb[key])} | {number(delta[key])} |")
        details+=['','| GT部件 | 类型正确 | Axis A → B（°） | Pivot A → B（% GT D） |','| --- | --- | --- | --- |']
        for j,k in zip(results['A']['joints'],results['B']['joints']):details.append(f"| {j['GT_part']} | {j['type_correct']} | {number(j['axis_degrees'])} → {number(k['axis_degrees'])} | {number(j['pivot_percent_GT_static_D'])} → {number(k['pivot_percent_GT_static_D'])} |")
        figures=root/f'figures_{name}';details+=['',f'[留出视角重建对比]({figures/"heldout_reconstruction.png"}) · [运动曲线]({figures/"motion_comparison.png"}) · [R0/A/B运动视频]({figures/"R0_A_B_motion.mp4"}) · [GT截面]({figures/"GT_sections.png"})。', '',
                  f"A/B 正式优化耗时：{json.loads((root/f'{name}_A_train/status.json').read_text())['elapsed_seconds']:.1f} / {json.loads((root/f'{name}_B_train/status.json').read_text())['elapsed_seconds']:.1f}秒；最终步均200。"]
    lines+=details+['','## 验证、解释与限制','',
                   f"两场景的真实数据梯度检查通过：图像项与运动项均对几何、方向和pivot有非零有限梯度；渲染投影误差<0.004像素，FK与已有实现误差<4e-8，共享形变场同时改变Mesh。30步 smoke 均完成。lambda=0两臂最大参数差异 {zero['max_absolute_parameter_difference']:.3g}；native/sweep两次A正式复跑最大差异 {replica['max_absolute_parameter_difference']:.3g}。", '',
                   '每个正式结果从相同共同初始化开始，200步固定最终checkpoint，无GT最佳步选择。两组同时开放每部件4x4x4控制点形变、方向和pivot；Gaussian opacity/尺度/颜色固定，位移约束、正则、学习率和样本路径相同。运动项使用每关节21点和同步路径的固定循环状态，每步3状态。所有保留Gaussian为最近邻目标，每部件1024源点；对双方几何可微。', '',
                   '独立运动评价：每关节与同步路径各41状态，源点最多4096/部件，排除训练源点；全密度目标。Mesh独立核查完整变形后的原始拓扑三角形，在同步路径21状态统计交线，采用相同初始资产对角线。尚未对全部多关节组合和每个独立路径计算Mesh交线；离散采样不能保证连续无碰撞。', '',
                   '90°实验是按原预测方向规定的假设范围，不是GT机械极限或原版预测完整运动；它不能证明真实机械全范围内安全。8961 native不足1°的限制仍在。45194原有类型错误保留，不能把类型错误部件的轴向量分数当有效关节精度。', '',
                   '图像指标来自留出相机；几何用候选自己的观察姿态FK，与GT对应部件在共同相机标定世界坐标直接比较，不重新对齐、不进行活动部件shape-only拟合。轴方向是忽略正负的夹角，pivot是预测原点到GT轴线的垂距，沿轴平移不作有效误差。这里使用图像注册对齐，与旧报告的GT静态AABB对齐不同，绝对数值不能跨协议直接比较；本轮A/B对齐完全相同。', '',
                   'Gaussian重叠是soft proxy；Mesh表面交线含合法接触，也受表面网格影响，都不等于实体穿透体积。Mesh实体合法性未认证，因此真实体积和穿透深度均 unavailable，不填0。', '',
                   '固定颜色/尺度和低维有界形变限制了可达到的重建质量。只有两个物体、一个种子与一个预声明权重，结果用于机制验证，不能据此宣称普适有效。碰撞下降而几何、外观或轴恶化属于权衡，不能单独称为重建改善。', '',
                   '监控：`/root/autodl-tmp/conda_envs/pact/bin/python repro/tools/monitor_reconstruction_ablation.py --root '+str(root)+' --watch`。原模型和源数据未覆盖；不推送官方上游，不提交大型产物。']
    content='\n'.join(lines)+'\n';(root/'REPORT.md').write_text(content);(REPO/'repro/RECONSTRUCTION_LOSS_ABLATION_RESULTS.md').write_text(content);(REPO/'repro/evidence/reconstruction-loss-ablation.json').write_text(json.dumps(evidence,indent=2,allow_nan=False));print(json.dumps({name:row['B_minus_A'] for name,row in evidence['experiments'].items()},indent=2))


if __name__=='__main__':main()
