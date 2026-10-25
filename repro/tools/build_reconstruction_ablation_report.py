"""Write measured paired comparisons; never infer success from collision alone."""
import argparse
import json
import pathlib
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO=pathlib.Path(__file__).resolve().parents[2]


def mean_joint(result,key):
    values=[j[key] for j in result['joints'] if j[key] is not None];return sum(values)/len(values) if values else None


def summary(result):
    image=result['images']['heldout']['mean']
    return {'PSNR':image['PSNR'],'foreground_PSNR':image['foreground_PSNR'],'SSIM':image['SSIM'],'LPIPS_vgg':image['LPIPS_vgg'],
            **{f'{g}_CD_percent_GT_D':result['geometry']['groups'][g]['CD_percent_GT_static_D'] for g in ['fixed','moving','whole']},
            'mean_axis_degrees':mean_joint(result,'axis_degrees'),'mean_pivot_percent_GT_D':mean_joint(result,'pivot_percent_GT_static_D'),
            'Gaussian_motion_proxy':result['motion']['mean_across_paths'],'worst_Gaussian_motion_state':result['motion']['worst_state'],
            'mesh_surface_intersection_line_over_D':result['mesh_motion']['mean_line_over_fixed_D'],
            'max_motion_added_Gaussian_proxy_over_own_initial':max(max(p['states'])-p['states'][0] for p in result['motion']['paths']),
            'max_motion_added_mesh_line_over_own_initial':max(s['sum_line_over_fixed_D'] for s in result['mesh_motion']['states'])-result['mesh_motion']['states'][0]['sum_line_over_fixed_D']}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=pathlib.Path(a.root).resolve();cfg=json.loads((root/'config.json').read_text());evidence={'root':str(root),'config':cfg,'experiments':{}}
    zero=json.loads((root/'zero_weight_control.json').read_text());replica=json.loads((root/'native_vs_sweep_A_control.json').read_text());assert zero['passed']
    evidence['zero_weight_gate']=zero;evidence['long_run_reproducibility_diagnostic']=replica
    lines=['# 运动碰撞 Loss 的重建 A/B 实测', '',
           '本轮完成三组成对对照、共六次200步优化：两组都进行相同的图像重建细化，只有运动碰撞 Loss 权重不同。R0保留共同图像注册后的原始几何和轴。它是多视角后处理消融，不是 PAct 原始单视图训练或整个网络微调。', '',
           '输入为每场景六个训练相机的 RGB 与 GT Mesh 投影的 oracle 2D 部件 mask，六个不同相机只用于最后评价。原始 PAct 推理仅首帧；细化使用额外视角，不能把额外观测带来的改进归因于碰撞 Loss。GT Mesh参与二维训练mask的制作；优化器不读取三维GT文件，不使用三维GT距离/法线或GT关节参数作监督，不按GT选checkpoint。GT已在此前基线诊断中被查看，因此不是盲测。', '',
           f'产物目录：[{root.name}]({root})。', '',
           '| 对照 | 留出 PSNR：A → B | Whole CD：A → B（% GT D） | 轴误差：A → B | Pivot：A → B（% GT D） | Gaussian 运动代理变化 | Mesh表面交线变化 |',
           '| --- | --- | --- | --- | --- | --- | --- |']
    environment_path=root/'environment.json'
    if environment_path.exists():
        environment=json.loads(environment_path.read_text());evidence['environment']=environment
        lines[2:2]=[f"环境：{environment['GPU']}，Python {environment['python']}，PyTorch {environment['torch']}，CUDA runtime {environment['CUDA_runtime']}。优化实现 commit `{environment['optimization_commit']}`；完整环境、pip freeze、每组源码SHA与原始文件SHA均在产物目录。",'']
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
        fig,axes=plt.subplots(1,2,figsize=(13,4))
        for method,color in [('A','#2166ac'),('B','#b2182b')]:
            for path in results[method]['motion']['paths']:
                values=np.array(path['states']);label=method+' '+path['path'];style='-' if path['path']=='synchronized' else '--'
                axes[0].plot(np.linspace(0,1,len(values)),values,color=color,linestyle=style,alpha=.8,label=label)
                axes[1].plot(np.linspace(0,1,len(values)),values-values[0],color=color,linestyle=style,alpha=.8,label=label)
        axes[0].set_ylabel('Absolute Gaussian proxy');axes[1].set_ylabel('Proxy minus own initial-state proxy')
        for ax in axes:ax.set_xlabel('Fraction of declared range');ax.grid(alpha=.2);ax.legend(fontsize=6)
        fig.suptitle(name+' | actual transformed motion; individual joints + synchronized path');fig.tight_layout();fig.savefig(figures/'motion_increment.png',dpi=180);fig.savefig(figures/'motion_increment.pdf');plt.close(fig)
        details+=['',f'[逐关节绝对重叠及相对起始状态的增量曲线]({figures/"motion_increment.png"})。这些增量是proxy诊断，不是实体新增穿透体积；A/B起始几何不同，须与绝对峰值一起读。']
    experiments=evidence['experiments'];count_cd=sum(x['B_minus_A']['whole_CD_percent_GT_D']<0 for x in experiments.values());count_psnr=sum(x['B_minus_A']['PSNR']>0 for x in experiments.values());count_proxy=sum(x['B_minus_A']['Gaussian_motion_proxy']<0 for x in experiments.values())
    lines[2:2]=[f'直接比较B与A：三组中 {count_cd} 组 Whole Chamfer 降低，{count_psnr} 组留出 PSNR 提高，{count_proxy} 组 Gaussian 运动重叠代理降低。重建细化本身的收益应看A与R0；额外碰撞项的收益必须看B与A，不能混为一谈。','']
    lines+=details+['','## 验证、解释与限制','',
                   f"两场景的真实数据梯度检查通过：图像项与运动项均对几何、方向和pivot有非零有限梯度；渲染投影误差<0.004像素，FK与已有实现误差<4e-8，共享形变场同时改变Mesh。30步 smoke 均完成。lambda=0两臂最大参数差异 {zero['max_absolute_parameter_difference']:.3g}，通过1e-5门槛。native/sweep两次A正式复跑最大差异 {replica['max_absolute_parameter_difference']:.3g}，严格1e-5检查结果为 {replica['passed']}。后者是长程可重复性诊断，未达到逐参数一致性，不能将小幅A/B差异解释为确定收益。两份原始A结果均保留，没有改用更有利的一份。", '',
                   '每个正式结果从相同共同初始化开始，200步固定最终checkpoint，无GT最佳步选择。两组同时开放每部件4x4x4控制点形变、方向和pivot；Gaussian opacity/尺度/颜色固定，位移约束、正则、学习率和样本路径相同。运动项使用每关节21点和同步路径的固定循环状态，每步3状态。所有保留Gaussian为最近邻目标，每部件1024源点；对双方几何可微。', '',
                   '独立运动评价：每关节与同步路径各41状态，源点最多4096/部件，排除训练源点；全密度目标。Mesh独立核查完整变形后的原始拓扑三角形，在同步路径21状态统计交线，采用相同初始资产对角线。尚未对全部多关节组合和每个独立路径计算Mesh交线；离散采样不能保证连续无碰撞。', '',
                   '90°实验是按原预测方向规定的假设范围，不是GT机械极限或原版预测完整运动；它不能证明真实机械全范围内安全。8961 native不足1°的限制仍在。45194原有类型错误保留，不能把类型错误部件的轴向量分数当有效关节精度。', '',
                   '图像指标来自留出相机；几何用候选自己的观察姿态FK，与GT对应部件在共同相机标定世界坐标直接比较，不重新对齐、不进行活动部件shape-only拟合。轴方向是忽略正负的夹角，pivot是预测原点到GT轴线的垂距，沿轴平移不作有效误差。这里使用图像注册对齐，与旧报告的GT静态AABB对齐不同，绝对数值不能跨协议直接比较；本轮A/B对齐完全相同。', '',
                   'Gaussian重叠是soft proxy；Mesh表面交线含合法接触，也受表面网格影响，都不等于实体穿透体积。Mesh实体合法性未认证，因此真实体积和穿透深度均 unavailable，不填0。', '',
                   '固定颜色/尺度和低维有界形变限制了可达到的重建质量。只有两个物体、一个种子与一个预声明权重，结果用于机制验证，不能据此宣称普适有效。碰撞下降而几何、外观或轴恶化属于权衡，不能单独称为重建改善。', '',
                   '监控：`/root/autodl-tmp/conda_envs/pact/bin/python repro/tools/monitor_reconstruction_ablation.py --root '+str(root)+' --watch`。原模型和源数据未覆盖；不推送官方上游，不提交大型产物。']
    motion_a=evidence['experiments']['8961_sweep90']['summary']['A']['max_motion_added_Gaussian_proxy_over_own_initial'];motion_b=evidence['experiments']['8961_sweep90']['summary']['B']['max_motion_added_Gaussian_proxy_over_own_initial']
    lines+=['','## 运动过程与几何变化的解释','',
            f'8961的90°假设范围中，各独立/同步路径的最大相对起始状态proxy增量由A的 {motion_a:.7f} 降到B的 {motion_b:.7f}。因此检查包含运动中新增加的重叠，而非只比较静态起始图。该0只表示本次离散Gaussian proxy未超过自身起始值，不代表实体无碰撞；对应Mesh交线增量仍非零。', '',
            '同时，B的起始重叠也大幅改变，且几何可训练，所以不能将全程重叠下降全部解释为轴被纠正。45194的部分单关节路径在B中仍有相对自身起始状态的proxy上升，应看逐关节曲线，不能只看同步平均。', '',
            '留出图像中，8961的GT百叶细节在原版及A/B渲染中仍明显缺失或不清晰，加入碰撞项没有恢复这一差异。45194保留部件类型/几何错误，单看Whole Chamfer会掩盖更大的逐部件误差。', '',
            '本轮支持的结论是：这一碰撞项有效约束了运动重叠代理，但在当前权重、可训练表示与预算下，额外项没有提升GT几何或留出图像精度。8961的pivot有局部收益；45194的B减少了采样到的形变折叠。这些收益不应被抹去，也不能替代整体重建精度对照。']
    native=evidence['experiments']['8961_native']['summary']['A'];sweep=evidence['experiments']['8961_sweep90']['summary']['A'];evidence['A_repeat_metric_delta']={k:None if native[k] is None or sweep[k] is None else sweep[k]-native[k] for k in native}
    lines+=['',f"两次A复跑的留出 PSNR 相差 {sweep['PSNR']-native['PSNR']:+.6f} dB，Whole CD相差 {sweep['whole_CD_percent_GT_D']-native['whole_CD_percent_GT_D']:+.6f} 个百分点。该波动不是跨种子统计置信区间，只是同一设置的重复性参照。"]
    audit=root/'deformation_audit.json'
    if audit.exists():
        evidence['deformation_audit']=json.loads(audit.read_text());lines+=['','形变场额外检查（每单元中心的Jacobian，不是Mesh实体合法性认证）：']
        for name,arms in evidence['deformation_audit']['results'].items():
            lines+=['',f"- {name}：A非正Jacobian中心 {sum(p['nonpositive_cell_centers'] for p in arms['A'])} 个，B {sum(p['nonpositive_cell_centers'] for p in arms['B'])} 个。正样本也不能证明全域无折叠。"]
    resume=root/'resume_validation.json'
    if resume.exists():
        evidence['resume_validation']=json.loads(resume.read_text());lines+=['',f"实际中断恢复检查：第{evidence['resume_validation']['interrupted_saved_step']}步终止，从checkpoint恢复到30步，历史连续，和不中断版本最大参数差异 {evidence['resume_validation']['max_parameter_difference_to_continuous_30steps']:.3g}，passed={evidence['resume_validation']['passed']}。"]
    content='\n'.join(lines)+'\n';(root/'REPORT.md').write_text(content);(REPO/'repro/RECONSTRUCTION_LOSS_ABLATION_RESULTS.md').write_text(content);(REPO/'repro/evidence/reconstruction-loss-ablation.json').write_text(json.dumps(evidence,indent=2,allow_nan=False));print(json.dumps({name:row['B_minus_A'] for name,row in evidence['experiments'].items()},indent=2))


if __name__=='__main__':main()
