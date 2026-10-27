"""Report only sealed measured results, retaining both positive and negative rows."""
import argparse
import json
import pathlib

import numpy as np


def fmt(x,digits=3):
    return 'N/A' if x is None else f'{x:.{digits}f}'


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=pathlib.Path(a.root);cfg=json.loads((root/'config.json').read_text());results=json.loads((root/'results.json').read_text());assert json.loads((root/'evaluation_status.json').read_text())['state']=='complete'
    lines=['# 冻结几何：直接关节优化与原文关节监督微调','',
           '本报告只包含本次真实完成的固定最终200步结果。所有局部部件形状、Gaussian 外观与尺寸、类型和层级冻结；没有 RGB 重建项、几何变形场或 GT 最优 checkpoint 选择。图像精度不作为主要指标。','',
           '两条路线需要分开解释。direct 为新增的逐物体参数后处理：A只保留原预测的二次先验，初始化处为驻点；B额外加入运动碰撞代理。head 为基于原文 masked articulation MSE 的冻结网络小规模微调：使用官方输出 MLP，A/B追加相同步数，B增加运动碰撞代理。它不是完整原版 flow-matching 续训，也不是“原版已有的逐场景细化”。','',
           f"head 训练对象：{', '.join(cfg['train_cases'])}；测试对象：{', '.join(cfg['test_cases'])}。训练对象 GT 用于监督与坐标归一化，测试 GT 只用于封存后的指标。局部划分对象不重叠，但未排除与预训练数据重叠。原始单视图输入使用 oracle 2D masks。",'',
           'head 缓存冻结投影后的四个单推理步特征（6/12/18/25），评估使用官方最后20步平均特征。没有伪称这些是随机 GT-noised latent 的完整训练分布。只训练官方 art_out_layers，输入投影、生成骨干和 decoder 不更新。masked MSE 额外限制到用户要求的连续关节通道，不修正错误类型。AdamW 为新建优化器，非作者 optimizer-state 恢复。','',
           'direct 开放轴方向、转动轴有效位置、q_start 和 q_end；方向限制5°、pivot限制初始资产对角线5%，端点相对初始值最多改变15°（转动）或0.1资产对角线（移动）。head 沿用官方表示的固定起点0，仅预测终点，没有新增起点输出头。','',
           '官方导出包含坐标轴吸附和 bbox pivot 投影。head 同时给出连续 pre-snap 诊断与官方后处理结果；碰撞 loss 作用在连续分支，没有使用 straight-through 梯度伪装成原版吸附可微。若连续分支改善但官方输出未改善，必须按此报告。','',
           '原始训练 encoder 用连续三个槽重复每个 range 标量，而官方 convert_data_range 使用交错槽平均，二者不完全互逆。本轮保留原版 encoder/decoder，不修复这一表示差异来改善指标。训练 GT 行程使用 b-a 的 signed span 并转为零起点 gauge；生成 rest-state 与 GT limit 零点未可靠对齐，绝对起止值 GT 误差 unavailable。预测起止值、实际行程变化全部保留。','',
           '位置误差为预测 pivot 到 GT 无限轴线的垂距，单位为 GT 静态部件对角线百分比；方向误差为 acos(abs(dot))。各方法复用同一新原版 baseline 的静态 AABB 相似变换，不对候选独立配准。移动关节的 pivot 不评分；错误类型不计有效轴误差。','',
           '## 轴和运动起止值', '',
           '| 路线 / 场景 / 输出 | GT部件 | 方向误差 ° | Pivot误差 %GT D | 起点 | 终点 | 行程/原始行程 |',
           '| --- | --- | --- | --- | --- | --- | --- |']
    for route,cases in results.items():
        for case,methods in cases.items():
            for method,result in methods.items():
                for row in result['joints']:
                    lines.append('| '+f"{route}/{case}/{method} | {row['GT_part']} | {fmt(row['axis_error_degrees'])} | {fmt(row['pivot_error_percent_GT_D'])} | {fmt(row['q_start'])} | {fmt(row['q_end'])} | {fmt(row['range_ratio_vs_original'])}"+' |')
    lines+=['','转动起止值单位为度，移动关节为导出坐标长度；方向符号相对原始轴统一。行程比小于1表示缩短，不能单凭自己的运动范围上碰撞降低就认定轴修正成功。','',
            '## 运动过程重叠', '',
            '| 路线 / 场景 / 输出 | 自身范围平均代理 | 共同原始范围平均代理 | 自身范围最坏代理 |',
            '| --- | --- | --- | --- |']
    for route,cases in results.items():
        for case,methods in cases.items():
            for method,result in methods.items():
                lines.append(f"| {route}/{case}/{method} | {fmt(result['own_range_motion']['mean_proxy'],6)} | {fmt(result['common_original_range_motion']['mean_proxy'],6)} | {fmt(result['own_range_motion']['worst_proxy'],6)} |")
    lines+=['','上述为全部单关节独立路径与同步路径的汇总。每条41状态，含训练21状态之间的插值状态；评价源中心与训练源中心不重叠。最近 Gaussian 中心距离和1-sigma间隙曲线见各 JSON。代理及其负间隙不是实体穿透深度。未证实原始 Mesh 为合法封闭实体，实体交叠体积/穿透深度均 unavailable，未写成0。','',
            '## A 与 B 的实测差异','',
            '| 路线 / 场景 / 分支 | 平均方向误差 A→B ° | 平均Pivot A→B %GT D | 共同范围代理相对变化 |',
            '| --- | --- | --- | --- |']
    averages=[]
    for route,cases in results.items():
        for case,methods in cases.items():
            branches=[('A','B','direct')] if route=='direct' else [('A_raw','B_raw','continuous'),('A_official','B_official','official')]
            for akey,bkey,branch in branches:
                A,B=methods[akey],methods[bkey]
                def avg(result,key):
                    xs=[r[key] for r in result['joints'] if r[key] is not None]
                    return float(np.mean(xs)) if xs else None
                aa,ba=avg(A,'axis_error_degrees'),avg(B,'axis_error_degrees');ap,bp=avg(A,'pivot_error_percent_GT_D'),avg(B,'pivot_error_percent_GT_D')
                original=A['common_original_range_motion']['mean_proxy'];rel=(B['common_original_range_motion']['mean_proxy']/original-1)*100 if original>1e-12 else None
                lines.append(f'| {route}/{case}/{branch} | {fmt(aa)} → {fmt(ba)} | {fmt(ap)} → {fmt(bp)} | {fmt(rel,2)}% |')
                averages.append({'route':route,'case':case,'branch':branch,'axis_A':aa,'axis_B':ba,'pivot_A':ap,'pivot_B':bp,'common_motion_proxy_relative_percent':rel})
    lines+=['','## A→B 的实际参数变化','',
            '| 路线 / 场景 / 分支 | GT部件 | 原点移动 %GT D | 轴实际旋转 ° | 起点变化 | 终点变化 |',
            '| --- | --- | --- | --- | --- | --- |']
    for route,cases in results.items():
        for case,methods in cases.items():
            baseline=json.loads((pathlib.Path(cfg['runs'][case])/'joint_evaluation/joint_errors.json').read_text());D=baseline['GT_static_diagonal']
            branches=[('A','B','direct')] if route=='direct' else [('A_raw','B_raw','continuous'),('A_official','B_official','official')]
            for akey,bkey,branch in branches:
                for x,y in zip(methods[akey]['joints'],methods[bkey]['joints']):
                    if x['q_start'] is None:continue
                    d0,d1=np.array(x['predicted_axis_direction_world']),np.array(y['predicted_axis_direction_world']);turn=float(np.degrees(np.arccos(np.clip(abs(d0@d1)/(np.linalg.norm(d0)*np.linalg.norm(d1)),0,1))))
                    displacement=100*np.linalg.norm(np.array(x['predicted_axis_origin_world'])-np.array(y['predicted_axis_origin_world']))/D if x['type']=='revolute' else None
                    lines.append(f"| {route}/{case}/{branch} | {x['GT_part']} | {fmt(displacement)} | {fmt(turn)} | {fmt(y['q_start']-x['q_start'])} | {fmt(y['q_end']-x['q_end'])} |")
    lines+=['','原点沿轴的移动不改变轴线；原点移动幅度不能单独当作轴位置的改善，需结合到GT轴的垂距与方向误差。','',
            '## 运行预算与耗时','',
            'direct每个对象各200步；head为四个训练对象共享200次更新，每次一个对象，不是每个对象200步。A/B预算一致，跨路线的监督和预算不同，不直接当作同一训练设置排名。','',
            '| 正式任务 | 步数 | 耗时 秒 | Torch峰值allocated MiB |',
            '| --- | --- | --- | --- |']
    for label in [*[f'direct_{c}_{arm}' for c in cfg['direct_cases'] for arm in ['A','B']],*[f'head_shared_{arm}' for arm in ['A','B']]]:
        status=json.loads((root/label/'status.json').read_text());assert status['state']=='complete'
        lines.append(f"| {label} | {status['step']} | {fmt(status['elapsed_seconds'],1)} | {fmt(status['peak_gpu_MiB'],1)} |")
    lines+=['','这是4个训练对象、2个测试对象、单seed、单碰撞权重的小规模实验，不构成泛化有效性的充分证据。几何冻结消除了本轮的形状变形自由度，但错误几何和错误类型仍会限制轴优化；不能将“存在可行碰撞下降”等同于找到GT机械轴。','',
            '## 核验与产物','',
            f'- 实验根目录：[{root.name}]({root})。',
            f'- 配置：[config.json]({root}/config.json)。',
            f'- 真数据梯度/FK/刚体/官方后处理核验：[gradient_and_rigidity_checks.json]({root}/gradient_and_rigidity_checks.json)。',
            f'- 零权重对照：[direct]({root}/direct_zero_control.json)、[head]({root}/head_zero_control.json)。',
            f'- 真实中断恢复：[resume_control.json]({root}/resume_control.json)。',
            '- 正式优化通过 tmux 顺序执行；每10步保存 optimizer、参数和历史，来源/配置 SHA 匹配后才能恢复。封存前重新核验全部冻结文件 SHA。各阶段日志、原版权重加载审计与所有失败尝试保留。']
    for route in results:
        for case in cfg['test_cases']:
            d=root/f'eval_{route}_{case}'
            lines+=['',f'### {route} / {case}','',f'![轴位置与方向]({d}/axes_vs_GT.png)','',f'![自身范围与共同范围碰撞曲线]({d}/motion_overlap.png)','',f'[固定部件刚体运动视频]({d}/rigid_motion.mp4)。']
    text='\n'.join(lines)+'\n';(root/'REPORT.md').write_text(text)
    repo=pathlib.Path(__file__).resolve().parents[2];(repo/'repro/FROZEN_JOINT_ROUTES_RESULTS.md').write_text(text)
    (root/'axis_summary.json').write_text(json.dumps(averages,indent=2,allow_nan=False))
    print(json.dumps(averages,indent=2),flush=True)


if __name__=='__main__':main()
