"""Measured-only report for axes-only optimization with exact frozen endpoints."""
import argparse
import json
import pathlib

import numpy as np

from axis_fixed_common import atomic_json


def fmt(x,n=3):return 'N/A' if x is None else f'{x:.{n}f}'
def average(result,key):
    xs=[r[key] for r in result['joints'] if r[key] is not None]
    return float(np.mean(xs)) if xs else None


def summaries(results):
    rows=[]
    for route,cases in results.items():
        for case,methods in cases.items():
            branches=[('A','B','direct')] if route=='direct' else [('A_raw','B_raw','continuous'),('A_official','B_official','official')]
            for A,B,branch in branches:
                a,b=methods[A],methods[B];metric='fixed_range_motion' if 'fixed_range_motion' in a else 'common_original_range_motion'
                base=a[metric]['mean_proxy'];rel=100*(b[metric]['mean_proxy']/base-1) if base>1e-12 else None
                rows.append({'route':route,'case':case,'branch':branch,'axis_A':average(a,'axis_error_degrees'),'axis_B':average(b,'axis_error_degrees'),
                    'pivot_A':average(a,'pivot_error_percent_GT_D'),'pivot_B':average(b,'pivot_error_percent_GT_D'),'motion_proxy_change_percent':rel,
                    'motion_proxy_A':a[metric]['mean_proxy'],'motion_proxy_B':b[metric]['mean_proxy']})
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=pathlib.Path(a.root);cfg=json.loads((root/'config.json').read_text());assert json.loads((root/'evaluation_status.json').read_text())['state']=='complete'
    data=json.loads((root/'results.json').read_text());parent=pathlib.Path(cfg['parent_root']);old=json.loads((parent/'results.json').read_text());summary=summaries(data);prior=summaries(old)
    lines=['# 只优化轴：冻结部件和原版起止数值','',
        '本轮直接检验：当部件几何与每个关节的原版预测起点、终点均固定时，加入运动碰撞Loss是否使轴位置和方向更接近GT。以下只包含完成的固定最终200步结果，原版和上一轮起止值可变结果保留。','',
        '两条路线各保留A不加/B加碰撞Loss。direct为逐物体轴后处理，A仅原预测二次先验（初始化为驻点）；B加同一定义的运动Gaussian代理。head为官方关节输出MLP的小规模GT轴监督微调，A/B只相差碰撞权重，不是完整flow-matching续训。','',
        f"沿用上一轮同一次原版生成的部件、缓存特征和原版关节初始化，不重新生成、也不从上轮训练后的A/B checkpoint继续。训练对象{', '.join(cfg['train_cases'])}，测试对象{', '.join(cfg['test_cases'])}。原始输入使用oracle 2D部件mask；训练GT仅监督轴，测试GT只用于封存后评价。未排除预训练集对象重叠，且不是盲测。",'',
        'direct仅开放方向切平面参数和转动轴有效pivot位置，沿用5°方向界限与0.05初始资产对角线pivot界限。range_delta无梯度且不在optimizer中。head仅使用方向与转动轴pivot输出，类型、语义、移动轴无意义的原点、所有range通道均回放该对象的原版预测。不能仅冻结输出层权重或detach范围：共享隐藏层更新仍可能改变数值，本轮在训练、推理和导出三处直接固定有效输出。','',
        '原版GT范围目标在打包时已清零且mask为0，optimizer不读取GT起止值作为目标。head仍沿用官方masked MSE表达式，但只在有效轴通道训练，缓存推理特征且仅训练输出MLP，须与完整原文训练区分。步数、种子、训练顺序、学习率、碰撞权重、点源采样和物理代理与上一轮一致。','',
        '起止数值始终使用原版JSON中的原始有符号值。轴方向仍可被优化甚至反向；数值范围冻结不等于物理运动方向一定不变。额外记录有向轴旋转和轴符号反向，不能让忽略正负的GT轴线角度掩盖运动方向反转。','',
        '## A/B汇总：GT轴误差与固定范围运动代理','',
        '位置误差为pivot到GT无限轴线的垂距，占GT静态部件对角线百分比；方向为无符号轴线角度（度）。两者越低越好。移动关节位置为N/A；错误类型保留并排除有效轴均值。所有候选使用同一原版baseline的静态AABB相似变换，无候选独立配准。','',
        '| 路线 / 场景 / 分支 | 方向误差 A→B ° | 位置误差 A→B %GT D | 固定范围代理变化 |',
        '| --- | --- | --- | --- |']
    for row in summary:lines.append(f"| {row['route']}/{row['case']}/{row['branch']} | {fmt(row['axis_A'])} → {fmt(row['axis_B'])} | {fmt(row['pivot_A'])} → {fmt(row['pivot_B'])} | {fmt(row['motion_proxy_change_percent'],2)}% |")
    lines+=['','## 逐轴误差和实际参数变化','',
        '| 路线 / 场景 / 分支 | GT部件 | 方向误差 A→B ° | 位置误差 A→B %GT D | 原点实际移动 %GT D | 轴实际旋转 ° | 有向旋转 ° |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for route,cases in data.items():
        for case,methods in cases.items():
            baseline=json.loads((pathlib.Path(cfg['runs'][case])/'joint_evaluation/joint_errors.json').read_text());D=baseline['GT_static_diagonal']
            branches=[('A','B','direct')] if route=='direct' else [('A_raw','B_raw','continuous'),('A_official','B_official','official')]
            for akey,bkey,branch in branches:
                for x,y in zip(methods[akey]['joints'],methods[bkey]['joints']):
                    if x['q_start'] is None:
                        lines.append(f"| {route}/{case}/{branch} | {x['GT_part']} | N/A: {x['type']} / GT {x['GT_type']} | N/A | N/A | N/A | N/A |")
                        continue
                    d0,d1=np.array(x['predicted_axis_direction_world']),np.array(y['predicted_axis_direction_world']);cos=np.clip(d0@d1/(np.linalg.norm(d0)*np.linalg.norm(d1)),-1,1);signed=float(np.degrees(np.arccos(cos)));turn=float(np.degrees(np.arccos(abs(cos))))
                    displacement=100*np.linalg.norm(np.array(x['predicted_axis_origin_world'])-np.array(y['predicted_axis_origin_world']))/D if x['type']=='revolute' else None
                    lines.append(f"| {route}/{case}/{branch} | {x['GT_part']} | {fmt(x['axis_error_degrees'])} → {fmt(y['axis_error_degrees'])} | {fmt(x['pivot_error_percent_GT_D'])} → {fmt(y['pivot_error_percent_GT_D'])} | {fmt(displacement)} | {fmt(turn)} | {fmt(signed)} |")
    lines+=['','原点沿轴平移不改变轴线；实际原点移动量不等于轴位置改善量，必须结合GT垂距和方向看。','',
        '## 固定起止值核验','',
        '| 场景 / 节点 | 原版固定起点 | 原版固定终点 | 单位 |',
        '| --- | --- | --- | --- |']
    for case in cfg['test_cases']:
        run=pathlib.Path(cfg['runs'][case]);obj=json.loads(pathlib.Path(json.loads((run/'status.json').read_text())['object']).read_text())
        for n in obj['diffuse_tree']:
            if n['joint']['type']=='fixed':continue
            unit='degrees' if n['joint']['type'] in ['revolute','continuous'] else 'export length units'
            lines.append(f"| {case}/{n['id']} | {n['joint']['range'][0]:.12g} | {n['joint']['range'][1]:.12g} | {unit} |")
    names=[*[f'direct_{c}_{arm}' for c in cfg['direct_cases'] for arm in ['A','B']],*[f'head_shared_{arm}' for arm in ['A','B']]]
    for name in names:
        history=json.loads((root/name/'history.json').read_text());assert len(history)==cfg['steps'] and all(h['max_endpoint_delta']==0 and h['range_grad_none'] for h in history)
    all_methods=[r for cases in data.values() for methods in cases.values() for r in methods.values()]
    assert all(r['max_numeric_endpoint_change']==0 for r in all_methods)
    lines+=['','六项正式训练的每步检查、所有预测/导出文件和全部评价的最大起止数值变化均为0。若有轴方向翻转，另列如下；它不表示range数值更新。','',
        '| 路线 / 场景 / 输出 | 相对原版方向反向的节点 |',
        '| --- | --- |']
    for route,cases in data.items():
        for case,methods in cases.items():
            for method,result in methods.items():
                nodes=[str(j['node_id']) for j in result['joints'] if j['axis_sign_reversed']]
                lines.append(f"| {route}/{case}/{method} | {', '.join(nodes) if nodes else 'none'} |")
    lines+=['','## 与上一轮起止可变实验对照','',
        '两轮使用同一批原版部件与特征，坐标/GT评价一致。新一轮移除运动区间自由度；head的范围监督也随之移除。下表是协议消融，不能将跨协议差异全归因于碰撞Loss。每轮内部的A/B才是加入该Loss的直接对照。','',
        '| 路线 / 场景 / 分支 | 上轮起止可变：方向 A→B ° | 本轮固定：方向 A→B ° | 上轮位置 A→B %GT D | 本轮位置 A→B %GT D |',
        '| --- | --- | --- | --- | --- |']
    indexed={(r['route'],r['case'],r['branch']):r for r in prior}
    for row in summary:
        prev=indexed[(row['route'],row['case'],row['branch'])]
        lines.append(f"| {row['route']}/{row['case']}/{row['branch']} | {fmt(prev['axis_A'])} → {fmt(prev['axis_B'])} | {fmt(row['axis_A'])} → {fmt(row['axis_B'])} | {fmt(prev['pivot_A'])} → {fmt(prev['pivot_B'])} | {fmt(row['pivot_A'])} → {fmt(row['pivot_B'])} |")
    lines+=['','## 读数限制与原版后处理','',
        '同时保留R0官方输出与R0_raw连续输出。官方导出仍采用原函数的坐标轴吸附及Gaussian动画→Mesh导出的bbox投影顺序；无straight-through伪梯度。连续分支和官方分支的改善可能不同，尤其0°可能来自硬吸附，不能自动称为方向学习更准。','',
        '所有运动路径固定为原版预测范围，每条训练21状态（独立关节及同步）、每步循环3状态；评价41状态和独立最多2048源Gaussian/部件，排除训练的512源中心，对全部保留目标做精确最近邻。不用GT关节角扩大或修正范围。8961原版范围很小，不能把t=1称为真实机械上的完全打开。','',
        '同一几何、轴、范围输入完全相同时，可复用上轮已封存的运动代理数组，仅以float32精确参数字节签名匹配，绝不按近似GT误差复用。新增候选均重新计算GT轴误差，复用来源见motion_cache_provenance.json。几何和范围均固定，因此本轮own/common两套数组应完全一致。','',
        'Gaussian软重叠、最近中心距离与1-sigma间隙是代理，不是实体交叠体积或穿透深度。未验证Mesh为合法封闭实体，实体指标为unavailable，未当作0。此处也没有以表面接触判定真实穿透。','',
        'direct受5°/0.05D范围限制；head只用四个训练对象、单seed、单权重，输入特征是冻结推理缓存，输出MLP微调不等价完整原版训练。原始几何与错误类型无法在本轮修正，不预设冻结后必然能找到GT轴。测试GT不用于选择checkpoint、seed或权重。','',
        '## 运行与核验','',
        'direct每场景每组200步；head每组共享200步，每步轮流一个训练对象（约50次/对象）。两条路线的监督数据不同，不能直接作同一训练协议排名。','',
        '| 正式任务 | 步数 | 耗时 秒 | Torch峰值allocated MiB |',
        '| --- | --- | --- | --- |']
    for name in names:
        status=json.loads((root/name/'status.json').read_text());assert status['state']=='complete';lines.append(f"| {name} | {status['step']} | {fmt(status['elapsed_seconds'],1)} | {fmt(status['peak_gpu_MiB'],1)} |")
    lines+=['',f'根目录：[{root.name}]({root})。', '',
        f'[范围/梯度/刚体核验]({root}/gradient_and_rigidity_checks.json)；[零权重direct]({root}/direct_zero_control.json)；[零权重head]({root}/head_zero_control.json)；[真实中断恢复]({root}/resume_control.json)。', '',
        '正式任务经tmux顺序执行，每10步保存参数、Adam、历史和状态；来源及配置SHA匹配后恢复。旧实验、旧源码和旧文件未覆盖。']
    for route in data:
        for case in cfg['test_cases']:
            folder=root/f'eval_{route}_{case}'
            lines+=['',f'### {route} / {case}','',f'![GT轴位置与方向]({folder}/axes_vs_GT.png)','',f'![固定范围运动曲线]({folder}/motion_overlap.png)','',f'[21状态固定部件运动视频]({folder}/rigid_motion.mp4)。']
    text='\n'.join(lines)+'\n';(root/'REPORT.md').write_text(text);repo=pathlib.Path(__file__).resolve().parents[2];(repo/'repro/AXIS_ONLY_FIXED_RANGES_RESULTS.md').write_text(text);atomic_json(root/'axis_summary.json',summary);print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
