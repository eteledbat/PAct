"""Measured direction/pivot ablation tables, actual parameter changes and plots."""
import argparse
import csv
import json
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from axis_mode_common import atomic_json,sha256
from build_axis_fixed_report import summaries,average,fmt,predicted_changes


def branches(route):
    return [('A','B','direct')] if route=='direct' else [('A_raw','B_raw','continuous'),('A_official','B_official','official + freeze')]


def parameter_changes(root,data,configs):
    rows=[]
    for mode,routes in data.items():
        cfg=configs[mode]
        for route,cases in routes.items():
            for case,methods in cases.items():
                gt=json.loads((pathlib.Path(cfg['runs'][case])/'joint_evaluation/joint_errors.json').read_text());R=np.array(gt['R_export_to_gtworld']);s=gt['scale'];D=gt['GT_static_diagonal']
                for ak,bk,branch in branches(route):
                    A=json.loads(pathlib.Path(methods[ak]['candidate']).read_text());B=json.loads(pathlib.Path(methods[bk]['candidate']).read_text());ad={n['id']:n for n in A['diffuse_tree']};matched={j['node_id']:j for j in methods[bk]['joints']}
                    for n in B['diffuse_tree']:
                        if n['joint']['type']=='fixed':continue
                        a=ad[n['id']]['joint'];b=n['joint'];da,db=np.array(a['axis']['direction']),np.array(b['axis']['direction']);cos=np.clip(da@db/np.linalg.norm(da)/np.linalg.norm(db),-1,1);delta=100*s*R@(np.array(b['axis']['origin'])-np.array(a['axis']['origin']))/D
                        row={'mode':mode,'route':route,'case':case,'branch':branch,'node_id':n['id'],'predicted_type':b['type'],'GT_match':n['id'] in matched,
                            'oriented_axis_rotation_A_to_B_deg':float(np.degrees(np.arccos(cos))),
                            'unsigned_axis_rotation_A_to_B_deg':float(np.degrees(np.arccos(abs(cos)))),
                            'origin_displacement_A_to_B_percent_GT_D':float(np.linalg.norm(delta)) if b['type'] in ['revolute','continuous'] else None,
                            'GT_axis_error_B_deg':matched[n['id']]['axis_error_degrees'] if n['id'] in matched else None,
                            'GT_pivot_error_B_percent_D':matched[n['id']]['pivot_error_percent_GT_D'] if n['id'] in matched else None}
                        for prefix,value in [('direction_A_export',da),('direction_B_export',db),('origin_A_export',a['axis']['origin']),('origin_B_export',b['axis']['origin']),('origin_delta_GT_percent_D',delta)]:
                            for k,v in zip('xyz',value):row[prefix+'_'+k]=float(v)
                        rows.append(row)
    atomic_json(root/'axis_parameter_changes.json',rows)
    with (root/'axis_parameter_changes.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return rows


def selected_plot(root,data):
    fig,axes=plt.subplots(3,2,figsize=(12,10),squeeze=False)
    for row,(route,branch,ak,bk) in enumerate([('direct','direct','A','B'),('head','continuous','A_raw','B_raw'),('head','official + freeze','A_official','B_official')]):
        for col,mode in enumerate(['direction','position']):
            metric='axis_error_degrees' if mode=='direction' else 'pivot_error_percent_GT_D';ax=axes[row,col];cases=list(data[mode][route]);x=np.arange(len(cases));a=[average(data[mode][route][c][ak],metric) for c in cases];b=[average(data[mode][route][c][bk],metric) for c in cases]
            init='H0' if branch=='continuous' else 'R0';initial=[average(data[mode][route][c][init],metric) for c in cases]
            ax.bar(x-.18,a,.34,label='A: no motion proxy',color='#2166ac');ax.bar(x+.18,b,.34,label='B: with motion proxy',color='#b2182b');ax.scatter(x,initial,c='black',marker='D',s=24,label='Original initialization',zorder=4)
            for xx,yy in zip(x-.18,a):ax.annotate(f'{yy:.2f}',(xx,yy),xytext=(0,3),textcoords='offset points',ha='center',fontsize=8)
            for xx,yy in zip(x+.18,b):ax.annotate(f'{yy:.2f}',(xx,yy),xytext=(0,3),textcoords='offset points',ha='center',fontsize=8)
            ax.set_xticks(x,cases);ax.set(title=f'{mode}-only / {route} / {branch}',ylabel='GT direction error (degrees)' if mode=='direction' else 'GT pivot error (% static D)');ax.grid(axis='y',alpha=.2);ax.margins(y=.2)
            if row==0 and col==0:ax.legend(fontsize=8)
    fig.suptitle('Selected axis variable only; original geometry, counterpart and endpoints fixed');fig.tight_layout();fig.savefig(root/'selected_axis_errors.png',dpi=180);fig.savefig(root/'selected_axis_errors.pdf');plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=pathlib.Path(a.root);top=json.loads((root/'config.json').read_text());assert json.loads((root/'evaluation_status.json').read_text())['state']=='complete';data=json.loads((root/'results.json').read_text());configs={m:json.loads((root/m/'config.json').read_text()) for m in top['modes']};parent=pathlib.Path(top['parent_root']);old=json.loads((parent/'results.json').read_text());summary=[];changes={}
    for mode,routes in data.items():
        for row in summaries(routes):summary.append({'mode':mode,**row})
        changes[mode]=predicted_changes(configs[mode],routes)
    atomic_json(root/'axis_summary.json',summary);atomic_json(root/'all_predicted_axis_changes.json',changes);params=parameter_changes(root,data,configs);selected_plot(root,data)
    lines=['# 8961 / 45194：只改方向与只改轴位置的A/B实测','',
        '本轮仅评价原来的8961、45194，分别做direction-only和position-only，两条路线均保留A不加/B加入相同运动Gaussian代理。12项正式任务全部固定最后200步，原始部件与起止值不变；没有重新生成、没有从此前训练后的A/B继续。','',
        'direct为逐物体参数后处理，A仅原预测二次先验、初始为驻点。head每模式从官方原始输出MLP重新开始，用原来四个对象的GT所选轴分量监督，共享200步，测试只看8961/45194。head使用冻结推理缓存，仅更新输出MLP，不能称为完整PAct/flow-matching续训。','',
        'direction-only把原版导出原点固定，在FK、连续导出和后处理后均只放开方向。position-only把原版导出方向固定，仅放开转动轴有效位置，平移origin保持固定。共享MLP与官方bbox投影均可能带动未选字段，本轮在实际前向及后处理之后明确回放，不能只冻结末层权重或detach。','',
        '## 所选参数的GT精度','',
        '方向为无符号轴线误差（°），位置为预测pivot到GT无限轴的垂距 / GT静态部件对角线D（%）。只对匹配且类型正确的关节取均值，错误类型与无GT匹配的活动部件保留，不重排或强制修正。','',
        '| 路线 / 场景 / 输出 | 只改方向：方向 A→B ° | 只改位置：位置 A→B %GT D |',
        '| --- | --- | --- |']
    indexed={(r['mode'],r['route'],r['case'],r['branch']):r for r in summary}
    for route in ['direct','head']:
        for case in top['test_cases']:
            for branch in (['direct'] if route=='direct' else ['continuous','official']):
                d=indexed[('direction',route,case,branch)];o=indexed[('position',route,case,branch)];lines.append(f"| {route}/{case}/{branch} | {fmt(d['axis_A'])} → {fmt(d['axis_B'])} | {fmt(o['pivot_A'])} → {fmt(o['pivot_B'])} |")
    lines+=['',f'![所选轴分量误差，黑点为各自初始参照]({root}/selected_axis_errors.png)']
    if (root/'interpretation.json').exists():
        notes=json.loads((root/'interpretation.json').read_text());assert notes['results_sha256']==sha256(root/'results.json');lines+=['','## 实测解释','',notes['text']]
    lines+=['','## 完整A/B与运动代理','',
        '| 模式 / 路线 / 场景 / 输出 | 方向误差 A→B ° | 位置误差 A→B %GT D | 运动代理变化 |',
        '| --- | --- | --- | --- |']
    for r in summary:lines.append(f"| {r['mode']}/{r['route']}/{r['case']}/{r['branch']} | {fmt(r['axis_A'])} → {fmt(r['axis_B'])} | {fmt(r['pivot_A'])} → {fmt(r['pivot_B'])} | {fmt(r['motion_proxy_change_percent'],2)}% |")
    lines+=['','## 原版与本轮连续初始参照','',
        'head的H0把所选原始连续输出与原版导出的冻结分量组合，因而不同于此前全连续R0_raw；不能把H0与原版官方R0混用。R0保持原版完整导出，head的官方初始结果经冻结回放复现R0。原版固定坐标/静态AABB对齐在每个候选间复用，不重新拟合。','',
        '| 模式 / 场景 / 初始参照 | 方向误差 ° | 位置误差 %GT D |',
        '| --- | --- | --- |']
    for mode,routes in data.items():
        for case,methods in routes['head'].items():
            for method in ['R0','H0']:lines.append(f"| {mode}/{case}/{method} | {fmt(average(methods[method],'axis_error_degrees'))} | {fmt(average(methods[method],'pivot_error_percent_GT_D'))} |")
    lines+=['','## 逐个GT匹配关节','',
        '| 模式 / 路线 / 场景 / 输出 / 节点 | 预测类型 / GT类型 | 方向 A→B ° | 位置 A→B %GT D |',
        '| --- | --- | --- | --- |']
    for mode,routes in data.items():
        for route,cases in routes.items():
            for case,methods in cases.items():
                for ak,bk,branch in branches(route):
                    for x,y in zip(methods[ak]['joints'],methods[bk]['joints']):lines.append(f"| {mode}/{route}/{case}/{branch}/{x['node_id']} | {x['type']} / {x['GT_type']} | {fmt(x['axis_error_degrees'])} → {fmt(y['axis_error_degrees'])} | {fmt(x['pivot_error_percent_GT_D'])} → {fmt(y['pivot_error_percent_GT_D'])} |")
    lines+=['','## 实际参数变化','',
        f"[全部活动预测节点的A/B方向与原点坐标CSV]({root}/axis_parameter_changes.csv) · [完整JSON]({root}/axis_parameter_changes.json)。坐标列明确区分export坐标与GT世界坐标下的位移百分比。包含未匹配活动部件，不将其GT误差伪造为0。",'',
        '| 模式 / 路线 / 场景 / 输出 / 节点 | A→B有向旋转 ° | A→B原点位移 %GT D |',
        '| --- | --- | --- |']
    for r in params:lines.append(f"| {r['mode']}/{r['route']}/{r['case']}/{r['branch']}/{r['node_id']} | {fmt(r['oriented_axis_rotation_A_to_B_deg'])} | {fmt(r['origin_displacement_A_to_B_percent_GT_D'])} |")
    lines+=['','direct的pivot仅开放垂轴的两个有效自由度，head仍沿用原版三维origin输出与所选GT MSE；沿轴平移不改变轴线，位移大小不等于精度改善。只改方向时固定的是原点坐标，整条轴线会随方向变化。所有相对原版的轴反向另存all_predicted_axis_changes.json。','',
        '## 冻结检查','',
        '所有正式步记录起止值与未选物理分量的变化，必须均为0；head每次optimizer更新后重新前向断言，所有连续/后处理导出逐节点校验，仅允许指定轴字段改变。方向模式GT pivot点垂距应不变，位置模式GT方向误差应不变。','',
        '| 模式 | 正式任务数 | 最大起止变化 | 最大未选字段变化 |',
        '| --- | --- | --- | --- |']
    for mode in top['modes']:
        sub=root/mode;cfg=configs[mode];names=[*[f'direct_{c}_{arm}' for c in cfg['test_cases'] for arm in ['A','B']],*[f'head_shared_{arm}' for arm in ['A','B']]]
        for name in names:
            h=json.loads((sub/name/'history.json').read_text());assert len(h)==200 and all(r['max_endpoint_delta']==0 and r['max_frozen_counterpart_delta']==0 and r['range_grad_none'] and r['frozen_counterpart_grad_none'] for r in h)
        for cases in data[mode].values():
            for methods in cases.values():assert all(v['max_numeric_endpoint_change']==0 and v['max_frozen_counterpart_change']==0 for v in methods.values())
        lines.append(f'| {mode} | 6 | 0 | 0 |')
    lines+=['','## 与此前方向+位置同时优化参照','',
        '以下是同一原版资产和GT对齐下的既有both实验，未重训或重选结果。单变量head同时改变所选监督分量与冻结的物理参照，连续初态也随H0定义变化，跨设置差异不能全部归因于碰撞Loss；每行内部A/B才是直接对照。','',
        '| 路线 / 场景 / 输出 | 此前both方向 A→B ° | 此前both位置 A→B %GT D | 此前both运动代理变化 |',
        '| --- | --- | --- | --- |']
    for r in summaries(old):lines.append(f"| {r['route']}/{r['case']}/{r['branch']} | {fmt(r['axis_A'])} → {fmt(r['axis_B'])} | {fmt(r['pivot_A'])} → {fmt(r['pivot_B'])} | {fmt(r['motion_proxy_change_percent'],2)}% |")
    lines+=['','## 已有预检中的梯度尺度','',
        '以下来自正式运行前已有预检：平均推理特征、同步路径固定三个状态[2,10,20]，输出MLP参数的梯度范数。它不是实际每步使用的单时刻特征/循环三状态梯度；未测梯度夹角，不能仅凭范数称为方向冲突。保留相同权重也不意味着方向与位置模式的物理/监督梯度比例相同，本轮没有用这些数值调参。','',
        '| 模式 / 训练对象 | 0.02×物理梯度范数 / GT MSE梯度范数 |',
        '| --- | --- |']
    for mode in top['modes']:
        preflight=json.loads((root/mode/'gradient_and_freeze_checks.json').read_text())
        for case,row in preflight['head'].items():
            ratio=.02*row['selected_collision_MLP_gradient_norm']/row['selected_MSE_MLP_gradient_norm'];lines.append(f'| {mode}/{case} | {ratio:.2f} |')
    lines+=['','## 类型、数据与读数限制','',
        '仍使用同一原版RGB及oracle二维部件mask；GT几何经mask进入输入，不能称RGB-only。GT只监督四个原训练对象的所选轴分量，测试GT不进入更新或checkpoint选择。测试对象此前已被检查，不是盲测，未排除PAct预训练重叠。','',
        '8961原版范围约2.60°/1.51°，运动信号小。45194仍有GT门被预测fixed、GT静态部件被预测为活动的角色错误；本轮固定类型/结构不修复，条件轴均值不能代表全机构正确。单seed、单权重和两个对象不支持稳定泛化结论。','',
        '沿用direct先验0.05、碰撞权重1，head碰撞权重0.02、lr1e-4、原24通道masked MSE分母；每次head共享200步、轮流4对象，约50次/对象。12项任务不是12个独立对象。训练21状态/路径、每步3状态、512源/部件；评价41状态、最多2048不同源/部件并排除训练源，targets全部保留。','',
        '所有运动沿原版有符号数值范围，不用GT扩大行程。软重叠、最近采样中心距离和1-sigma间隙只是Gaussian代理。未证明Mesh合法封闭，真实交叠体积和穿透深度保持unavailable，未填0。','',
        '连续输出与官方后处理+冻结回放均保留。后处理仍可能吸附可变方向或投影可变pivot，不能把吸附后的0°归为学习改善。冻结字段在后处理之后恢复，这是本轮按要求限定自由度的协议，不冒充不加限制的原版导出。','',
        '完全相同的Gaussian几何、采样、轴和范围才可按float32实际FK输入字节签名复用已封存运动数组；不按近似轴或GT误差复用，来源见motion_cache_provenance.json，GT轴误差重新评分。','',
        '## 实际运行','',
        '| 模式 / 任务 | 步数 | 耗时 秒 | Torch峰值allocated MiB |',
        '| --- | --- | --- | --- |']
    for mode in top['modes']:
        sub=root/mode;names=[*[f'direct_{c}_{arm}' for c in top['test_cases'] for arm in ['A','B']],*[f'head_shared_{arm}' for arm in ['A','B']]]
        for name in names:
            d=json.loads((sub/name/'status.json').read_text());assert d['state']=='complete';lines.append(f"| {mode}/{name} | {d['step']} | {fmt(d['elapsed_seconds'],1)} | {fmt(d['peak_gpu_MiB'],1)} |")
    lines+=['',f'根目录：[{root.name}]({root})。预检、smoke、零权重控制、真实10→30步恢复、配置、源码SHA、原始checkpoint和Adam均保存；tmux顺序执行，独立监控脚本不依赖持续调用Codex。旧实验与部件不覆盖。']
    for mode in top['modes']:
        for route in ['direct','head']:
            for case in top['test_cases']:
                folder=root/mode/f'eval_{route}_{case}';lines+=['',f'### {mode} / {route} / {case}','',f'![GT轴位置与方向]({folder}/axes_vs_GT.png)','',f'![41状态运动重叠曲线]({folder}/motion_overlap.png)','',f'[21帧刚体运动视频]({folder}/rigid_motion.mp4)。']
    text='\n'.join(lines)+'\n';(root/'REPORT.md').write_text(text);repo=pathlib.Path(__file__).resolve().parents[2];(repo/'repro/AXIS_DIRECTION_POSITION_RESULTS.md').write_text(text);print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
