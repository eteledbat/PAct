"""Measured-only report for new test cases and the all-fixed failure."""
import argparse
import json
import pathlib

import numpy as np

from axis_fixed_common import atomic_json
from build_axis_fixed_report import summaries,average,fmt,predicted_changes


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=pathlib.Path(a.root);cfg=json.loads((root/'config.json').read_text())
    assert json.loads((root/'evaluation_status.json').read_text())['state']=='complete'
    data=json.loads((root/'results.json').read_text());rows=summaries(data);changes=predicted_changes(cfg,data)
    atomic_json(root/'axis_summary.json',rows);atomic_json(root/'all_predicted_axis_changes.json',changes)
    lines=['# 追加场景：冻结部件和数值起止值，只优化轴','',
        f"正式测试对象为{', '.join(cfg['test_cases'])}。direct各A/B固定200步；head直接复用上一轮封存的A/B输出MLP checkpoint，在新对象上测试，没有新增训练步或调参。A不加、B加运动Gaussian代理。全部局部部件、类型、层级及原版native起止值冻结。",'',
        '20411为用户指定的平移关节案例，origin不影响运动，因此轴位置/pivot精度为N/A，只评价方向。10655含转动关节，可评价轴位置；原版已有部件/类型失败，不能称为纯净的pivot缺陷样本。','',
        '第二个对象最初固定为101284（frame0、seed42），原版推理成功生成三个部件，但全部为fixed。在冻结类型的协议下没有可优化轴，失败原样保留，未改成GT类型。随后按已有原版输出存在可评估活动关节的条件选10655，没有按优化效果挑样本。该选择及历史GT访问意味着不是盲测。','',
        '## A/B轴精度与运动重叠','',
        '方向为无符号轴线角度，位置为预测pivot到GT无限轴线的垂距 / GT静态部件对角线D。均值仅包含匹配且类型正确的关节；类型错判不能通过忽略该关节美化为全模型准确率。Gaussian代理不是实体穿透体积。','',
        '| 路线 / 场景 / 输出 | 方向 A→B ° | 位置 A→B %GT D | 运动代理变化 |',
        '| --- | --- | --- | --- |']
    for r in rows:lines.append(f"| {r['route']}/{r['case']}/{r['branch']} | {fmt(r['axis_A'])} → {fmt(r['axis_B'])} | {fmt(r['pivot_A'])} → {fmt(r['pivot_B'])} | {fmt(r['motion_proxy_change_percent'],2)}% |")
    notes=root/'interpretation.json'
    if notes.exists():
        from axis_fixed_common import sha256
        content=json.loads(notes.read_text());assert content['results_sha256']==sha256(root/'results.json');lines+=['','## 实测解释','',content['text']]
    lines+=['','## 原版与类型覆盖','',
        '| 场景 / 原版输出 | 方向误差 ° | 位置误差 %GT D | GT匹配关节类型正确数 |',
        '| --- | --- | --- | --- |']
    for case,methods in data['head'].items():
        correct=sum(j['type_correct'] for j in methods['R0']['joints']);total=len(methods['R0']['joints'])
        for method in ['R0','R0_raw']:lines.append(f"| {case}/{method} | {fmt(average(methods[method],'axis_error_degrees'))} | {fmt(average(methods[method],'pivot_error_percent_GT_D'))} | {correct}/{total} |")
    lines+=['','## 每个GT匹配关节的轴变化','',
        '| 路线 / 场景 / 输出 / 预测节点 | GT类型 / 预测类型 | 方向误差 A→B ° | 位置误差 A→B %GT D | 轴有向旋转 A→B ° | 原点位移 A→B %GT D |',
        '| --- | --- | --- | --- | --- | --- |']
    for route,cases in data.items():
        for case,methods in cases.items():
            gt=json.loads((pathlib.Path(cfg['runs'][case])/'joint_evaluation/joint_errors.json').read_text());D=gt['GT_static_diagonal']
            branches=[('A','B','direct')] if route=='direct' else [('A_raw','B_raw','continuous'),('A_official','B_official','official')]
            for ak,bk,branch in branches:
                for x,y in zip(methods[ak]['joints'],methods[bk]['joints']):
                    turn=None;move=None
                    if x['q_start'] is not None:
                        d0,d1=np.array(x['predicted_axis_direction_world']),np.array(y['predicted_axis_direction_world']);turn=float(np.degrees(np.arccos(np.clip(d0@d1/np.linalg.norm(d0)/np.linalg.norm(d1),-1,1))))
                        if x['type'] in ['revolute','continuous']:move=100*np.linalg.norm(np.array(x['predicted_axis_origin_world'])-np.array(y['predicted_axis_origin_world']))/D
                    lines.append(f"| {route}/{case}/{branch}/node{x['node_id']} | {x['GT_type']} / {x['type']} | {fmt(x['axis_error_degrees'])} → {fmt(y['axis_error_degrees'])} | {fmt(x['pivot_error_percent_GT_D'])} → {fmt(y['pivot_error_percent_GT_D'])} | {fmt(turn)} | {fmt(move)} |")
    lines+=['','原点沿轴移动不改变轴线，位移不等于精度改善。平移关节位置不可识别，不把N/A当0。','',
        '## 固定起止值与方向符号','',
        '| 场景 / 节点 | 预测类型 | 固定起点 | 固定终点 |',
        '| --- | --- | --- | --- |']
    for case in cfg['test_cases']:
        run=pathlib.Path(cfg['runs'][case]);status=json.loads((run/'status.json').read_text());obj=json.loads(pathlib.Path(status['object']).read_text())
        for n in obj['diffuse_tree']:lines.append(f"| {case}/{n['id']} | {n['joint']['type']} | {n['joint']['range'][0]:.12g} | {n['joint']['range'][1]:.12g} |")
    lines+=['','转动范围为degrees，平移范围为导出模型长度单位。所有正式direct步、迁移head的实际FK输出及导出均断言数值完全不变。轴方向反向仍可能改变物理运动，因此另查所有活动预测节点（包含无GT匹配者）：','',
        '| 路线 / 场景 / 输出 | 相对原版方向反向的节点 |',
        '| --- | --- |']
    for route,cases in changes.items():
        for case,methods in cases.items():
            for method,joints in methods.items():lines.append(f"| {route}/{case}/{method} | {', '.join(str(j['node_id']) for j in joints if j['axis_sign_reversed']) or 'none'} |")
    lines+=['','## 每条运动路径的留出代理','',
        '路径按原版活动节点顺序；单关节路径仍统计全物体部件对，其余关节保持原版起点。','',
        '| 路线 / 场景 / 输出 | 路径（预测节点） | A→B均值变化 |',
        '| --- | --- | --- |']
    for route,cases in data.items():
        for case,methods in cases.items():
            run=pathlib.Path(cfg['runs'][case]);obj=json.loads(pathlib.Path(json.loads((run/'status.json').read_text())['object']).read_text());active=[n['id'] for n in obj['diffuse_tree'] if n['joint']['type']!='fixed']
            branches=[('A','B','direct')] if route=='direct' else [('A_raw','B_raw','continuous'),('A_official','B_official','official')]
            for ak,bk,branch in branches:
                for x,y in zip(methods[ak]['fixed_range_motion']['paths'],methods[bk]['fixed_range_motion']['paths']):
                    name=x['path'];nid=active[int(name.rsplit('_',1)[-1])] if name.startswith('only_index_') else 'all';base=x['mean_proxy'];change=100*(y['mean_proxy']/base-1) if base>1e-12 else None
                    lines.append(f"| {route}/{case}/{branch} | {name} (node {nid}) | {fmt(change,2)}% |")
    lines+=['','## 设置、GT与读数限制','',
        '沿用上轮参数：direct方向/位置lr 0.002/0.0005D、5°/0.05D界限、先验0.05、B代理权重1；head复用上轮lr=1e-4、权重0.02、训练200步的A/B结果，训练ID为10489/31249/45503/9016。没有新head优化，也没有新GT监督。它是冻结推理特征上的官方输出MLP微调权重迁移，不等价完整PAct续训。direct A是原预测驻点先验，不冒充原版生成训练续跑。','',
        '训练运动路径21状态，每步3状态、512 Gaussian源/部件；评价每路径41状态，最多2048个源/部件，排除训练源，全部保留targets。部件几何与外观只读，FK刚体运动。head仅取有效方向/转动轴pivot，非轴通道回放该对象原版预测，FK/导出均固定native limits。','',
        '原版输入含GT Mesh投影的oracle二维部件mask，并非RGB-only。GT轴不进入direct优化或新增head更新（本次没有head更新）；测试GT轴在所有候选封存后评分。20411的GT此前已被研究，10655也有旧评估，故不声称首次访问或盲测。未排除PAct预训练重叠。','',
        '20411沿用实际六状态GLB验证的平移方向与固定identity坐标，10655沿用VideoArtGS固定坐标约定。每个场景只由原版静态部件AABB确定一次尺度和平移，候选不另行配准，不用GT选最终步数。','',
        'R0与R0_raw同时保留；官方输出含原版坐标轴硬吸附和GS动画→Mesh导出bbox投影，连续分支的轴学习效果须单看，不能把后处理的0°都当成优化收益。','',
        'Gaussian重叠、采样最近中心距离与1-sigma间隙均为代理，未证明Mesh为合法封闭实体，真实交叠体积与穿透深度unavailable，不填0。所有曲线使用原版预测范围，范围很小时不称真实完全打开。','',
        '首次预检对20411请求无效pivot梯度触发unused-tensor拒绝；只修预检以接受平移pivot不参与计算图，未改优化Loss或轴结果，失败日志保留。原版新推理与旧缓存存在运行漂移，记录在feature_capture.json；本轮A/B始终使用同一次新推理配对，不混用旧部件与新特征。','',
        '## 实际运行','',
        '| direct正式任务 | 步数 | 耗时 秒 | Torch峰值allocated MiB |',
        '| --- | --- | --- | --- |']
    for case in cfg['test_cases']:
        for arm in ['A','B']:
            name=f'direct_{case}_{arm}';status=json.loads((root/name/'status.json').read_text());assert status['state']=='complete';lines.append(f"| {name} | {status['step']} | {fmt(status['elapsed_seconds'],1)} | {fmt(status['peak_gpu_MiB'],1)} |")
    lines+=['','head两组只做新对象前向推理，各新增优化步数0，原训练checkpoint保持SHA不变。源配置、来源、命令、梯度检查、smoke/零权重/实际恢复对照均保存；独立监控见协议。', '',f'根目录：[{root.name}]({root})。', '',f'[预检]({root}/gradient_and_rigidity_checks.json) · [恢复控制]({root}/resume_control.json) · [101284失败]({root}/ineligible_cases.json)。']
    for route in data:
        for case in cfg['test_cases']:
            folder=root/f'eval_{route}_{case}';lines+=['',f'### {route}/{case}','',f'![GT轴位置和方向]({folder}/axes_vs_GT.png)','',f'![固定范围运动重叠]({folder}/motion_overlap.png)','',f'[21帧刚体运动视频]({folder}/rigid_motion.mp4)。']
    text='\n'.join(lines)+'\n';(root/'REPORT.md').write_text(text);repo=pathlib.Path(__file__).resolve().parents[2];(repo/'repro/AXIS_EXTRA_CASES_RESULTS.md').write_text(text);print(json.dumps(rows,indent=2),flush=True)


if __name__=='__main__':main()
