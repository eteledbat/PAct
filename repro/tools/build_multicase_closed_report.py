"""Aggregate every selected case, including negative changes and missing stages."""
import argparse
import csv
import json
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from collision_refinement_common import sha256


def mean(xs):
    xs=[float(x) for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def fmt(x,d=2):return 'N/A' if x is None else f'{x:.{d}f}'


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=pathlib.Path(a.root).resolve()
    protocol=json.loads((root/'protocol.json').read_text());cases=protocol['cases'];data={};coverage={}
    for case in cases:
        out=root/case;coverage[case]={k:(out/file).exists() for k,file in [('fit','fit_results.json'),('evaluation','evaluation.json'),('videos','render_results.json'),('verification','verification.json')]}
        if coverage[case]['evaluation']:data[case]=json.loads((out/'evaluation.json').read_text())
    methods=['original','prior','optimized'];summary={}
    for m in methods:
        values=list(data.values());joints=[v['joint_methods'][m] for v in values];rows=[r for v in joints for r in v['rows']]
        n=sum(v['GT_joint_count'] for v in joints)
        summary[m]={'case_count':len(values),
          'fit_IoU_macro':mean(v['fit_image_metrics'][m]['IoU'] for v in values),
          'fit_PSNR_macro':mean(v['fit_image_metrics'][m]['PSNR'] for v in values),
          'unfitted_IoU_macro':mean(v['unfitted_open_mean'].get(m,{}).get('IoU') for v in values),
          'unfitted_PSNR_macro':mean(v['unfitted_open_mean'].get(m,{}).get('PSNR') for v in values),
          'unfitted_case_count':sum(m in v['unfitted_open_mean'] for v in values),
          'surface_CD_percent_D_macro':mean(v['surface_reconstruction'][m]['CD_percent_D'] for v in values),
          'surface_Fscore_5pct_D_macro':mean(v['surface_reconstruction'][m]['Fscore_5pct_D'] for v in values),
          'GT_joint_count':n,'independently_matched_GT_count':sum(r['matched'] for r in rows),
          'type_correct_GT_count':sum(r['type_correct'] for r in rows),
          'type_correct_GT_coverage_micro':sum(r['type_correct'] for r in rows)/n if n else None,
          'axis_correct_count':sum(r['axis_correct'] for r in rows),
          'axis_correct_rate_micro':sum(r['axis_correct'] for r in rows)/n if n else None,
          'axis_correct_rate_macro':mean(v['axis_correct_rate'] for v in joints),
          'conditional_mean_angle_degrees':mean(r['angle_degrees'] for r in rows),
          'conditional_angle_count':sum(r['angle_degrees'] is not None for r in rows),
          'conditional_mean_pivot_percent_D':mean(r['pivot_percent_D'] for r in rows),
          'conditional_pivot_count':sum(r['pivot_percent_D'] is not None for r in rows),
          'predicted_active_joint_count':sum(v['predicted_active_joint_count'] for v in joints),
          'extra_active_joint_count':sum(v['extra_active_joint_count'] for v in joints),
          'geometry_supported_axis_correct_rate_micro':sum(r.get('geometry_supported_axis_correct',False) for r in rows)/n if n else None}
    deltas={c:{'fit_IoU':v['fit_image_metrics']['optimized']['IoU']-v['fit_image_metrics']['prior']['IoU'],
               'unfitted_IoU':v['unfitted_open_mean'].get('optimized',{}).get('IoU',0)-v['unfitted_open_mean'].get('prior',{}).get('IoU',0) if v['unfitted_open_mean'] else None,
               'surface_CD_percent_D':v['surface_reconstruction']['optimized']['CD_percent_D']-v['surface_reconstruction']['prior']['CD_percent_D']} for c,v in data.items()}
    common=[]
    for v in data.values():
        for old,new in zip(v['joint_methods']['original']['rows'],v['joint_methods']['prior']['rows']):
            if old['type_correct'] and new['type_correct'] and old['angle_degrees'] is not None and new['angle_degrees'] is not None:common.append((old,new))
    common_stats={'joint_count':len(common),'angle_original':mean(a['angle_degrees'] for a,b in common),'angle_prior':mean(b['angle_degrees'] for a,b in common),
                  'pivot_original':mean(a['pivot_percent_D'] for a,b in common if a['pivot_percent_D'] is not None and b['pivot_percent_D'] is not None),
                  'pivot_prior':mean(b['pivot_percent_D'] for a,b in common if a['pivot_percent_D'] is not None and b['pivot_percent_D'] is not None)}
    result={'selected_cases':cases,'coverage':coverage,'summary':summary,'deltas':deltas,'common_matched_joint_subset':common_stats,
            'improved_fit_IoU_cases':sum(v['fit_IoU']>0 for v in deltas.values()),
            'improved_unfitted_IoU_cases':sum(v['unfitted_IoU'] is not None and v['unfitted_IoU']>0 for v in deltas.values()),
            'improved_surface_CD_cases':sum(v['surface_CD_percent_D']<0 for v in deltas.values()),
            'case_evaluations':{c:sha256(root/c/'evaluation.json') for c in data}}
    (root/'aggregate.json').write_text(json.dumps(result,indent=2)+'\n')
    columns=['case','method','fit_IoU','fit_PSNR','unfitted_IoU','unfitted_PSNR','surface_CD_percent_D','surface_Fscore_5pct_D','type_GT_coverage','independent_GT_coverage','axis_correct_rate','angle_degrees','pivot_percent_D','GT_joints','extra_active_joints']
    with (root/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for c,v in data.items():
            for m in methods:
                fit=v['fit_image_metrics'][m];nov=v['unfitted_open_mean'].get(m,{});j=v['joint_methods'][m];g=v['surface_reconstruction'][m]
                writer.writerow(dict(case=c,method=m,fit_IoU=fit['IoU'],fit_PSNR=fit['PSNR'],unfitted_IoU=nov.get('IoU'),unfitted_PSNR=nov.get('PSNR'),
                                     surface_CD_percent_D=g['CD_percent_D'],surface_Fscore_5pct_D=g['Fscore_5pct_D'],type_GT_coverage=j['type_correct_GT_coverage'],
                                     independent_GT_coverage=j['independent_joint_coverage'],axis_correct_rate=j['axis_correct_rate'],angle_degrees=j['conditional_mean_angle_degrees'],
                                     pivot_percent_D=j['conditional_mean_pivot_percent_D'],GT_joints=j['GT_joint_count'],extra_active_joints=j['extra_active_joint_count']))
    plt.rcParams.update({'font.size':10})
    fig,axes=plt.subplots(1,3,figsize=(14,4.5));colors=['#888888','#337ab7','#e38728'];names=['Original','Motion prior','Image fit']
    for ax,key,title in zip(axes,['fit_IoU_macro','unfitted_IoU_macro','axis_correct_rate_micro'],['Fitted closed image (10 cases)','Unfitted open-view IoU (9 cases)','Axis success <=5deg / 5% D']):
        y=[100*summary[m][key] for m in methods];bars=ax.bar(names,y,color=colors)
        for b,h in zip(bars,y):ax.text(b.get_x()+b.get_width()/2,h+1,f'{h:.1f}%',ha='center')
        ax.set_ylim(0,110);ax.set_ylabel('%');ax.set_title(title);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(root/'overall_metrics.png',dpi=180);fig.savefig(root/'overall_metrics.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,6))
    for c,d in deltas.items():
        if d['unfitted_IoU'] is None:continue
        x,y=100*d['fit_IoU'],100*d['unfitted_IoU'];ax.scatter(x,y,s=50);ax.annotate(c,(x,y),xytext=(4,4),textcoords='offset points',fontsize=9)
    ax.axhline(0,c='grey',lw=1);ax.axvline(0,c='grey',lw=1);ax.set(xlabel='Fitted-image IoU change (percentage points)',ylabel='Unfitted open-view IoU change (percentage points)',title='Image fitting: improvement versus transfer');ax.grid(alpha=.2);fig.tight_layout();fig.savefig(root/'fit_vs_unfitted_change.png',dpi=180);plt.close(fig)
    lines=['# 多 case：运动先验与闭合图形状优化','',
      f'预先选定现有 {len(cases)} 个 case：'+', '.join(cases)+'。所有已完成和失败状态均列入覆盖表，不按指标排除案例。', '',
      '三个阶段为原版 PAct、显式运动先验修正、单张闭合图部件形状优化。每个 case 使用相同的 100 步图像配准及 240 步几何优化，展示固定最终步；没有根据 GT 轴误差或未拟合视角挑选 checkpoint。', '',
      '## 整体结果','',
      '| 指标 | 原版 | 运动先验 | 图像优化 |','| --- | ---: | ---: | ---: |']
    metrics=[('闭合拟合图 IoU，case 宏均值 %','fit_IoU_macro',100),('闭合拟合图 PSNR，dB','fit_PSNR_macro',1),('未拟合开放状态视角 IoU，9 case 宏均值 %','unfitted_IoU_macro',100),('未拟合开放状态视角 PSNR，dB','unfitted_PSNR_macro',1),('三维表面 CD，% GT 静态对角线，越低越好','surface_CD_percent_D_macro',1),('独立且类型正确的 GT 关节覆盖率 %','type_correct_GT_coverage_micro',100),('轴正确率：方向≤5°、铰链位置≤5% D %','axis_correct_rate_micro',100),('匹配且类型正确关节的轴角均值，度','conditional_mean_angle_degrees',1),('有效铰链的轴位置均值，% D','conditional_mean_pivot_percent_D',1)]
    for title,key,scale in metrics:lines.append('| '+title+' | '+' | '.join(fmt(summary[m][key]*scale if summary[m][key] is not None else None) for m in methods)+' |')
    lines+=['',f'几何优化相对运动先验：拟合图 IoU 改善 {result["improved_fit_IoU_cases"]}/{len(data)}；未拟合视角 IoU 改善 {result["improved_unfitted_IoU_cases"]}/{summary["prior"]["unfitted_case_count"]}；三维表面 CD 改善 {result["improved_surface_CD_cases"]}/{len(data)}。', '',
      f'独立且类型正确的关节数：原版 {summary["original"]["type_correct_GT_count"]}/{summary["original"]["GT_joint_count"]}，先验/图像优化 {summary["prior"]["type_correct_GT_count"]}/{summary["prior"]["GT_joint_count"]}。轴正确数：原版 {summary["original"]["axis_correct_count"]}/{summary["original"]["GT_joint_count"]}，先验/图像优化 {summary["prior"]["axis_correct_count"]}/{summary["prior"]["GT_joint_count"]}。', '',
      '**图像优化阶段固定了关节，因此先验版与图像优化版的轴数值和正确率相同。轴改善应归因于运动结构/连接先验，不能归因于几何拟合。**', '',
      f'在两阶段共同匹配且类型正确的 {common_stats["joint_count"]} 个关节子集上，轴角均值 {fmt(common_stats["angle_original"])}→{fmt(common_stats["angle_prior"])}°；有效铰链位置 {fmt(common_stats["pivot_original"])}→{fmt(common_stats["pivot_prior"])}% D。条件均值不包含缺失/错误类型，必须与关节覆盖率同时看。', '',
      f'![整体指标]({root}/overall_metrics.png)', '',f'![拟合改善是否传递到其他视角]({root}/fit_vs_unfitted_change.png)', '',
      '## 逐 case 重建指标与视频','',
      '视频均为左原版、中运动先验、右图像优化；上下为两个共同相机。主视频保留原版输出区间，先验/优化版从几何估计闭合姿态开始，保留原始绝对行程；原 fixed 片段被解锁时的 90° 行程单独标注为假设。小于 10° 的预测转轴另外给出明确标注的 90° 诊断视频，不计作真实行程改善。','',
      '| case | 拟合 IoU 先验→优化 % | 未拟合视角 IoU 先验→优化 % | 三维 CD 先验→优化 %D | 三路视频 |','| --- | ---: | ---: | ---: | --- |']
    for c,v in data.items():
        fit=v['fit_image_metrics'];nov=v['unfitted_open_mean'];geo=v['surface_reconstruction']
        n='N/A' if not nov else f'{100*nov["prior"]["IoU"]:.2f}→{100*nov["optimized"]["IoU"]:.2f}'
        lines.append(f'| {c} | {100*fit["prior"]["IoU"]:.2f}→{100*fit["optimized"]["IoU"]:.2f} | {n} | {geo["prior"]["CD_percent_D"]:.2f}→{geo["optimized"]["CD_percent_D"]:.2f} | [视频]({root/c/(c+"_three_versions.mp4")}) |')
    lines+=['','## 逐 case 关节与轴','',
      '| case | GT 关节数 | 类型正确覆盖：原版→先验 | 轴正确：原版→先验 | 条件轴角：原版→先验 ° | 条件位置：原版→先验 %D |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for c,v in data.items():
        x,y=v['joint_methods']['original'],v['joint_methods']['prior'];n=x['GT_joint_count']
        lines.append(f'| {c} | {n} | {round(x["type_correct_GT_coverage"]*n)}/{n}→{round(y["type_correct_GT_coverage"]*n)}/{n} | {x["axis_correct_count"]}/{n}→{y["axis_correct_count"]}/{n} | {fmt(x["conditional_mean_angle_degrees"])}→{fmt(y["conditional_mean_angle_degrees"])} | {fmt(x["conditional_mean_pivot_percent_D"])}→{fmt(y["conditional_mean_pivot_percent_D"])} |')
    lines+=['','## 评价口径','',
      '- IoU 是预测前景与 RGB alpha 的二维轮廓 IoU，阈值均为 0.5，不是实体体积 IoU。闭合图也是拟合目标，必须与未拟合视角分开。三阶段使用同一个在先验版上得到的图像尺度/平移配准，原版不单独重新配准。',
      '- 闭合图的原版指标是闭合姿态诊断：使用由部件几何估计的关节量，但通过原版轴线/FK 渲染；共享轴只能取第一实例的闭合量，不能同时满足两门。它不是原版自然生成参考姿态的分数。主视频的原版仍展示其原始预测区间，因此其起始姿态与闭合图诊断可能不同。',
      '- 未拟合视角固定为 state0/time0 的帧 25、50、75、100、125、149；这些相机没有参与单图形状拟合，帧0是原 PAct 条件图，已排除。所有方法保持原生成参考姿态 q=0，不根据验证图重新调整姿态、尺度或相机。该指标同时反映形状和生成参考姿态偏差。20411 只有固定相机六状态输入，无法提供相同的新相机评价，因此其该项为 N/A，不填 0，也不混入 9-case 均值。',
      '- 三维表面 CD 采用 16,000 个面积采样点，双向非平方欧氏距离的均值，除以 GT 静态部件对角线 D；没有把它称为实体穿透或体积误差。GT 参考为原生成条件状态；不做逐部件刚性对齐，因此包含姿态/装配误差。',
      '- GT 仅在全部拟合候选封存后的独立进程中读取。位置评价使用真正生成柜体/固定体的原始 AABB 与 GT 静态体求一次相似变换；三阶段共用，不用候选各自重新配准，不做 ICP 或轴拟合。',
      '- GT 运动部件与预测关节用固定坐标下的几何表面距离作类型/轴无关的一对一匹配。原版按它实际输出的每个源刚体完整几何匹配；先验/优化版共同使用几何优化前的物理分组匹配，并冻结该对应关系。一个原始共享关节只能为一个 GT 关节计分。这样也保留了 31249 中原始错误类型但独立运动的第二个门片，避免按修正后的主门片归属遗漏这条原始预测。',
      '- 轴角使用无符号 acos(abs(dot))，不评价轴正负/机械运动范围。铰链位置为预测轴原点到 GT 无限轴线的垂距，占 D 百分比。移动关节的原点不可辨识，位置为 N/A。',
      '- 轴正确率按 GT 全部关节为分母：类型正确、独立关节存在、方向≤5°，且铰链位置≤5% D。缺失、错误类型、共用轴造成的独立关节缺失都算失败。条件方向/位置均值另列有效数量。',
      '- 附加的 geometry_supported_axis_correct_rate 要求上述轴正确且至少 30% GT 部件表面在预测对应部件的 5% D 内，仅作几何支撑诊断；不能用一个很小的活动碎片代表完整门。',
      '- 原始 PAct 输入使用此前的二维 oracle 部件 mask；本轮闭合图前板矩形与运动结构均为显式人工先验，不是无需分割的自动重建，也不是新训练的通用模型。未排除原预训练数据重叠。',
      '- 不对单图不可见厚度/背面作自由重建，不改变颜色/透明度，不根据 GT 调轴或换 checkpoint；实体碰撞、运动范围精度和交互后的稳定性未评价。', '',
      '45194 此处按统一批次重新运行并重新评价。旧试验的输入节点对应不能用来评价新的物理门实例，旧对齐也可能把抽屉当成静态体；因此本报告的轴数值不应与旧报告直接拼成趋势。统一批次还为所有 case 明确设置了几何闭合参考、按连接距离选轴，以及共享的图像平面配准，和旧的个案探索有实现差异。', '',
      '评价对应关系经过一次明确复核：初次版本由修正后主部件反推原始关节归属，会漏掉 31249 中那条错误类型但独立存在的原始活动部件。最终原版改按实际源刚体完整几何独立匹配；该修正不读轴误差、不改变任何训练结果或阈值。初次结果保存在 evaluation_v1_inherited_ownership，逐 case 评价另保留该口径作为敏感性记录。', '',
      '## 主要正负结果','',
      '- 10489：两根轴均通过严格标准，闭合图和未拟合视角均改善，是本批较稳定的正例。',
      '- 31249：分离共享门轴、把错误标为抽屉的重复门片绑定到右门后，轴通过数由 3/4 到 4/4；但形状拟合后的三维 CD 略有退化，不能称所有指标都改善。',
      '- 45194：类型正确的独立关节覆盖由 3/4 到 4/4，闭合图与未拟合视角改善；两根门轴方向正确，但平均位置仍约 6.86% D，所以严格轴通过数仍是 2/4，仅两条滑轨通过。',
      '- 8961：轴位置修正有效，但原预测行程不足 1°。主视频几乎不动是原始行程问题；额外 90° 诊断仅用于观察轴，不能称为预测行程修复。',
      '- 100481：篮筐手柄不符合当前设定的 X 向手柄轴先验。两根轴仍有 90° 方向误差，未拟合视角 IoU 和三维 CD 退化；这是该先验不能直接泛化到其他运动族的反例。该错误没有在看到 GT 后重新调参掩盖。',
      '- 9016：闭合图 IoU 很高，但未拟合视角略退化，轴位置误差仍约 20.5% D，说明轮廓拟合无法保证正确的三维连接位置。',
      '- 101808：原版上门只有小片段/融合几何，解锁该片段虽让关节类型覆盖完整，但没有重建出可靠完整门；两根轴均未通过严格标准，三维 CD 也退化。',
      '- 10655：多视角轮廓改善，但轴位置约 7.0% D，仍未达到 5% D 门槛。20411 的滑轨方向通过，但没有同类新相机数据，不参与 9-case 未拟合视角均值。', '',
      '## 文件与覆盖','',f'[完整 CSV]({root}/metrics.csv) · [汇总 JSON]({root}/aggregate.json) · [预先固定协议]({root}/protocol.json) · [所有先验与标注]({root}/recipes.json)','',
      '| case | 拟合 | 评价 | 视频 | 文件核验 |','| --- | --- | --- | --- | --- |']
    for c,s in coverage.items():lines.append('| '+c+' | '+' | '.join('完成' if s[k] else '缺失/失败' for k in ['fit','evaluation','videos','verification'])+' |')
    for c,v in data.items():
        out=root/c;links=[f'[闭合拟合图]({out}/closed_target_and_three_versions.png)',f'[原生优化资产]({out}/optimized_asset/object.json)',f'[逐关节评价]({out}/evaluation.json)']
        if (out/'unfitted_open_view_comparison.png').exists():links.insert(1,f'[未拟合视角对比]({out}/unfitted_open_view_comparison.png)')
        lines+=['',f'### {c}','',' · '.join(links)]
        for diagnostic in out.glob('*assumed90_diagnostic.mp4'):lines+=['',f'[假设 90° 诊断视频，非原预测范围]({diagnostic})']
    report='\n'.join(lines)+'\n';(root/'REPORT.md').write_text(report)
    (REPO_PATH:=pathlib.Path(__file__).resolve().parents[2]/'repro/MULTICASE_CLOSED_IMAGE_RESULTS.md').write_text(report)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
