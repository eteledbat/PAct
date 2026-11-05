"""Compare sealed automatic fits with the manual fit using one closed-state CD protocol."""
import argparse
import copy
import csv
import pathlib

import numpy as np
import torch
import trimesh

from auto_part_semantics import write_json
from collision_refinement_common import surface
from evaluate_freeart_partnet_case import gt_parts
from evaluate_multicase_closed import distance_scores
from evaluate_observed_drawer_pose import build_mesh
from evaluate_two_extrema import gt_mesh
from fit_closed_extrema_geometry import CASES
from fit_observed_drawer_pose import load
from fit_single_origin_extrema_travel import source
from fit_auto_closed_geometry import read
from multicase_closed_common import Model


NAMES=['previous_auto','latest_selected','manual_style_mask_pose',
       'manual_style_geometric_pose','manual_style_bbox_anchor','manual_semantic']


def frozen(groups,pose):
    tree=copy.deepcopy(groups)
    for g in tree:
        value=pose.get(str(g['id']),pose.get(g['id'],0.))
        g['joint']['range']=[float(value)]*2
    return tree


def evaluate(outputs,case):
    previous=outputs/'closed_travel_reference_geometry_v011_20261031_v2'/case
    latest=outputs/'closed_auto_geometry_selected_20261104'/case
    mask_pose=outputs/'closed_auto_manual_style_cd_20261104_v1'/case
    geometric=outputs/'closed_auto_manual_style_geometric_pose_20261104_v1'/case
    bbox=outputs/'closed_auto_manual_style_bbox_anchor_20261104_v1'/case
    human=outputs/'closed_multicase_20261028_v1'/case
    paths={'previous_auto':previous,'latest_selected':latest,
           'manual_style_mask_pose':mask_pose,
           'manual_style_geometric_pose':geometric,
           'manual_style_bbox_anchor':bbox,
           'manual_semantic':human}
    if any(not p.exists() for p in paths.values()):
        raise FileNotFoundError({k:str(v) for k,v in paths.items() if not v.exists()})
    torch.set_num_threads(4)
    A=read(previous/'metrics.json')['alignment']
    R=np.asarray(A['R_export_to_GT']);scale=float(A['scale'])
    shift=np.asarray(A['translation']);D=float(A['GT_static_diagonal'])
    frame=int(read(previous/'fit_seal.json')['closed_frame'])
    _,published=source(case)
    scene=pathlib.Path(read(published/'config.json')['source_scene'])
    truth=(trimesh.util.concatenate(gt_parts(scene)[0][frame]) if case=='20411'
           else gt_mesh(scene,case,frame))
    gt_cloud=surface(truth,16000,177)
    candidates={'previous_auto':(previous,'observed_pose_geometry.pt',read(previous/'fit_seal.json')['observed_pose'])}
    if (latest/'auto_closed_geometry.pt').exists():
        latest_checkpoint='auto_closed_geometry.pt'
        latest_pose=read(latest/'auto_closed_fit.json')['pose']
    else:
        latest_checkpoint='observed_pose_geometry.pt'
        latest_pose=read(latest/'fit_seal.json')['observed_pose']
    candidates['latest_selected']=(latest,latest_checkpoint,latest_pose)
    for name,path in [('manual_style_mask_pose',mask_pose),
                      ('manual_style_geometric_pose',geometric),
                      ('manual_style_bbox_anchor',bbox)]:
        candidates[name]=(path,'manual_style_geometry.pt',read(path/'manual_style_fit.json')['pose'])
    human_groups=read(human/'motion_groups.json')['groups']
    candidates['manual_semantic']=(human,'checkpoint.pt',
        {str(g['id']):g['observed_q'] for g in human_groups})
    result={}
    for name,(path,checkpoint,pose) in candidates.items():
        if name=='manual_semantic':
            model=Model(read(path/'config.json'),read(path/'motion_groups.json'),path)
            model.load_state_dict(torch.load(path/checkpoint,map_location='cuda',weights_only=False)['model'])
        else:model=load(path,checkpoint=checkpoint)
        model.requires_grad_(False)
        pred_mesh=build_mesh(model,optimized=True,tree=frozen(model.groups,pose),progress=0.)
        points=scale*(surface(pred_mesh,16000,178)@R.T)+shift
        result[name]=distance_scores(points,gt_cloud,D)
        del model,pred_mesh,points
        torch.cuda.empty_cache()
    return dict(case=case,closed_frame=frame,CD_percent_D=result,alignment=A)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--outputs',type=pathlib.Path,required=True)
    p.add_argument('--report-root',type=pathlib.Path,required=True)
    p.add_argument('--case')
    a=p.parse_args();outputs=a.outputs.resolve();root=a.report_root.resolve();root.mkdir(parents=True,exist_ok=True)
    rows=[];details=[]
    for case in ([a.case] if a.case else CASES):
        data=evaluate(outputs,case)
        details.append(data)
        row=dict(case=case,**{name:data['CD_percent_D'][name]['CD_percent_D'] for name in NAMES})
        rows.append(row)
        write_json(root/(case+'.json'),data)
        print(case,{k:round(row[k],3) for k in NAMES},flush=True)
    if a.case:return
    means={name:float(np.mean([r[name] for r in rows])) for name in NAMES}
    with (root/'CD_COMPARISON.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['case',*NAMES]);w.writeheader();w.writerows(rows)
    write_json(root/'SUMMARY.json',dict(rows=rows,means=means,
        protocol='All methods at the SAME closed GT frame; 16000 area-uniform whole-object surface points, seeds 177/178, symmetric non-squared Euclidean CD, reported as percent original GT static-body diagonal. Same frozen original-body similarity for all candidates. GT meshes only read post-fit; no GT-based candidate selection.'))
    lines=['# 关闭状态 3D CD：自动流程与人工语义版同口径比较','',
           '数值为双向非平方表面 Chamfer 距离，占 GT 固定体对角线百分比，越低越好。所有方法使用同一关闭状态 GT 网格、原始柜体配准和 16,000 个面积采样点；GT 只在拟合封存后用于评价。与旧报告的原始输入状态 CD=3.90%D 不是同一 GT 姿态，不能直接比较。','',
           '活动结果已回滚至 `../closed_auto_geometry_active/`，该链接指向上一自动版 `closed_travel_reference_geometry_v011_20261031_v2/`。本表后三组自动拟合只是隔离实验，没有覆盖活动结果或历史产物。','',
           '| Case | 回滚基线 | 上次选定（已回滚） | 人工损失+mask姿态 | +几何关闭姿态 | +包围盒锚点 | 人工语义 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append('| '+row['case']+' | '+' | '.join(f"{row[name]:.3f}" for name in NAMES)+' |')
    lines.append('| 十例均值 | '+' | '.join(f"{means[name]:.3f}" for name in NAMES)+' |')
    wins=sum(r['manual_style_geometric_pose']<r['previous_auto'] for r in rows)
    gap=100*(means['manual_style_geometric_pose']/means['manual_semantic']-1)
    by_case={r['case']:r for r in rows}
    hard=by_case['8961']
    lines+=['',
        f"统一的几何关闭姿态方案在 {wins}/10 例优于回滚基线，十例平均 {means['manual_style_geometric_pose']:.3f}%D，距人工语义版 {means['manual_semantic']:.3f}%D 约 {gap:.1f}%。8961 退化至 {hard['manual_style_geometric_pose']:.3f}%D（回滚基线 {hard['previous_auto']:.3f}%D）；100481 和 10655 仍明显落后于人工版。AABB 锚点方案均值 {means['manual_style_bbox_anchor']:.3f}%D，没有稳定额外收益。上次选定版的均值 {means['latest_selected']:.3f}%D 略低于本次统一方案，但用户已要求回滚其结果；不能把本次方案称为全指标最优。",'',
        '本次自动实验复用了原生 PAct 几何和既有自动语义。采用人工版相同的 RGB、轮廓、语义颜色、部件框及形变正则权重，100 步配准和 240 步形变，最终步保存；人工矩形框改由关闭帧实例 mask 自动计算，对称部件按语义和位置自动配对。几何关闭姿态方案还让抽屉取收回位、侧开门按生成门板与柜体正面确定关闭角；轴不变。它与人工版仍有部件归属和框来源差异，101808 的独立上门仍未由回滚基线恢复。','',
        '关闭帧实例 mask 是数据集 GT 网格投影出的二维实例 ID，不是普通 RGB 自动分割。对比的实验代码和输入未读取人工配方、GT 三维几何或 GT 关节进行拟合；人工语义结果只在此评价阶段载入。CD 是封存后的评价指标，没有把 GT CD 当作训练损失或逐例挑选依据。','',
        '逐例明细见 [CSV](CD_COMPARISON.csv)、[对比图](CD_COMPARISON.png) 和 `<case>.json`。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('MEANS',{k:round(v,4) for k,v in means.items()},flush=True)


if __name__=='__main__':main()
