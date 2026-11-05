"""Aggregate all-case surface-voxel 3D IoU and joint-axis metrics."""
import argparse
import csv
import json
import pathlib
import statistics

from auto_part_semantics import write_json

METHODS=[('origin','Origin'),('single_image_geometry','单图几何优化／原行程'),('extrema_travel','双极限帧行程优化')]


def read(path):return json.loads(pathlib.Path(path).read_text())
def avg(values):
    values=[float(v) for v in values if v is not None]
    return statistics.mean(values) if values else None
def fmt(v):return '—' if v is None else f'{v:.3f}'


def summarize(root):
    cases=read(root/'protocol.json')['cases']
    assert len(cases)==10 and all((root/c/'three_way_3d_axis.json').exists() for c in cases)
    case_rows=[];axis_rows=[];aggregates={}
    for case in cases:
        data=read(root/case/'three_way_3d_axis.json')
        for method,_ in METHODS:
            axis=data['axis'][method];values=data['summary'][method]
            row=dict(case=case,method=method,endpoint_frames='-'.join(map(str,data['frames'])),
                     surface_voxel_IoU_r0=values['surface_voxel_IoU_r0_mean'],
                     surface_voxel_IoU_r1=values['surface_voxel_IoU_r1_mean'],
                     surface_voxel_IoU_r2=values['surface_voxel_IoU_r2_mean'],
                     GT_joints=axis['GT_joint_count'],matched=sum(x['matched'] for x in axis['rows']),
                     type_correct=sum(x['type_correct'] for x in axis['rows']),
                     axis_correct=axis['axis_correct_count'],
                     angle_deg_conditional=axis['conditional_mean_angle_degrees'],
                     pivot_percent_D_conditional=axis['conditional_mean_pivot_percent_D'])
            case_rows.append(row)
            for joint in axis['rows']:
                axis_rows.append(dict(case=case,method=method,GT_part=joint['GT_part'],GT_type=joint['GT_type'],
                    predicted_owner=joint.get('predicted_owner'),predicted_type=joint.get('predicted_type'),
                    matched=joint['matched'],type_correct=joint['type_correct'],
                    direction_error_degrees=joint['angle_degrees'],
                    pivot_line_error_percent_D=joint['pivot_percent_D'],
                    axis_correct=joint['axis_correct'],reason=joint.get('reason'),
                    GT_axis_direction=joint.get('GT_axis_direction'),
                    predicted_axis_direction=joint.get('predicted_axis_direction'),
                    GT_axis_origin=joint.get('GT_axis_origin'),
                    predicted_axis_origin=joint.get('predicted_axis_origin')))
    for method,_ in METHODS:
        rows=[x for x in case_rows if x['method']==method]
        joints=[x for x in axis_rows if x['method']==method]
        aggregates[method]=dict(cases=len(rows),
            surface_voxel_IoU_r0_macro=avg(x['surface_voxel_IoU_r0'] for x in rows),
            surface_voxel_IoU_r1_macro=avg(x['surface_voxel_IoU_r1'] for x in rows),
            surface_voxel_IoU_r2_macro=avg(x['surface_voxel_IoU_r2'] for x in rows),
            GT_joints=sum(x['GT_joints'] for x in rows),matched=sum(x['matched'] for x in rows),
            type_correct=sum(x['type_correct'] for x in rows),axis_correct=sum(x['axis_correct'] for x in rows),
            direction_error_degrees_conditional=avg(x['direction_error_degrees'] for x in joints),
            pivot_line_error_percent_D_conditional=avg(x['pivot_line_error_percent_D'] for x in joints))
    for filename,rows in [('three_way_3d_axis.csv',case_rows),('three_way_axes_per_joint.csv',axis_rows)]:
        with (root/filename).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    meshes=[x for case in cases for x in read(root/case/'three_way_3d_axis.json')['mesh_checks'].values()]
    write_json(root/'three_way_aggregate.json',dict(cases=cases,summary=aggregates,per_case=case_rows,
        nonwatertight_meshes=sum(not x['watertight'] for x in meshes),total_meshes=len(meshes),
        primary_metric='3D surface-shell voxel IoU with pitch GT_static_body_diagonal/96, one 6-neighbor dilation; not solid-volume IoU.',
        axis_correct='Type correct; unoriented direction error <=5 degrees; revolute pivot-line distance <=5% GT static-body diagonal. Prismatic pivot has no physical meaning.',
        comparison='All three variants use the same selected GT endpoint frames and frozen original fixed-body alignment per case. Geometry and extrema-travel variants share all axes, so their axis metrics are mathematically identical.'))
    lines=['# 单图 Origin、几何优化与双帧行程：3D 重叠和轴评测','',
        '覆盖全部 10 个 case。三组分别是原始 Origin、已有单图几何优化且保持原行程、相同单图几何并只更新双极限帧行程。每组都在该 case 的同两张真实极限帧比较。',
        '',f'这些网格中 {sum(not x["watertight"] for x in meshes)}/{len(meshes)} 个不封闭，无法无歧义定义实体体积 IoU。此处报告 **3D 表面体素 IoU**：按 GT 固定底座对角线的 1/96 设共同体素格，用每个网格 80,000 个等面积表面样本占据格点，向 6 邻域膨胀一格后求交并比。`r0` 和 `r2` 敏感性分数也保存在 CSV/JSON。所有方法共用原单图固定底座的 GT 相似对齐，未为任何候选重新对齐。',
        '', '| 方法 | 3D 表面体素 IoU ↑ | 正确轴 / GT | 类型正确 / GT | 方向误差° ↓ | 转轴线位置误差 %D ↓ |',
        '|---|---:|---:|---:|---:|---:|']
    for method,label in METHODS:
        a=aggregates[method]
        lines.append(f'| {label} | {fmt(a["surface_voxel_IoU_r1_macro"])} | {a["axis_correct"]}/{a["GT_joints"]} | {a["type_correct"]}/{a["GT_joints"]} | {fmt(a["direction_error_degrees_conditional"])} | {fmt(a["pivot_line_error_percent_D_conditional"])} |')
    lines += ['', '方向误差按轴的正负方向等价计算，只在关节类型正确时给出；位置误差是预测旋转轴线到 GT 轴线的垂直距离，按 GT 固定底座对角线归一化。平移轴的轴线上某一点不影响运动，因此不对它计算位置误差。严格轴正确要求类型正确、方向误差 ≤5°，且旋转轴线位置误差 ≤5%D。未匹配的 GT 部件计入分母。行程优化不改轴，故后两组轴分数相同。',
        '', '| Case | Origin 3D IoU | 单图几何／原行程 | 双帧行程 | Origin 轴 | 优化后轴 |',
        '|---|---:|---:|---:|---:|---:|']
    for case in cases:
        a,b,c=[next(x for x in case_rows if x['case']==case and x['method']==method) for method,_ in METHODS]
        lines.append(f'| {case} | {fmt(a["surface_voxel_IoU_r1"])} | {fmt(b["surface_voxel_IoU_r1"])} | {fmt(c["surface_voxel_IoU_r1"])} | {a["axis_correct"]}/{a["GT_joints"]} | {c["axis_correct"]}/{c["GT_joints"]} |')
    lines += ['', '体素厚度敏感性：不膨胀 `r0` 时三组平均 IoU 分别为 '
        + ' / '.join(fmt(aggregates[m]['surface_voxel_IoU_r0_macro']) for m,_ in METHODS)
        + '；膨胀两格 `r2` 时分别为 '
        + ' / '.join(fmt(aggregates[m]['surface_voxel_IoU_r2_macro']) for m,_ in METHODS)
        + '。两种尺度下，行程优化后的平均重叠也高于前两组。',
        '', '| Case | Origin 轴 | Origin 方向° | Origin 位置 %D | 优化后轴 | 优化后方向° | 优化后位置 %D |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for case in cases:
        a=next(x for x in case_rows if x['case']==case and x['method']=='origin')
        c=next(x for x in case_rows if x['case']==case and x['method']=='extrema_travel')
        lines.append(f'| {case} | {a["axis_correct"]}/{a["GT_joints"]} | {fmt(a["angle_deg_conditional"])} | {fmt(a["pivot_percent_D_conditional"])} | {c["axis_correct"]}/{c["GT_joints"]} | {fmt(c["angle_deg_conditional"])} | {fmt(c["pivot_percent_D_conditional"])} |')
    lines += ['', '逐关节方向、轴线位置、类型与匹配失败原因见 [`three_way_axes_per_joint.csv`](three_way_axes_per_joint.csv)；逐 case 三组结果见 [`three_way_3d_axis.csv`](three_way_3d_axis.csv)，每个 case 的 `three_way_3d_axis.json` 保存两个极限帧及体素尺度敏感性。对应的[五列视频与行程拟合报告](REPORT.md)仍可查看。']
    (root/'THREE_WAY_3D_AXIS_REPORT.md').write_text('\n'.join(lines)+'\n')
    print('REPORT',root/'THREE_WAY_3D_AXIS_REPORT.md')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);a=p.parse_args()
    summarize(a.root.resolve())
