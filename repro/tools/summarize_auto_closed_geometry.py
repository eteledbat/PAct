"""Create the ten-case automatic/manual geometry audit and closed-view sheet."""
import argparse
import pathlib

import numpy as np
from PIL import Image,ImageDraw,ImageFont

from fit_auto_closed_geometry import CASES,read


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=pathlib.Path,required=True)
    p.add_argument('--selection-root',type=pathlib.Path)
    a=p.parse_args()
    batch=a.root.resolve()
    out=a.selection_root.resolve() if a.selection_root else batch
    selection={r['case']:r for r in read(out/'SELECTION.json')['cases']} if a.selection_root else None
    outputs=batch.parent
    rows=[]
    for case in CASES:
        root=batch/case
        selected=selection[case] if selection else dict(selected='geometric_pose',selected_root=str(root))
        chosen_root=pathlib.Path(selected['selected_root'])
        fit=read(root/'auto_closed_fit.json')
        evaluation=read(root/'auto_closed_evaluation.json')
        old=evaluation['same_image_2d']
        voxel=evaluation['surface_voxel_3d_IoU']
        axis=evaluation['axis']
        if selected['selected']=='previous_auto':
            comparison=evaluation
            chosen_2d=old['previous_auto']['IoU']
            chosen_psnr=old['previous_auto']['PSNR']
            chosen_3d=voxel['previous_auto']['surface_voxel_IoU_r1']
            chosen_axis=axis['previous_auto']['axis_correct_count']
            chosen_parts=read(chosen_root/'fit_seal.json')['geometry_fit']['part_mean_IoU']
            chosen_image=chosen_root/'observed_pose_geometry_after.png'
        else:
            chosen_eval=read(chosen_root/'auto_closed_evaluation.json')
            comparison=chosen_eval
            chosen_fit=read(chosen_root/'auto_closed_fit.json')
            chosen_2d=chosen_eval['same_image_2d']['new_auto']['IoU']
            chosen_psnr=chosen_eval['same_image_2d']['new_auto']['PSNR']
            chosen_3d=chosen_eval['surface_voxel_3d_IoU']['new_auto']['surface_voxel_IoU_r1']
            chosen_axis=chosen_eval['axis']['new_auto']['axis_correct_count']
            chosen_parts=chosen_fit['after']['part_mean_IoU']
            chosen_image=chosen_root/'auto_closed_after.png'
        common_2d=comparison['same_image_2d']
        common_3d=comparison['surface_voxel_3d_IoU']
        common_axis=comparison['axis']
        rows.append(dict(case=case,
            selected=selected['selected'],chosen_image=str(chosen_image),
            prior_2d=common_2d['previous_auto']['IoU'],new_2d=chosen_2d,manual_2d=common_2d['manual']['IoU'],
            prior_psnr=common_2d['previous_auto']['PSNR'],new_psnr=chosen_psnr,manual_psnr=common_2d['manual']['PSNR'],
            prior_3d=common_3d['previous_auto']['surface_voxel_IoU_r1'],
            new_3d=chosen_3d,
            manual_3d=common_3d['manual']['surface_voxel_IoU_r1'],
            prior_axis=common_axis['previous_auto']['axis_correct_count'],
            new_axis=chosen_axis,
            manual_axis=common_axis['manual']['axis_correct_count'],
            gt_joints=common_axis['manual']['GT_joint_count'],
            part_iou=chosen_parts,
            recovered=bool(fit['recovery'].get('recovered'))))
    header='| Case | 2D IoU 上一自动 / 本次 / 人工 | 3D IoU 上一自动 / 本次 / 人工 | 轴正确数 上一自动 / 本次 / 人工 |'
    lines=[
        '# 自动几何重建与人工指导版同图对照',
        '',
        '输入为已发布单张图 PAct 几何、自动 VLM 部件语义和所选关闭帧 RGB/实例 mask。',
        '自动恢复漏掉的独立部件；侧开门的关闭角由其网格主方向与柜体正面估计；候选铰链须满足柜体接触；抽屉在已知关闭帧使用收纳位；之后拟合相机尺度、部件形变与自动 mask 框。没有读取旧人工配方、框或优化权重参与拟合。',
        '',
        '关闭帧 mask 是数据集 GT 网格投影出的实例 ID，**不是人工标注**，但仍是带有真值信息的 2D 输入；仅给普通 RGB 的场景还需要自动实例分割与跨帧跟踪。GT 3D 网格和关节只在全部拟合后用于本表评估。',
        '',
        '候选选择只读自动拟合在输入关闭帧的轮廓 IoU：优先保留几何关闭姿态，其他自动候选只有高出 0.02 才替换。人工版与 GT 3D/轴指标未参与选择。' if selection else '本表为统一几何关闭姿态候选，未进行后选。',
        '',header,
        '|---|---|---:|---:|---:|' if selection else '|---|---:|---:|---:|',
    ]
    if selection:lines[-2]='| Case | 自动候选 | 2D IoU 上一自动 / 本次 / 人工 | 3D IoU 上一自动 / 本次 / 人工 | 轴正确数 上一自动 / 本次 / 人工 |'
    for r in rows:
        mode=f"{r['selected']} | " if selection else ''
        lines.append(f"| {r['case']} | {mode}{r['prior_2d']:.3f} / {r['new_2d']:.3f} / {r['manual_2d']:.3f} | {r['prior_3d']:.3f} / {r['new_3d']:.3f} / {r['manual_3d']:.3f} | {r['prior_axis']} / {r['new_axis']} / {r['manual_axis']} |")
    mean=lambda key:float(np.mean([r[key] for r in rows]))
    mode='selected | ' if selection else ''
    lines.append(f"| 十例均值 / 总数 | {mode}{mean('prior_2d'):.3f} / {mean('new_2d'):.3f} / {mean('manual_2d'):.3f} | {mean('prior_3d'):.3f} / {mean('new_3d'):.3f} / {mean('manual_3d'):.3f} | {sum(r['prior_axis'] for r in rows)} / {sum(r['new_axis'] for r in rows)} / {sum(r['manual_axis'] for r in rows)}（GT {sum(r['gt_joints'] for r in rows)}） |")
    lines.extend(['',f"同一关闭图 PSNR 十例均值：上一自动 {mean('prior_psnr'):.2f} dB，本次 {mean('new_psnr'):.2f} dB，人工版 {mean('manual_psnr'):.2f} dB。"])
    lines+=['',
        '2D 为同一关闭图的二值轮廓 IoU，属于拟合视角表现。3D 为同一 GT 静止柜体配准、每网格 8 万个面积均匀表面点、固定体对角线 / 96 的共享体素网格，半径 1 体素膨胀后的表面 IoU；不是实体体积 IoU。轴正确要求类型匹配、无符号方向误差 ≤5°、转轴线位置误差 ≤固定体对角线的 5%。',
        '',
        f"自动语义恢复：{', '.join(r['case'] for r in rows if r['recovered']) or '无'}。",
        '',
        '正面图并不能唯一确定部件深度、厚度和完整行程；需要多视角或整段运动一致性与碰撞约束才能继续缩小 3D 差距。',
        '',
        '同图对比见 [closed_view_comparison.png](closed_view_comparison.png)。每例候选原目录的优化前后与 mask 见 `<case>/auto_closed_before.png`、`<case>/auto_closed_after.png`。',
    ]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    width=4*256
    row_h=288
    sheet=Image.new('RGB',(width,36+row_h*len(CASES)),(248,248,248))
    draw=ImageDraw.Draw(sheet)
    for j,title in enumerate(['GT closed frame','Old manual fit','Previous automatic','New automatic']):
        draw.text((j*256+8,8),title,font=font,fill=(0,0,0))
    for i,r in enumerate(rows,1):
        case=r['case']
        paths=[outputs/'closed_multicase_20261028_v1'/case/'optimized_closed_fit.png',
               outputs/'closed_multicase_20261028_v1'/case/'optimized_closed_fit.png',
               outputs/'closed_travel_reference_geometry_v011_20261031_v2'/case/'observed_pose_geometry_after.png',
               pathlib.Path(r['chosen_image'])]
        crops=[(0,0,256,256),(256,0,512,256),(256,0,512,256),(256,0,512,256)]
        y=36+(i-1)*row_h
        draw.text((8,y+3),f"{case}    2D IoU {r['prior_2d']:.3f} -> {r['new_2d']:.3f}   manual {r['manual_2d']:.3f}",font=font,fill=(0,0,0))
        for j,(path,box) in enumerate(zip(paths,crops)):
            with Image.open(path) as pic:sheet.paste(pic.convert('RGB').crop(box),(j*256,y+28))
    sheet.save(out/'closed_view_comparison.png')
    print('SUMMARY',out,lines[-10],flush=True)


if __name__=='__main__':main()
