"""Visualize current/semantic masks and report the sealed two-case ablation."""
import argparse,json,pathlib,sys
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from collision_refinement_common import sha256
from multicase_closed_common import REPO,write_json
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--masks-only',action='store_true');a=p.parse_args();root=pathlib.Path(a.root).resolve();proto=json.loads((root/'protocol.json').read_text());baseline=pathlib.Path(proto['baseline']);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15);large=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
read=lambda p:json.loads(p.read_text())
def rgb_white(path):
 pic=Image.open(path).convert('RGBA');back=Image.new('RGBA',pic.size,'white');back.alpha_composite(pic);return back.convert('RGB')
for case in proto['cases']:
 out=root/case;cfg=read(out/'config.json');anno=read(out/'semantic_annotations.json');manifest=read(out/'motion_groups.json');source_run=pathlib.Path(manifest['source_run']);provenance=read(source_run/'input_provenance.json');source=rgb_white(provenance['source_image']);input_mask=Image.open(pathlib.Path(provenance['input_dir'])/'gt_part_mask_preview.png').convert('RGB');closed=rgb_white(out/'closed_reference.png');old=Image.open(baseline/case/'front_panel_annotations.png').convert('RGB');new=Image.open(out/'front_panel_annotations.png').convert('RGB');labels=np.array(Image.open(out/'semantic_labels_800.png'));overlay=np.array(closed).astype(float)
 for gid in anno['names']:
  color=np.array(cfg['palette'][str(gid)])*255;select=labels==int(gid);overlay[select]=.55*overlay[select]+.45*color
 overlay=Image.fromarray(overlay.astype('uint8'));canvas=Image.new('RGB',(960,920),'white');draw=ImageDraw.Draw(canvas)
 tiles=[('Existing input RGB (frame 0)',source),('Existing input INSTANCE mask (oracle)',input_mask),('Closed RGB used for fitting',closed),('Previous fitting mask: coarse boxes',old),('New fitting mask: visible part polygons',new),('New semantic masks over closed RGB',overlay)]
 for i,(title,pic) in enumerate(tiles):
  x=(i%3)*320;y=(i//3)*370;draw.text((x+5,y+8),title,font=font,fill='black');canvas.paste(pic.resize((320,320),Image.Resampling.NEAREST if i in [1,3,4] else Image.Resampling.LANCZOS),(x,y+34))
 draw.text((8,745),f'{case}: semantic functional parts',font=large,fill='black')
 for j,(gid,name) in enumerate(anno['names'].items()):
  x=8+(j%2)*480;y=778+(j//2)*32;color=tuple(int(x*255) for x in cfg['palette'][str(gid)]);draw.rectangle((x,y,x+18,y+18),fill=color);draw.text((x+25,y),f'{gid}: {name}',font=font,fill='black')
 if case=='45194':draw.text((8,890),'15: occluded drawer insert; rigid child of 13, no invented visible mask.',font=font,fill='black')
 else:draw.text((8,890),'Handle pivots are annotated on this RGB; no GT axis/depth used in fitting.',font=font,fill='black')
 canvas.save(out/'current_vs_semantic_masks.png')
 print('MASK ATLAS',case,flush=True)
if a.masks_only:sys.exit(0)
records={};rows=[]
for case in proto['cases']:
 out=root/case;old=read(baseline/case/'evaluation.json');new=read(out/'evaluation.json');parts=read(out/'part_mask_metrics.json');records[case]=dict(baseline=old,semantic=new,part_masks=parts)
 for variant,e in [('previous',old),('semantic',new)]:
  for method in ['original','prior','optimized']:
   image=e['fit_image_metrics'][method];joint=e['joint_methods'][method];unseen=e['unfitted_open_mean'][method]
   rows.append(dict(case=case,variant=variant,stage=method,fit_IoU=image['IoU'],heldout_IoU=unseen['IoU'],axis_correct=joint['axis_correct_count'],joint_count=joint['GT_joint_count']))
write_json(root/'comparison.json',dict(records=records,rows=rows,failed_initial_attempt=str(root.parent/'semantic_parts_100481_45194_20261028_v1'),initial_attempt_note='First trial retained wrong source ownership and replaced box objective; basket panel rotated as a handle. Preserved rather than hidden. Final experiment corrects ownership and retains original objective before adding semantic losses.'))
lines=['# 逐部件语义约束：100481 / 45194','',
 '每个 case 均保留 Origin → Motion prior → Closed-image fit 三阶段，并另加 GT 网格参考。旧实验和原始生成几何全部保留。','',
 '当前 mask 有两层：PAct 起始输入已使用 frame0 的 GT 投影实例 mask（oracle 实例分割，无语义名称）；旧闭合图优化另用矩形框内前景作弱部件标签。100481 的矩形把大面积篮体误标成提手。','',
 '新闭合图标注是同一张 RGB 上人工描绘的可见部件多边形。保留原 RGB、轮廓、部件颜色与门板框损失，并为所有可见功能部件增加平衡 Dice / CE 约束；门把手跟随门，抽屉把手和内件跟随抽屉。45194 的 group15 在闭合图不可见，用刚性父子连接约束，不虚构可见 mask。','',
 '语义约束：100481 先按完整连通分量纠正所有权：原 source0 大面板归静态篮体；source2 的左右前腿与 source1 的左右后腿分别组成两个独立提手。所有顶点、面和 Gaussian 均一一保留。再从错误的 door 标签/横轴先验改为篮体固定、两个提手分别绕前后方向的连接轴旋转；铆钉在闭合 RGB 标注，投到生成部件自身的前平面，闭合角由 mask 搜索确定。45194 保留已有两扇独立竖轴门、两个滑轨抽屉及附属内件。','',
 '保持旧 RGB 注册参数固定；两例均运行最终 240 步，使用相同损失权重。图像优化冻结轴、关节类型、树和行程。几何只使用原有有界形变能力；提手轴向厚度和轴向位移固定，不补全缺失拓扑。训练进程禁止读取 GT 网格/关节/深度和未用视角；评估在拟合封存后独立运行。','',
 '各视频统一运动进度，保留各自原行程；GT 使用其标注行程，以中性灰真实网格显示。','',
 '| Case | 当前/新 mask | GT + 三阶段 RGB 视频 | 三阶段部件 mask 视频 | 新旧优化结果 + GT |','|---|---|---|---|---|']
for case in proto['cases']:
 lines.append(f'| {case} | [查看]({case}/current_vs_semantic_masks.png) | [播放]({case}/{case}_four_versions_with_GT.mp4) | [播放]({case}/{case}_semantic_masks_three_stages.mp4) | [播放]({case}/{case}_previous_vs_semantic_with_GT.mp4) |')
lines+=['','| Case | 版本 | 阶段 | 拟合图 IoU | 未用视角 IoU | 轴正确数 |','|---|---|---|---:|---:|---:|']
for row in rows:
 lines.append(f"| {row['case']} | {row['variant']} | {row['stage']} | {100*row['fit_IoU']:.2f}% | {100*row['heldout_IoU']:.2f}% | {row['axis_correct']}/{row['joint_count']} |")
lines+=['','| Case | 版本 | Origin 部件 mIoU | Motion prior 部件 mIoU | Closed-image fit 部件 mIoU |','|---|---|---:|---:|---:|']
for case in proto['cases']:
 for variant in ['previous','semantic']:
  values=records[case]['part_masks'][variant]
  lines.append('| '+case+' | '+variant+' | '+' | '.join(f'{100*values[m]["mean_IoU"]:.2f}%' for m in ['original','prior','optimized'])+' |')
lines+=['','100481：新运动先验把轴方向平均误差从 90° 降到 0°，但轴位置平均误差仍为静态主体对角线的约 14.71%，未通过 5% 阈值，完整轴正确数仍为 0/2。45194 的完整轴正确数为 2/4，图像拟合阶段没有修改轴。','',
        '100481 的可见部件 mIoU 与整体闭合图 IoU 在 Motion prior 阶段最高，最后的图像拟合有退化；45194 的部件 mIoU 提升，但整体轮廓和未用视角 IoU 下降。当前适合保留 100481 的语义所有权/运动修正，45194 的整体重建仍以旧图像优化版更好。','']
lines+=['','**结果判断**：100481 的语义归属修正有效，但 Closed-image fit 的闭合图整体 IoU 低于其 Motion prior 阶段；45194 的新语义版本未超过旧版本的整体 IoU。不能把更强的部件语义损失等同于所有重建指标都会改善。','','轴正确标准沿用原实验：方向误差≤5°，转轴到 GT 轴距离≤静态主体对角线5%。静态主体对齐与旧实验一致；对应仍在拟合封存后按几何匹配，100481 的功能部件归属已修正，因此同时保存新旧对应，不用轴误差选择匹配。','','[逐部件 mask 指标](part_mask_summary.json) 使用新 RGB 多边形作为可见功能部件标签，不能视为独立的 GT 分割评测；另报告未参与拟合的原始视角与 GT 轴指标，以区分拟合收益和泛化收益。','',
'100481 的生成提手缺少连接前后腿的 U 形顶桥；修正语义、轴和可见 mask 不会恢复缺失的 U 形结构。单张闭合图也不能唯一确定轴深度、隐藏部件或真实机械行程。']
(root/'REPORT.md').write_text('\n'.join(lines)+'\n')
write_json(root/'part_mask_summary.json',{case:records[case]['part_masks'] for case in proto['cases']})
print('REPORT',root/'REPORT.md',flush=True)
