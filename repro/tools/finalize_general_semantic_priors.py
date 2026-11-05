"""Seal the reusable-rule regression and added upward-door case."""
import argparse,ast,json,pathlib,shutil,sys
import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from collision_refinement_common import sha256,transforms
from multicase_closed_common import REPO,write_json
p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=pathlib.Path(a.root).resolve();read=lambda p:json.loads(pathlib.Path(p).read_text());protocol=read(root/'protocol.json');previous=pathlib.Path(protocol['previous']);cases=protocol['cases'];states=read(root/'controller_status.json');assert states['state']=='complete'
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
records={};videos=[];regression={};protection=read(root/'previous_preservation.json');assert all(sha256(pathlib.Path(path))==h for path,h in protection.items())
for case in cases:
 out=root/case;cfg=read(out/'config.json');manifest=read(out/'motion_groups.json');fit=read(out/'fit_results.json');ev=read(out/'evaluation.json');prior=read(out/'axis_prior_results.json');mask_info=read(out/'semantic_labels.json')
 assert not fit['GT_forbidden_reads'] and not prior['GT_forbidden_reads'];assert mask_info['original_mask_changed_pixels']==0
 for key,file in [('checkpoint_sha256','checkpoint.pt'),('config_sha256','config.json'),('manifest_sha256','motion_groups.json'),('optimized_parts_sha256','optimized_parts.npz')]:assert sha256(out/file)==fit[key]
 assert all(sha256(pathlib.Path(row['path']))==row['sha256'] for row in manifest['geometry_files'])
 # Every Gaussian/vertex/triangle retained exactly once; original source
 # counts are read from native source groups recorded in baseline preparation.
 membership=np.load(out/'membership.npz');source_counts={}
 def ply_header_counts(path):
  counts={}
  with open(path,'rb') as f:
   for raw in f:
    line=raw.decode('ascii').strip()
    if line.startswith('element '):
     _,element,n=line.split();counts[element]=int(n)
    if line=='end_header':break
  return counts
 for nid in manifest['sources']:
  subset=[g for g in manifest['groups'] if g['source_node']==int(nid)];source_counts[nid]={}
  for suffix in ['vertices','faces','gaussians']:
   header=ply_header_counts(manifest['sources'][nid]['gs' if suffix=='gaussians' else 'mesh']);expected=header['face' if suffix=='faces' else 'vertex']
   ids=np.concatenate([membership[g['membership_prefix']+'_'+suffix] for g in subset]);assert np.array_equal(np.sort(ids),np.arange(expected));source_counts[nid][suffix]=len(ids)
 for progress in np.linspace(0,1,21):
  tr=transforms(manifest['groups'],float(progress))
  for g in manifest['groups']:
   if g['joint']['type']=='fixed' and g['parent']!=-1:assert np.allclose(tr[g['id']],tr[g['parent']],atol=1e-12)
 for filename in ['render_results.json','gt_video_results.json']:
  for v in read(out/filename)['videos']:
   path=out/v['file'];assert sha256(path)==v['sha256']
   with imageio.get_reader(path) as reader:
    n=reader.count_frames();assert n==v['frames']
    for i in range(n):reader.get_data(i)
   videos.append({**v,'case':case,'all_frames_decoded':True})
 assert ev['joint_methods']['prior']['axis_correct_count']==ev['joint_methods']['optimized']['axis_correct_count']
 if (previous/case/'evaluation.json').exists():
  oldev=read(previous/case/'evaluation.json');oldman=read(previous/case/'motion_groups.json');oldactive={g['id']:g for g in oldman['groups'] if g['joint']['type']!='fixed'};newactive={g['id']:g for g in manifest['groups'] if g['joint']['type']!='fixed'}
  unchanged=set(oldactive)==set(newactive) and all(oldactive[k]['joint']==newactive[k]['joint'] for k in oldactive)
  regression[case]=dict(kinematics_unchanged=unchanged,handle_rule_accepted=prior.get('accepted'),old_axis=oldev['joint_methods']['prior']['axis_correct_count'],new_axis=ev['joint_methods']['prior']['axis_correct_count'],old_input_IoU=oldev['fit_image_metrics']['optimized']['IoU'],new_input_IoU=ev['fit_image_metrics']['optimized']['IoU'],old_unfitted_IoU=oldev['unfitted_open_mean'].get('optimized',{}).get('IoU'),new_unfitted_IoU=ev['unfitted_open_mean'].get('optimized',{}).get('IoU'))
  if not prior.get('accepted'):assert unchanged,'Uncertain handle evidence must retain original family prior'
 # Input/mask overlay for direct inspection.
 rgba=np.asarray(Image.open(out/'input_reference.png').convert('RGBA')).astype(float)/255;rgb=(rgba[:,:,:3]*rgba[:,:,3:]+1-rgba[:,:,3:]);labels=np.array(Image.open(out/'semantic_ids_800.png'));seg=np.ones_like(rgb)
 for owner in np.unique(labels):
  if owner:seg[labels==owner]=cfg['palette'][str(owner)]
 canvas=Image.new('RGB',(1200,450),'white');draw=ImageDraw.Draw(canvas)
 for j,(title,pic) in enumerate([('Original conditioning RGB',rgb),('Exact original instance mask',seg),('Semantic overlay',.6*rgb+.4*seg)]):
  tile=Image.fromarray((pic.clip(0,1)*255).astype(np.uint8)).resize((400,400));canvas.paste(tile,(400*j,30));draw.text((400*j+8,7),title,font=font,fill='black')
 draw.text((8,432),case+' | input mask boundaries unchanged | semantic family / attachment priors',font=font,fill='black');canvas.save(out/'original_mask_semantic_comparison.png')
 shutil.copyfile(out/'closed_target_and_three_versions.png',out/'input_target_and_three_versions.png')
 write_json(out/'status.json',dict(state='complete',**fit))
 records[case]=dict(evaluation=ev,axis_prior=prior,source_partitions=source_counts,fit_seconds=fit['fit_seconds'])
write_json(root/'results.json',dict(cases=records,regression=regression))
# Aggregate like-for-like stages over all eleven cases.
aggregate={}
for method in ['original','prior','optimized']:
 valid=[records[c]['evaluation'] for c in cases];joint=[e['joint_methods'][method] for e in valid];held=[e['unfitted_open_mean'][method]['IoU'] for e in valid if method in e['unfitted_open_mean']]
 aggregate[method]=dict(input_IoU=float(np.mean([e['fit_image_metrics'][method]['IoU'] for e in valid])),unfitted_IoU=float(np.mean(held)),unfitted_case_count=len(held),axis_correct=sum(j['axis_correct_count'] for j in joint),joint_count=sum(j['GT_joint_count'] for j in joint))
write_json(root/'aggregate.json',aggregate)
# Selection RGBs are kept separately from actual training images.
lines=['# 通用提手规则与上翻门：11-case 实验','',
 '**已完成原有 10 个 case 回归，并新增 103015 的三扇上翻窗门。** 新视频顺序为 GT → Origin → 通用运动先验 → 原始 mask 几何优化。旧实验保留。','',
 '提手入口不再使用 case ID。原始模型的语义名称是候选证据；程序结合细长/完整 U 形几何、原始实例 mask、独立连接点及父子关系选择规则。门/抽屉上的固定拉手跟随父部件，独立活动提手由分离连接点拟合轴线。方向接近规范坐标轴12°以内时采用轴对齐先验，否则保留几何估计方向。证据不足时保留既有运动，写出拒绝原因。','',
 '完整 U 形连通提手使用接触点聚类；分片式提手合并同一实例的支撑片段。当前真实独立提手回归样本仍主要是100481；任意轴方向、完整U形、固定拉手与退化连接另有几何单元测试，不能据此宣称已验证所有真实提手类别。','',
 '上翻门根据语义族选择上侧连接边，拟合水平铰链线，轴穿过门板中面；开启符号由自由边向外、向上的运动决定。103015 的现有几何中，一个下部面板曾被当成门，两扇上窗曾合并到同一源节点。本轮按原始实例 mask 的连通分量投票纠正归属，保持所有原顶点、面和 Gaussian。','',
 '原始 mask 为已有 oracle/GT 投影实例分割，本轮不改变边界；类别/上翻族仍允许人工语义注释，尚未训练新的自动语义识别器。103015 的样本类型由源 RGB 运动确认；只有原始第0帧用于训练，20411沿用原生成输入05。','',
 '每个 case 从原生 PAct 几何重新开始：100步RGB/轮廓配准，冻结配准与关节后做240步原始mask几何拟合。GT轴、网格和未用视角在所有拟合封存后才进入评价。关节不会在几何阶段继续优化。','',
 '| 阶段 | 输入轮廓 IoU | 未用视角 IoU | 完整轴通过数 |','|---|---:|---:|---:|']
for method,name in [('original','Origin'),('prior','通用运动先验'),('optimized','原始mask几何优化')]:
 a=aggregate[method];lines.append(f"| {name} | {100*a['input_IoU']:.2f}% | {100*a['unfitted_IoU']:.2f}% ({a['unfitted_case_count']}例) | {a['axis_correct']}/{a['joint_count']} |")
lines+=['','轴标准：方向误差≤5°；旋转关节的轴位置误差≤GT静态主体对角线5%。方向0°不代表位置正确。单帧q0的重建损失不能直接确定真实运动轴。','',
 '| Case | 规则结果 | 输入／mask／叠加 | GT＋三阶段视频 | 拟合前后图片 | 轴通过 Origin→prior |','|---|---|---|---|---|---|']
for case in cases:
 e=records[case]['evaluation'];prior=records[case]['axis_prior'];description=prior['dispatched']+('：采用' if prior.get('accepted') else '：保留既有先验')
 lines.append(f"| {case} | {description} | [图片]({case}/original_mask_semantic_comparison.png) | [播放]({case}/{case}_four_versions_with_GT.mp4) | [图片]({case}/input_target_and_three_versions.png) | {e['joint_methods']['original']['axis_correct_count']}→{e['joint_methods']['prior']['axis_correct_count']} / {e['joint_methods']['prior']['GT_joint_count']} |")
lines+=['','重点 case 的方向与位置分开评价：','', '| Case | 平均方向误差 Origin→prior | 平均轴位置误差 Origin→prior（主体对角线） | 完整轴通过 |','|---|---:|---:|---:|']
for case in ['100481','103015']:
 e=records[case]['evaluation']['joint_methods'];o=e['original'];n=e['prior']
 lines.append(f"| {case} | {o['conditional_mean_angle_degrees']:.2f}° → {n['conditional_mean_angle_degrees']:.2f}° | {o['conditional_mean_pivot_percent_D']:.2f}% → {n['conditional_mean_pivot_percent_D']:.2f}% | {n['axis_correct_count']}/{n['GT_joint_count']} |")
lines+=['','两例的方向都已纠正，但位置仍未通过5%阈值。103015 输入轮廓IoU从98.57%降至92.37%，部件mask mIoU则有提升，说明当前多目标几何损失存在取舍，不能宣称本轮几何普遍变好。10项单元测试通过；新实验视频全部逐帧解码核验。','']
lines+=['','103015 两扇窗继承的原始预测行程只有约2.6°，主视频保留该幅度。另有 [上翻90°诊断](103015/103015_assumed90_with_GT.mp4)，其中仅预测列使用假设90°，GT保留标注行程；诊断仅检查转轴，不表示真实预测行程。','',
 '首轮过宽的细长形状规则曾误选门板，已在GT评价前停止并保留于同级 general_semantic_priors_20261029_v1；当前v2要求明确提手语义证据，拒绝仅靠细长门板触发。','',
 '原有case的逐项回归：[JSON](results.json)。未触发提手规则的case应保持原运动参数；100481可能因新的连接点估计而有位置和拟合差异。几何优化仍使用固定最终240步，保留下降与退化的case，不按GT挑选结果。','',
 '代码：[通用规则](../../tools/semantic_motion_rules.py)、[实验入口](../../tools/general_semantic_multicase.py)、[单元测试](../../tools/test_semantic_motion_rules.py)。']
(root/'REPORT.md').write_text('\n'.join(lines)+'\n')
# Contact sheet of all comparison midpoints.
canvas=Image.new('RGB',(1024,11*256),'white')
for i,case in enumerate(cases):
 pic=Image.open(root/case/'four_versions_with_GT_t0.50.png').resize((1024,512));tile=pic.crop((0,0,1024,256));canvas.paste(tile,(0,i*256))
canvas.save(root/'all_cases_midpoint.png')
for name in ['general_semantic_multicase.py','semantic_motion_rules.py','test_semantic_motion_rules.py','finalize_general_semantic_priors.py']:
 source=REPO/'repro/tools'/name;ast.parse(source.read_text());shutil.copyfile(source,root/'source_snapshot'/name)
for name,h in protocol['frozen_before_fit_source_sha256'].items():assert sha256(REPO/'repro/tools'/name)==h
write_json(root/'FINAL_SEAL.json',dict(state='complete',cases=cases,video_count=len(videos),videos=videos,all_video_frames_decoded=True,unchanged_original_mask_pixels=True,previous_outputs_preserved=True,source_partition_conservation=True,report_sha256=sha256(root/'REPORT.md'),results_sha256=sha256(root/'results.json'),aggregate_sha256=sha256(root/'aggregate.json'),source_snapshot_sha256={p.name:sha256(p) for p in (root/'source_snapshot').glob('*.py')}))
(REPO/'repro/current_general_semantic_priors.txt').write_text(str(root)+'\n');print('SEALED',len(cases),'cases',len(videos),'videos',aggregate,flush=True)
