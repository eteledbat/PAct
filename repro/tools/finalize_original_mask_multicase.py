"""Verify all masks/geometry/motion/video/source integrity; build batch report."""
import argparse,csv,json,pathlib,os,shutil,sys,copy,subprocess
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
import cv2,imageio.v2 as imageio,numpy as np,torch
from PIL import Image,ImageDraw,ImageFont
from plyfile import PlyData
from original_mask_multicase import CASES,BASE,REPO,load_model,read,COLORS,FONT
from collision_refinement_common import sha256,transforms
from multicase_closed_common import write_json

def mean(v):
 values=[float(x) for x in v if x is not None];return float(np.mean(values)) if values else None

def fmt(v,scale=1,digits=2):return 'N/A' if v is None else f'{scale*v:.{digits}f}'

def finish(batch):
 assert sha256(REPO/'repro/tools/original_mask_multicase.py')==read(batch/'protocol.json')['code_sha256']
 torch.set_num_threads(4);verification={};evaluations={};comparisons={};fitdata={};csvrows=[];diagnostics={}
 for case in CASES:
  root=batch/case;cfg=read(root/'config.json');manifest=read(root/'motion_groups.json');fit=read(root/'fit_results.json');ev=read(root/'evaluation.json');cmp=read(root/'baseline_comparison.json');sem=read(root/'semantic_labels.json');evaluations[case]=ev;comparisons[case]=cmp;fitdata[case]=fit
  previous_comparison=read(batch/'comparison_mask_camera_v1'/case/'baseline_comparison.json')
  assert previous_comparison['old_checkpoint_sha256']==cmp['old_checkpoint_sha256'] and previous_comparison['new_checkpoint_sha256']==cmp['new_checkpoint_sha256']
  assert previous_comparison['new_evaluation']==cmp['new_evaluation'] and previous_comparison['old_published_evaluation']==cmp['old_published_evaluation']
  for title,values in cmp['common_registration']['training_frame_metrics'].items():
   for key in ['IoU','PSNR','foreground_PSNR','foreground_L1']:assert np.isclose(values[key],previous_comparison['common_registration']['training_frame_metrics'][title][key],atol=1e-6)
  for title,values in cmp['common_registration']['held_out_mean'].items():
   for key,value in values.items():assert np.isclose(value,previous_comparison['common_registration']['held_out_mean'][title][key],atol=1e-6)
  if case!='100481':
   shared=cmp['common_registration']['training_frame_metrics'];assert abs(shared['Old motion prior']['part_mask']['mean_IoU']-shared['New motion prior']['part_mask']['mean_IoU'])<.002
  lab=np.array(Image.open(root/'original_instance_labels_800.png'));raw=cv2.imread(sem['source_original_mask'],cv2.IMREAD_UNCHANGED);raw=raw if raw.ndim==2 else raw[:,:,0];assert np.array_equal(lab,raw);ids=np.array(Image.open(root/'semantic_ids_800.png'));restored=np.zeros_like(lab)
  for k,value in sem['label_to_semantics'].items():restored[ids==value['owner']]=int(k)
  assert np.array_equal(restored,lab);assert sha256(sem['source_original_mask'])==sem['source_mask_sha256'];assert sem['changed_pixels']==0
  for key,name in [('checkpoint_sha256','checkpoint.pt'),('optimized_parts_sha256','optimized_parts.npz'),('config_sha256','config.json'),('manifest_sha256','motion_groups.json'),('observation_sha256','observation.npz')]:assert sha256(root/name)==fit[key],(case,name)
  assert fit['case']==case and fit['source_tool_sha256']==sha256(REPO/'repro/tools/original_mask_multicase.py')
  assert len(read(root/'registration_results.json')['history'])==100
  hist=read(root/'history.json');assert len(hist)==240 and hist[-1]['step']==240;diagnostics[case]=dict(first_step_loss=hist[0]['loss'],final_step_loss=hist[-1]['loss'],final_loss_exceeds_first=hist[-1]['loss']>hist[0]['loss'],part_mask_mIoU_before=fit['part_mask_metrics']['prior']['mean_IoU'],part_mask_mIoU_after=fit['part_mask_metrics']['optimized']['mean_IoU'],fixed_final_step=240)
  assert cfg['alignment_steps']==100 and cfg['geometry_steps']==240;assert cfg['annotation_boxes_original_pixels']=={} and cfg['annotation_polygons_original_pixels']=={}
  assert all(g['box'] is None and g['observed_q']==0 for g in manifest['groups'])
  for filename in ['registration_results.json','axis_prior_results.json','fit_results.json']:assert not read(root/filename)['GT_forbidden_reads']
  assert sha256(cfg['source_image'])==cfg['source_image_sha256'];assert sha256(cfg['source_camera'])==cfg['camera_sha256'];assert all(sha256(p['path'])==p['sha256'] for p in manifest['geometry_files']);assert sha256(manifest['source_object'])==manifest['source_object_sha256']
  m=load_model(root,checkpoint=True);m.requires_grad_(False);meshes={p['node']['id']:p['mesh'] for p in m.asset['parts']};partition=[];maxrest=0.;maxrigid=0.
  for nid,mesh in meshes.items():
   groups=[g for g in m.groups if g['source_node']==nid];ngs=len(PlyData.read(manifest['sources'][str(nid)]['gs'])['vertex'].data)
   for key,n in [('vertices',len(mesh.vertices)),('faces',len(mesh.faces)),('gaussians',ngs)]:
    selected=np.concatenate([m.membership[g['membership_prefix']+'_'+key] for g in groups]);assert np.array_equal(np.sort(selected),np.arange(n)),(case,nid,key)
   partition.append(dict(source=nid,vertices=len(mesh.vertices),faces=len(mesh.faces),gaussians=ngs))
  for p in m.parts:
   g=p['group'];native=np.asarray(meshes[g['source_node']].vertices)[m.membership[g['membership_prefix']+'_vertices']];back=p['vertex_local']@p['B'].cpu().numpy().T+p['anchor'].cpu().numpy();maxrest=max(maxrest,float(abs(native-back).max()))
  assert maxrest<1e-6
  with torch.no_grad():
   sample={p['group']['id']:m.deform(i)[0][::997].cpu().numpy().astype(float) for i,p in enumerate(m.parts)}
   for t in np.linspace(0,1,21):
    tr=transforms(m.groups,t)
    for g in m.groups:
     if g['joint']['type']=='fixed' and g['parent']!=-1:assert np.allclose(tr[g['id']],tr[g['parent']],atol=1e-12)
     x=sample[g['id']];T=tr[g['id']];y=x@T[:3,:3].T+T[:3,3];maxrigid=max(maxrigid,float(abs(np.linalg.norm(x-x[0],axis=1)-np.linalg.norm(y-y[0],axis=1)).max()))
  assert maxrigid<1e-12
  for key in ['axis_correct_count','conditional_mean_angle_degrees','conditional_mean_pivot_percent_D']:assert ev['joint_methods']['prior'][key]==ev['joint_methods']['optimized'][key]
  oldprior=read(BASE/case/'motion_groups.json')
  same9=None
  if case!='100481':
   same9=all(g['joint']==next(v for v in oldprior['groups'] if v['id']==g['id'])['joint'] for g in m.groups);assert same9
  videos=[]
  for filename in ['render_results.json','gt_video_results.json']:
   for row in read(root/filename)['videos']:
    path=root/row['file'];assert sha256(path)==row['sha256']
    with imageio.get_reader(path) as reader:
     count=reader.count_frames();meta=reader.get_meta_data()
     for frame in reader:assert frame.shape[:2]==(meta['size'][1],meta['size'][0])
    assert count==row['frames'];videos.append({**row,'all_frames_decoded':True})
  comp=cmp['video_comparison'];path=root/comp['file'];assert sha256(path)==comp['sha256']
  with imageio.get_reader(path) as reader:
   count=reader.count_frames()
   for frame in reader:assert frame.shape[:2]==(832,1536)
  assert count==84;videos.append(dict(file=path.name,frames=count,sha256=sha256(path),all_frames_decoded=True))
  ver=dict(original_mask_changed_pixels=0,inverse_semantic_remap_exact=True,source_partition_bijections=partition,max_original_rest_reconstruction_error=maxrest,max_rigid_length_error=maxrigid,fixed_children_follow_parent=True,all_observed_q_zero=True,axes_types_tree_ranges_frozen_in_fit=True,prior_joints_equal_baseline_for_nonbasket_case=same9,source_files_unchanged=True,all_video_frames_decoded=True,videos=videos);write_json(root/'verification.json',ver);verification[case]=ver
  del m;torch.cuda.empty_cache();print('VERIFIED',case,len(videos),flush=True)
  for method in ['original','prior','optimized']:
   j=ev['joint_methods'][method];s=ev['surface_reconstruction'][method];n=ev['unfitted_open_mean'].get(method,{})
   csvrows.append(dict(case=case,method=method,input_IoU=fit['image_metrics'][method]['IoU'],input_PSNR=fit['image_metrics'][method]['PSNR'],part_mask_mIoU=fit['part_mask_metrics'][method]['mean_IoU'],heldout_IoU=n.get('IoU'),heldout_PSNR=n.get('PSNR'),CD_percent_D=s['CD_percent_D'],Fscore_1pct_D=s['Fscore_1pct_D'],Fscore_5pct_D=s['Fscore_5pct_D'],GT_joints=j['GT_joint_count'],type_correct_count=round(j['type_correct_GT_coverage']*j['GT_joint_count']),axis_correct_count=j['axis_correct_count'],axis_angle_degrees=j['conditional_mean_angle_degrees'],axis_position_percent_D=j['conditional_mean_pivot_percent_D']))
 jointrows_csv=[]
 for c in CASES:
  for m in ['original','prior','optimized']:
   for row in evaluations[c]['joint_methods'][m]['rows']:
    jointrows_csv.append(dict(case=c,method=m,GT_part=row['GT_part'],GT_type=row['GT_type'],predicted_owner=row.get('predicted_owner'),predicted_type=row.get('predicted_type'),matched=row['matched'],type_correct=row['type_correct'],axis_correct=row['axis_correct'],angle_degrees=row['angle_degrees'],pivot_percent_D=row['pivot_percent_D'],surface_coverage_5pct_D=row.get('GT_surface_coverage_at_5pct_D'),reason=row.get('reason')))
 with (batch/'joint_metrics.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(jointrows_csv[0]));writer.writeheader();writer.writerows(jointrows_csv)
 baseline_hashes=read(batch/'baseline_hashes.json');assert all(sha256(p)==digest for p,digest in baseline_hashes.items()),'Baseline output changed'
 write_json(batch/'optimization_diagnostics.json',diagnostics)
 write_json(batch/'VERIFICATION.json',dict(cases=verification,all_original_masks_changed_pixels=0,baseline_all_files_unchanged=True,baseline_file_count=len(baseline_hashes),all_source_files_unchanged=True))
 with (batch/'metrics.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(csvrows[0]));writer.writeheader();writer.writerows(csvrows)
 agg={}
 for method in ['original','prior','optimized']:
  selected=[r for r in csvrows if r['method']==method];jointrows=[r for c in CASES for r in evaluations[c]['joint_methods'][method]['rows']];N=sum(r['GT_joints'] for r in selected)
  agg[method]=dict(input_IoU=mean(r['input_IoU'] for r in selected),input_PSNR=mean(r['input_PSNR'] for r in selected),part_mask_mIoU=mean(r['part_mask_mIoU'] for r in selected),heldout_IoU=mean(r['heldout_IoU'] for r in selected),heldout_PSNR=mean(r['heldout_PSNR'] for r in selected),heldout_case_count=sum(r['heldout_IoU'] is not None for r in selected),CD_percent_D=mean(r['CD_percent_D'] for r in selected),Fscore_1pct_D=mean(r['Fscore_1pct_D'] for r in selected),Fscore_5pct_D=mean(r['Fscore_5pct_D'] for r in selected),GT_joint_count=N,type_correct_count=sum(r['type_correct_count'] for r in selected),type_correct_rate=sum(r['type_correct_count'] for r in selected)/N,axis_correct_count=sum(r['axis_correct_count'] for r in selected),axis_correct_rate=sum(r['axis_correct_count'] for r in selected)/N,conditional_axis_angle_degrees=mean(r['angle_degrees'] for r in jointrows),conditional_axis_position_percent_D=mean(r['pivot_percent_D'] for r in jointrows))
 oldevals={c:comparisons[c]['old_published_evaluation'] for c in CASES};oldrows=[r for c in CASES for r in oldevals[c]['joint_methods']['optimized']['rows']];oldN=sum(oldevals[c]['joint_methods']['optimized']['GT_joint_count'] for c in CASES)
 old=dict(published_heldout_IoU=mean(oldevals[c]['unfitted_open_mean'].get('optimized',{}).get('IoU') for c in CASES),published_heldout_PSNR=mean(oldevals[c]['unfitted_open_mean'].get('optimized',{}).get('PSNR') for c in CASES),CD_percent_D=mean(oldevals[c]['surface_reconstruction']['optimized']['CD_percent_D'] for c in CASES),axis_correct_count=sum(oldevals[c]['joint_methods']['optimized']['axis_correct_count'] for c in CASES),axis_correct_rate=sum(oldevals[c]['joint_methods']['optimized']['axis_correct_count'] for c in CASES)/oldN,type_correct_count=sum(r['type_correct'] for r in oldrows),GT_joint_count=oldN,conditional_axis_angle_degrees=mean(r['angle_degrees'] for r in oldrows),conditional_axis_position_percent_D=mean(r['pivot_percent_D'] for r in oldrows))
 common={}
 for title in ['Origin','Old motion prior','Old closed-image fit','New motion prior','New original-mask fit']:
  common[title]=dict(input_IoU=mean(comparisons[c]['common_registration']['training_frame_metrics'][title]['IoU'] for c in CASES),input_PSNR=mean(comparisons[c]['common_registration']['training_frame_metrics'][title]['PSNR'] for c in CASES),input_part_mask_mIoU=mean(comparisons[c]['common_registration']['training_frame_metrics'][title]['part_mask']['mean_IoU'] for c in CASES),heldout_IoU=mean(comparisons[c]['common_registration']['held_out_mean'].get(title,{}).get('IoU') for c in CASES),heldout_PSNR=mean(comparisons[c]['common_registration']['held_out_mean'].get(title,{}).get('PSNR') for c in CASES))
 improvements=dict(new_fit_vs_new_prior_input=sum(evaluations[c]['fit_image_metrics']['optimized']['IoU']>evaluations[c]['fit_image_metrics']['prior']['IoU'] for c in CASES),new_fit_vs_new_prior_heldout=sum(evaluations[c]['unfitted_open_mean'].get('optimized',{}).get('IoU',0)>evaluations[c]['unfitted_open_mean'].get('prior',{}).get('IoU',0) for c in CASES if c!='20411'),new_vs_old_shared_heldout=sum(comparisons[c]['common_registration']['held_out_mean']['New original-mask fit']['IoU']>comparisons[c]['common_registration']['held_out_mean']['Old closed-image fit']['IoU'] for c in CASES if c!='20411'),new_vs_old_surface_CD=sum(evaluations[c]['surface_reconstruction']['optimized']['CD_percent_D']<oldevals[c]['surface_reconstruction']['optimized']['CD_percent_D'] for c in CASES))
 write_json(batch/'aggregate.json',dict(new_methods=agg,old_optimized=old,common_registration=common,improvement_counts=improvements))
 headers=['case','旧→新 同配准输入IoU %','旧→新 同配准未拟合IoU %','旧→新 CD %D','旧→新 轴通过数','新四路视频','新旧视频对比']
 lines=['# 原始输入 mask 语义约束：10-case 重建与旧实验对比','', '**10/10 case 已完成；原始 mask 边界改变像素总数为 0。** 视频顺序均为 **GT → Origin → Motion prior → Original-mask geometry fit**。','', '[原始输入与 mask 总览](original_masks_inventory.png) · [完整指标 CSV](metrics.csv) · [逐关节 CSV](joint_metrics.csv) · [汇总 JSON](aggregate.json) · [固定实验协议](protocol.json) · [全部核验](VERIFICATION.json) · [优化收敛诊断](optimization_diagnostics.json) · [部件投影相机修订记录](comparison_evaluation_revision.json)','', '800×800原始 mask 原样保存，训练时使用最近邻缩放至256×256。本轮给原始 mask 中已有的实例编号添加语义，不画框、不圈多边形、不改变区域。语义名称由原始 RGB 人工注明，不是自动语义识别；原始 mask 是此前输入已有的 oracle/GT 投影实例 mask。100481 使用刚确认的左右提手流程，其余9个 case 保持旧实验的物理分组、门/抽屉类型与几何连接轴先验。所有 case 从原版 PAct 网格/Gaussian 重新开始，不读取旧几何拟合 checkpoint。','', '重新进行100步原始图 RGB/轮廓尺度与图像平面配准，然后固定关节类型、轴、层级和行程，进行240步有界几何优化。损失为 RGB、整体轮廓、原始部件语义颜色、逐部件 Dice 与平衡交叉熵；原来手画的闭合图矩形损失已移除。全部10个拟合结果封存后才进行 GT 评价，不用 GT 轴、深度或未拟合视角调整结果。','', '## 新流程三个阶段','', '| 指标 | Origin | Motion prior | 原始 mask 几何优化 |','|---|---:|---:|---:|']
 for label,key,scale in [('原始输入整体 IoU %','input_IoU',100),('原始输入 PSNR dB','input_PSNR',1),('未拟合视角 IoU %（9 case）','heldout_IoU',100),('未拟合视角 PSNR dB（9 case）','heldout_PSNR',1),('表面 CD %D，越低越好','CD_percent_D',1),('表面 F-score @1%D %','Fscore_1pct_D',100),('表面 F-score @5%D %','Fscore_5pct_D',100),('类型正确独立关节覆盖率 %','type_correct_rate',100),('完整轴正确率 %','axis_correct_rate',100),('条件平均轴角 °','conditional_axis_angle_degrees',1),('条件平均轴位置 %D','conditional_axis_position_percent_D',1)]:lines.append('| '+label+' | '+' | '.join(fmt(agg[m][key],scale) for m in ['original','prior','optimized'])+' |')
 lines+=['',f"几何优化相对本轮 Motion prior：输入 IoU 改善 **{improvements['new_fit_vs_new_prior_input']}/10**，未拟合视角 IoU 改善 **{improvements['new_fit_vs_new_prior_heldout']}/9**。运动先验与几何优化的轴分数相同，因为几何阶段冻结关节。",'', '## 与旧闭合图优化比较','', '旧实验使用闭合图拟合，本轮使用原始输入图，因此两者各自训练图的 IoU **不能直接横向比较**。下面同时给出：①在同一个本轮固定配准、同一组相机下重新评价旧/新几何；②旧实验原样发布的未拟合视角分数。前者用于减少配准差异，后者保留完整旧流程的实际表现。都没有重新拟合旧几何。','', '| 指标 | 旧闭合图优化 | 新原始 mask 优化 |','|---|---:|---:|']
 for label,key,scale in [('同配准原始输入 IoU %','input_IoU',100),('同配准原始输入 PSNR dB','input_PSNR',1),('同配准输入部件 mIoU %','input_part_mask_mIoU',100),('同配准未拟合视角 IoU %','heldout_IoU',100),('同配准未拟合视角 PSNR dB','heldout_PSNR',1)]:lines.append(f"| {label} | {fmt(common['Old closed-image fit'][key],scale)} | {fmt(common['New original-mask fit'][key],scale)} |")
 for label,a,b,scale in [('各自原配准的未拟合 IoU %',old['published_heldout_IoU'],agg['optimized']['heldout_IoU'],100),('各自原配准的未拟合 PSNR dB',old['published_heldout_PSNR'],agg['optimized']['heldout_PSNR'],1),('三维表面 CD %D',old['CD_percent_D'],agg['optimized']['CD_percent_D'],1),('完整轴正确率 %',old['axis_correct_rate'],agg['optimized']['axis_correct_rate'],100),('条件平均轴角 °',old['conditional_axis_angle_degrees'],agg['optimized']['conditional_axis_angle_degrees'],1),('条件平均轴位置 %D',old['conditional_axis_position_percent_D'],agg['optimized']['conditional_axis_position_percent_D'],1)]:lines.append(f'| {label} | {fmt(a,scale)} | {fmt(b,scale)} |')
 lines+=['',f"新优化相对旧优化：同配准未拟合 IoU 改善 **{improvements['new_vs_old_shared_heldout']}/9**，表面 CD 改善 **{improvements['new_vs_old_surface_CD']}/10**。旧/新完整轴分别通过 **{old['axis_correct_count']}/{oldN}**、**{agg['optimized']['axis_correct_count']}/{agg['optimized']['GT_joint_count']}**。",'', '输入部件 mIoU 是训练 mask 的诊断，不能当作独立分割泛化指标。旧版也按其物理分组重着色；100481 旧物理分组确实把篮子面板当作提手，因此该分数同时反映分组修正。Origin 的重着色分数不代表 PAct 原始语义准确率。','', '## 逐 case 指标与视频','', '| '+' | '.join(headers)+' |','|'+'---|'*len(headers)]
 comparisonrows=[]
 for c in CASES:
  cmp=comparisons[c];shared=cmp['common_registration'];a=shared['training_frame_metrics']['Old closed-image fit']['IoU'];b=shared['training_frame_metrics']['New original-mask fit']['IoU'];ha=shared['held_out_mean'].get('Old closed-image fit',{}).get('IoU');hb=shared['held_out_mean'].get('New original-mask fit',{}).get('IoU');oa=oldevals[c]['joint_methods']['optimized'];nb=evaluations[c]['joint_methods']['optimized'];cdold=oldevals[c]['surface_reconstruction']['optimized']['CD_percent_D'];cdnew=evaluations[c]['surface_reconstruction']['optimized']['CD_percent_D'];n=nb['GT_joint_count']
  lines.append(f"| {c} | {fmt(a,100)}→{fmt(b,100)} | {fmt(ha,100)}→{fmt(hb,100)} | {fmt(cdold)}→{fmt(cdnew)} | {oa['axis_correct_count']}/{n}→{nb['axis_correct_count']}/{n} | [新视频]({c}/{c}_four_versions_with_GT.mp4) | [上下对比]({c}/{c}_old_vs_original_mask_four_versions.mp4) |")
  comparisonrows.append(dict(case=c,old_shared_input_IoU=a,new_shared_input_IoU=b,old_shared_heldout_IoU=ha,new_shared_heldout_IoU=hb,old_CD_percent_D=cdold,new_CD_percent_D=cdnew,old_axis_correct=oa['axis_correct_count'],new_axis_correct=nb['axis_correct_count'],GT_joints=n))
 with (batch/'old_new_comparison.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(comparisonrows[0]));writer.writeheader();writer.writerows(comparisonrows)
 lines+=['','[逐 case 新旧对比 CSV](old_new_comparison.csv)','','## Mask、共同相机图片与逐关节明细','', '| case | 原始 mask 加语义 | 旧/新部件投影 | 同配准原始图 | 同配准未拟合视角 | 全部评价 |','|---|---|---|---|---|---|']
 for c in CASES:
  held=f'[图片]({c}/old_new_shared_camera_unfitted_views.png)' if c!='20411' else 'N/A'
  lines.append(f'| {c} | [图片]({c}/original_mask_semantic_comparison.png) | [图片]({c}/old_new_part_mask_comparison.png) | [图片]({c}/old_new_original_input_comparison.png) | {held} | [JSON]({c}/evaluation.json) |')
 lines+=['','## 运动展示和指标边界','','新四路主视频上下为两个共同相机，四列按 GT、Origin、Motion prior、原始 mask 几何优化排列。Origin 使用原始预测区间；9个非提手 case 沿用旧版“几何估计闭合姿态 + 原始绝对行程”的先验播放区间。100481 沿用上轮确认的输入姿态向外旋转，GT 同样从输入姿态转至真实折叠限位。播放进度同步，预测幅度不替换成 GT 行程。被解锁的原 fixed 片段所用90°、极小原行程的90°诊断视频均为明确标注的假设，不视为预测行程正确。','','新旧上下对比视频各取一个观察视角；上排旧实验、下排新实验，各排保留自己的相机和运动起点，标题已标明，因此它只供视觉浏览，不是逐像素同姿态对比。精确形状对比请看同配准输入/未拟合图片。20411 没有同类未拟合相机，不把它的 N/A 填0；其 GT 使用真实00→05运动状态，原始模型区间保留，未用GT修复行程。','','IoU 为前景轮廓 IoU（alpha>0.5）；PSNR 在白背景合成 RGB 上计算。未拟合视角固定为25、50、75、100、125、149，保持原生成 q0，不作验证时配准/姿态优化。三维 CD 为16000个面积采样点的双向非平方欧氏距离均值，归一化到 GT 静态主体对角线 D。三维与轴评价共用原始静态体的一次尺度/平移，不做 ICP 或逐部件对齐。','','运动关节用类型/轴无关的固定坐标几何距离作一对一匹配；原版共享轴只能记一个实例。完整轴正确要求独立关节且类型正确、无符号方向误差≤5°、旋转轴位置误差≤5%D；平移轴原点不可辨识，位置记 N/A。条件角度/位置均值须与覆盖率及完整通过数一起看。','','本轮原 mask 能约束已有几何的可见投影，无法凭一张图补出缺失完整部件、背面和连接结构。运动先验不是自动学习到的通用模型；9个非提手 case 的轴保持旧先验，因此单独换 mask 不会改善这些已冻结的轴分数。100481 改善了轴方向，但轴位置仍未通过严格阈值。结果保留全部正负案例，不按 GT 分数调整先验或选择 checkpoint。10489 出现遮挡次序变化后部件 mIoU 明显下降，说明当前固定学习率与多目标几何优化并非所有 case 都稳定；收敛诊断已单独保存，本轮没有据此替换最终240步结果。','','所有旧实验文件和原版 PAct 源资产均已核对哈希保持不变。']
 subprocess.run([sys.executable,str(REPO/'repro/tools/plot_original_mask_multicase.py'),'--root',str(batch)],check=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
 lines+=['','[逐 case 指标图](old_new_metrics.png) · [可导出 PDF](old_new_metrics.pdf) · [全部新四路视频中间帧](all_cases_four_versions_midpoint.png)']
 (batch/'REPORT.md').write_text('\n'.join(lines)+'\n')
 # A compact contact sheet of the NEW four-way midpoint, linked by report.
 sheet=Image.new('RGB',(1536,10*(192+24)),'white');d=ImageDraw.Draw(sheet)
 for i,c in enumerate(CASES):
  pic=Image.open(batch/c/'four_versions_with_GT_t0.50.png').crop((0,0,1536,384)).resize((768,192));d.text((5,i*216+3),c,font=FONT,fill='black');sheet.paste(pic,(220,i*216+24))
 sheet.save(batch/'all_cases_four_versions_midpoint.png')
 for name in ['original_mask_multicase.py','compare_original_mask_multicase.py','finalize_original_mask_multicase.py','plot_original_mask_multicase.py']:shutil.copyfile(REPO/'repro/tools'/name,batch/'source_snapshot'/name)
 seal=dict(state='complete',cases=CASES,original_mask_changed_pixels=0,baseline_all_files_unchanged=True,video_count=sum(len(v['videos']) for v in verification.values()),all_video_frames_decoded=True,report_sha256=sha256(batch/'REPORT.md'),metrics_sha256=sha256(batch/'metrics.csv'),comparison_csv_sha256=sha256(batch/'old_new_comparison.csv'),verification_sha256=sha256(batch/'VERIFICATION.json'),aggregate_sha256=sha256(batch/'aggregate.json'),protocol_sha256=sha256(batch/'protocol.json'),source_snapshot_sha256={p.name:sha256(p) for p in (batch/'source_snapshot').glob('*.py')},percase={c:dict(fit_sha256=sha256(batch/c/'fit_results.json'),evaluation_sha256=sha256(batch/c/'evaluation.json'),comparison_sha256=sha256(batch/c/'baseline_comparison.json')) for c in CASES})
 write_json(batch/'FINAL_SEAL.json',seal);(REPO/'repro/current_original_mask_multicase.txt').write_text(str(batch)+'\n');print(json.dumps(dict(aggregate=agg,old=old,common=common,improvements=improvements,seal=seal),indent=2),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();finish(pathlib.Path(a.root).resolve())
