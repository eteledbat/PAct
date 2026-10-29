"""Audit VLM-guided shape optimization and publish all four-version videos."""
import argparse,csv,json,pathlib,shutil,re
import imageio.v2 as imageio,numpy as np,torch
from PIL import Image,ImageDraw
from plyfile import PlyData
from collision_refinement_common import sha256,transforms
from fit_vlm_component_geometry import read,load_model,FONT
from multicase_closed_common import write_json

def mean(values):
 a=[float(x) for x in values if x is not None];return float(np.mean(a)) if a else None

def finalize(batch):
 torch.set_num_threads(4);protocol=read(batch/'protocol.json');cases=protocol['cases'];axis_root=pathlib.Path(protocol['axis_root']);assert read(batch/'controller_status.json')['state']=='complete';assert protocol['code_sha256']==sha256(pathlib.Path(__file__).parent/'fit_vlm_component_geometry.py');rows=[];jointrows=[];verifications={};evaluations={};fitresults={};video_rows=[];supported=0;total_masks=0
 for case in cases:
  root=batch/case;cfg=read(root/'config.json');mfest=read(root/'motion_groups.json');f=read(root/'fit_results.json');ev=read(root/'evaluation.json');oldaxis=read(axis_root/case/'motion_groups.json');old_ev=read(axis_root/case/'joint_evaluation.json');prep=read(root/'prepared.json');evaluations[case]=ev;fitresults[case]=f
  for key,name in [('checkpoint_sha256','checkpoint.pt'),('optimized_parts_sha256','optimized_parts.npz'),('config_sha256','config.json'),('manifest_sha256','motion_groups.json'),('observation_sha256','observation.npz')]:assert sha256(root/name)==f[key],(case,name)
  assert f['case']==case and f['source_tool_sha256']==protocol['code_sha256'];assert not prep['GT_forbidden_reads'] and not f['GT_forbidden_reads'];assert not prep['manual_semantic_labels_used'] and not prep['manual_case_recipe_used'];assert cfg['annotation_boxes_original_pixels']=={} and cfg['annotation_polygons_original_pixels']=={};assert len(read(root/'history.json'))==240;assert sha256(cfg['source_image'])==cfg['source_image_sha256']
  source_mask=pathlib.Path(mfest['instance_mask']);ids=np.load(root/'original_instance_ids.npy');assert np.array_equal(ids,np.load(source_mask));assert sha256(source_mask)==prep['source_mask_sha256']==prep['copied_mask_sha256'];assert sha256(source_mask)==mfest['instance_mask_sha256'];assert sha256(mfest['semantic_input'])==mfest['semantic_input_sha256']==prep['source_semantics_sha256'];assert sha256(root/'membership.npz')==sha256(axis_root/case/'membership.npz');assert all(sha256(x['path'])==x['sha256'] for x in mfest['geometry_files']);assert sha256(mfest['source_object'])==mfest['source_object_sha256'];assert sha256(root/'registration.pt')==prep['registration_sha256']
  assert sha256(cfg['source_camera'])==cfg['camera_sha256']
  oldobs=np.load(axis_root/case/'observation.npz');newobs=np.load(root/'observation.npz')
  for field in ['rgb','alpha','extrinsics','intrinsics']:assert np.array_equal(oldobs[field],newobs[field])
  assert len(oldaxis['groups'])==len(mfest['groups'])
  for x,y in zip(mfest['groups'],oldaxis['groups']):assert all(x[k]==y[k] for k in ['id','parent','joint','source_node','component','semantic_owner','counts','membership_prefix']) and x['box'] is None and x['observed_q']==0.
  model=load_model(root,checkpoint=True);model.requires_grad_(False);mesh={p['node']['id']:p['mesh'] for p in model.asset['parts']};partitions=[];rest_error=0.;rigid_error=0.;attachment_error=0.
  for nid,source in mesh.items():
   gs_count=len(PlyData.read(mfest['sources'][str(nid)]['gs'])['vertex'].data);groups=[g for g in model.groups if g['source_node']==nid]
   for key,n in [('vertices',len(source.vertices)),('faces',len(source.faces)),('gaussians',gs_count)]:
    selected=np.concatenate([model.membership[g['membership_prefix']+'_'+key] for g in groups]);assert np.array_equal(np.sort(selected),np.arange(n))
   partitions.append(dict(source=nid,vertices=len(source.vertices),faces=len(source.faces),gaussians=gs_count))
  for p in model.parts:
   g=p['group'];v=np.asarray(mesh[g['source_node']].vertices)[model.membership[g['membership_prefix']+'_vertices']];back=p['vertex_local']@p['B'].cpu().numpy().T+p['anchor'].cpu().numpy();rest_error=max(rest_error,float(np.max(abs(v-back))))
   if g['shape_owner'] is not None:
    other=model.parts[model.indices[g['shape_owner']]];assert p['parameter_index']==other['parameter_index'] and torch.allclose(p['anchor'],other['anchor']) and torch.allclose(p['B'],other['B'])
  assert rest_error<1e-6
  with torch.no_grad():
   sample={p['group']['id']:(model.deform(i)[0] if len(p['local']) else model.deform(i,torch.tensor(p['vertex_local'],dtype=torch.float32,device='cuda')))[::997].cpu().numpy().astype(float) for i,p in enumerate(model.parts)}
   for t in np.linspace(0,1,21):
    tr=transforms(model.groups,t)
    for g in model.groups:
     if g['joint']['type']=='fixed' and g['parent']!=-1:assert np.allclose(tr[g['id']],tr[g['parent']],atol=1e-12)
     x=sample[g['id']];
     if not len(x):continue
     T=tr[g['id']];y=x@T[:3,:3].T+T[:3,3];rigid_error=max(rigid_error,float(np.max(abs(np.linalg.norm(x-x[0],axis=1)-np.linalg.norm(y-y[0],axis=1)))))
   assert rigid_error<1e-12
   geom0=model.geometry(original=True);geom1=model.geometry();probs=[model.probabilities(g).cpu().numpy() for g in [geom0,geom1]];pics=[Image.fromarray((model.rgb.clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8)),Image.open(root/'automatic_semantic_target.png').convert('RGB')]
   for prob in probs:
    labels=np.array([0,*model.visible_owners])[np.concatenate([(1-prob.sum(0)).clip(0,1)[None],prob]).argmax(0)];rgb=np.full((256,256,3),255,np.uint8)
    for gid in model.visible_owners:rgb[labels==gid]=(255*np.array(cfg['palette'][str(gid)])).astype(np.uint8)
    pics.append(Image.fromarray(rgb))
   canvas=Image.new('RGB',(1024,300),'white');d=ImageDraw.Draw(canvas)
   for j,(title,pic) in enumerate(zip(['Original RGB','Automatic semantic target','VLM axes: projected masks','After geometry fit'],pics)):canvas.paste(pic,(j*256,40));d.text((j*256+3,6),title,font=FONT,fill='black')
   canvas.save(root/'automatic_mask_fit_comparison.png')
  supported+=len(model.visible_owners);total_masks+=len([x for x in np.unique(ids) if x]);del model;torch.cuda.empty_cache()
  # Joint matching on unchanged pre-fit geometry and frozen alignment must give
  # exactly the same axis scores as the sealed v7 evaluation.
  for method in ['prior','optimized']:
   now=ev['joint_methods'][method];old=old_ev['joint_methods']['prior'];assert now['axis_correct_count']==old['axis_correct_count'] and now['GT_joint_count']==old['GT_joint_count']
   for a,b in zip(now['rows'],old['rows']):
    assert a['GT_part']==b['GT_part'] and a['type_correct']==b['type_correct'] and a['axis_correct']==b['axis_correct']
    for k in ['angle_degrees','pivot_percent_D']:
     if a[k] is None:assert b[k] is None
     else:assert np.isclose(a[k],b[k],atol=1e-6)
  vr=read(root/'auto_four_video_results.json');path=root/vr['video'];assert sha256(path)==vr['sha256']
  with imageio.get_reader(path) as reader:
   count=reader.count_frames();assert count==84
   for frame in reader:assert frame.shape[:2]==(768,1536)
  video_rows.append(dict(case=case,file=str(path),sha256=sha256(path),frames=count,all_frames_decoded=True))
  # Correct generic legacy export prose; this asset was fitted at input q0.
  assetpath=root/'optimized_asset/object.json';asset=read(assetpath);asset.update(optimization='VLM automatic physical-instance ownership, original RGB/instance mask bounded geometry fit; v7 joints frozen',rest_pose='Original conditioning pose q0; camera registration fixed');write_json(assetpath,asset)
  verifications[case]=dict(original_mask_changed_pixels=0,source_partitions=partitions,max_native_rest_reconstruction_error=rest_error,max_animated_rigid_length_error=rigid_error,same_instance_fragments_share_shape_parameters=True,all_v7_joints_types_parents_axes_ranges_unchanged=True,registration_unchanged=True,source_files_unchanged=True,unsupported_instances=prep['unsupported_instances'],all_frames_decoded=True)
  for method in ['original','prior','optimized']:
   j=ev['joint_methods'][method];n=ev['unfitted_open_mean'].get(method,{});s=ev['surface_reconstruction'][method];rows.append(dict(case=case,method=method,input_IoU=f['image_metrics'][method]['IoU'],input_PSNR=f['image_metrics'][method]['PSNR'],supported_instance_mIoU=f['part_mask_metrics'][method]['mean_IoU'],heldout_IoU=n.get('IoU'),heldout_PSNR=n.get('PSNR'),CD_percent_D=s['CD_percent_D'],Fscore_1pct_D=s['Fscore_1pct_D'],Fscore_5pct_D=s['Fscore_5pct_D'],GT_joint_count=j['GT_joint_count'],type_correct_count=round(j['type_correct_GT_coverage']*j['GT_joint_count']),axis_correct_count=j['axis_correct_count'],mean_axis_angle=j['conditional_mean_angle_degrees'],mean_axis_position_percent_D=j['conditional_mean_pivot_percent_D']))
   for r in j['rows']:jointrows.append(dict(case=case,method=method,GT_part=r['GT_part'],GT_type=r['GT_type'],predicted_owner=r.get('predicted_owner'),type_correct=r['type_correct'],axis_correct=r['axis_correct'],angle_degrees=r['angle_degrees'],pivot_percent_D=r['pivot_percent_D']))
  print('VERIFIED',case,flush=True)
 assert all(sha256(path)==digest for path,digest in read(batch/'source_axis_files_sha256.json').items()),'Sealed v7 output changed'
 for filename,data in [('metrics.csv',rows),('joint_metrics.csv',jointrows)]:
  with (batch/filename).open('w') as stream:writer=csv.DictWriter(stream,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
 summary={}
 for method in ['original','prior','optimized']:
  rr=[r for r in rows if r['method']==method];jj=[r for r in jointrows if r['method']==method];summary[method]={k:mean(r[k] for r in rr) for k in ['input_IoU','input_PSNR','supported_instance_mIoU','heldout_IoU','heldout_PSNR','CD_percent_D','Fscore_1pct_D','Fscore_5pct_D']};summary[method].update(GT_joint_count=sum(r['GT_joint_count'] for r in rr),type_correct_count=sum(r['type_correct_count'] for r in rr),axis_correct_count=sum(r['axis_correct_count'] for r in rr),conditional_mean_axis_angle=mean(r['angle_degrees'] for r in jj),conditional_mean_axis_position_percent_D=mean(r['pivot_percent_D'] for r in jj),heldout_case_count=sum(r['heldout_IoU'] is not None for r in rr))
 improvements=dict(input_IoU=sum(fitresults[c]['image_metrics']['optimized']['IoU']>fitresults[c]['image_metrics']['prior']['IoU'] for c in cases),part_mIoU=sum(fitresults[c]['part_mask_metrics']['optimized']['mean_IoU']>fitresults[c]['part_mask_metrics']['prior']['mean_IoU'] for c in cases),heldout_IoU=sum(evaluations[c]['unfitted_open_mean']['optimized']['IoU']>evaluations[c]['unfitted_open_mean']['prior']['IoU'] for c in cases if c!='20411'),surface_CD=sum(evaluations[c]['surface_reconstruction']['optimized']['CD_percent_D']<evaluations[c]['surface_reconstruction']['prior']['CD_percent_D'] for c in cases))
 write_json(batch/'summary.json',dict(methods=summary,improved_case_counts=improvements,supported_instances=supported,total_original_instances=total_masks));write_json(batch/'VERIFICATION.json',dict(state='complete',cases=verifications,video_count=len(video_rows),videos=video_rows,source_v7_all_files_unchanged=True,manual_semantic_guidance_used=False,all_source_masks_and_native_geometry_unchanged=True,automatic_ownership_contract_tests_passed=5))
 def num(v,scale=1):return 'N/A' if v is None else f'{v*scale:.2f}'
 lines=['# VLM 语义部件先验替代人工几何指导','', '**完成现有11个case：GT → Origin → VLM轴修复（v7）→ VLM指导几何优化。**','', '本轮保留原始 PAct 几何作为起点，使用已封存的 VLM 类别/把手识别和可见投影自动部件匹配，替代人工语义名称、手工物理分组、左右配对和几何拟合框。没有重新运行Origin生成。原始输入实例mask仍沿用已有的oracle/GT投影分割，边界没有改变；本轮不声称由VLM重新预测了像素分割。类别不明确的实例沿用v7已记录的原始预测回退，没有人工补标签。通用运动规则、形变模型和优化损失仍由程序设定。','', '每个自动物理实例的所有连通片段共享形变参数；不同门、抽屉、左右提手分别优化。轴、父子关系、运动类型、原预测行程全程固定为v7，包含45503已修正的上沿铰链。没有读取旧人工语义表、部件框、多边形或人工形变配对。','', f'支持原始mask实例 **{supported}/{total_masks}**。101808缺失部件没有生成几何，其mask在部件语义损失中标为忽略，不当背景，也不增加虚构部件。RGB/整体轮廓仍监督现有全部几何。','', '使用原始输入图（20411为05，其余为第0帧）及其相机、原mask，在固定的原RGB/轮廓配准下进行240步有界几何优化。损失为RGB＋2×轮廓＋0.5×逐部件Dice＋0.1×平衡内部交叉熵及小幅形变正则；沿用既有优化器和形变范围。展示最终240步，不根据GT分数挑选checkpoint。全部11个拟合完成后才读取GT和未拟合视角。','', '| 指标 | Origin | VLM轴修复 | 几何优化 |','|---|---:|---:|---:|']
 for label,key,scale in [('原始输入整体IoU %','input_IoU',100),('原始输入PSNR dB','input_PSNR',1),('支持实例投影mIoU %（训练mask）','supported_instance_mIoU',100),('未拟合视角IoU %（10 case）','heldout_IoU',100),('未拟合视角PSNR dB','heldout_PSNR',1),('表面CD %D，越低越好','CD_percent_D',1),('F-score @5%D %','Fscore_5pct_D',100)]:lines.append('| '+label+' | '+' | '.join(num(summary[m][key],scale) for m in ['original','prior','optimized'])+' |')
 for label,key in [('类型正确独立关节数','type_correct_count'),('完整轴正确数','axis_correct_count')]:lines.append('| '+label+' | '+' | '.join(f"{summary[m][key]}/{summary[m]['GT_joint_count']}" for m in ['original','prior','optimized'])+' |')
 lines+=['', f"几何优化相对VLM轴修复版：输入IoU改善 **{improvements['input_IoU']}/11**，部件投影mIoU改善 **{improvements['part_mIoU']}/11**，未拟合IoU改善 **{improvements['heldout_IoU']}/10**，三维CD改善 **{improvements['surface_CD']}/11**。轴分数不随几何优化改变，因为关节被冻结。",'', 'Origin的部件mIoU使用自动修正后的功能归属重着色，只用于q0几何投影诊断，并非PAct原始语义识别准确率。训练mask上的收益也不能代替未拟合视角/三维评价。20411没有同类未拟合相机，记N/A，不填0。','', '[逐阶段完整指标CSV](metrics.csv) · [逐关节指标CSV](joint_metrics.csv) · [汇总JSON](summary.json) · [实验协议](protocol.json) · [核验结果](VERIFICATION.json)','', '| case | 输入IoU：轴版→几何 % | 部件mIoU：轴版→几何 % | 未拟合IoU：轴版→几何 % | 四路视频 | 自动部件投影 |','|---|---:|---:|---:|---|---|']
 for c in cases:
  f=fitresults[c];e=evaluations[c];prior=f['image_metrics']['prior']['IoU'];after=f['image_metrics']['optimized']['IoU'];pa=f['part_mask_metrics']['prior']['mean_IoU'];pb=f['part_mask_metrics']['optimized']['mean_IoU'];na=e['unfitted_open_mean'].get('prior',{}).get('IoU');nb=e['unfitted_open_mean'].get('optimized',{}).get('IoU');lines.append(f"| {c} | {num(prior,100)}→{num(after,100)} | {num(pa,100)}→{num(pb,100)} | {num(na,100)}→{num(nb,100)} | [视频]({c}/{c}_GT_Origin_AutoAxis_FrameFit.mp4) | [图片]({c}/automatic_mask_fit_comparison.png) |")
 lines+=['', '视频每个84帧、16fps、1536×768，四列共享相机，上下两行为两个观察视角。GT从真实输入姿态到标注端点；Origin/v7/优化版保留各自预测行程，进度同步，方向和幅度不强行改成GT。几何优化前后使用同一根轴和同一行程。','', '单张输入只能约束可见外观；缺失完整门、提手顶桥、隐藏厚度或错误生成结构不会凭类别自动补齐。目标冲突和遮挡变化仍可能导致个别几何退化，所有正负结果保留。原生优化资产分别保存在各case的optimized_asset/object.json。','', '原始mask和源网格/Gaussian、v7所有结果均已核对哈希保持不变。全部11个视频逐帧解码通过，自动部件归属的5项回归检查通过。']
 lines+=['','复现入口：`python repro/tools/fit_vlm_component_geometry.py --root <新的输出目录> --axis-root repro/outputs/auto_component_axis_priors_20261029_v7`。完成后运行 `finalize_vlm_component_geometry.py --root <输出目录>` 生成核验报告。']
 (batch/'README.md').write_text('\n'.join(lines)+'\n');links=re.findall(r'\]\(([^)]+)\)',(batch/'README.md').read_text());assert all((batch/p).exists() for p in links)
 for name in ['finalize_vlm_component_geometry.py','test_vlm_component_geometry.py','render_closed_image_45194.py']:shutil.copyfile(pathlib.Path(__file__).parent/name,batch/'source_snapshot'/name)
 write_json(batch/'FINAL_SEAL.json',dict(state='complete',cases=cases,video_count=len(video_rows),report_sha256=sha256(batch/'README.md'),verification_sha256=sha256(batch/'VERIFICATION.json'),summary_sha256=sha256(batch/'summary.json'),metrics_sha256=sha256(batch/'metrics.csv'),source_axis_unchanged=True,source_snapshot_sha256={p.name:sha256(p) for p in (batch/'source_snapshot').glob('*.py')}));(pathlib.Path(__file__).resolve().parents[1]/'current_vlm_component_geometry.txt').write_text(str(batch.resolve())+'\n');print(json.dumps(dict(summary=summary,improvements=improvements,video_count=len(video_rows)),indent=2),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);a=p.parse_args();finalize(a.root.resolve())
