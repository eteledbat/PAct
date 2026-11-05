"""Aggregate, compare, and verify the released-pipeline multicase replay."""
import argparse,csv,datetime,json,pathlib,re,shutil,subprocess,sys
import imageio.v2 as imageio,numpy as np
from PIL import Image,ImageDraw,ImageFont
from fit_observed_drawer_pose import read,REPO
from collision_refinement_common import sha256
from multicase_closed_common import write_json

LABELS={'origin':'Origin','vlm_axis':'VLM轴修正，无几何','vlm_pose':'VLM轴＋观测位移，无几何','vlm_frozen_pose_geometry':'VLM轴＋几何，固定q0','vlm_pose_geometry':'VLM轴＋观测位移＋几何'}
MAIN=['origin','vlm_axis','vlm_pose_geometry']
def mean(v):return float(np.mean(v)) if len(v) else None
def finalize(batch):
 protocol=read(batch/'protocol.json');cases=protocol['cases'];assert len(cases)==10
 codeproof=read(batch/'published_code_verification.json');assert codeproof['all_published_tools_byte_identical'];assert all(sha256(REPO/p)==h for p,h in codeproof['files_sha256'].items());assert read(batch/'cached_stage_provenance.json')['all_semantics_masks_axes_match_sealed_sources'];assert all(r['RGB_pixels_identical_to_original'] for r in read(batch/'origin_input_audit.json'));protected=read(batch/'protected_source_files.json');assert all(sha256(p)==h for p,h in protected.items())
 historical=read(batch/'protected_historical_files.json');assert all(sha256(p)==h for p,h in historical.items())
 rows=[];checks=[];jointrows=[];poserows=[];case_data={};overview=[]
 for case in cases:
  root=batch/case;r=root/'vlm';e=read(root/'evaluation.json');man=read(r/'motion_groups.json');seal=read(r/'fit_seal.json');prep=read(r/'prepared.json');pose=read(r/'pose_estimation.json');video=read(root/'video_results.json');assert e['GT_only_after_all_fit_seals'] and not seal['GT_forbidden_reads'];assert (root/'evaluation.json').stat().st_mtime_ns>=(batch/'ALL_FITS_SEAL.json').stat().st_mtime_ns;assert video['all_frames_decoded']
  assert sha256(r/'fit_seal.json')==read(batch/'ALL_FITS_SEAL.json')['seals'][case];assert sha256(r/'motion_groups.json')==seal['source_manifest_sha256'];assert sha256(r/'observation.npz')==seal['source_observation_sha256'];assert sha256(r/'original_instance_ids.npy')==prep['mask_sha256'];assert sha256(r/'registration.pt')==prep['common_registration_sha256'];assert all(sha256(p['path'])==p['sha256'] for p in man['geometry_files']);assert {str(g['id']):dict(parent=g['parent'],joint=g['joint']) for g in man['groups']}==man['source_kinematics'];assert all(g.get('box') is None for g in man['groups']);assert not man['pairs']
  for name in ['frozen_pose_geometry','observed_pose_geometry']:
   result=read(r/(name+'_results.json'));assert sha256(r/(name+'.pt'))==result['checkpoint_sha256'];assert len(read(r/(name+'_history.json')))==240;assert result['selected_training_loss']<=result['last_training_loss']+1e-6
  for gid,q in pose['observed_pose'].items():
   g=next(g for g in man['groups'] if str(g['id'])==gid);assert g['joint']['type']=='prismatic';lo,hi=sorted(g['joint']['range']);assert lo-1e-8<=q<=hi+1e-8;poserows.append(dict(case=case,group=gid,semantic_instance=g.get('semantic_owner'),observed_q=q,native_range_lower=lo,native_range_upper=hi,mechanical_range_optimized=False))
  export=read(r/'optimized_asset/object.json');export_nodes=export['diffuse_tree'];assert all(next(n for n in export_nodes if n['id']==g['id'])['joint']==g['joint'] for g in man['groups'])
  for event in seal['training_reads']:
   assert not any(s in event for s in ['/gt/','/gt_mesh/','joint_evaluation','joint_info','/depth/','/closed_multicase_'])
   if '/FreeArt3D/' in event and event.endswith('_pure.png'):assert event.endswith('/05_pure.png')
   if '/images/' in event and event.endswith('.png'):assert event.endswith('/000000.png')
  assert sha256(root/video['video'])==video['sha256']
  case_data[case]=e
  for method,v in e['methods'].items():
   a=v['axis'];h=v['heldout_mean'];traj=e['native_range_six_state_surface'].get(method,[])
   rows.append(dict(case=case,method=method,input_IoU_percent=100*v['image']['IoU'],instance_mIoU_percent=100*v['parts']['mean_IoU'],input_PSNR=v['image']['PSNR'],input_pose_CD_percent_D=v['surface_at_fitted_input_pose']['CD_percent_D'],input_pose_Fscore_5pct_D=v['surface_at_fitted_input_pose']['Fscore_5pct_D'],heldout_IoU_percent=100*h['IoU'] if h else None,heldout_PSNR=h['PSNR'] if h else None,GT_joint_count=a['GT_joint_count'],type_correct_count=sum(x['type_correct'] for x in a['rows']),axis_correct_count=a['axis_correct_count'],axis_mean_angle_degrees=a['conditional_mean_angle_degrees'],axis_mean_pivot_percent_D=a['conditional_mean_pivot_percent_D'],six_state_CD_percent_D=mean([x['CD_percent_D'] for x in traj]),observed_pose=json.dumps(v['observed_pose'])))
   if method in MAIN:
    for j in a['rows']:jointrows.append(dict(case=case,method=method,**j))
  checks.append(dict(case=case,original_masks_unchanged=True,native_geometry_sources_unchanged=True,kinematics_and_mechanical_ranges_unchanged=True,only_original_input_RGB_used=True,GT_forbidden_reads=[],updates240_per_ablation=True,checkpoint_selected_by_training_only=True,video_all84_frames_decoded=video['all_frames_decoded']))
  overview.append(Image.open(root/'four_versions_t0.50.png').convert('RGB').resize((768,384)))
 def csvwrite(path,data):
  keys=list(dict.fromkeys(k for row in data for k in row))
  with path.open('w') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(data)
 csvwrite(batch/'metrics.csv',rows);csvwrite(batch/'joint_metrics.csv',jointrows);csvwrite(batch/'observed_pose.csv',poserows)
 aggregates={}
 for method in LABELS:
  rs=[x for x in rows if x['method']==method];agg={k:mean([x[k] for x in rs if x[k] is not None]) for k in ['input_IoU_percent','instance_mIoU_percent','input_PSNR','input_pose_CD_percent_D','input_pose_Fscore_5pct_D','heldout_IoU_percent','heldout_PSNR','six_state_CD_percent_D']};agg.update(case_count=len(rs),heldout_case_count=sum(x['heldout_IoU_percent'] is not None for x in rs),GT_joint_count=sum(x['GT_joint_count'] for x in rs),type_correct_count=sum(x['type_correct_count'] for x in rs),axis_correct_count=sum(x['axis_correct_count'] for x in rs));agg['type_correct_rate']=agg['type_correct_count']/agg['GT_joint_count'];agg['axis_correct_rate']=agg['axis_correct_count']/agg['GT_joint_count'];aggregates[method]=agg
 improvements={}
 for key,higher in [('input_IoU_percent',True),('instance_mIoU_percent',True),('input_pose_CD_percent_D',False),('heldout_IoU_percent',True),('six_state_CD_percent_D',False)]:
  base={x['case']:x[key] for x in rows if x['method']=='vlm_axis' and x[key] is not None};full={x['case']:x[key] for x in rows if x['method']=='vlm_pose_geometry' and x[key] is not None};delta={c:full[c]-base[c] for c in base};improvements[key]=dict(improved=sum(d>1e-8 if higher else d<-1e-8 for d in delta.values()),count=len(delta),regressed_cases=[c for c,d in delta.items() if (d<-1e-8 if higher else d>1e-8)],case_deltas=delta)
 oldroot=REPO/'repro/outputs/closed_multicase_20261028_v1';old=read(oldroot/'aggregate.json');write_json(batch/'old_closed_image_aggregate_reference.json',old)
 summary=dict(protocol=protocol,methods=aggregates,improvements_vs_axes_only=improvements,per_case=rows,historical_closed_image_comparison='Historical version used extra closed RGB plus manual rectangles and pairing. Not equal-supervision training. Do not compare its closed-fit IoU directly with new original-input IoU. Same-state held-out views below provide a descriptive reference only. Same axes threshold, potentially different frozen physical correspondences.',old_closed_image_aggregate=old)
 write_json(batch/'summary.json',summary)
 # Same source-state heldout views permit a descriptive, explicitly unequal-supervision old/new comparison.
 comparisons=[]
 for case in cases:
  oe=read(oldroot/case/'evaluation.json');ne=case_data[case];ov=oe.get('unfitted_open_mean',{}).get('optimized');nv=ne['methods']['vlm_pose_geometry']['heldout_mean'];oa=oe['joint_methods']['optimized'];na=ne['methods']['vlm_pose_geometry']['axis'];comparisons.append(dict(case=case,old_extra_closed_image_heldout_IoU_percent=100*ov['IoU'] if ov else None,new_single_input_heldout_IoU_percent=100*nv['IoU'] if nv else None,old_axis_correct_count=oa['axis_correct_count'],new_axis_correct_count=na['axis_correct_count'],GT_joint_count=na['GT_joint_count']))
 csvwrite(batch/'historical_comparison.csv',comparisons)
 # Compact contact sheet with case labels and the four video columns.
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18);sheet=Image.new('RGB',(1536,5*412),'white');d=ImageDraw.Draw(sheet)
 for i,(c,im) in enumerate(zip(cases,overview)):
  x=(i%2)*768;y=(i//2)*412;d.text((x+8,y+4),c,font=font,fill='black');sheet.paste(im,(x,y+28))
 sheet.save(batch/'all_cases_mid_motion.jpg',quality=92)
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,3,figsize=(15,4));colors=['#888888','#559ac7','#dd8844']
 for ax,(metric,title) in zip(axes,[('input_IoU_percent','Input silhouette IoU (%)'),('input_pose_CD_percent_D','Input-pose surface CD (%D, lower)'),('heldout_IoU_percent','Held-out view IoU (%) / 9 cases')]):
  ax.bar(['Origin','VLM axes','VLM pose+geometry'],[aggregates[m][metric] for m in MAIN],color=colors);ax.set_title(title);ax.tick_params(axis='x',rotation=15)
 fig.tight_layout();fig.savefig(batch/'metrics_overview.png',dpi=180);fig.savefig(batch/'metrics_overview.pdf');plt.close(fig)
 text='''# GitHub v0.1.0：10-case 批量验证

四列视频顺序：**GT → Origin → VLM 轴修正（不优化几何）→ VLM 轴修正＋观测位移估计＋几何优化**。

使用提交 `d0baa9b0f6bf589a0bff1c760954c5b464c6cd38` / tag `v0.1.0-auto-semantics-pose`。已核验该提交所有已跟踪工具与本地源码逐字节一致。新增脚本仅负责批量调度、独立评价、四列展示和报告；没有更改发布版拟合函数。未加入后续行程优化。

本轮从已封存 VLM 语义、把手证据和 v7 轴结果继续，从同一 Origin 原生几何重新执行拟合；没有重新运行随机 VLM/PAct 推理，也没有复用旧的已优化几何。只有 VLM 分支，不新增人工语义或手选区域。

## 结果

10-case 宏平均；新视角为有新相机的9-case宏平均。类型和完整轴指标按GT关节数微平均。IoU为前景轮廓IoU，非实体体积IoU。

| 方法 | 输入IoU % ↑ | 部件mIoU % ↑ | 输入姿态CD %D ↓ | 新视角IoU % ↑ | 类型正确 | 完整轴正确 |
|---|---:|---:|---:|---:|---:|---:|
'''
 for k in LABELS:
  a=aggregates[k];text+=f"| {LABELS[k]} | {a['input_IoU_percent']:.2f} | {a['instance_mIoU_percent']:.2f} | {a['input_pose_CD_percent_D']:.3f} | {a['heldout_IoU_percent']:.2f} | {a['type_correct_count']}/{a['GT_joint_count']} | {a['axis_correct_count']}/{a['GT_joint_count']} |\n"
 a=aggregates['origin'];b=aggregates['vlm_pose_geometry'];text+=f"\n**主要结论：二维轮廓/部件拟合改善、轴修复有效，但三维没有稳定改善。** 输入姿态CD仅{a['input_pose_CD_percent_D']:.3f}→{b['input_pose_CD_percent_D']:.3f}%D，只有4/10case改善；F-score@5%D从{100*a['input_pose_Fscore_5pct_D']:.2f}%降到{100*b['input_pose_Fscore_5pct_D']:.2f}%。完整运动序列CD从{a['six_state_CD_percent_D']:.3f}上升到{b['six_state_CD_percent_D']:.3f}%D。新视角PSNR也从{a['heldout_PSNR']:.3f}降到{b['heldout_PSNR']:.3f}dB，不能把训练mask提升解释为全面提升。\n"
 text+='\n完整轴标准：独立部件且类型正确，轴方向误差≤5°，旋转轴线位置误差≤GT固定体对角线5%；平移轴原点不评价。几何阶段冻结轴，所以纯轴正确率相同。部件mask指标在相同自动归属上计算，不代表Origin自己输出了类别文字。\n\n'
 text+='| Case | 输入IoU：轴→几何 % | 输入姿态CD：轴→几何 %D | 新视角IoU：轴→几何 % | 轴正确/GT | 视频 / 输入姿态 |\n|---|---:|---:|---:|---:|---|\n'
 for c in cases:
  pair={x['method']:x for x in rows if x['case']==c};a=pair['vlm_axis'];b=pair['vlm_pose_geometry'];h='无新相机' if a['heldout_IoU_percent'] is None else f"{a['heldout_IoU_percent']:.2f} → {b['heldout_IoU_percent']:.2f}";text+=f"| {c} | {a['input_IoU_percent']:.2f} → {b['input_IoU_percent']:.2f} | {a['input_pose_CD_percent_D']:.3f} → {b['input_pose_CD_percent_D']:.3f} | {h} | {b['axis_correct_count']}/{b['GT_joint_count']} | [四列视频]({c}/{c}_GT_Origin_VLMAxis_VLMPoseGeo.mp4) · [输入姿态]({c}/input_pose_comparison.png) · [位移消融]({c}/pose_ablation.png) |\n"
 text+='\n几何相对仅修正轴的改善计数：\n\n'
 for k,v in improvements.items():text+=f"- {k}：{v['improved']}/{v['count']} 改善；退化case：{', '.join(v['regressed_cases']) or '无'}。\n"
 text+='\n六个均匀进度状态的全物体表面CD（GT与预测保留各自区间，包含行程与运动误差）宏平均：'+ '；'.join(f"{LABELS[k]} {aggregates[k]['six_state_CD_percent_D']:.3f}%D" for k in MAIN)+'。此项是诊断，不强制GT与预测在同一角度或位移。\n'
 text+='''
## 输入、优化和评价边界

- 拟合只用原始单张RGB、原实例mask和已知相机；mask来自已有GT投影的oracle 2D分割。准确表述为“单张RGB＋给定实例mask”，不是纯RGB自动分割。
- Origin是现有原版PAct输出，输入RGBA和实例编号，没有VLM类别文字。mask边界完全不变。101808缺失几何对应的mask区域按发布版忽略，不补造部件。
- VLM语义/物理归属指导通用规则轴；上/下翻门先看把手位置，再回退柜体位置。原始来源和规则输出均保留。
- 原生几何固定时，对**全部已有prismatic组**进行原预测区间内49点＋局部有界搜索、2轮坐标搜索，包括不确定碎片；不是后来行程实验仅处理语义主抽屉的筛选。revolute观测角仍固定0。
- 冻结观测q，再优化几何240步Adam，保存训练目标最小状态。并行报告固定q0几何消融，采用相同训练最优选择政策。该政策与此前固定最终240步的11-case版本不同。
- 轴、父子关系、机械行程全部保持不变。观测q是输入照片的当前位移，不是机械限位；没有行程优化。
- 所有10case拟合先封存，GT仅用于之后评价和视频。GT网格/轴/额外关闭RGB/深度/其他相机不参与拟合。使用固定原生body对齐，不按候选重新对齐或按GT挑checkpoint。
- 输入IoU、部件mIoU是训练图诊断；表面CD使用16000点、双向非平方均值 / 固定GTbody对角线D，包含姿态和形状误差。
- 新视角使用原状态0的相机25/50/75/100/125/149，复用输入估计q，无新视角姿态/相机重拟合。20411没有同类新相机，未填指标。
- 轴指标沿用拟合前已封存的类型/轴误差无关几何匹配。几何优化不改变轴数值；不把形状覆盖率当作轴位置精度。

## 视频怎样看

每个视频84帧、16fps、1536×768，两行为同一组共享相机。20411 GT真实00关闭→05打开；其余GT原输入状态→较远标注端点。三个预测列保留各自发布版的完整数值行程，未用GT行程替换。共同进度不等于同一物理角度/位移。

观测q只用于确定几何拟合时的输入状态，不烘焙进canonical几何或机械区间。输入姿态静态图单独显示最终模型在q_obs下的效果。第三列仅修正轴、观测保持q0；另提供“VLM轴＋观测位移但无几何”数值消融和图片。

## 与旧closed_multicase版本的关系

旧目录使用额外关闭RGB、人工矩形框和配对约束，当前版只用原输入RGB、原mask和自动语义，监督条件不同。旧版关闭训练图IoU不能直接与当前原输入IoU比较。`historical_comparison.csv`给出同原状态新相机IoU和严格轴指标的描述性对照；不能据此把差异全部归因于VLM语义。这组历史对照的相机相同，但各版本固定配准也不同，因此还包含配准差异。旧结果、原生资产和mask均不覆盖。

[完整指标CSV](metrics.csv) · [关节指标](joint_metrics.csv) · [观测位移](observed_pose.csv) · [汇总JSON](summary.json) · [旧版描述性对照](historical_comparison.csv) · [中间运动总览](all_cases_mid_motion.jpg) · [指标图](metrics_overview.png) · [实验协议](protocol.json) · [源码版本核对](published_code_verification.json) · [缓存阶段来源](cached_stage_provenance.json) · [Origin输入核对](origin_input_audit.json) · [核验结果](VERIFICATION.json)
'''
 (batch/'REPORT.md').write_text(text)
 write_json(batch/'VERIFICATION.json',dict(state='complete',cases=cases,checks=checks,published_source_unmodified=True,source_file_count=len(protected),all_protected_sources_unchanged=True,historical_file_count=len(historical),all_historical_files_unchanged=True,all_fits_sealed_before_GT=True,GT_not_used_for_checkpoint_or_rule_selection=True,mechanical_ranges_not_optimized=True))
 for target in re.findall(r'\]\(([^)]+)\)',text):assert (batch/target).exists(),target
 for f in ['replay_github_v010_multicase.py','evaluate_github_v010_multicase.py','render_github_v010_multicase.py','finalize_github_v010_multicase.py','complete_github_v010_multicase.py']:(batch/'source_snapshot'/f).write_bytes((REPO/'repro/tools'/f).read_bytes())
 write_json(batch/'FINAL_SEAL.json',dict(state='complete',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),excluded_mutable_files=['*.log','postprocess_status.json','FINAL_SEAL.json'],files_sha256={str(p.relative_to(batch)):sha256(p) for p in batch.rglob('*') if p.is_file() and p.suffix!='.log' and p.name not in ['FINAL_SEAL.json','postprocess_status.json']}));(REPO/'repro/current_github_v010_multicase.txt').write_text(str(batch)+'\n');print(json.dumps(aggregates,indent=2),flush=True);print('COMPLETE',batch,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);a=p.parse_args();finalize(a.root.resolve())
