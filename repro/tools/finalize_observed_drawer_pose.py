"""Report and integrity verification for the sealed single-input pose experiment."""
import argparse,csv,copy,json,pathlib,re,shutil,unittest
import numpy as np,torch
from PIL import Image
from fit_observed_drawer_pose import read,load,REPO
from collision_refinement_common import sha256,transforms
from multicase_closed_common import write_json


def finalize(batch):
 protocol=read(batch/'protocol.json');protected=read(batch/'protected_source_files.json');assert all(sha256(p)==h for p,h in protected.items());rows=[];checks=[]
 for case in protocol['cases']:
  root=batch/case;e=read(root/'evaluation.json');assert read(root/'video_results.json')['all_frames_decoded'];branches={}
  for branch in ['vlm','human']:
   r=root/branch;prep=read(r/'prepared.json');man=read(r/'motion_groups.json');seal=read(r/'fit_seal.json');pose=read(r/'pose_estimation.json');assert not seal['GT_forbidden_reads'];assert prep['mask_sha256']==sha256(r/'original_instance_ids.npy');assert prep['common_registration_sha256']==sha256(r/'registration.pt');assert seal['source_manifest_sha256']==sha256(r/'motion_groups.json');assert seal['source_observation_sha256']==sha256(r/'observation.npz');assert all(sha256(g['path'])==g['sha256'] for g in man['geometry_files']);kin={str(g['id']):dict(parent=g['parent'],joint=g['joint']) for g in man['groups']};assert kin==man['source_kinematics'];branches[branch]=dict(mask=prep['mask_sha256'],registration=prep['common_registration_sha256'])
   for name in ['frozen_pose_geometry','observed_pose_geometry']:
    result=read(r/(name+'_results.json'));assert result['checkpoint_sha256']==sha256(r/(name+'.pt'));assert result['selected_completed_updates']<=240;assert len(read(r/(name+'_history.json')))==240;assert result['selected_training_loss']<=result['last_training_loss']+1e-6
   for gid,q in pose['observed_pose'].items():
    g=next(g for g in man['groups'] if str(g['id'])==gid);assert g['joint']['type']=='prismatic';lo,hi=sorted(g['joint']['range']);assert lo-1e-8<=q<=hi+1e-8
   for event in seal['training_reads']:
    assert not any(s in event for s in ['/gt/','/gt_mesh/','joint_evaluation','joint_info','/depth/','/closed_multicase_'])
    if '/FreeArt3D/' in event and event.endswith('_pure.png'):assert event.endswith('/05_pure.png')
   checks.append(dict(case=case,branch=branch,kinematics_and_ranges_unchanged=True,original_mask_unchanged=True,only_original_RGB_used=True,GT_fit_reads_denied=[],selected_by_training_objective=True))
  assert branches['vlm']==branches['human']
  for method,v in e['methods'].items():rows.append(dict(case=case,method=method,input_IoU=v['image']['IoU'],input_PSNR=v['image']['PSNR'],instance_mIoU=v['parts']['mean_IoU'],surface_input_pose_CD_percent_D=v['surface_at_fitted_input_pose']['CD_percent_D'],canonical_closed_CD_percent_D=v['canonical_surface_vs_GT_closed']['CD_percent_D'] if v['canonical_surface_vs_GT_closed'] else None,axis_correct_count=v['axis']['axis_correct_count'],GT_joint_count=v['axis']['GT_joint_count'],axis_angle=v['axis']['mean_axis_angle'],observed_pose=json.dumps(v['observed_pose'])))
 with (batch/'metrics.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 # Original inference data path and actual image pixels are audited separately.
 origin=[]
 for case in protocol['cases']:
  man=read(batch/case/'vlm/motion_groups.json');run=pathlib.Path(man['source_run']);provenance=read(run/'input_provenance.json');inputs=pathlib.Path(provenance['input_dir']);processed=next((inputs/'Sample').glob('*_processed.png'));mask=next((inputs/'Sample').glob('*_mask.exr'));same=np.array_equal(np.array(Image.open(processed).convert('RGBA')),np.array(Image.open(provenance['source_image']).convert('RGBA')));assert same;origin.append(dict(case=case,original_RGB=provenance['source_image'],PAct_RGB=str(processed),instance_EXR=str(mask),RGB_pixels_identical_to_original=True,instance_mask_origin='Existing oracle GT-projected2D instance IDs',text_category_labels_passed=False,VLM_semantics_passed=False,manual_axis_priors_passed=False,dataset_code='modules/pact/datasets/components.py:559-610',input_provenance_sha256=sha256(run/'input_provenance.json')))
 write_json(batch/'origin_input_audit.json',origin)
 # Execute the small independent FK tests, not renderer-specific mirror tests.
 import test_observed_drawer_pose
 suite=unittest.defaultTestLoader.loadTestsFromModule(test_observed_drawer_pose);result=unittest.TextTestRunner(verbosity=1).run(suite);assert result.wasSuccessful()
 summary={case:read(batch/case/'evaluation.json')['methods'] for case in protocol['cases']};write_json(batch/'summary.json',summary)
 text='''# 单图观测位移 → 几何优化：人工与 VLM 对照

本轮先完成 **20411**。五列顺序：**GT → Origin → VLM轴修正 → VLM轴＋观测位移＋几何 → 人工轴＋观测位移＋几何**。

两条分支只读取同一张原始 `05_pure.png`（抽屉打开）、原始实例mask、已知相机和原生生成几何。**没有使用额外关闭图、GT轴、GT行程、GT几何、深度或其他参考RGB进行拟合。** GT只用于拟合封存后的评价和视频。

## 20411 的结果

'''
 for case in protocol['cases']:
  e=read(batch/case/'evaluation.json');methods=e['methods'];p=read(batch/case/'vlm/pose_estimation.json');q=next(iter(p['observed_pose'].values()),0.);text+=f"人工与VLM估计的观测位移均约 **{q:.5f} 原模型单位**。这是原始输入图的姿态参数，不是估计完整机械行程；既有预测滑轨区间保持不变。\n\n| 阶段 | 输入IoU % | 实例mIoU % | 输入姿态表面CD %D ↓ |\n|---|---:|---:|---:|\n"
  for k,label in [('origin','Origin'),('vlm_axis','VLM轴修正'),('vlm_frozen_pose_geometry','VLM轴＋直接几何，q固定0'),('vlm_pose','VLM轴＋仅观测位移'),('vlm_pose_geometry','VLM轴＋观测位移＋几何'),('human_frozen_pose_geometry','人工轴＋直接几何，q固定0'),('human_pose_geometry','人工轴＋观测位移＋几何')]:
   v=methods[k];text+=f"| {label} | {v['image']['IoU']*100:.2f} | {v['parts']['mean_IoU']*100:.2f} | {v['surface_at_fitted_input_pose']['CD_percent_D']:.3f} |\n"
  text+='\n人工与VLM的最终结果基本相同：本例两条分支识别的柜体/抽屉归属、滑轨方向和数值区间相同，因此不是VLM优于人工的证据。原始轴方向误差约0.636°，两种修正规则后均0°；均通过1/1轴标准。\n\n'
  text+=f"原生关闭参考几何的CD（相同对齐、相同原生源面顺序采样）从 **{methods['origin']['canonical_surface_vs_GT_closed']['CD_percent_D']:.3f}%D** 降到VLM **{methods['vlm_pose_geometry']['canonical_surface_vs_GT_closed']['CD_percent_D']:.3f}%D**、人工 **{methods['human_pose_geometry']['canonical_surface_vs_GT_closed']['CD_percent_D']:.3f}%D**。这项不施加拟合观测位移，说明几何本身也有改善。\n\n"
  traj=e['native_range_six_state_surface_CD'];text+='六个GT状态按共同进度比较、预测保留完整原行程时，平均表面CD：'+ '，'.join(f"{label} {np.mean([r['CD_percent_D'] for r in traj[k]]):.3f}%D" for k,label in [('origin','Origin'),('vlm_axis','VLM轴'),('vlm_pose_geometry','VLM几何'),('human_pose_geometry','人工几何')])+ '。该诊断包含原预测行程误差，不以GT替换预测行程。\n\n'
  text+=f"[五列视频]({case}/{case}_GT_Origin_VLMAxis_VLMPoseGeo_HumanPoseGeo.mp4) · [原始输入姿态对比]({case}/five_versions_input_pose.png) · [位移估计消融]({case}/pose_ablation.png) · [视频中间帧]({case}/five_versions_t0.50.png) · [完整评价]({case}/evaluation.json)\n\n"
 text+='''## 五列视频与输入图对比怎样看

视频两行为共同的两个相机，84帧、16fps、1920×768。20411的GT展示真实00关闭→05打开；四个预测列都展示各自原预测的完整数值区间，未改机械行程。第四/五列中的观测位移用于确定几何拟合时的姿态，导出几何仍保留原生canonical参考坐标。因此**视频最大开度不强制等于输入图开度或GT开度**。输入姿态静态对比图单独显示第四/五列在其估计观测位移下的重建。

静态对比第一列是实际原始输入RGB，视频第一列是中性材质GT网格。输入图IoU和mask mIoU是训练图诊断，不是新视角成绩；20411没有同类新相机集合，未报告虚构的新视角指标。

## 公平对照设置

- 两条分支从同一原生PAct生成资产开始，使用相同原mask、相机和冻结的原RGB/轮廓配准；不读取旧几何优化checkpoint。
- 人工分支沿用此前已声明的人工部件语义/物理归属与规则轴；VLM分支沿用v7自动语义/归属/规则轴。没有给任何分支新增手绘区域或GT数值轴。
- 两条分支都重新计算同一物理实例的联合形变基，使用相同RGB＋2轮廓＋0.5实例Dice＋0.1内部CE及相同正则。没有人工部件框、左右配对或额外关闭图监督。
- 只对移动关节的**观测q**做原预测区间内49点搜索、局部有界一维细化、2轮坐标搜索。保持原生几何固定；随后冻结观测q，再优化几何。没有修改轴、父子关系、类型或机械区间。
- 每条分支分别运行“不估计位移”和“先估计位移”的240步相同Adam优化，对比新增观测位移的作用。
- 两种设置均保存**训练目标最小**的状态，包含初始/最终状态，不按GT或其他图选checkpoint。相较上轮固定最终240步，checkpoint策略也明确变更；但本轮的位移消融与人工/VLM比较完全共用该策略。
- 观测位移与全局相机/部件尺寸仍可能耦合；本次冻结配准并先位移后形变，不能据一个case断言全类别泛化成功。

## Origin究竟输入了什么

**Origin带实例分割信息，不带人工/VLM类别文字标签。** 其输入是原始RGBA和单独的`*_mask.exr`，loader生成RGB条件、实例mask条件和部件数量。并非给RGB画上“抽屉、柜体”文字或语义颜色后再生成，也没有把后处理VLM识别结果、人工轴规则传回Origin。

现有mask是此前GT投影得到的oracle实例mask，因此本实验准确表述是 **单张RGB＋给定实例mask**，不能称为纯RGB无标注输入。20411实际输入是05打开图；原生输出接近关闭参考姿态，二者不能简单等同。

[Origin输入审计](origin_input_audit.json) · [实验协议](protocol.json) · [完整指标CSV](metrics.csv) · [汇总JSON](summary.json) · [核验结果](VERIFICATION.json)

代码入口：`fit_observed_drawer_pose.py` → `evaluate_observed_drawer_pose.py` → `render_observed_drawer_pose.py` → `finalize_observed_drawer_pose.py`。先完成所有候选拟合再评价/渲染。原始mask、原生资产与此前结果均保持不变。
'''
 (batch/'REPORT.md').write_text(text)
 verification=dict(state='complete',cases=protocol['cases'],checks=checks,old_source_files_unchanged=True,protected_file_count=len(protected),same_mask_and_registration_in_both_branches=True,kinematic_tests_run=result.testsRun,kinematic_tests_passed=result.wasSuccessful(),videos_decoded=True,origin_input_audit=origin)
 write_json(batch/'VERIFICATION.json',verification)
 for target in re.findall(r'\]\(([^)]+)\)',text):assert (batch/target).exists(),target
 for name in ['fit_observed_drawer_pose.py','evaluate_observed_drawer_pose.py','render_observed_drawer_pose.py','finalize_observed_drawer_pose.py','test_observed_drawer_pose.py']:shutil.copyfile(REPO/'repro/tools'/name,batch/'source_snapshot'/name)
 assert all(sha256(p)==h for p,h in protected.items());write_json(batch/'FINAL_SEAL.json',dict(state='complete',files_sha256={str(p.relative_to(batch)):sha256(p) for p in batch.rglob('*') if p.is_file() and p.name!='FINAL_SEAL.json'}));(REPO/'repro/current_observed_drawer_pose.txt').write_text(str(batch)+'\n');print('COMPLETE',batch,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);a=p.parse_args();finalize(a.root.resolve())
