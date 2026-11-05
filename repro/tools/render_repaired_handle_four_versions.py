"""Fresh geometry fit on attachment-repaired handle axes, then four-column video."""
import argparse,copy,datetime,json,os,pathlib,shutil,subprocess,sys
import numpy as np
from collision_refinement_common import sha256
from multicase_closed_common import REPO,write_json
from fit_observed_drawer_pose import read

CASE='100481'
OLD=REPO/'repro/outputs/github_v010_multicase_20261030_v1'
AXES=REPO/'repro/outputs/handle_attachment_axis_20261030_v1'
AUTO=REPO/'repro/outputs/vlm_component_geometry_20261029_v1'

def initialize(batch):
 batch.mkdir(exist_ok=False);assert read(AXES/'FINAL_SEAL.json')['state']=='complete'
 write_json(batch/'protocol.json',dict(cases=[CASE],created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),versions=['GT','Origin','VLM repaired handle axes only','VLM repaired axes + fresh geometry'],axis_source=str(AXES/CASE),training='Original single conditioning RGB, exact original instance mask, frozen original RGB/silhouette camera registration and native PAct geometry only.',geometry='Published fit_observed_drawer_pose.fit;240-step Adam, training-minimum checkpoint, q0 ablation and observed-q branch. No revolute observed angle or mechanical range optimization.',GT_policy='Only read GT alignment and meshes after fit_seal exists, for video display.',old_results='Preserve previous ten-case batch and attachment-axis diagnostic outputs.'))
 protected={str(p):sha256(p) for src in [OLD/CASE,AXES] for p in src.rglob('*') if p.is_file()};write_json(batch/'protected_sources.json',protected)
 snap=batch/'source_snapshot';snap.mkdir()
 for name in ['render_repaired_handle_four_versions.py','fit_observed_drawer_pose.py','fit_vlm_component_geometry.py','render_github_v010_multicase.py','semantic_motion_rules.py','auto_component_axis_prior.py']:
  shutil.copyfile(REPO/'repro/tools'/name,snap/name)

def prepare(batch):
 import fit_observed_drawer_pose as fit
 source=AUTO/CASE;corrected=read(AXES/CASE/'motion_groups.json');man=copy.deepcopy(read(source/'motion_groups.json'));byid={g['id']:g for g in corrected['groups']};assert set(byid)=={g['id'] for g in man['groups']};changes=[]
 staged=batch/'axis_prior'/CASE;staged.mkdir(parents=True,exist_ok=False)
 for g in man['groups']:
  new=byid[g['id']];assert g['parent']==new['parent'] and g['source_node']==new['source_node'] and g['semantic_owner']==new['semantic_owner'];assert g['joint']['range']==new['joint']['range'];assert g['joint']['type']==new['joint']['type']
  if g['joint']!=new['joint']:
   assert g['semantic_owner'] in [2,3];changes.append(dict(group=g['id'],semantic_instance=g['semantic_owner'],old=copy.deepcopy(g['joint']),new=copy.deepcopy(new['joint'])))
  g['joint']=copy.deepcopy(new['joint'])
 assert len(changes)==2
 for name in ['config.json','membership.npz','original_instance_ids.npy','observation.npz','registration.pt']:shutil.copyfile(source/name,staged/name)
 assert sha256(staged/'membership.npz')==corrected['membership_sha256']
 man.update(source_axis_root=str(AXES/CASE),source_axis_manifest_sha256=sha256(AXES/CASE/'motion_groups.json'),source_axis_membership_sha256=sha256(AXES/CASE/'membership.npz'),axis_repair='Body-facing support endpoints replace support-bar centers; no shape/range change')
 write_json(staged/'motion_groups.json',man);write_json(batch/'axis_changes.json',changes)
 # Only input routing changes. The published prepare recomputes all physical
 # instance deformation bases from raw geometry around the NEW axes.
 fit.AUTO=batch/'axis_prior';fit.prepare(batch,CASE,'vlm')
 r=batch/CASE/'vlm';prepared=read(r/'prepared.json');prepared.update(corrected_axis_manifest_sha256=sha256(AXES/CASE/'motion_groups.json'),new_geometry_fit_from_native=True,old_optimized_checkpoint_reused=False);write_json(r/'prepared.json',prepared)


def render(batch):
 r=batch/CASE/'vlm';assert read(r/'fit_seal.json')['state']=='complete';write_json(batch/'ALL_FITS_SEAL.json',dict(state='complete',seals={CASE:sha256(r/'fit_seal.json')},GT_not_used_for_fitting=True))
 # Display alignment only, sealed before video and never loaded by fitting.
 alignment=read(REPO/f'repro/outputs/general_semantic_priors_20261029_v2/{CASE}/evaluation.json')['alignment'];write_json(batch/CASE/'config.json',read(r/'config.json'));write_json(batch/CASE/'evaluation.json',dict(case=CASE,alignment=alignment,GT_only_after_fit_seal=True,evaluation_scope='Frozen display alignment only; no new GT metric claims.'))
 import render_github_v010_multicase as video
 video.TITLES=['GT | reference mesh','Origin | original PAct','VLM | repaired handle axes','VLM | repaired axes + geometry'];video.render(batch,CASE)


def finalize(batch):
 root=batch/CASE;r=root/'vlm';protected=read(batch/'protected_sources.json');assert all(sha256(p)==h for p,h in protected.items());seal=read(r/'fit_seal.json');assert not seal['GT_forbidden_reads'];man=read(r/'motion_groups.json');target={str(g['id']):dict(parent=g['parent'],joint=g['joint']) for g in read(AXES/CASE/'motion_groups.json')['groups']};assert man['source_kinematics']==target;assert all(sha256(p['path'])==p['sha256'] for p in man['geometry_files']);assert sha256(r/'original_instance_ids.npy')==sha256(AUTO/CASE/'original_instance_ids.npy');assert sha256(r/'motion_groups.json')==seal['source_manifest_sha256'];assert sha256(r/'observation.npz')==seal['source_observation_sha256']
 video=read(root/'video_results.json');assert video['all_frames_decoded'] and video['frames']==84;assert sha256(root/video['video'])==video['sha256'];assert (root/'evaluation.json').stat().st_mtime_ns>(r/'fit_seal.json').stat().st_mtime_ns
 before=read(OLD/CASE/'vlm/observed_pose_geometry_results.json');after=read(r/'observed_pose_geometry_results.json');write_json(batch/'fit_summary.json',dict(old_axis_geometry=before,new_axis_geometry=after,note='Training-image diagnostics, not new GT surface or axis evaluation.'))
 # Give the same video a descriptive link-friendly filename.
 destination=root/f'{CASE}_GT_Origin_RepairedAxis_RepairedAxisGeometry.mp4';os.link(root/video['video'],destination) if not destination.exists() else None
 text='''# 100481：连接端提手轴修复后的四列视频

顺序：**GT → Origin → 修复后的VLM轴（无几何优化）→ 修复后的VLM轴＋重新几何优化**。两行是两个共享相机。

[修复后的四列视频](100481/100481_GT_Origin_RepairedAxis_RepairedAxisGeometry.mp4) · [中间帧](100481/four_versions_t0.50.png) · [起始帧](100481/four_versions_t0.00.png) · [末帧](100481/four_versions_t1.00.png)

第三列使用原生几何和刚修复的提手连接端轴。第四列从同一原生几何出发，以新轴重新计算形变基、重新运行240步几何拟合，按训练目标最小状态保存；没有沿用旧轴拟合的checkpoint。输入仍是原RGB、原始mask及冻结相机配准，不用GT、额外关闭图或其他视角监督。

100481没有prismatic关节，观测姿态仍为q0。提手的行程幅度与之前完全相同，本轮只更新轴位置并重新拟合几何。84帧、16fps、1536×768，全部帧已解码。GT使用冻结的原body对齐展示，所有拟合封存后才读取；本轮没有新增GT准确率结论。

后侧生成支杆存在缺失，新轴仍是依据连接端的近似估计。共同播放进度不等于GT与预测具有相同的开角；所有预测保留各自原行程。原mask、原生资产、旧批量结果和上轮轴诊断均不覆盖。

[轴变化](axis_changes.json) · [训练图指标](fit_summary.json) · [核验](VERIFICATION.json)
'''
 (batch/'REPORT.md').write_text(text);write_json(batch/'VERIFICATION.json',dict(state='complete',case=CASE,fresh_geometry_fit_on_new_axes=True,axes_and_ranges_frozen_to_corrected_source=True,original_masks_unchanged=True,native_sources_unchanged=True,previous_outputs_unchanged=True,protected_file_count=len(protected),GT_forbidden_fit_reads=[],GT_render_after_fit_seal=True,video_frames=84,all_frames_decoded=True))
 write_json(batch/'FINAL_SEAL.json',dict(state='complete',files_sha256={str(p.relative_to(batch)):sha256(p) for p in batch.rglob('*') if p.is_file() and p.suffix!='.log' and p.name not in ['FINAL_SEAL.json','controller_status.json']}));print('COMPLETE',destination,flush=True)


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--stage',choices=['all','prepare','fit','render','finalize'],default='all');a=p.parse_args();batch=a.root.resolve()
 if a.stage!='all':
  if a.stage=='prepare':prepare(batch)
  elif a.stage=='fit':
   from fit_observed_drawer_pose import fit
   fit(batch/CASE/'vlm')
  elif a.stage=='render':render(batch)
  else:finalize(batch)
  return
 if not batch.exists():initialize(batch)
 states={}
 for stage,marker in [('prepare',CASE+'/vlm/prepared.json'),('fit',CASE+'/vlm/fit_seal.json'),('render',CASE+'/video_results.json'),('finalize','FINAL_SEAL.json')]:
  if (batch/marker).exists():continue
  with (batch/(stage+'.log')).open('w') as log:proc=subprocess.run([sys.executable,__file__,'--root',str(batch),'--stage',stage],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4','OPENCV_IO_ENABLE_OPENEXR':'1'})
  states[stage]='complete' if proc.returncode==0 else 'failed';write_json(batch/'controller_status.json',states);print(stage,states[stage],flush=True)
  if proc.returncode:raise RuntimeError(stage+' failed')
 states['state']='complete';write_json(batch/'controller_status.json',states)
if __name__=='__main__':main()
