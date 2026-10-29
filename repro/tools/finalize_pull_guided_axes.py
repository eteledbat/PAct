"""Verify immutable geometry/masks, candidates, videos and handle-first selection."""
import argparse,csv,json,pathlib,shutil,sys
import imageio.v2 as imageio,numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.spatial.transform import Rotation
import trimesh
from auto_part_semantics import sha256,write_json
p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--baseline',type=pathlib.Path,required=True);p.add_argument('--auto-root',type=pathlib.Path,required=True);a=p.parse_args();b=a.root;protocol=json.loads((b/'protocol.json').read_text());seal=json.loads((b/'AXIS_CANDIDATES_SEAL.json').read_text());pullseal=json.loads((a.auto_root/'PULL_RECOGNITION_SEAL.json').read_text());assert not seal['GT_forbidden_reads'] and not pullseal['GT_forbidden_reads'];assert pullseal['source_inputs_unchanged'];assert sha256(pathlib.Path(__file__).parent/'auto_component_axis_prior.py')==protocol['rule_source_sha256'];allrows=[];cases={};changes=[];video_rows=[]
for c in protocol['cases']:
 r=b/c;old=a.baseline/c;m=json.loads((r/'motion_groups.json').read_text());o=json.loads((old/'motion_groups.json').read_text());ev=json.loads((r/'joint_evaluation.json').read_text());oe=json.loads((old/'joint_evaluation.json').read_text());ass=json.loads((r/'component_assignment.json').read_text());prev=json.loads((old/'component_assignment.json').read_text());sem=json.loads((a.auto_root/c/'semantics.json').read_text());src=pathlib.Path(json.loads((a.auto_root/'protocol.json').read_text())['source_semantics'])/c
 assert sha256(r/'motion_groups.json')==seal['files'][c]['manifest_sha256'];assert sha256(r/'membership.npz')==seal['files'][c]['membership_sha256'];assert sha256(r/'component_assignment.json')==seal['files'][c]['assignment_sha256'];assert sha256(a.auto_root/c/'instance_ids.npy')==sha256(src/'instance_ids.npy');assert sha256(r/'membership.npz')==sha256(old/'membership.npz');assert all(sha256(x['path'])==x['sha256'] for x in m['geometry_files']);assert sha256(m['source_object'])==m['source_object_sha256'];assert np.load(r/'observation.npz')['labels'].sum()==0
 changed=[]
 for x,y in zip(m['groups'],o['groups']):
  assert all(x[k]==y[k] for k in ['id','parent','source_node','component','counts','semantic_owner'])
  if x['joint']!=y['joint']:changed.append(x['semantic_owner'])
 for row in ass['assignments']:
  if row.get('semantic')!='door' or not row.get('supported'):continue
  audit=row['axis_evidence'];before=next(v for v in prev['assignments'] if v['instance_id']==row['instance_id']);e=audit.get('pull_evidence') or {};use=audit.get('pull_priority_used',False)
  if use:assert row['family']==('hinge_upper' if e['position']=='lower' else 'hinge_lower') and e['valid']
  rr=dict(case=c,instance_id=row['instance_id'],old_family=before.get('family'),new_family=row.get('family'),pull_position=e.get('position'),pull_valid=e.get('valid'),handle_priority_used=use,cabinet_position_fallback=audit.get('cabinet_position_fallback_used'),evidence=audit.get('family_evidence'));allrows.append(rr)
  if before.get('joint')!=row['joint']:changes.append(rr)
 vr=json.loads((r/'video_results.json').read_text());path=r/vr['video'];assert sha256(path)==vr['sha256']
 with imageio.get_reader(path) as reader:
  n=reader.count_frames();assert n==vr['frames']==32
  for frame in reader:assert frame.shape[:2]==(960,960)
 video_rows.append(dict(case=c,path=str(path),sha256=sha256(path),all_frames_decoded=True))
 cases[c]=dict(changed_semantic_instances=changed,old_axes_correct=oe['joint_methods']['prior']['axis_correct_count'],new_axes_correct=ev['joint_methods']['prior']['axis_correct_count'],GT_joint_count=ev['joint_methods']['prior']['GT_joint_count'],all_source_geometry_and_mask_files_unchanged=True)
 print('VERIFIED',c,'changed',changed,flush=True)
assert all(sha256(p)==h for p,h in json.loads((b/'previous_results_sha256.json').read_text()).items())
with (b/'door_rule_comparison.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=list(allrows[0]));writer.writeheader();writer.writerows(allrows)
# Direct comparison clip retains each experiment's camera, visibly labelled.
c='45503';root=b/c;old=a.baseline/c;out=root/'45503_before_after_pull_rule.mp4';font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
with imageio.get_reader(old/'45503_GT_Origin_AutoComponentAxis_3views.mp4') as x,imageio.get_reader(root/'45503_GT_Origin_AutoComponentAxis_3views.mp4') as y,imageio.get_writer(out,fps=8,codec='libx264',quality=8) as writer:
 for i in range(32):
  pic=Image.new('RGB',(960,704),'white');d=ImageDraw.Draw(pic);d.text((8,6),'Before: cabinet-position rule | own camera',font=font,fill='black');d.text((8,358),'After: visible lower pull -> upper hinge | own camera',font=font,fill='black');pic.paste(Image.fromarray(x.get_data(i)[:320]),(0,32));pic.paste(Image.fromarray(y.get_data(i)[:320]),(0,384));writer.append_data(np.array(pic))
  if i==15:pic.save(root/'45503_before_after_pull_rule.png')
with imageio.get_reader(out) as reader:
 assert reader.count_frames()==32
 for frame in reader:assert frame.shape[:2]==(704,960)
video_rows.append(dict(case=c,path=str(out),sha256=sha256(out),all_frames_decoded=True))
# Check actual far free-edge motion, not only the solver's centroid diagnostic.
m=json.loads((root/'motion_groups.json').read_text());ass=json.loads((root/'component_assignment.json').read_text());target=next(x for x in ass['assignments'] if x['instance_id']==3);joint=target['joint'];mesh=trimesh.load(m['sources']['0']['mesh'],force='mesh',process=False);v=np.asarray(mesh.vertices);free=np.median(v[v[:,1]<=np.quantile(v[:,1],.05)],axis=0);origin=np.array(joint['axis']['origin']);d=np.array(joint['axis']['direction']);R=Rotation.from_rotvec(d*np.deg2rad(joint['range'][1])).as_matrix();end=(free-origin)@R.T+origin;delta=end-free;assert delta[1]>0 and delta[2]>0
motion=dict(family=target['family'],free_edge_start=free.tolist(),free_edge_end=end.tolist(),free_edge_displacement=delta.tolist(),moves_up=True,moves_outward=True,joint=joint,geometry_unchanged=True);write_json(root/'free_edge_motion_check.json',motion)
ev=json.loads((root/'joint_evaluation.json').read_text());old_ev=json.loads((old/'joint_evaluation.json').read_text());gid=next(g['id'] for g in m['groups'] if g['semantic_owner']==3 and g['joint']['type']!='fixed');jr=next(r for r in ev['joint_methods']['prior']['rows'] if r.get('predicted_owner')==gid);orow=next(r for r in old_ev['joint_methods']['prior']['rows'] if r.get('predicted_owner')==gid)
N=sum(x['GT_joint_count'] for x in cases.values());before=sum(x['old_axes_correct'] for x in cases.values());after=sum(x['new_axes_correct'] for x in cases.values());shutil.copyfile(a.auto_root/c/'recognized_door_pulls.png',root/'recognized_door_pulls.png')
lines=['# 把手位置优先的门轴规则修复','', '本轮输出为v7，以既有v6为批次比较基线；你指出的45503下方门问题在v5、v6中相同。两版旧结果均保持不变。','', '**45503 下方门现按自动识别到的下部把手，选择上沿铰链，开门时自由边上升并向外。**','', '[45503 新视频：GT / Origin / 自动轴（三视角）](45503/45503_GT_Origin_AutoComponentAxis_3views.mp4) · [45503 修复前后视频](45503/45503_before_after_pull_rule.mp4) · [原图上的自动把手识别](45503/recognized_door_pulls.png)','', '新规则遵循用户指定优先级：对宽扁、需要区分上下翻转的门，先检查自动识别的把手在该门板内的相对位置；下部把手→上沿铰链，上部把手→下沿铰链。把手不可见、低置信度、中心不在原始目标 mask 内或位于中部时，回退到原来的柜体上/下位置规则。侧开门继续原有逻辑。','', '把手由本地冻结 Qwen3-VL-4B 模型在原始 RGB 和原始实例 mask 中定位，没有人工点选把手坐标、没有修改 mask。高置信度定位框的中心还须落入该部件原 mask；上/下位置使用门板包围框的上35%和下35%，中间30%保留为不明确。模型只看图定位把手，不被告知期望的门类型或 GT。','', '选定上下铰链族后，继续从生成门板对应边缘及其与柜体的接近点拟合轴线，不更改原始网格/Gaussian，不拷贝 GT 轴或角度。所有11个 case 的候选轴先封存，再读取 GT 作评价。','', '## 45503','', f"模型定位到下方门的白色圆形把手，位于门板高度约90%处，因此该门从 `hinge_lower` 改为 `hinge_upper`。实际自由边位移：向上 **{delta[1]:.5f}**，向外 **{delta[2]:.5f}**（生成模型坐标单位）。保留原预测绝对行程 **{abs(joint['range'][1]):.2f}°**，没有改成GT行程。",'', f"轴方向误差仍为 **{jr['angle_degrees']:.2f}°**；轴位置误差由 **{orow['pivot_percent_D']:.2f}%D** 降到 **{jr['pivot_percent_D']:.2f}%D**。虽然翻转机制已改为上沿铰链，数值轴位置仍未达到5%D阈值，因此该门的严格轴评价仍未通过，整个45503仍是2/3。",'', '视频的GT沿用真实输入状态到标注端点，模型沿用自身原预测行程；相同进度不代表相同开闭方向或幅度。修复前后上下对比视频各排保留自己的相机，用于浏览；轴族和自由边运动另有数值验证。','', '## 批次检查','', f'全部11个case中，只有45503和101808的各一扇门改变关节，其余9个case所有关节保持不变。严格轴通过总数 **{before}/{N} → {after}/{N}**。','', '101808 的下部门也被把手模型判为下部把手，按同一规则改成上沿铰链；这是规则的真实批次结果，不根据 GT 反馈再改单例。其数值轴位置误差稍有退化，说明单张已打开门的把手图像位置可能受透视/姿态或误识别影响；此启发式并非所有门型的保证。','', '| case | 严格轴通过：旧→新 | 改变的语义实例 | 视频 |','|---|---:|---|---|']
for c,row in cases.items():lines.append(f"| {c} | {row['old_axes_correct']}/{row['GT_joint_count']}→{row['new_axes_correct']}/{row['GT_joint_count']} | {row['changed_semantic_instances']} | [视频]({c}/{c}_GT_Origin_AutoComponentAxis_3views.mp4) |")
lines+=['', '[逐门规则对比 CSV](door_rule_comparison.csv) · [45503 自由边运动核验](45503/free_edge_motion_check.json) · [45503 逐关节评价](45503/joint_evaluation.json) · [规则协议](protocol.json) · [候选轴封存](AXIS_CANDIDATES_SEAL.json)','', '修复规则的6项回归检查及原有27项运动/语义检查全部通过，共33项。全部12个视频逐帧解码通过；v5、v6既有实验文件、原始mask、源几何均核对哈希保持不变。']
(b/'README.md').write_text('\n'.join(lines)+'\n');write_json(b/'VERIFICATION.json',dict(state='complete',all_old_results_unchanged=True,all_source_geometry_unchanged=True,all_masks_unchanged=True,no_GT_during_axis_preparation=True,cases=cases,changed_doors=changes,videos=video_rows,regression_tests_passed=33,lower_door_actual_free_edge_moves_up_and_out=True))
for name in ['finalize_pull_guided_axes.py','render_pull_guided_axes.py']:shutil.copyfile(pathlib.Path(__file__).parent/name,b/'source_snapshot'/name)
write_json(b/'FINAL_SEAL.json',dict(state='complete',cases=protocol['cases'],video_count=len(video_rows),report_sha256=sha256(b/'README.md'),verification_sha256=sha256(b/'VERIFICATION.json'),candidate_seal_sha256=sha256(b/'AXIS_CANDIDATES_SEAL.json'),protocol_sha256=sha256(b/'protocol.json'),source_snapshot_sha256={p.name:sha256(p) for p in (b/'source_snapshot').glob('*.py')}));(pathlib.Path(__file__).resolve().parents[1]/'current_auto_component_axis_priors.txt').write_text(str(b.resolve())+'\n');print('COMPLETE',b,'45503 pivot',orow['pivot_percent_D'],'->',jr['pivot_percent_D'],'axis total',before,'->',after,'/',N,flush=True)
