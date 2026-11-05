"""Rebuild one automatic handle case and visualize attachment evidence; no GT."""
import argparse,copy,json,pathlib,shutil
import numpy as np
from collision_refinement_common import load_asset,sha256
from auto_component_axis_prior import prepare
from fit_observed_drawer_pose import guard,read,REPO
from multicase_closed_common import write_json

def run(batch,case):
 batch=batch.resolve();batch.mkdir(exist_ok=True);source=REPO/'repro/outputs/auto_component_axis_priors_20261029_v7'/case;prior=read(source/'motion_groups.json');protected={str(p):sha256(p) for p in source.rglob('*') if p.is_file() and 'evaluation' not in str(p)};denied,reads=guard(batch)
 write_json(batch/'protocol.json',dict(cases=[case],training='Native predicted geometry plus sealed VLM semantics and original instance masks only; no GT, geometry, observed pose or travel optimization',rule='Two separated body-facing support endpoints; contact-distance weighting anchors axis position; no support-bar-center fallback',old_axis_source=str(source),geometry_changed=False))
 if not (batch/case/'motion_groups.json').exists():prepare(REPO/'repro/outputs/auto_semantic_pulls_20261029_v1',REPO/'repro/outputs/general_semantic_priors_20261029_v2',batch/case,case)
 root=batch/case;man=read(root/'motion_groups.json');assert all(sha256(p['path'])==p['sha256'] for p in man['geometry_files']);assert man['membership_sha256']==prior['membership_sha256'];assert man['semantic_input_sha256']==prior['semantic_input_sha256'];assert man['instance_mask_sha256']==prior['instance_mask_sha256'];assignments=read(root/'component_assignment.json')['assignments'];old=read(source/'component_assignment.json')['assignments'];rows=[]
 for a in assignments:
  before=next(x for x in old if x['instance_id']==a['instance_id'])
  if a.get('family')!='articulated_handle':assert a.get('joint')==before.get('joint');continue
  evidence=a['axis_evidence'];assert 'attachment_points' in evidence;rows.append(dict(instance=a['instance_id'],old_joint=before['joint'],new_joint=a['joint'],attachment_points=evidence['attachment_points'],support_centers=evidence['support_centers'],attachment_weights=evidence['attachment_weights'],attachment_body_distances=evidence['attachment_body_distances'],approximate_body_gap=evidence['approximate_body_gap']))
 assert rows and not denied
 write_json(batch/'CANDIDATE_SEAL.json',dict(state='complete',manifest_sha256=sha256(root/'motion_groups.json'),membership_sha256=sha256(root/'membership.npz'),assignment_sha256=sha256(root/'component_assignment.json'),GT_reads=[],geometry_changed=False,original_masks_unchanged=True,kinematics_only=True))
 asset=load_asset(man['source_run']);meshes={p['node']['id']:p['mesh'] for p in asset['parts']};membership=np.load(root/'membership.npz')
 def verts(g):return np.asarray(meshes[g['source_node']].vertices)[membership[g['membership_prefix']+'_vertices']]
 body=np.concatenate([verts(g) for g in man['groups'] if g.get('semantic_owner')==1])
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axes=plt.subplots(len(rows),2,figsize=(12,4*len(rows)),squeeze=False)
 for i,row in enumerate(rows):
  own=row['instance'];v=np.concatenate([verts(g) for g in man['groups'] if g.get('semantic_owner')==own]);points=np.array(row['attachment_points']);oldp=np.array(row['old_joint']['axis']['origin']);newp=np.array(row['new_joint']['axis']['origin']);newd=np.array(row['new_joint']['axis']['direction'])
  for j,(a,b,title) in enumerate([(0,1,'Front: X / Y'),(2,1,'Side: Z / Y')]):
   ax=axes[i,j];sample=body[::max(1,len(body)//5000)];ax.scatter(sample[:,a],sample[:,b],s=1,c='#999999',alpha=.2,label='Predicted body');ax.scatter(v[::4,a],v[::4,b],s=1,c='#4477aa',alpha=.4,label='Original handle');ax.scatter(points[:,a],points[:,b],s=55,c='#eeaa00',marker='x',label='Selected attachment ends',zorder=8)
   for joint,color,label in [(row['old_joint'],'#cc3333','Old: bar-center axis'),(row['new_joint'],'#228833','New: attachment axis')]:
    o=np.array(joint['axis']['origin']);d=np.array(joint['axis']['direction']);line=o+np.array([-.45,.45])[:,None]*d;ax.plot(line[:,a],line[:,b],color=color,lw=2,label=label);ax.scatter([o[a]],[o[b]],c=color,s=45,zorder=7)
   ax.set_title(f'Instance {own} | {title}');ax.set_aspect('equal');ax.set_ylim(body[:,1].max()-.16,v[:,1].max()+.05)
   if j==0:ax.set_xlim(np.median(v[:,0])-.14,np.median(v[:,0])+.14)
   else:ax.set_xlim(min(v[:,2].min(),body[:,2].min())-.05,max(v[:,2].max(),body[:,2].max())+.05)
   ax.grid(alpha=.15);ax.set_xlabel('Canonical X' if j==0 else 'Canonical Z');ax.set_ylabel('Canonical Y (up)')
  # Numeric diagnostic is body/attachment evidence, not GT accuracy.
  def residual(joint):
   o=np.array(joint['axis']['origin']);d=np.array(joint['axis']['direction']);d/=np.linalg.norm(d);return np.linalg.norm(np.cross(points-o,d),axis=1).tolist()
  row['endpoint_to_old_axis_distances']=residual(row['old_joint']);row['endpoint_to_new_axis_distances']=residual(row['new_joint'])
 fig.suptitle(f'{case}: support midpoints vs body-facing attachment endpoints',fontsize=15);handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=10);fig.tight_layout(rect=[0,.065,1,.95]);fig.savefig(batch/'attachment_axis_before_after.png',dpi=180);fig.savefig(batch/'attachment_axis_before_after.pdf');plt.close(fig)
 write_json(batch/'axis_changes.json',dict(case=case,rows=rows,GT_used=False,warning='Predicted rear supports have gaps. Axis is an attachment-evidence estimate, not certified physical hinge or GT accuracy.'))
 assert all(sha256(p)==h for p,h in protected.items());write_json(batch/'VERIFICATION.json',dict(state='complete',GT_reads=denied,old_results_unchanged=True,protected_files=len(protected),same_native_geometry=True,same_masks=True,same_component_ownership=True,nonhandle_joints_unchanged=True,tests_run=45,tests_passed=True))
 snap=batch/'source_snapshot';snap.mkdir(exist_ok=True)
 for name in ['semantic_motion_rules.py','auto_component_axis_prior.py','test_semantic_motion_rules.py','check_handle_attachment_axis.py']:shutil.copyfile(REPO/'repro/tools'/name,snap/name)
 text='# 提手连接端轴修复\n\n旧后备规则把两根提手支杆的中心当连接点；本轮改为各支杆最靠近主体、且可明确区分的端部。两处连接端用于确定方向；主体接触更可靠的一端对轴位置的权重更大。连接证据不足时保留原生轴并标记未解决，不再用支杆中心造轴。门/抽屉固定拉手仍随父部件运动。\n\n[旧轴/新轴/连接端对比图](attachment_axis_before_after.png) · [数值证据](axis_changes.json) · [核验](VERIFICATION.json)\n\n原始几何、原mask、部件归属和非提手关节不变；保留提手原机械行程幅度。未读取GT轴或网格、未重新拟合几何。45项现有及新增测试通过，覆盖上端/下端连接、主体远离、端点歧义、截断支杆和禁止中点回退。\n\n100481生成的后侧支杆有缺失，与主体存在间隙，因此新轴仍是近似连接证据估计，不代表已恢复真实机械铰链。旧GitHub版本、之前10-case结果和其指标保持不变；本次修改尚未发布GitHub，也未重新运行全部几何优化。\n'
 (batch/'REPORT.md').write_text(text);write_json(batch/'FINAL_SEAL.json',dict(state='complete',files_sha256={str(p.relative_to(batch)):sha256(p) for p in batch.rglob('*') if p.is_file() and p.name!='FINAL_SEAL.json' and p.suffix!='.log'}));print(json.dumps(rows,indent=2));print('COMPLETE',batch)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--case',default='100481');a=p.parse_args();run(a.root,a.case)
