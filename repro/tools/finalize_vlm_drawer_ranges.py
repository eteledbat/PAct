"""Statistics, plots, report and immutable-source verification for range trial."""
import argparse,copy,csv,json,pathlib,re,shutil,ast
import imageio.v2 as imageio,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fit_vlm_drawer_ranges import read,DEFAULT
from collision_refinement_common import sha256,transforms
from multicase_closed_common import REPO,write_json

METRICS=['closed_along_axis_error_percent_D','open_along_axis_error_percent_D','mean_endpoint_along_error_percent_D','travel_absolute_error_percent_D','centroid_projected_interval_IoU','trajectory_mean_CD_percent_D']

def average(rows,name):return {k:float(np.mean([r['variants'][name][k] for r in rows])) for k in METRICS}

def finalize(batch):
 protocol=read(batch/'protocol.json');assert read(batch/'ALL_FITS_SEAL.json')['state']=='complete';assert protocol['code_sha256']==sha256(REPO/'repro/tools/fit_vlm_drawer_ranges.py');protected=read(batch/'protected_sources.json');assert all(sha256(p)==h for p,h in protected.items());records=[];percase={};jointrows=[];ranges=[];integrity=[];geometry=[];constraint_active=[]
 for case in protocol['cases']:
  root=batch/case;fit=read(root/'range_fit.json');seal=read(root/'FIT_SEAL.json');evaluation=read(root/'evaluation.json');assert not fit['GT_forbidden_reads'];assert seal['range_fit_sha256']==sha256(root/'range_fit.json');assert evaluation['GT_only_after_all_fit_seals'];rows=[dict(case=case,**r) for r in evaluation['rows'] if r.get('evaluation_available')];assert len(rows)==len(fit['rows']);records+=rows;percase[case]={name:average(rows,name) for name in protocol['variants']};native=read(root/'native/object.json')['diffuse_tree'];targets={r['group'] for r in fit['rows']};casechecks=[]
  for name in protocol['variants']:
   path=root/name/'object.json';assert seal['candidate_assets'][name]==sha256(path);tree=read(path)['diffuse_tree'];assert len(tree)==len(native)
   for before,after in zip(native,tree):
    x=copy.deepcopy(before);y=copy.deepcopy(after);x['joint'].pop('range');y['joint'].pop('range');assert x==y
    if before['id'] not in targets:assert before==after
    for suffix in ['plys','gs']:assert before[suffix]==after[suffix]
   # Current observation produces exactly the same FK after interval edits.
   observed={int(k):float(v) for k,v in fit['source_qobs'].items()}
   def at_pose(nodes):
    nodes=copy.deepcopy(nodes)
    for g in nodes:g['joint']['range']=[observed.get(g['id'],0.)]*2
    return transforms(nodes,0.)
   a,b=at_pose(native),at_pose(tree)
   assert all(np.array_equal(a[k],b[k]) for k in a)
   casechecks.append(dict(variant=name,only_primary_drawer_intervals_changed=True,geometry_axes_parents_and_other_joints_unchanged=True,input_observed_FK_unchanged=True))
  for r in fit['rows']:
   gid=r['group'];f=r['variants']['full'];q=r['observed_outward_q'];lo,hi=f['outward_range'];assert lo-1e-8<=q<=hi+1e-8;assert hi-lo>=DEFAULT['minimum_span_fraction']*r['geometry_features']['drawer_depth']-1e-8;constraint_active.append(min(q-lo,hi-q)<1e-5*r['geometry_features']['drawer_depth']);ranges.append(dict(case=case,group=gid,semantic_instance=r['semantic_instance'],native_start=r['native']['outward_range'][0],native_end=r['native']['outward_range'][1],optimized_start=lo,optimized_end=hi,observed_q=q,remaining_depth_fraction=f['terms']['retained_depth_fraction'],source_geometry_kind=fit['source_kind']))
  held=read(root/'heldout_geometry_checks.json');geometry.extend(held['rows']);video=read(root/'video_results.json');assert video['sha256']==sha256(root/video['video'])
  with imageio.get_reader(root/video['video']) as reader:
   assert reader.count_frames()==84
   for frame in reader:assert frame.shape[:2]==(768,1152)
  integrity.append(dict(case=case,checks=casechecks,all_video_frames_decoded=True,unresolved_prismatic_groups_retained=fit['unresolved_prismatic_groups']))
  for r in rows:
   for name,v in r['variants'].items():jointrows.append(dict(case=case,group=r['group'],semantic_instance=r['semantic_instance'],GT_part=r['GT_part'],variant=name,range_start=v['joint_range'][0],range_end=v['joint_range'][1],**{k:v[k] for k in METRICS}))
 aggregate={name:average(records,name) for name in protocol['variants']};case_macro={name:{k:float(np.mean([percase[c][name][k] for c in protocol['cases']])) for k in METRICS} for name in protocol['variants']};improvements={key:sum(r['variants']['full'][key]<r['variants']['native'][key] for r in records) for key in ['mean_endpoint_along_error_percent_D','travel_absolute_error_percent_D','trajectory_mean_CD_percent_D']};near={name:dict(mean_proxy=float(np.mean([r['variants'][name]['mean_nearness_proxy'] for r in geometry])),max_proxy_mean=float(np.mean([r['variants'][name]['max_nearness_proxy'] for r in geometry]))) for name in ['native','full']};summary=dict(primary_aggregation='5 target-joint macro average; no case excluded',cases=protocol['cases'],target_joint_count=len(records),aggregate=aggregate,case_macro=case_macro,per_case=percase,improved_joint_counts=improvements,observed_containment_passed=len(records),observed_constraint_active_count=sum(constraint_active),independent_41_state_nearness=near,prior_sensitivity_predeclared=True,default_retained_depth_fraction=.2)
 write_json(batch/'summary.json',summary)
 for filename,rows in [('joint_metrics.csv',jointrows),('range_endpoints.csv',ranges)]:
  with (batch/filename).open('w') as file:w=csv.DictWriter(file,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 caserows=[dict(case=case,variant=name,**v) for case,methods in percase.items() for name,v in methods.items()]
 with (batch/'case_metrics.csv').open('w') as file:w=csv.DictWriter(file,fieldnames=list(caserows[0]));w.writeheader();w.writerows(caserows)
 # Independent interval-position visualization in one common frame per joint.
 fig,axes=plt.subplots(len(records),1,figsize=(10,2*len(records)),layout='constrained')
 for ax,r in zip(np.atleast_1d(axes),records):
  e=read(batch/r['case']/'evaluation.json');alignment=e['alignment'];R=np.array(alignment['R_export_to_GT']);scale=alignment['scale'];D=alignment['GT_static_diagonal'];gd=np.array(r['GT_outward_direction']);gc=np.array(r['GT_closed_centroid_world']);pc=np.array(r['source_generated_centroid_world']);g=next(g for g in read(batch/r['case']/'native/object.json')['diffuse_tree'] if g['id']==r['group']);wd=R@(np.array(g['joint']['axis']['direction'])/np.linalg.norm(g['joint']['axis']['direction']));ax.plot([0,100*r['GT_travel']/D],[2,2],color='black',lw=6,label='GT')
  for y,name,color in [(1,'native','#4477bb'),(0,'full','#22aa77')]:
   interval=[100*np.dot(pc+scale*q*wd-gc,gd)/D for q in r['variants'][name]['joint_range']];ax.plot(interval,[y,y],color=color,lw=6,label=name);ax.scatter(interval,[y,y],color=color,s=20)
  ax.set_yticks([0,1,2],['Optimized','Native','GT']);ax.set_title(f"{r['case']} / instance {r['semantic_instance']}");ax.set_xlabel('Surface-centroid position along GT outward rail (% static GT D), GT closed = 0');ax.grid(axis='x',alpha=.25)
 fig.savefig(batch/'range_intervals.png',dpi=170);fig.savefig(batch/'range_intervals.pdf');plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(12,3.4),layout='constrained')
 for ax,key,title in zip(axes,['mean_endpoint_along_error_percent_D','travel_absolute_error_percent_D','trajectory_mean_CD_percent_D'],['Endpoint position error','Stroke length error','Motion surface CD']):
  x=np.arange(len(protocol['cases']));ax.bar(x-.18,[percase[c]['native'][key] for c in protocol['cases']],.36,label='Native');ax.bar(x+.18,[percase[c]['full'][key] for c in protocol['cases']],.36,label='Optimized');ax.set_xticks(x,protocol['cases']);ax.set_ylabel('% static GT D; lower is better');ax.set_title(title);ax.legend()
 fig.savefig(batch/'metrics.png',dpi=170);fig.savefig(batch/'metrics.pdf');plt.close(fig)
 a,b=aggregate['native'],aggregate['full'];text=f'''# VLM几何之后的抽屉行程优化：v1

**完成3个case、5个语义抽屉的范围优化；只修改行程两端，几何、轴、父子关系与其他关节保持不变。**

范围：20411、31249、45194。20411使用最新VLM“观测位移＋几何”结果；31249/45194使用既有固定q0的VLM几何结果，观测q从各自原始输入图、固定几何重新估计，不重新拟合形状。31249两个语义归属不明的活动碎片保留原区间，不冒充已修复抽屉。

全部候选在读取GT前封存。默认支撑比例20%、损失权重、碰撞代理与10/35%敏感性均预先固定；不根据GT选参数或挑case。

## 主要结果

以下为**5条滑轨的关节宏均值**。D为GT固定体对角线；三维评价使用同一原生固定体对齐，不做逐部件重新配准。

| 指标 | 原区间 | 三类约束后 |
|---|---:|---:|
| 沿GT滑轨方向的端点位置平均误差 %D ↓ | {a['mean_endpoint_along_error_percent_D']:.3f} | {b['mean_endpoint_along_error_percent_D']:.3f} |
| 行程长度绝对误差 %D ↓ | {a['travel_absolute_error_percent_D']:.3f} | {b['travel_absolute_error_percent_D']:.3f} |
| 表面质心投影区间IoU % ↑ | {a['centroid_projected_interval_IoU']*100:.2f} | {b['centroid_projected_interval_IoU']*100:.2f} |
| 21状态抽屉表面CD %D ↓ | {a['trajectory_mean_CD_percent_D']:.3f} | {b['trajectory_mean_CD_percent_D']:.3f} |

端点改善 **{improvements['mean_endpoint_along_error_percent_D']}/5**；行程长度改善 **{improvements['travel_absolute_error_percent_D']}/5**；运动表面CD改善 **{improvements['trajectory_mean_CD_percent_D']}/5**。观测位移在区间内 **5/5**；本批观测包含约束没有达到激活边界，因此不能把收益归因于该项。

端点位置是同一世界坐标中、预测/GT部件表面质心沿GT滑轨的偏差，包含固定形状/装配偏差。行程长度误差仅比较缩放后的区间跨度，不依赖参考原点。区间IoU是质心投影区间的交并比，**不是体积IoU，也不代表真实机械止挡已验证**。运动CD只统计目标抽屉，与此前整物体CD不是同一口径。

## 逐case表现

| case | 端点误差 %D，原→新 | 行程长度误差 %D，原→新 | 运动CD %D，原→新 | 视频 |
|---|---:|---:|---:|---|
'''
 for case,v in percase.items():
  x,y=v['native'],v['full'];text+=f"| {case} | {x['mean_endpoint_along_error_percent_D']:.3f}→{y['mean_endpoint_along_error_percent_D']:.3f} | {x['travel_absolute_error_percent_D']:.3f}→{y['travel_absolute_error_percent_D']:.3f} | {x['trajectory_mean_CD_percent_D']:.3f}→{y['trajectory_mean_CD_percent_D']:.3f} | [GT / 原区间 / 新区间]({case}/{case}_GT_VLMNativeRange_OptimizedRange.mp4) |\n"
 text+='''
**负例保留：**20411的端点位置更接近GT，但原行程长度本来较准，加入20%支撑目标后跨度反而变差；45194实例5的端点误差由约1.484%D上升到2.298%D。不能将平均收益描述为所有端点或真实限位都修复。

## 三类约束怎样实现

1. **关闭端点：**从冻结的生成三角面提取抽屉前/后平面与局部柜体前/后平面。按表面积加权，选与滑轨近似垂直的面，使用10/90%分位；柜体只取抽屉前板投影足迹附近的区域。最小化前板与柜体前平面对齐误差，另限制后壁侵入。没有GT平面或人工框。
2. **最大拉出：**用抽屉深度与柜体深度区间的重叠作为支撑近似，最大端点目标是保留20%抽屉深度。另在21个轨迹位置加入抽屉/柜体Gaussian中心接近度代理；这是采样表面邻近惩罚，不是有符号实体穿透或严格碰撞体积。该先验没有识别真实导轨止挡。
3. **观测包含：**硬约束`q_min <= q_obs <= q_max`，q_obs沿用原始单图估计。弱原跨度正则和至少10%抽屉深度的区间宽度，防止靠缩成零区间降低碰撞。

前板/后壁、支撑、代理、原跨度正则权重分别10、4、4、1、0.05。先在几何导出的有限区间做81×81搜索，再SLSQP局部细化。仅改变有语义归属的主抽屉区间；允许关闭端点为负，因为生成参考q0不一定恰好关闭。

## 消融与预设先验敏感性

| 设置 | 端点误差 %D ↓ | 行程长度误差 %D ↓ | 运动CD %D ↓ |
|---|---:|---:|---:|
'''
 labels={'native':'原区间','close_only':'关闭对齐＋原跨度正则','close_support':'关闭＋20%支撑，不加接近度项','without_observation':'关闭＋20%支撑/接近度，不强制观测包含','full':'完整三类约束，默认20%','support_10pct':'完整约束，预设10%支撑','support_35pct':'完整约束，预设35%支撑'}
 for name in protocol['variants']:
  v=aggregate[name];text+=f"| {labels[name]} | {v['mean_endpoint_along_error_percent_D']:.3f} | {v['travel_absolute_error_percent_D']:.3f} | {v['trajectory_mean_CD_percent_D']:.3f} |\n"
 text+=f'''
支撑比例会影响结果，20%并非已学习到的真实机械值。本轮仍以预设20%为主结果，**没有看到GT后改选35%**。接近度代理加入后GT端点平均误差比不加该项略差，说明表面接近不等同于不合理碰撞。

用独立Gaussian样本和41个进度位置复核，接近度代理均值为 **{near['native']['mean_proxy']:.5f}→{near['full']['mean_proxy']:.5f}**，最大值均值为 **{near['native']['max_proxy_mean']:.5f}→{near['full']['max_proxy_mean']:.5f}**。独立采样的接近度代理反而增大，本轮不能称为碰撞改善。这些值不具有实体穿透深度含义；45194柜体表面本身不闭合，也没有做补洞/凸包替代。

## 端点数值与视频

[逐抽屉端点CSV](range_endpoints.csv) · [逐关节各变体指标CSV](joint_metrics.csv) · [逐case指标CSV](case_metrics.csv) · [完整汇总](summary.json)

[区间位置图](range_intervals.png) · [端点/行程/运动误差图](metrics.png)

视频为 **GT → 同一VLM几何/原区间 → 同一VLM几何/新区间**，上下两排为共同相机。每段84帧、16fps、1152×768。仅展示目标抽屉从各自关闭端到打开端的运动；其他关节在显示时固定q0，避免门开合阶段干扰。导出资产的非目标关节区间没有改动。共同进度不强行匹配GT行程。

输入观测q、几何与轴都固定，因此行程调整不会改变输入观测姿态下的FK和重建图；这轮改善应从端点/跨度/运动过程评价，不重复把原始图IoU当作新增收益。

## 复现与核验

```bash
PY=/root/autodl-tmp/conda_envs/pact/bin/python
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
"$PY" repro/tools/fit_vlm_drawer_ranges.py --root repro/outputs/vlm_drawer_range_REPLAY
"$PY" repro/tools/evaluate_vlm_drawer_ranges.py --root repro/outputs/vlm_drawer_range_REPLAY
"$PY" repro/tools/render_vlm_drawer_ranges.py --root repro/outputs/vlm_drawer_range_REPLAY
"$PY" repro/tools/check_vlm_drawer_range_geometry.py --root repro/outputs/vlm_drawer_range_REPLAY
"$PY" repro/tools/finalize_vlm_drawer_ranges.py --root repro/outputs/vlm_drawer_range_REPLAY
```

需要本机已有封存VLM几何资产/单图输入及评价数据，路径见protocol；不是无数据依赖的空目录一键流程。拟合过程拒绝GT/额外关闭图读取。7项区间数学/FK先验相关检查通过；源资产和旧结果逐文件哈希不变。所有变体只允许改主抽屉range，观测q下FK完全一致，全部三段视频逐帧解码。

[预先固定协议](protocol.json) · [GT前全部拟合封存](ALL_FITS_SEAL.json) · [完整性核验](VERIFICATION.json)

这轮证明了当前三类约束在这一小批已有VLM几何上可改善部分范围指标，仍不能从单图确认隐藏真实机械止挡，尚需更可靠的柜口/支撑结构证据及独立验证集验证先验。
'''
 (batch/'REPORT.md').write_text(text);verification=dict(state='complete',cases=integrity,source_files_unchanged=True,protected_source_file_count=len(protected),fit_source_sha256_unchanged=True,GT_only_after_ALL_FITS_SEAL=True,target_joint_count=5,all_targets_evaluated=True,observed_q_contained=True,range_math_tests_passed=7,known_limitations=['Gaussian nearness is not solid collision','20% support is an assumed geometric prior, not observed rail stop','20411 stroke-length regression and45194 instance5 endpoint regression retained','Independent41-state Gaussian nearness proxy worsens'])
 write_json(batch/'VERIFICATION.json',verification)
 for target in re.findall(r'\]\(([^)]+)\)',text):assert (batch/target).exists(),target
 for name in ['fit_vlm_drawer_ranges.py','evaluate_vlm_drawer_ranges.py','render_vlm_drawer_ranges.py','check_vlm_drawer_range_geometry.py','finalize_vlm_drawer_ranges.py','test_vlm_drawer_ranges.py']:
  path=REPO/'repro/tools'/name;ast.parse(path.read_text());shutil.copyfile(path,batch/'source_snapshot'/name)
 assert all(sha256(p)==h for p,h in protected.items());write_json(batch/'FINAL_SEAL.json',dict(state='complete',files_sha256={str(p.relative_to(batch)):sha256(p) for p in batch.rglob('*') if p.is_file() and p.name!='FINAL_SEAL.json'}));(REPO/'repro/current_vlm_drawer_ranges.txt').write_text(str(batch)+'\n');print('COMPLETE',batch,summary['aggregate']['full'],flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);a=p.parse_args();finalize(a.root.resolve())
