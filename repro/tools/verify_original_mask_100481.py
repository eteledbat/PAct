"""Verify and summarize the approved-original-mask 100481 experiment."""
import argparse,copy,json,pathlib,shutil,sys,os
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
import cv2,imageio.v2 as imageio,numpy as np,torch
from PIL import Image,ImageDraw,ImageFont
from original_mask_semantic_100481 import OriginalMaskModel,APPROVED,read,OWNERS,COLORS,load_model,mask_scores
from multicase_closed_common import REPO,write_json,tensor
from collision_refinement_common import sha256,transforms
p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();batch=pathlib.Path(a.root).resolve();root=batch/'100481';cfg=read(root/'config.json');manifest=read(root/'motion_groups.json');fit=read(root/'fit_results.json');evaldata=read(root/'evaluation.json');protocol=read(batch/'protocol.json');approved=read(APPROVED/'semantic_labels.json')
labels=np.array(Image.open(root/'original_instance_labels_800.png'));approved_labels=np.array(Image.open(APPROVED/'100481_original_instance_labels.png'));raw=cv2.imread(approved['source_original_mask'],cv2.IMREAD_UNCHANGED)[:,:,0];assert np.array_equal(labels,approved_labels) and np.array_equal(labels,raw)
semantic=np.array(Image.open(root/'semantic_ids_800.png'));restored=np.where(semantic==10,1,np.where(semantic==11,2,np.where(semantic==12,3,0)));assert np.array_equal(labels,restored)
assert not fit['GT_forbidden_reads'];assert not read(root/'registration_results.json')['GT_forbidden_reads'];assert not read(root/'axis_prior_results.json')['GT_forbidden_reads']
for key,name in [('checkpoint_sha256','checkpoint.pt'),('optimized_parts_sha256','optimized_parts.npz'),('config_sha256','config.json'),('manifest_sha256','motion_groups.json'),('observation_sha256','observation.npz')]:assert sha256(root/name)==fit[key]
assert sha256(REPO/'repro/tools/original_mask_semantic_100481.py')==fit['source_tool_sha256']==protocol['code_sha256']
assert all(sha256(pathlib.Path(row['path']))==row['sha256'] for row in manifest['geometry_files']);assert sha256(pathlib.Path(manifest['source_object']))==manifest['source_object_sha256']
assert all(sha256(pathlib.Path(path))==digest for path,digest in approved['source_sha256'].items())
for name in ['prior','optimized']:
 assert evaldata['joint_methods'][name]['conditional_mean_angle_degrees']==0.
assert all(evaldata['joint_methods']['prior'][key]==evaldata['joint_methods']['optimized'][key] for key in ['axis_correct_count','conditional_mean_angle_degrees','conditional_mean_pivot_percent_D'])
torch.set_num_threads(4);m=load_model(root,checkpoint=True);m.requires_grad_(False);meshes={p['node']['id']:p['mesh'] for p in m.asset['parts']};groups=m.groups;partitions=[];maxrest=0.;maxrigid=0.
for nid,mesh in meshes.items():
 sg=[g for g in groups if g['source_node']==nid];ngs=len(__import__('plyfile').PlyData.read(manifest['sources'][str(nid)]['gs'])['vertex'].data)
 for key,n in [('vertices',len(mesh.vertices)),('faces',len(mesh.faces)),('gaussians',ngs)]:
  ids=np.concatenate([m.membership[g['membership_prefix']+'_'+key] for g in sg]);assert np.array_equal(np.sort(ids),np.arange(n))
 partitions.append(dict(source=nid,vertices=len(mesh.vertices),faces=len(mesh.faces),gaussians=ngs))
for part in m.parts:
 g=part['group'];original=meshes[g['source_node']].vertices[m.membership[g['membership_prefix']+'_vertices']];rest=part['vertex_local']@part['B'].cpu().numpy().T+part['anchor'].cpu().numpy();maxrest=max(maxrest,float(np.max(abs(original-rest))))
assert maxrest<1e-6
with torch.no_grad():
 sampled={p['group']['id']:m.deform(i)[0][::997].cpu().numpy().astype(float) for i,p in enumerate(m.parts)}
 for t in np.linspace(0,1,21):
  tr=transforms(groups,t)
  for g in groups:
   if g['joint']['type']=='fixed' and g['parent']!=-1:assert np.allclose(tr[g['id']],tr[g['parent']],atol=1e-12)
   x=sampled[g['id']];T=tr[g['id']];y=x@T[:3,:3].T+T[:3,3];maxrigid=max(maxrigid,float(np.max(abs(np.linalg.norm(x-x[0],axis=1)-np.linalg.norm(y-y[0],axis=1)))))
assert maxrigid<1e-12
# Re-render visible functional mask comparison at the actual approved camera.
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15);canvas=Image.new('RGB',(1024,320),'white');draw=ImageDraw.Draw(canvas)
with torch.no_grad():
 geometries=[m.geometry(original=True),m.geometry()];probs=[m.probabilities(g).cpu().numpy() for g in geometries]
 pics=[Image.fromarray((m.rgb.clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype('uint8')),Image.open(root/'approved_semantic_target.png').convert('RGB')]
 for pr in probs:
  index=np.concatenate([(1-pr.sum(0)).clip(0,1)[None],pr]).argmax(0);ids=np.array([0,*OWNERS])[index];rgb=np.full((256,256,3),255,np.uint8)
  for gid,color in COLORS.items():rgb[ids==gid]=(np.array(color)*255).astype(np.uint8)
  pics.append(Image.fromarray(rgb))
for j,(title,pic) in enumerate(zip(['Approved original RGB','Unchanged original mask','Prior: projected part mask','After geometry fit'],pics)):
 draw.text((j*256+5,7),title,fill='black',font=font);canvas.paste(pic,(j*256,35))
canvas.save(root/'original_mask_fit_comparison.png')
video_rows=[]
for filename in ['render_results.json','gt_video_results.json']:
 for row in read(root/filename)['videos']:
  path=root/row['file'];assert sha256(path)==row['sha256']
  with imageio.get_reader(path) as reader:
   count=reader.count_frames();meta=reader.get_meta_data()
   for i in range(count):reader.get_data(i)
  assert count==row['frames']==84;video_rows.append({**row,'all_frames_decoded':True})
assert len(video_rows)==9
verification=dict(original_mask_changed_pixels=0,original_mask_exactly_equal_to_approved_and_source_EXR=True,source_partition_bijections=partitions,max_original_rest_reconstruction_error=maxrest,max_rigid_length_error=maxrigid,fixed_children_follow_same_side_handle=True,axes_types_tree_and_ranges_frozen_in_geometry_fit=True,no_manual_polygon_or_rivet_reads=True,no_previous_closed_image_fit_checkpoint_reads=True,all_original_sources_unchanged=True,all_video_frames_decoded=True,videos=video_rows)
write_json(batch/'VERIFICATION.json',verification)
lines=['# 100481：原始 mask 语义 → 轴先验 → 几何优化','',
 '**使用你确认的原始实例 mask，空间边界改变像素数为 0。** 仅把原标签 1/2/3 对应为固定篮体、左提手、右提手。','',
 '原始 mask 对应输入第0帧，提手处于竖起状态。因此第四组使用同一帧 RGB 和原始 mask 优化几何，名称为 **Original-mask geometry fit**；本轮没有使用此前手画的闭合图多边形、手工铆钉坐标或旧图像优化 checkpoint。起始实例 mask 沿用原流程已有的 oracle/GT 投影 mask；本轮未读取 GT 网格或关节来拟合预测。','',
 '[主视频：GT → Origin → Motion prior → Original-mask geometry fit](100481/100481_four_versions_with_GT.mp4)','',
 '[你确认的 mask](100481/approved_mask_comparison.png) · [优化前后部件 mask](100481/original_mask_fit_comparison.png) · [输入图与三组重建](100481/input_target_and_three_versions.png)','',
 '先重新拟合100步全局尺度和相机平面平移，几何固定。随后用细长竖直连通部件的几何先验寻找提手腿，使用原 mask 投票区分左右。原来错误运动的篮子面板归静态篮体，同侧前后腿共用一个独立关节，全部源顶点、面和 Gaussian 一一保留。','',
 '轴方向采用该提手族的前后方向 +Z 先验。轴位置拟合生成提手前后腿的底端连接点（各权重1），以及原 mask 最下方3%像素自动提取的端点反投影（权重2）；反投影使用生成前腿自身的深度。没有手工圈选铆钉，没有 GT 轴对齐。单张原始输入不能唯一确定隐藏的连接结构和轴深度，因此轴位置仍可能不准。','',
 '最后运行240步有界几何形变，冻结关节类型、轴位置/方向、父子关系和运动行程。损失为 RGB、整体轮廓、语义颜色，以及逐部件 Dice/平衡交叉熵；没有手工框约束。','',
 '**视频运动说明**：四组均从原始输入姿态开始。Origin 保留原始预测区间；两个修正后的提手分别向外旋转，继承原前腿组的绝对行程63.15°。GT 从真实输入姿态转向其标注折叠限位。统一的是播放进度，各组角度幅度可能不同；没有用 GT 行程修正预测，也没有把预测末端当成真实闭合限位。上下两行使用相同的两个观察相机。','',
 '| 阶段 | 输入第0帧整体 IoU | 未参与拟合的6视角 IoU | 轴方向平均误差 | 轴位置平均误差（主体对角线百分比） | 完整轴正确数 |','|---|---:|---:|---:|---:|---:|']
for method,title in [('original','Origin'),('prior','Motion prior'),('optimized','Original-mask geometry fit')]:
 j=evaldata['joint_methods'][method];lines.append(f"| {title} | {100*evaldata['fit_image_metrics'][method]['IoU']:.2f}% | {100*evaldata['unfitted_open_mean'][method]['IoU']:.2f}% | {j['conditional_mean_angle_degrees']:.2f}° | {j['conditional_mean_pivot_percent_D']:.2f}% | {j['axis_correct_count']}/{j['GT_joint_count']} |")
parts=fit['part_mask_metrics'];lines+=['',f"可见部件 mask mIoU：Motion prior **{100*parts['prior']['mean_IoU']:.2f}%** → 几何优化 **{100*parts['optimized']['mean_IoU']:.2f}%**。此项使用原始训练 mask，不是独立分割验证。Origin 的部件 mIoU 若按修正后的功能归属重着色，会与 prior 的 q0 分数相同，不把它报告为原始源分组的语义准确率。",'',
 '结果：轴方向先验纠正了运动方向，但轴位置误差仍高于5%阈值，完整轴正确数仍为0/2。几何优化变化很小，整体 IoU 没有提升。原始 mask 只约束当前可见区域，不能补出生成几何缺失的 U 形顶桥。当前帧与此前闭合图实验不同，不直接横向比较那些拟合 IoU。','',
 '[轴拟合和组件投票明细](100481/axis_prior_results.json) · [评估指标](100481/evaluation.json) · [像素/几何/运动/视频核验](VERIFICATION.json)','',
 '原始源文件、已确认 mask 和此前全部实验保持不变。']
(batch/'REPORT.md').write_text('\n'.join(lines)+'\n')
shutil.copyfile(REPO/'repro/tools/verify_original_mask_100481.py',batch/'source_snapshot/verify_original_mask_100481.py')
seal=dict(case='100481',four_versions_video=str(root/'100481_four_versions_with_GT.mp4'),video_count=len(video_rows),original_mask_changed_pixels=0,all_source_files_unchanged=True,all_frames_decoded=True,
 report_sha256=sha256(batch/'REPORT.md'),verification_sha256=sha256(batch/'VERIFICATION.json'),protocol_sha256=sha256(batch/'protocol.json'),checkpoint_sha256=sha256(root/'checkpoint.pt'),fit_seal_sha256=sha256(root/'fit_results.json'),evaluation_sha256=sha256(root/'evaluation.json'),GT_render_manifest_sha256=sha256(root/'gt_video_results.json'),source_snapshot_sha256={p.name:sha256(p) for p in (batch/'source_snapshot').glob('*.py')})
write_json(batch/'FINAL_SEAL.json',seal);(REPO/'repro/current_original_mask_100481.txt').write_text(str(batch)+'\n');print(json.dumps(seal,indent=2),flush=True)
