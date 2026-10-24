"""Write only observed PAct results, with explicit unavailable GT/solid metrics."""
import datetime,json,os,pathlib,subprocess
ROOT=pathlib.Path(__file__).resolve().parents[2];dest=ROOT/'repro/REPRODUCTION_REPORT.md';run=pathlib.Path((ROOT/'repro/current_run.txt').read_text().strip())
def read(p,default=None):return json.loads(p.read_text()) if p.exists() else default
def link(p,label):return f'[{label}]({os.path.relpath(p,dest.parent)})'
smoke=read(run/'verified.json',{});motion=read(run/'motion_analysis/metrics.json',{});curve=read(run/'motion_analysis/collision_states.json',[]);render=read(run/'mesh_previews/render_info.json',{});env=read(ROOT/'repro/evidence/environment-smoke.json',{});source=read(ROOT/'repro/evidence/provenance.json',{});loads=[json.loads(s) for s in (run/'checkpoint_loads.jsonl').read_text().splitlines()];objects=read(pathlib.Path(smoke['object']),{}) if smoke else {};audit=motion.get('mesh_audit',smoke.get('parts',[]))
lines=['# PAct 复现与物理诊断：首个官方示例','',f'更新：{datetime.datetime.now(datetime.timezone.utc).isoformat()}。','', '**范围：官方 Dishwasher_001 的真实推理 smoke 与运动诊断。不是 PARIS Storage45135 完整 benchmark 复现；该场景尚未运行。**','', '## 1. 来源与环境','',f'官方源码 `Mobiuslqm/PAct`，commit `{source["commit"]}`；官方模型 `PAct000/PAct`，revision `{source["model_revision"]}`，3,972,142,331 字节、14 文件，已逐一核对大小和官方 LFS SHA256。','',f'工作分支 `repro/pact-baseline-20261024`；独立环境 `/root/autodl-tmp/conda_envs/pact`。Python {env.get("python")}; torch {env.get("torch")}; CUDA runtime {env.get("cuda")}。GPU RTX4080 SUPER，报告容量约32GiB。','', '官方 YAML 为 torch2.4.0/CUDA11.8，本机独立克隆为2.4.0+cu121/CUDA12.1，部分编译依赖版本有偏差；实际导入和推理已验证，不声称完整 YAML 逐项相同。'+link(ROOT/'repro/CHANGES.md','全部改动和偏差')+'；'+link(ROOT/'repro/evidence/environment-smoke.json','环境验证')+'。原 FreeArt3D、VideoArtGS、Part²GS 环境未修改。','', 'DINOv2 从已有缓存复制，记录本地 SHA256，使用新克隆的官方固定源码，严格 state_dict 加载通过；本地计算的 SHA 并非独立发布的官方校验值。'+link(ROOT/'repro/evidence/dinov2-provenance.json','DINOv2 来源')+'。','', '## 2. 输入、采样与运行','', '输入为官方仓库 Dishwasher_001 RGBA 和 EXR 二维部件 mask，识别为2部件。mask 是作者提供，自动/GT 来源未独立验证，不能描述成完全无 mask 的单图输入。没有 GT mesh、axis 或 joint type 作为本次推理输入。','', f'原采样：SS25步、SLAT25步，CFG均为7，最后20步 feature 聚合关节，seed42。加载、推理、写出和校验总计约 {smoke.get("elapsed_seconds",0):.2f} 秒；不是纯采样速度，也不直接与论文A800的耗时对比。','', f'共记录 {len(loads)} 次模型 state_dict 加载，所有缺失/多余 key 列表为空：'+link(run/'checkpoint_loads.jsonl','加载审计')+'。','', '首次实际推理在 exploded video 写出时因原版 imageio.imwrite 接收4D数组失败；修成视频接口 mimsave 后，用相同配置在新目录重跑通过。失败目录保留；没有选择更好关节/碰撞结果作为成功依据。','', '输出包括原版60帧关节动画、21帧 orbit 分离预览、raw mesh PLY、Gaussian PLY 与 object.json；未做可选 GLB 贴图烘焙。'+link(run/'command.json','命令和模型来源')+'；'+link(run/'input_manifest.json','输入哈希')+'。','', '## 3. 预测和几何','', '| 部件 | 预测类型 | 预测运动范围 | 顶点 | 面 | Mesh封闭 |','|---|---|---|---:|---:|---|']
for part in audit:
 node=next((n for n in objects.get('diffuse_tree',[]) if n['id']==part['id']),{});j=node.get('joint',{});lines.append(f'| {part["name"]} | {j.get("type")} | {j.get("range")} | {part["vertices"]} | {part["faces"]} | {part["watertight"]} |')
lines+=['', '范围为预测值，revolute单位度；几何距离为模型归一化坐标，不能称为米。原始 mesh 未补洞、重建或优化。base 不封闭，因此真实实体交集体积/实体穿透深度 unavailable，不填0。','', '原版 rescale_axis 会将 revolute 轴吸附到最近 X/Y/Z 方向，当前导出 door 轴为(1,0,0)。这是经过启发式处理的轴，不能代表原始连续回归方向误差为0。原版 animate/export 还分别依据 Gaussian/Mesh bbox 投影 origin；保留这些处理，没有进行新的 refinement。','', '## 4. 21状态运动诊断','']
if motion:
 pairs=[v for state in curve for v in state['pairs']];lines += [f'均匀采样导出预测范围内21状态，使用原版 _local_transform 和 object.json。无GT拟合。分析耗时 {motion["elapsed_seconds"]:.2f} 秒。','', 'Gaussian PLY 保存时比 Mesh/JSON 多一次固定旋转，评估依据代码撤销该序列化旋转，对齐到 Mesh/JSON；不是ICP/GT对齐。原始文件未改。','', 'opacity≥0.05 后使用全部保留点，计算双向、opacity加权的最近中心 Gaussian 核；sigma取三轴activated scales均值。该定义不同于全点对核平均，也不是真实体积。最小中心距离在全部保留跨部件点之间求得，1σ球间隙仅针对最近中心点对。','', '| 统计 | 实测 |','|---|---:|',f'| 最大最近中心 Gaussian overlap proxy | {max(x["soft_overlap_nearest_center_proxy"] for x in pairs):.8g} |',f'| 最小中心距离 | {min(x["minimum_center_distance"] for x in pairs):.8g} |',f'| 最小最近对1σ球间隙 proxy | {min(x["minimum_1sigma_gap_among_nearest_center_pairs_proxy"] for x in pairs):.8g} |','| 真实实体交叠体积 | unavailable：base Mesh不封闭 |','| 真实实体穿透深度 | unavailable |','', '高 Gaussian 核或负球间隙也可能来自正常接触、Gaussian尺度和表面表示；不能仅据此断言严重实体穿模。当前没有对照实验，因此不能归因某个物理项有效或无效。']
else:lines+=['21状态分析正在运行，数值尚未完成，不填入最终统计。']
comparison=run/'input_vs_generated.png'
if comparison.exists():lines+=['',f'![实际输入与生成结果，视角未配准]({os.path.relpath(comparison,dest.parent)})']
for name,label in [('collision_curve.png','21状态碰撞代理曲线'),('gaussians_state00.png','预测起始状态及轴'),('gaussians_state10.png','预测中间状态及轴'),('gaussians_state20.png','预测终止状态及轴')]:
 path=run/'motion_analysis'/name
 if path.exists():lines+=['',f'![{label}]({os.path.relpath(path,dest.parent)})']
hotspots=read(run/'motion_analysis/hotspot_summary.json',[])
if hotspots:
 lines+=['','高代理值区域仅定位最近核候选，不是实际穿透标签；终止状态热点的空间范围见 '+link(run/'motion_analysis/hotspot_summary.json','热点边界与阈值')+'。']
 for state in [0,10,20]:
  path=run/'motion_analysis'/f'hotspot_state{state:02d}.png'
  if path.exists():lines+=['',f'![高代理值区域，状态{state}]({os.path.relpath(path,dest.parent)})']
if motion:
 lines+=['','base为一个连通表面，28条边界边；door为两个封闭表面分量，最大分量占面数约53.2%。两个表面分量也可能对应内外壳，不能直接称为碎片；未做自交完整验证。audit中的signed_volume只是有向四面体积分诊断，开放base的该值不是真实实体体积。']
 lines+=['',f'代理曲线最低在状态8/20；终止/起始值比约 {curve[-1]["pairs"][0]["soft_overlap_nearest_center_proxy"]/curve[0]["pairs"][0]["soft_overlap_nearest_center_proxy"]:.2f}。该比值只描述本代理，不是实体穿透变化倍率。']
for state in [0,10,20]:
 path=run/'mesh_previews'/f'mesh_state{state:02d}_3views.png'
 if path.exists():lines+=['',f'![实际三角 Mesh，状态{state}]({os.path.relpath(path,dest.parent)})']
for path,label in [(run/'mesh_previews/raw_mesh_motion_21.mp4','21状态原始Mesh运动视频'),(run/'motion_analysis/collision_states.json','逐状态数值与变换'),(run/'motion_analysis/mesh_audit.json','Mesh边界、碎片和合法性诊断')]:
 if path.exists():lines+=['',link(path,label)]
for path in run.rglob('00__articulation_animation.mp4'):lines+=['',link(path,'原版Gaussian关节运动视频')]
lines+=['','## 5. GT指标与协议限制','', '| 指标 | 状态 |','|---|---|','| PSNR / SSIM / LPIPS | unavailable：无标定多视角目标，不将不同视角图像直接比较 |','| Fixed / Moving / Whole CD | unavailable：无已核验对应GT几何与对齐 |','| Joint Type Accuracy / Axis Error / Pivot Error / Range Error | unavailable：没有可靠GT关节 |','', '这是官方释放预训练模型的推理验证，没有本机从头训练 PAct、按对象优化或按GT误差选择 checkpoint。GT-independent Selection 对本次对象级推理没有Part²GS那样的best/final区别；不能声称预训练本身无GT监督，也未验证该示例与训练集是否重叠。','', 'Storage45135正式实验需补齐可追溯二维部件mask、数据预处理/坐标对齐及训练测试集重叠审计；官方示例没有冒充该场景。后续阶段在这些条件核实后再启动。','', '## 6. 保存与监控','',f'成功输出：`{run}`。大型模型/视频/点云只在本机，Git忽略；工具、报告、compact evidence提交独立分支，未向上游推送。','', '```bash','/root/autodl-tmp/conda_envs/pact/bin/python /root/autodl-tmp/PAct/repro/tools/run_status.py --watch','```','', 'Part²GS依用户要求保持暂停，完整checkpoint在iteration5020后，可从5021恢复；本轮没有重启。']
split=read(ROOT/'repro/evidence/storage45135-split-audit.json',{})
if split:
 lines+=['','## 7. Storage45135 划分预审','', '依据 [SINGAPO 官方固定版本 data_split.json]('+split['file_url']+')，StorageFurniture/45135 在 test；train 同时含6条45135-*增强实例。本记录不能证明PAct实际训练用了同一几何，但增强来源与实际训练manifest尚未核验，不能先称为严格未见对象测试。'+link(ROOT/'repro/evidence/storage45135-split-audit.json','固定版本与匹配记录')+'。']
lines+=['','本轮PAct执行的是预训练推理，没有Part²GS的逐对象Repel/Repulsion/Contact等训练步骤或优化器。本轮未做这些Loss的消融或Joint梯度优化；运动由原版回归关节和几何后处理生成。']
dest.write_text('\n'.join(lines)+'\n')
summary={'run':str(run),'scope':'Official real-world tooling smoke, NOT Storage45135 final benchmark','smoke':smoke,'motion_analysis':motion,'mesh_preview':render,'checkpoint_loads':loads,'source':source,'environment':env};(ROOT/'repro/evidence/smoke-results.json').write_text(json.dumps(summary,indent=2,allow_nan=False));print(dest)
