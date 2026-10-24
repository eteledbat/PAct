"""Run one real official example with the pinned pretrained PAct pipeline in isolation."""
import datetime,hashlib,json,os,pathlib,subprocess,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[2];stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');run=ROOT/'repro/outputs'/('official_dishwasher001_'+stamp);run.mkdir(parents=True,exist_ok=False)
status=run/'status.json'
def save(state,**extra):
 tmp=status.with_suffix('.tmp');tmp.write_text(json.dumps({'state':state,'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra},indent=2));tmp.replace(status)
(root_current:=ROOT/'repro/current_run.txt').write_text(str(run)+'\n')
try:
 model=json.loads((ROOT/'repro/evidence/model-download-status.json').read_text());assert model['state']=='verified'
 inputs=ROOT/'repro/inputs/official_dishwasher001'/'Dishwasher';inputs.mkdir(parents=True,exist_ok=True);manifest=[]
 for name in ['Dishwasher_001_processed.png','Dishwasher_001_mask.exr']:
  src=ROOT/'assets/real_world_examples/Dishwasher'/name;dest=inputs/name;content=src.read_bytes()
  if dest.exists():assert dest.read_bytes()==content
  else:dest.write_bytes(content)
  manifest.append({'source':str(src.relative_to(ROOT)),'sha256':hashlib.sha256(content).hexdigest(),'size':len(content)})
 (run/'input_manifest.json').write_text(json.dumps({'scope':'Official real-world tooling smoke; NOT PARIS Storage45135 or final benchmark','mask_source':'Author-provided EXR; automatic/GT provenance not independently established','files':manifest},indent=2))
 env=os.environ.copy();env.update({'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4','PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1','TORCH_HOME':str(ROOT/'repro/cache/torch'),'HF_HOME':str(ROOT/'repro/cache/huggingface'),'XDG_CACHE_HOME':str(ROOT/'repro/cache'),'TORCH_EXTENSIONS_DIR':str(ROOT/'repro/cache/torch_extensions'),'SPCONV_ALGO':'native','PACT_DINOV2_REPO':str(ROOT/'repro/vendor/dinov2'),'PACT_LOAD_AUDIT':str(run/'checkpoint_loads.jsonl')})
 argv=[sys.executable,'repro/tools/guarded_infer.py','--model',model['path'],'--data_dir',str(inputs.parent),'--outdir',str(run),'--batch_size','1','--ss_steps','25','--slat_steps','25','--ss_cfg_strength','7','--slat_cfg_strength','7','--arti_mean_num','20','--render_num_frames','21','--export_arti_objects','--save_gs','--no-save_video_grid','--no-save_cond_vis_grid']
 record={'argv':argv,'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'environment_overrides':{k:env[k] for k in ['CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','TORCH_HOME','HF_HOME','XDG_CACHE_HOME','TORCH_EXTENSIONS_DIR','SPCONV_ALGO','PACT_DINOV2_REPO']},'model':model,'sampling_note':'Original full 25+25 denoising steps, CFG 7, last20 articulation features, seed42 default; 21 orbit preview frames only affects visualization','original_export':'Original raw mesh PLY plus Gaussians; no optional GLB texture baking in this smoke'}
 (run/'command.json').write_text(json.dumps(record,indent=2));(run/'source.patch').write_bytes(subprocess.check_output(['git','diff','HEAD'],cwd=ROOT));started=time.monotonic();save('running',run=str(run),argv=argv)
 with (run/'inference.log').open('wb') as log:
  process=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
  while process.poll() is None:
   gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
   save('running',run=str(run),pid=process.pid,elapsed_seconds=time.monotonic()-started,gpu_memory_utilization=gpu.stdout.strip());time.sleep(10)
 if process.returncode:raise RuntimeError(f'Official inference exited {process.returncode}; see {run}/inference.log')
 objects=list(run.rglob('object.json'));videos=list(run.rglob('*.mp4'));assert len(objects)==1,(objects,videos)
 import numpy as np,trimesh
 obj=json.loads(objects[0].read_text());checks=[]
 for node in obj['diffuse_tree']:
  assert np.isfinite(node['joint']['axis']['origin']).all() and np.isfinite(node['joint']['axis']['direction']).all(),node
  for key in ['plys','gs']:
   assert node.get(key),f'Missing {key} for node {node["id"]}'
   for relative in node[key]:assert (objects[0].parent/relative).is_file(),relative
  mesh=trimesh.load(objects[0].parent/node['plys'][0],process=False,force='mesh');assert len(mesh.vertices)>0 and len(mesh.faces)>0 and np.isfinite(mesh.vertices).all()
  checks.append({'id':node['id'],'name':node['name'],'joint_type':node['joint']['type'],'vertices':len(mesh.vertices),'faces':len(mesh.faces),'watertight':bool(mesh.is_watertight),'winding_consistent':bool(mesh.is_winding_consistent)})
 assert len(videos)>=2,'Missing actual motion / exploded videos'
 result={'passed':True,'scope':'Official supplied sample only, not benchmark reproduction','elapsed_seconds':time.monotonic()-started,'object':str(objects[0]),'parts':checks,'videos':[str(p) for p in videos]};(run/'verified.json').write_text(json.dumps(result,indent=2));save('complete',**result);print(json.dumps(result,indent=2),flush=True)
except BaseException as e:
 save('failed',run=str(run),error=f'{type(e).__name__}: {e}');raise
