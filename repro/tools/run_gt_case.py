"""One pinned, fixed-seed PAct inference on an existing GT-mask case. No GT axis input."""
import argparse,datetime,json,os,pathlib,shutil,subprocess,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[2];p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--inputs',type=pathlib.Path);a=p.parse_args();inputs=a.inputs.resolve() if a.inputs else ROOT/'repro/inputs'/f'video_gt_{a.scene}_frame000000';provenance=json.loads((inputs/'input_provenance.json').read_text());assert provenance['silhouette_iou']>.98
run=ROOT/'repro/outputs'/f'gt_{a.scene}_{datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")}';run.mkdir(exist_ok=False);(ROOT/f'repro/current_gt_{a.scene}.txt').write_text(str(run)+'\n');shutil.copy2(inputs/'input_provenance.json',run/'input_provenance.json');started=time.monotonic();peak=0

def save(state,**kwargs):
 p=run/'status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps({'state':state,'scene':a.scene,'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-started,**kwargs},indent=2));tmp.replace(p)
try:
 model=json.loads((ROOT/'repro/evidence/model-download-status.json').read_text());assert model['state']=='verified'
 env=os.environ.copy();env.update({'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4','PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1','TORCH_HOME':str(ROOT/'repro/cache/torch'),'HF_HOME':str(ROOT/'repro/cache/huggingface'),'XDG_CACHE_HOME':str(ROOT/'repro/cache'),'TORCH_EXTENSIONS_DIR':str(ROOT/'repro/cache/torch_extensions'),'SPCONV_ALGO':'native','PACT_DINOV2_REPO':str(ROOT/'repro/vendor/dinov2'),'PACT_LOAD_AUDIT':str(run/'checkpoint_loads.jsonl'),'PACT_JOINT_TRACE':str(run/'joint_trace')})
 argv=[sys.executable,'repro/tools/guarded_infer.py','--model',model['path'],'--data_dir',str(inputs),'--outdir',str(run),'--batch_size','1','--ss_steps','25','--slat_steps','25','--ss_cfg_strength','7','--slat_cfg_strength','7','--arti_mean_num','20','--render_num_frames','21','--export_arti_objects','--save_gs','--no-save_video_grid','--no-save_cond_vis_grid'];(run/'command.json').write_text(json.dumps({'argv':argv,'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'fixed_seed':42,'protocol':provenance['protocol'],'GT_joint_axis_passed_to_model':False,'selection':provenance.get('selection','Fixed first calibrated frame, fixed official model revision, no GT-based sample/checkpoint/seed selection')},indent=2));(run/'source.patch').write_bytes(subprocess.check_output(['git','diff','HEAD'],cwd=ROOT))
 with (run/'inference.log').open('wb') as log:
  proc=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
  while proc.poll() is None:
   gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True).stdout.strip();peak=max(peak,int(gpu.split(',')[0]));save('running',pid=proc.pid,gpu_memory_utilization=gpu,peak_observed_gpu_mib=peak);time.sleep(10)
 if proc.returncode:raise RuntimeError(f'Inference failed with code {proc.returncode}; inspect {run}/inference.log')
 objects=list(run.rglob('object.json'));assert len(objects)==1;obj=json.loads(objects[0].read_text());assert len(obj['diffuse_tree'])==provenance['part_count'],'Original export removed parts; correspondence must be reviewed before scoring'
 import numpy as np
 raw=np.load(run/'joint_trace/raw_head_00.npy');assert len(raw)==provenance['part_count'] and np.isfinite(raw).all()
 for n in obj['diffuse_tree']:
  assert np.isfinite(n['joint']['axis']['origin']).all() and np.isfinite(n['joint']['axis']['direction']).all()
  for rel in n.get('plys',[]):assert (objects[0].parent/rel).is_file()
 save('complete',object=str(objects[0]),part_count=len(obj['diffuse_tree']),peak_observed_gpu_mib=peak);print('COMPLETE',run,flush=True)
except BaseException as e:save('failed',error=f'{type(e).__name__}: {e}',peak_observed_gpu_mib=peak);raise
