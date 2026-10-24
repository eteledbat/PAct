"""Read-only progress monitor independent of Codex."""
import argparse,datetime,json,pathlib,subprocess,time
p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');p.add_argument('--interval',type=int,default=20);a=p.parse_args();root=pathlib.Path(__file__).resolve().parents[2]
while True:
 print(datetime.datetime.now(datetime.timezone.utc).isoformat(),flush=True)
 current=root/'repro/current_run.txt'
 if current.exists():
  run=pathlib.Path(current.read_text().strip());print('Run:',run,flush=True)
  for p in [run/'status.json',run/'motion_analysis/metrics.json',run/'mesh_previews/render_info.json']:
   if p.exists():
    j=json.loads(p.read_text());print(p.name, {k:j[k] for k in ['state','passed','states','elapsed_seconds','error'] if k in j},flush=True)
  log=run/'motion-analysis.log'
  if log.exists():
   lines=[x for x in log.read_text(errors='replace').splitlines() if x.startswith(('STATE','ANALYSIS COMPLETE'))];print('Motion:',lines[-1] if lines else 'initializing',flush=True)
 print(subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout.strip(),flush=True)
 if not a.watch:break
 time.sleep(a.interval)
