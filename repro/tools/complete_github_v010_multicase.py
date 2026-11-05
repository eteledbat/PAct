"""Wait for all sealed fits, then evaluate/render/finalize; no candidate selection."""
import argparse,os,pathlib,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from fit_observed_drawer_pose import read,REPO
from multicase_closed_common import write_json
from collision_refinement_common import sha256

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);a=p.parse_args();batch=a.root.resolve()
 if not (batch/'protected_historical_files.json').exists():write_json(batch/'protected_historical_files.json',{str(p):sha256(p) for p in (REPO/'repro/outputs/closed_multicase_20261028_v1').rglob('*') if p.is_file()})
 while not (batch/'ALL_FITS_SEAL.json').exists():
  if (batch/'controller_status.json').exists():
   s=read(batch/'controller_status.json');assert not any(v=='failed' for v in s.values()),s
  time.sleep(5)
 states={}
 for stage,script,marker in [('evaluate','evaluate_github_v010_multicase.py','evaluation.json'),('render','render_github_v010_multicase.py','video_results.json')]:
  def job(case):
   if (batch/case/marker).exists():return case,0
   with (batch/f'{case}_{stage}.log').open('w') as f:proc=subprocess.run([sys.executable,str(REPO/'repro/tools'/script),'--root',str(batch),'--case',case],cwd=REPO,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4','OPENCV_IO_ENABLE_OPENEXR':'1'})
   return case,proc.returncode
  with ThreadPoolExecutor(max_workers=2) as pool:
   futures=[pool.submit(job,case) for case in read(batch/'protocol.json')['cases']]
   for future in as_completed(futures):
    case,returncode=future.result();key=case+':'+stage;states[key]='complete' if returncode==0 else 'failed';write_json(batch/'postprocess_status.json',states);print(key,states[key],flush=True)
    if returncode:raise RuntimeError(key+' failed')
 with (batch/'finalize.log').open('w') as f:proc=subprocess.run([sys.executable,str(REPO/'repro/tools/finalize_github_v010_multicase.py'),'--root',str(batch)],cwd=REPO,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4'})
 assert proc.returncode==0,'finalize failed';states['state']='complete';write_json(batch/'postprocess_status.json',states);print('ALL COMPLETE',flush=True)
if __name__=='__main__':main()
