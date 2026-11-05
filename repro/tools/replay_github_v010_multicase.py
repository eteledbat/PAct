"""Batch wrapper around the unmodified published v0.1.0 fitting functions."""
import argparse,datetime,json,os,pathlib,subprocess,sys
from fit_observed_drawer_pose import initialize,read,REPO
from collision_refinement_common import sha256
from multicase_closed_common import write_json

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);a=p.parse_args();batch=a.root.resolve()
 if not batch.exists():
  cases=read(REPO/'repro/outputs/closed_multicase_20261028_v1/protocol.json')['cases'];initialize(batch,cases)
  protocol=read(batch/'protocol.json');protocol.update(branches=['vlm'],versions=['GT','Origin','VLM axes only (no geometry)','VLM axes + observed pose + geometry'],release_commit='d0baa9b0f6bf589a0bff1c760954c5b464c6cd38',release_tag='v0.1.0-auto-semantics-pose',cached_semantic_and_axis_stages='Reuse sealed Qwen VLM semantics, pull evidence and v7 axes from the published release; do not rerun stochastic VLM or original PAct generation.',scope='All and only ten cases in closed_multicase_20261028_v1/protocol.json. New fitting from native geometry; no optimized geometry/range experiment reused.',evaluation='After all ten VLM fit seals, frozen original body similarity and pre-fit axis correspondence; original conditioning view, six held-out state0 cameras where available, input-pose surface CD and full-range videos. GT never selects parameters.',human_prior_root_not_used=True)
  protocol.pop('human_prior_root',None);write_json(batch/'protocol.json',protocol)
  manifest=read(REPO/'repro/versions/v0.1.0-auto-semantics-pose.manifest.json');write_json(batch/'release_manifest.json',manifest)
  files=subprocess.check_output(['git','ls-tree','-r','--name-only',protocol['release_commit'],'repro/tools'],cwd=REPO,text=True).splitlines();proof={}
  for f in files:
   original=subprocess.check_output(['git','show',protocol['release_commit']+':'+f],cwd=REPO);local=(REPO/f).read_bytes();assert original==local,f;proof[f]=sha256(REPO/f)
  write_json(batch/'published_code_verification.json',dict(commit=protocol['release_commit'],all_published_tools_byte_identical=True,files_sha256=proof))
  for f in files:
   target=batch/'source_snapshot'/pathlib.Path(f).name;target.write_bytes((REPO/f).read_bytes())
  (batch/'source_snapshot'/pathlib.Path(__file__).name).write_bytes(pathlib.Path(__file__).read_bytes())
 states=read(batch/'controller_status.json') if (batch/'controller_status.json').exists() else {}
 for stage,marker in [('prepare','prepared.json'),('fit','fit_seal.json')]:
  for case in read(batch/'protocol.json')['cases']:
   key=f'{case}:vlm:{stage}';r=batch/case/'vlm'
   if (r/marker).exists():states[key]='complete';continue
   with (batch/f'{case}_{stage}.log').open('w') as log:
    proc=subprocess.run([sys.executable,str(REPO/'repro/tools/fit_observed_drawer_pose.py'),'--root',str(batch),'--case',case,'--branch','vlm','--stage',stage],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4','OPENCV_IO_ENABLE_OPENEXR':'1'})
   states[key]='complete' if proc.returncode==0 else 'failed';write_json(batch/'controller_status.json',states);print(key,states[key],flush=True)
   if proc.returncode:raise RuntimeError(key+' failed')
 seals={c:sha256(batch/c/'vlm/fit_seal.json') for c in read(batch/'protocol.json')['cases']};write_json(batch/'ALL_FITS_SEAL.json',dict(state='complete',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seals=seals,GT_evaluation_not_started=True));states['state']='fit_complete';write_json(batch/'controller_status.json',states);print('ALL TEN FITS SEALED',flush=True)
if __name__=='__main__':main()
