"""Isolate RGB-only fitting workers from post-seal GT evaluation workers."""
import argparse,json,os,pathlib,subprocess,sys,shutil
from collision_refinement_common import sha256
from multicase_closed_common import REPO,write_json
p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=pathlib.Path(a.root).resolve();cfg=json.loads((root/'protocol.json').read_text());cases=cfg['cases'];states={c:{} for c in cases}
snap=root/'source_snapshot';snap.mkdir(exist_ok=True)
for name in ['semantic_parts_ablation.py','run_semantic_parts_ablation.py','multicase_closed_common.py','evaluate_multicase_closed.py','render_multicase_closed.py','add_gt_multicase_videos.py']:
 shutil.copyfile(REPO/'repro/tools'/name,snap/name)
protected={}
for case in cases:
 base=pathlib.Path(cfg['baseline'])/case
 for name in ['config.json','motion_groups.json','observation.npz','membership.npz','checkpoint.pt','optimized_parts.npz','fit_results.json','evaluation.json','render_results.json']:
  protected[str(base/name)]=sha256(base/name)
write_json(root/'baseline_preservation.json',protected)
for stage in ['prior','fit','evaluate','render','gt']:
 for case in cases:
  # A sealed result is never re-fit during an output-only retry.
  marker={'prior':'motion_prior_search.json','fit':'fit_results.json','evaluate':'evaluation.json','render':'render_results.json','gt':'gt_video_results.json'}[stage]
  if (root/case/marker).exists():states[case][stage]='complete';continue
  env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4')
  with (root/case/f'{stage}.log').open('w') as log:
   proc=subprocess.run([sys.executable,str(REPO/'repro/tools/semantic_parts_ablation.py'),'--root',str(root),'--stage',stage,'--case',case],env=env,cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
  states[case][stage]='complete' if proc.returncode==0 else 'failed';write_json(root/'controller_status.json',states);print(stage,case,states[case][stage],flush=True)
  if proc.returncode:raise RuntimeError(f'{stage} failed: {case}')
assert all(sha256(pathlib.Path(p))==digest for p,digest in protected.items())
write_json(root/'controller_status.json',dict(state='complete',cases=states,baseline_unchanged=True))
