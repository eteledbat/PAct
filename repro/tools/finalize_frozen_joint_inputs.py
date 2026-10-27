"""Seal the same-run original assets/features, then prepare declared train GT."""
import argparse
import json
import pathlib
import subprocess
import sys

from collision_refinement_common import sha256

REPO=pathlib.Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=pathlib.Path(a.root)
    cfg=json.loads((root/'config.json').read_text());cfg.setdefault('source_runs',dict(cfg['runs']))
    files=[]
    for c in cfg['train_cases']+cfg['test_cases']:
        run=root/'features'/c;status=json.loads((run/'status.json').read_text());assert status['same_run_feature_asset_association']
        cfg['runs'][c]=str(run)
        obj=pathlib.Path(status['object']);nodes=json.loads(obj.read_text())['diffuse_tree']
        ps=[obj]+[obj.parent/rel for n in nodes for key in ['plys','gs'] for rel in n[key]]
        files.extend({'case':c,'path':str(p),'sha256':sha256(p)} for p in ps)
        if not (run/'joint_evaluation/joint_errors.json').exists():
            with (root/f'baseline_GT_evaluation_{c}.log').open('wb') as log:
                subprocess.run([sys.executable,str(REPO/'repro/tools/evaluate_gt_joints.py'),'--run',str(run)],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
    cfg['baseline_association']='Features and geometry from the SAME new official fixed-seed inference; older originals retained separately. Cross-run replay drift reported, not assumed bitwise equality.'
    # No optimization has started; this finalizes inputs before the config SHA is used.
    assert not any(root.glob('*/checkpoint.pt')),'Cannot change protocol after optimization starts'
    (root/'config.json').write_text(json.dumps(cfg,indent=2))
    if not (root/'older_source_assets.json').exists():(root/'older_source_assets.json').write_bytes((root/'frozen_assets.json').read_bytes())
    (root/'frozen_assets.json').write_text(json.dumps(files,indent=2))
    if not (root/'training_targets').exists():
        subprocess.run([sys.executable,str(REPO/'repro/tools/prepare_frozen_head_targets.py'),'--root',str(root)],cwd=REPO,check=True)
    print('INPUTS SEALED',flush=True)


if __name__=='__main__':main()
