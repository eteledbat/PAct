"""Reuse verified same-run original assets/features; strip range GT targets."""
import argparse
import datetime
import json
import os
import pathlib
import subprocess

import torch

from axis_fixed_common import atomic_json,sha256

REPO=pathlib.Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser();p.add_argument('--parent');a=p.parse_args()
    parent=pathlib.Path(a.parent or (REPO/'repro/current_frozen_joint_routes.txt').read_text().strip()).resolve()
    integrity=json.loads((parent/'integrity.json').read_text());assert integrity['state']=='complete'
    assert sha256(parent/'config.json')==integrity['config_sha256']
    assert sha256(parent/'results.json')==integrity['results_sha256']
    files=json.loads((parent/'frozen_assets.json').read_text());assert all(sha256(f['path'])==f['sha256'] for f in files)
    cfg=json.loads((parent/'config.json').read_text());now=datetime.datetime.now(datetime.timezone.utc)
    root=REPO/'repro/outputs'/('axis_only_fixed_ranges_'+now.strftime('%Y%m%dT%H%M%SZ'));root.mkdir(exist_ok=False)
    cfg.update(created_utc=now.isoformat(),experiment='Axis position/direction only; geometry AND all native numeric endpoints fixed',
        parent_root=str(parent),parent_integrity_sha256=sha256(parent/'integrity.json'),parent_config_sha256=sha256(parent/'config.json'),
        git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        endpoints_frozen=True,GT_ranges_in_training_target=False,
        direct_objective='Same .05 normalized axis/pivot prediction prior, plus same motion proxy if B; range parameter excluded from optimizer',
        head_objective='Official masked articulation MSE restricted to effective axis channels6:12; no range GT targets; same motion proxy if B',
        head_start='Original native start AND end frozen at training, prediction and export',
        head_output_freeze='Only direction and revolute pivot from MLP; all other channels replayed from SAME original per-object prediction, not just detached gradients',
        checkpoint_initialization='Original official MLP states and unchanged original assets; NEVER previous trained A/B weights',
        axis_sign_policy='No extra hemisphere constraint; report oriented axis reversals separately from unsigned GT line error')
    cfg.pop('direct_endpoint_bound',None)
    atomic_json(root/'config.json',cfg)
    (root/'features').symlink_to(os.path.relpath(parent/'features',root),target_is_directory=True)
    for filename in ['frozen_assets.json','older_source_assets.json']:(root/filename).write_bytes((parent/filename).read_bytes())
    targets=root/'training_targets';targets.mkdir()
    manifests={}
    for case in cfg['train_cases']:
        assert case not in cfg['test_cases']
        old=parent/'training_targets'/f'{case}.pt';pack=torch.load(old,map_location='cpu',weights_only=False)
        for key in ['target','mask']:
            value=pack[key].clone();value[:,:6]=0;value[:,12:]=0;pack[key]=value
        assert torch.count_nonzero(pack['target'][:,12:18])==0 and torch.count_nonzero(pack['mask'][:,12:18])==0
        torch.save(pack,targets/f'{case}.pt')
        manifests[case]={'parent_target_sha256':sha256(old),'axis_target_sha256':sha256(targets/f'{case}.pt'),'GT_ranges_removed_before_optimizer':True,'supervision':'Same training GT axis positions/directions and validity mask; other target channels zeroed'}
    atomic_json(root/'training_target_provenance.json',manifests)
    for case in cfg['runs']:
        cache=root/'features'/case/'cache.pt'
        assert cache.is_file()
        protocol=json.loads((parent/'head_shared_A/protocol.json').read_text())
        if case in cfg['train_cases']:assert sha256(cache)==protocol['feature_targets_sha256'][case]['features']
    (REPO/'repro/current_axis_only_fixed_ranges.txt').write_text(str(root)+'\n')
    print(root,flush=True)


if __name__=='__main__':main()
