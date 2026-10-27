"""Pin original cases/initialization, and create physically separate target packs."""
import datetime
import json
import pathlib
import subprocess

import torch

from axis_mode_common import atomic_json,sha256

REPO=pathlib.Path(__file__).resolve().parents[2]


def main():
    parent=pathlib.Path((REPO/'repro/current_axis_only_fixed_ranges.txt').read_text().strip())
    integrity=json.loads((parent/'integrity.json').read_text());assert integrity['state']=='complete'
    assert sha256(parent/'results.json')==integrity['results_sha256']
    cfg=json.loads((parent/'config.json').read_text());assert cfg['test_cases']==['8961','45194']
    files=json.loads((parent/'frozen_assets.json').read_text());assert all(sha256(x['path'])==x['sha256'] for x in files)
    now=datetime.datetime.now(datetime.timezone.utc);root=REPO/'repro/outputs'/('axis_mode_ablation_'+now.strftime('%Y%m%dT%H%M%SZ'));root.mkdir(exist_ok=False)
    manifest={'created_utc':now.isoformat(),'parent_root':str(parent),'parent_integrity_sha256':sha256(parent/'integrity.json'),
        'parent_config_sha256':sha256(parent/'config.json'),'modes':['direction','position'],'test_cases':cfg['test_cases'],
        'train_cases':cfg['train_cases'],'formal_trials':12,'steps':200,'initialization':'Original released MLP and same original assets, never previous trained A/B',
        'source_git_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()}
    atomic_json(root/'config.json',manifest)
    hashes={c:sha256(parent/'features'/c/'cache.pt') for c in cfg['runs']};atomic_json(root/'feature_cache_sha256.json',hashes)
    for mode in manifest['modes']:
        sub=root/mode;sub.mkdir();(sub/'features').symlink_to(parent/'features',target_is_directory=True);(sub/'training_targets').mkdir()
        child=dict(cfg);child.update(created_utc=now.isoformat(),mode=mode,route_freeze='Physical FK AND both continuous/official exports freeze original exported counterpart, plus original native limits',
            experiment=f'{mode}-only axis ablation, same original objects and native endpoints',parent_root=str(parent),parent_integrity_sha256=manifest['parent_integrity_sha256'],parent_config_sha256=manifest['parent_config_sha256'],
            head_targets=f'Only {mode} GT channels retained; all other target/mask entries physically zero',
            head_objective='Original 24-channel masked MSE denominator, restricted to selected effective channels; same collision weight',
            head_postprocessing='Official snap/bbox projection, followed by replay of requested frozen physical counterpart and native ranges',
            checkpoint_initialization='Original official MLP; no previous trained A/B used',git_sha=manifest['source_git_sha'])
        atomic_json(sub/'config.json',child)
        for name in ['frozen_assets.json','older_source_assets.json']:(sub/name).write_bytes((parent/name).read_bytes())
        for case in cfg['train_cases']:
            pack=torch.load(parent/'training_targets'/f'{case}.pt',map_location='cpu',weights_only=False)
            lo,hi=(6,9) if mode=='direction' else (9,12)
            for key in ['target','mask']:
                value=torch.zeros_like(pack[key]);value[:,lo:hi]=pack[key][:,lo:hi];pack[key]=value
            torch.save(pack,sub/'training_targets'/f'{case}.pt')
    (REPO/'repro/current_axis_mode_ablation.txt').write_text(str(root)+'\n');print(root,flush=True)


if __name__=='__main__':main()
