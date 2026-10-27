"""Fix object splits, budgets and provenance before either optimization route."""
import datetime
import json
import pathlib
import subprocess

from collision_refinement_common import sha256

REPO = pathlib.Path(__file__).resolve().parents[2]


def main():
    now = datetime.datetime.now(datetime.timezone.utc)
    root = REPO/'repro/outputs'/('frozen_joint_routes_'+now.strftime('%Y%m%dT%H%M%SZ'))
    root.mkdir(exist_ok=False)
    train = ['10489','31249','45503','9016']
    test = ['8961','45194']
    runs = {c:(REPO/f'repro/current_gt_{c}.txt').read_text().strip() for c in train+test}
    assert not set(train)&set(test)
    cfg = {'created_utc':now.isoformat(), 'runs':runs,'train_cases':train,'test_cases':test,
           'direct_cases':test,'steps':200,'smoke_steps':30,'sources_per_part':512,
           'training_states_per_path':21,'evaluation_states_per_path':41,
           'direct_prior_weight':.05,'direct_collision_weight':1.,'head_collision_weight':.02,
           'head_lr':.0001,'seed':42,'selection':'Fixed final step; no GT checkpoint selection',
           'direct_objective':'Normalized quadratic axis/pivot/endpoint prior to original prediction, plus motion proxy if B; NO RGB/mask fitting',
           'head_objective':'Official masked articulation MSE restricted to requested continuous channels, plus motion proxy if B',
           'head_trainable':'Released art_out_layers only; feature projections and all geometry-generating weights frozen',
           'head_start':'Zero, as in original representation; only direct route opens both endpoints',
           'head_features':'Frozen single INFERENCE timestep pooled features at 6/12/18/25; evaluation uses official last20 mean-feature regression. Not full official flow-matching continuation',
           'head_targets':'GT training-object joint labels transformed by fixed source rotation and static-AABB similarity; signed span in zero-start gauge; official 24-channel encoder',
           'GT_train_cases':train,'GT_test_in_optimizer':False,
           'type_tree_semantics_frozen':True,'local_geometry_appearance_frozen':True,
           'head_postprocessing':'Report both continuous pre-snap diagnostic and official axis snapping/bbox projection; no straight-through estimator',
           'pretrained_overlap':'Local train/test object IDs are disjoint; pretrained PAct data overlap has not been excluded',
           'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()}
    (root/'config.json').write_text(json.dumps(cfg,indent=2))
    files=[]
    for c,run in runs.items():
        run=pathlib.Path(run); status=json.loads((run/'status.json').read_text())
        assert status['state']=='complete'
        obj=pathlib.Path(status['object']); nodes=json.loads(obj.read_text())['diffuse_tree']
        paths=[obj]+[obj.parent/rel for n in nodes for key in ['plys','gs'] for rel in n[key]]
        files.extend({'case':c,'path':str(p),'sha256':sha256(p)} for p in paths)
    (root/'frozen_assets.json').write_text(json.dumps(files,indent=2))
    (REPO/'repro/current_frozen_joint_routes.txt').write_text(str(root)+'\n')
    print(root)


if __name__=='__main__': main()
