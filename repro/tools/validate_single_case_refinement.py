"""Check immutable inputs, finite fixed-step optimization and actual artifacts."""
import argparse, copy, json, pathlib
import imageio.v2 as imageio
import numpy as np
from PIL import Image
from collision_refinement_common import load_asset,sha256,transforms

p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
root=pathlib.Path(a.root);config=json.loads((root/'experiment.json').read_text());name,run=next(iter(config['cases'].items()))
assert len(config['cases'])==1
asset=load_asset(run);original=asset['object'];variant=config['variants'][0];trial=root/f'{name}_{variant}'
assert json.loads((root/'controller_status.json').read_text())['state']=='complete'
assert json.loads((trial/'status.json').read_text())['state']=='complete'
seal=json.loads((trial/'prediction_seal.json').read_text());assert seal['frozen_files_unchanged'] and seal['GT_reads_blocked']==[]
assert sha256(trial/'candidate.json')==seal['candidate_sha256']
protocol=json.loads((trial/'protocol.json').read_text());assert protocol['steps']==config['steps']==200
for row in protocol['frozen_files']:
    assert sha256(row['mesh'])==row['mesh_sha256'] and sha256(row['gs'])==row['gs_sha256']
for file,digest in protocol['source_sha256'].items():assert sha256(file)==digest
history=json.loads((trial/'history.json').read_text());assert [r['step'] for r in history]==list(range(1,201))
for row in history:
    assert np.isfinite([row['loss'],row['gaussian_proxy'],row['gradient_norms']['direction']]).all()
    assert row['gradient_norms']['pivot'] is None
candidate=json.loads((trial/'candidate.json').read_text());clean=copy.deepcopy(candidate);clean.pop('geometry_source_object')
direction_changes=[]
for old,new in zip(original['diffuse_tree'],clean['diffuse_tree']):
    assert old['id']==new['id']
    if old['joint']['type']=='prismatic':
        d0=np.array(old['joint']['axis']['direction']);d1=np.array(new['joint']['axis']['direction'])
        angle=float(np.degrees(np.arccos(np.clip(d0@d1/np.linalg.norm(d0)/np.linalg.norm(d1),-1,1))))
        assert angle<=5.001;direction_changes.append({'id':old['id'],'degrees':angle})
        new['joint']['axis']['direction']=old['joint']['axis']['direction']
assert original==clean,'A field other than the allowed prismatic direction changed'
assert all(np.array_equal(transforms(original['diffuse_tree'],0)[nid],transforms(candidate['diffuse_tree'],0)[nid]) for nid in transforms(original['diffuse_tree'],0))
metrics={};tri={}
for method in ['original',variant]:
    ev=json.loads((root/f'eval_{name}_{method}/metrics.json').read_text());mesh=json.loads((root/f'mesh_{name}_{method}/metrics.json').read_text())
    assert len(ev['states'])==41 and len(mesh['states'])==21
    assert sum(x['interleaved_heldout_time'] for x in ev['states'])==20
    assert all(np.isfinite(row['mean_pair_proxy']) for row in ev['states'])
    assert all(np.isfinite(p['line_length_over_diagonal']) for row in mesh['states'] for p in row['pairs'])
    assert all(c['max_GPU_NN_distance_excess_over_CPU_on_31_real_points']<2e-6 for c in ev['independent_GPU_CPU_NN_checks'])
    assert ev['solid_collision']['volume'] is None and ev['solid_collision']['penetration_depth'] is None
    assert all(r['pivot_to_GT_axis_percent_static_diagonal'] is None for r in ev['GT']['joints'])
    metrics[method]=ev;tri[method]=mesh
assert metrics['original']['initial_state_mean_pair_proxy']==metrics[variant]['initial_state_mean_pair_proxy']
assert tri['original']['states'][0]==tri[variant]['states'][0]
image=np.asarray(Image.open(root/f'previews_{name}/mesh_state00_comparison.png'));H,W=image.shape[:2];width=W//2;height=H//2
assert all(np.array_equal(image[row*height+30:(row+1)*height,:width],image[row*height+30:(row+1)*height,width:]) for row in [0,1])
videos={}
for path in [root/f'previews_{name}/original_{variant}.mp4',root/'GT_mesh_previews/GT_original_refined.mp4']:
    reader=imageio.get_reader(str(path));count=reader.count_frames();reader.close();assert count==21;videos[str(path.relative_to(root))]={'frames':count,'sha256':sha256(path)}
for path in [root/'figures/collision_curves.png',root/'figures/GT_direction_overlay.png',root/f'hotspots_{name}/regions.json']:
    assert path.is_file()
provenance=json.loads((asset['run']/'input_provenance.json').read_text());assert provenance['silhouette_iou']>.98 and provenance['part_count']==2
assert all(sha256(row['path'])==row['sha256'] for row in provenance['files'])
result={'passed':True,'case':name,'source_mesh_GS_SHA_unchanged':True,'only_allowed_joint_direction_changed':True,
        'direction_changes':direction_changes,'steps':len(history),'finite_loss_and_direction_gradients':True,'prismatic_pivot':'not_applicable; no fake pivot gradient',
        'GT_reads_blocked_during_optimization':seal['GT_reads_blocked'],'selection':'Original fixed seed42 / refined fixed final step200',
        'Gaussian_states_per_method':41,'heldout_times_per_method':20,'full_triangle_states_per_method':21,
        'zero_state_motion_metrics_and_unlabelled_preview_pixels_identical':True,'videos':videos,'true_solid_collision_metrics':'unavailable, never zero-filled',
        'input_source_SHA_unchanged':True}
(root/'integrity.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
