"""Seal pull-guided automatic axis candidates before any GT evaluation."""
import argparse,pathlib,json,hashlib,shutil,sys,os
from auto_part_semantics import sha256,write_json
from auto_component_axis_prior import prepare
p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--auto-root',type=pathlib.Path,required=True);p.add_argument('--registration-root',type=pathlib.Path,required=True);a=p.parse_args();b=a.root;b.mkdir(exist_ok=True);src=a.auto_root;rec=json.loads((src/'PULL_RECOGNITION_SEAL.json').read_text());assert rec['state']=='complete' and rec['source_inputs_unchanged'];protocol=json.loads((src/'protocol.json').read_text());snapshot=b/'source_snapshot';snapshot.mkdir(exist_ok=True)
for name in ['auto_component_axis_prior.py','semantic_motion_rules.py','semantic_axis_router.py','recognize_door_pulls.py','test_door_pull_priority.py','build_pull_guided_axes.py','render_auto_component_three_views.py','evaluate_auto_component_axes.py']:shutil.copyfile(pathlib.Path(__file__).parent/name,snapshot/name)
denied=[];reads=[]
def guard(event,args):
 if event=='open' and isinstance(args[0],(str,bytes)):
  path=os.fsdecode(args[0])
  if any(s in path for s in ['/gt/','/gt_mesh/','mobility_v2','part_info.json','joint_evaluation','/auto_component_axis_priors_20261029_v5/','/auto_component_axis_priors_20261029_v6/']):denied.append(path);raise PermissionError('GT/old joint read forbidden: '+path)
  if path.endswith(('.json','.npz','.pt','.npy','.png')):reads.append(path)
sys.addaudithook(guard)
write_json(b/'protocol.json',dict(cases=protocol['cases'],automatic_semantics=str(src),registration=str(a.registration_root),GT_used_during_preparation=False,rule_priority='Horizontal candidates: visible lower pull -> upper hinge; upper pull -> lower hinge; absent/uncertain/middle -> original cabinet-position fallback. Side doors unchanged.',pull_recognition_seal_sha256=sha256(src/'PULL_RECOGNITION_SEAL.json'),rule_source_sha256=sha256(pathlib.Path(__file__).parent/'auto_component_axis_prior.py'),previous_results='v5 and v6 preserved',geometry_policy='Immutable native mesh/Gaussians; only joints and component grouping exported',geometry_changed=False))
for case in protocol['cases']:
 if not (b/case/'motion_groups.json').exists():prepare(src,a.registration_root,b/case,case)
assert not denied
write_json(b/'AXIS_CANDIDATES_SEAL.json',dict(state='complete',cases=protocol['cases'],GT_forbidden_reads=denied,training_reads=sorted(set(reads)),files={case:dict(manifest_sha256=sha256(b/case/'motion_groups.json'),membership_sha256=sha256(b/case/'membership.npz'),assignment_sha256=sha256(b/case/'component_assignment.json')) for case in protocol['cases']}));print('SEALED all candidate axes; no GT reads',flush=True)
