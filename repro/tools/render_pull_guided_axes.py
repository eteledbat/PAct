"""Post-seal render; reuse only clips with identical full kinematics/membership."""
import argparse,json,pathlib,shutil,subprocess,sys,os
from collision_refinement_common import sha256
from multicase_closed_common import write_json
p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--baseline',type=pathlib.Path,required=True);p.add_argument('--reference-root',type=pathlib.Path,required=True);a=p.parse_args();seal=json.loads((a.root/'AXIS_CANDIDATES_SEAL.json').read_text());assert seal['state']=='complete';states={}
for case in seal['cases']:
 root=a.root/case;old=a.baseline/case;m=json.loads((root/'motion_groups.json').read_text());prev=json.loads((old/'motion_groups.json').read_text());same=len(m['groups'])==len(prev['groups']) and all(all(x.get(k)==y.get(k) for k in ['id','parent','joint','source_node','component','semantic_owner']) for x,y in zip(m['groups'],prev['groups'])) and sha256(root/'membership.npz')==sha256(old/'membership.npz')
 if (root/'video_results.json').exists():states[case]='complete';continue
 if same:
  for name in [f'{case}_GT_Origin_AutoComponentAxis_3views.mp4','three_views_t0.00.png','three_views_t0.50.png','three_views_t1.00.png','video_results.json']:shutil.copyfile(old/name,root/name)
  data=json.loads((root/'video_results.json').read_text());data['reuse_verified_from']=str(old);data['identical_kinematics_and_membership']=True;write_json(root/'video_results.json',data);states[case]='reused_identical'
 else:
  with (root/'render.log').open('w') as log:subprocess.run([sys.executable,str(pathlib.Path(__file__).parent/'render_auto_component_three_views.py'),'--root',str(a.root),'--reference-root',str(a.reference_root),'--case',case],check=True,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4'})
  states[case]='rerendered'
 write_json(a.root/'render_status.json',states);print('VIDEO',case,states[case],flush=True)
