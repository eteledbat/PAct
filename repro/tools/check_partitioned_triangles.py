"""Validate accelerated all-triangle checks against completed legacy states."""
import argparse,json,pathlib,time
from collision_refinement_common import load_asset,candidate_nodes,sha256
from partitioned_triangle_motion import PartitionedTrianglePairs
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--reference',required=True);p.add_argument('--candidate');p.add_argument('--out',required=True);p.add_argument('--partition-size',type=int,default=12000);a=p.parse_args()
asset=load_asset(a.run);nodes=candidate_nodes(asset,a.candidate);reference=json.loads(pathlib.Path(a.reference).read_text());checker=PartitionedTrianglePairs(asset,partition_size=a.partition_size);rows=[]
for expected in reference:
    start=time.monotonic();actual,_=checker.state(nodes,expected['t']);deltas=[]
    for old,new in zip(expected['pairs'],actual):
        assert old['ids']==new['ids']
        for k in ['contacting_triangle_pairs','nonzero_segments']:assert old[k]==new[k],(k,old[k],new[k])
        for k in ['line_length_over_diagonal','transverse_line_length_over_diagonal']:
            delta=abs(old[k]-new[k]);deltas.append(delta);assert delta<2e-6,(k,old[k],new[k])
    rows.append({'state':expected['state'],'t':expected['t'],'seconds':time.monotonic()-start,'exact_counts_match':True,'max_line_over_D_absolute_difference':max(deltas),'pairs':actual})
    result={'passed':len(rows)==len(reference),'geometry_decimated':False,'partition_size':a.partition_size,'chunk_counts':[len(c) for c in checker.chunks],'candidate_sha256':sha256(a.candidate or asset['object_path']),'reference':a.reference,'reference_states':reference,'states':rows}
    pathlib.Path(a.out).write_text(json.dumps(result,indent=2));print(json.dumps(rows[-1]),flush=True)
