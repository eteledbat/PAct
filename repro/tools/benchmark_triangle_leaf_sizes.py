"""Check a VTK search-tree parameter against already measured full triangles."""
import argparse,json,pathlib,time
from collision_refinement_common import load_asset,sha256
from inspect_triangle_motion import TrianglePairs
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--reference',required=True);p.add_argument('--out',required=True);p.add_argument('--leaf-cells',type=int,required=True);a=p.parse_args()
asset=load_asset(a.run);reference=json.loads(pathlib.Path(a.reference).read_text());checker=TrianglePairs(asset,leaf_cells=a.leaf_cells);rows=[]
for expected in reference[:2]:
    start=time.monotonic();actual,_=checker.state(asset['nodes'],expected['t']);match=True
    for old,new in zip(expected['pairs'],actual):
        assert old['ids']==new['ids']
        for k in ['contacting_triangle_pairs','nonzero_segments']:
            assert old[k]==new[k],(k,old[k],new[k])
        for k in ['line_length_over_diagonal','transverse_line_length_over_diagonal']:
            assert abs(old[k]-new[k])<1e-7,(k,old[k],new[k])
    rows.append({'state':expected['state'],'t':expected['t'],'elapsed_seconds':time.monotonic()-start,'exact_counts_match':True,'lengths_within_1e-7':True,'pairs':actual})
    pathlib.Path(a.out).write_text(json.dumps({'leaf_cells':a.leaf_cells,'source_object_sha256':sha256(asset['object_path']),'states':rows,'passed':len(rows)==min(2,len(reference))},indent=2));print(json.dumps(rows[-1]),flush=True)
