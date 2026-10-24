"""Analytic checks for direction/line/pivot errors, never experimental results."""
import ast,json,pathlib,numpy as np
root=pathlib.Path(__file__).resolve().parents[2];tree=ast.parse((root/'repro/tools/evaluate_gt_joints.py').read_text());functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['unit','angle','position']];env={'np':np,'gdiag':2.};exec(compile(ast.Module(body=functions,type_ignores=[]),'metric-math','exec'),env);angle=env['angle'];position=env['position'];g=np.array([0.,0.,1.]);zero=np.zeros(3)
assert angle([0,0,-1],g)==0
assert np.isclose(angle([.5,0,np.sqrt(.75)],g),30)
a=position([.03,.04,10.],[0,0,-1],zero,g);assert np.isclose(a['pivot_to_gt_axis'],.05) and np.isclose(a['axis_line_distance'],.05)
b=position([1.,0,0],[1.,0,1.],zero,g);assert np.isclose(b['pivot_to_gt_axis'],1.) and np.isclose(b['axis_line_distance'],0.)
assert angle(zero,g) is None and position(zero,zero,zero,g)['axis_line_distance'] is None
R=np.array([[0,-1,0],[1,0,0],[0,0,1.]]);s=3.7;t=np.array([12.,-6,9.]);origin=np.array([.3,.4,1]);direction=np.array([.2,.5,1]);first=position(origin,direction,zero,g);after=position(s*(R@origin)+t,R@direction,t,R@g)
assert np.isclose(angle(direction,g),angle(R@direction,R@g))
for key in ['axis_line_distance','pivot_to_gt_axis']:assert np.isclose(after[key],s*first[key])
report={'passed':True,'scope':'Analytic math checks only; not synthetic experimental scene results','checks':['axis-sign invariance','known30degree angle','parallel-offset3-4-5 distance with arbitrary along-axis origin','intersecting nonparallel lines have zero line distance but nonzero pivot error','undefined zero-axis remains unavailable','rigid-frame and scale covariance']};(root/'repro/evidence/joint-metric-math.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
