"""21-state diagnostics of original PAct exports. No optimization or GT inputs.
Nearest-center soft overlap is an explicitly defined Gaussian proxy, not volume.
"""
import argparse,copy,datetime,json,math,pathlib,sys,time
import numpy as np
from scipy.spatial import cKDTree
from plyfile import PlyData
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=pathlib.Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from modules.pact.pipelines import PActPipeline  # Bootstrap original circular imports in their working entry order.
from modules.utils.articulation_utils import _local_transform,_build_hierarchy
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();run=pathlib.Path(a.run).resolve();assert run.is_relative_to(ROOT/'repro/outputs')
assert json.loads((run/'verified.json').read_text())['passed']
obj_path=next(run.rglob('object.json'));nodes=json.loads(obj_path.read_text())['diffuse_tree'];out=run/'motion_analysis';out.mkdir(exist_ok=False)
started=time.monotonic();colors=['#3784be','#e88b31','#43a875','#bf6592'];threshold=.05
# save_ply applies R by default, after the export wrapper applied inverse(R).
# Undo that final serialization rotation to match mesh and object.json exactly.
R=np.array([[1,0,0],[0,0,-1],[0,1,0]],dtype=np.float64);undo=np.linalg.inv(R)
parts=[];audits=[]
for i,node in enumerate(nodes):
 mesh=trimesh.load(obj_path.parent/node['plys'][0],force='mesh',process=False)
 v=PlyData.read(obj_path.parent/node['gs'][0])['vertex'].data
 xyz=np.column_stack([v[k] for k in ['x','y','z']]).astype(np.float64)@undo.T
 alpha=1/(1+np.exp(-v['opacity'].astype(np.float64)));sigma=np.exp(np.column_stack([v[k] for k in ['scale_0','scale_1','scale_2']]).astype(np.float64)).mean(1);keep=alpha>=threshold
 assert keep.any() and np.isfinite(xyz).all() and np.isfinite(sigma).all()
 parts.append({'node':node,'mesh':mesh,'xyz':xyz[keep],'alpha':alpha[keep],'sigma':sigma[keep]})
 counts=np.bincount(mesh.edges_unique_inverse);boundary=int((counts==1).sum());nonmanifold=int((counts>2).sum())
 components=trimesh.graph.connected_components(mesh.face_adjacency,nodes=np.arange(len(mesh.faces)),engine='scipy');sizes=sorted([len(c) for c in components],reverse=True)
 audits.append({'id':node['id'],'name':node['name'],'vertices':len(mesh.vertices),'faces':len(mesh.faces),'face_components':len(sizes),'largest_component_face_fraction':sizes[0]/len(mesh.faces),'boundary_edges':boundary,'nonmanifold_edges':nonmanifold,'watertight':bool(mesh.is_watertight),'winding_consistent':bool(mesh.is_winding_consistent),'signed_volume':float(mesh.volume),'self_intersection_checked':False,'self_intersection':None,'gaussian_count':len(xyz),'retained_count':int(keep.sum()),'retained_opacity_fraction':float(alpha[keep].sum()/alpha.sum()),'mesh_bounds':mesh.bounds.tolist(),'aligned_gaussian_bounds':np.array([xyz.min(0),xyz.max(0)]).tolist()})
 print('LOADED',node['name'],len(xyz),'retained',keep.sum(),'watertight',mesh.is_watertight,flush=True)

def pair_metrics(x,y,aa,ab,sa,sb,hotspots=False):
 da,ia=cKDTree(y).query(x,workers=4);db,ib=cKDTree(x).query(y,workers=4)
 ka=ab[ia]*np.exp(-da**2/(2*(sa**2+sb[ia]**2)));kb=aa[ib]*np.exp(-db**2/(2*(sb**2+sa[ib]**2)))
 gaps=np.concatenate([da-sa-sb[ia],db-sb-sa[ib]])
 metrics={'soft_overlap_nearest_center_proxy':float((np.dot(aa,ka)+np.dot(ab,kb))/(aa.sum()+ab.sum())),'minimum_center_distance':float(min(da.min(),db.min())),'minimum_1sigma_gap_among_nearest_center_pairs_proxy':float(gaps.min()),'fraction_negative_nearest_pair_1sigma_gap':float((gaps<0).mean())}
 return metrics,ka,kb
# Independent all-pairs check on real point subsets, for nearest-distance/kernel math.
p0,p1=parts[:2];x,y=p0['xyz'][:31],p1['xyz'][:43];d=np.linalg.norm(x[:,None,:]-y[None,:,:],axis=-1)
m,ka,kb=pair_metrics(x,y,p0['alpha'][:31],p1['alpha'][:43],p0['sigma'][:31],p1['sigma'][:43]);assert np.isclose(m['minimum_center_distance'],d.min())
ia=d.argmin(1);expected=p1['alpha'][:43][ia]*np.exp(-d.min(1)**2/(2*(p0['sigma'][:31]**2+p1['sigma'][:43][ia]**2)));assert np.allclose(ka,expected)
order=_build_hierarchy(nodes);lookup={n['id']:n for n in nodes};curve=[];snapshots={}
for state,t in enumerate(np.linspace(0,1,21)):
 world={-1:np.eye(4)}
 for nid in order:world[nid]=world.get(lookup[nid].get('parent',-1),np.eye(4))@_local_transform(lookup[nid],float(t))
 transformed=[part['xyz']@world[part['node']['id']][:3,:3].T+world[part['node']['id']][:3,3] for part in parts];pairs=[]
 for i in range(len(parts)):
  for j in range(i+1,len(parts)):
   metrics,hi,hj=pair_metrics(transformed[i],transformed[j],parts[i]['alpha'],parts[j]['alpha'],parts[i]['sigma'],parts[j]['sigma']);metrics['parts']=[nodes[i]['id'],nodes[j]['id']];pairs.append(metrics)
   # Top 0.5% nearest-kernel values locate high proxy regions; no physical label.
   if state in [0,10,20]:
    for idx,h in [(i,hi),(j,hj)]:
     n=min(3000,max(1,int(.005*len(h))));chosen=np.argpartition(h,-n)[-n:];np.savez_compressed(out/f'hotspot_state{state:02d}_part{idx}.npz',xyz=transformed[idx][chosen],proxy=h[chosen])
 if state in [0,10,20]:snapshots[state]=(transformed,world)
 curve.append({'state':state,'t':float(t),'pairs':pairs,'transforms':{str(k):v.tolist() for k,v in world.items() if k!=-1}})
 print('STATE',state,'proxy',pairs[0]['soft_overlap_nearest_center_proxy'],'min_center',pairs[0]['minimum_center_distance'],flush=True)
(out/'collision_states.json').write_text(json.dumps(curve,indent=2,allow_nan=False));(out/'mesh_audit.json').write_text(json.dumps(audits,indent=2,allow_nan=False))
fig,axes=plt.subplots(3,1,figsize=(9,8),sharex=True)
fields=['soft_overlap_nearest_center_proxy','minimum_center_distance','minimum_1sigma_gap_among_nearest_center_pairs_proxy'];labels=['Nearest-center Gaussian overlap proxy','Minimum retained-center distance','Nearest-pair 1σ sphere gap proxy']
for ax,key,label in zip(axes,fields,labels):
 for pair in range(len(curve[0]['pairs'])):ax.plot([v['t'] for v in curve],[v['pairs'][pair][key] for v in curve],marker='.',label=str(curve[0]['pairs'][pair]['parts']))
 ax.set_ylabel(label,fontsize=9);ax.grid(alpha=.25)
axes[2].axhline(0,color='grey',linestyle='--');axes[2].set_xlabel('Normalized predicted motion range');fig.suptitle('Official Dishwasher_001: Gaussian proxies, NOT solid collision');fig.tight_layout();fig.savefig(out/'collision_curve.png',dpi=160);plt.close(fig)
rng=np.random.default_rng(0)
for state,(xyzs,world) in snapshots.items():
 fig=plt.figure(figsize=(15,5))
 for view,azim in enumerate([25,125,225]):
  ax=fig.add_subplot(1,3,view+1,projection='3d')
  for idx,x in enumerate(xyzs):
   selection=rng.choice(len(x),min(12000,len(x)),replace=False);pts=x[selection];ax.scatter(*pts.T,s=.15,c=colors[idx%len(colors)],alpha=.3,rasterized=True)
   node=nodes[idx]
   if node['joint']['type']!='fixed':
    tr=world[node['id']];o=np.asarray(node['joint']['axis']['origin']);axis=np.asarray(node['joint']['axis']['direction']);ax.quiver(*o,*(axis*.45),color='red',linewidth=2)
  ax.set(xlim=(-.85,.85),ylim=(-.85,.85),zlim=(-.85,1.3),xlabel='x',ylabel='y (up)',zlabel='z');ax.set_box_aspect([1,1,1.25]);ax.view_init(elev=15,azim=azim,vertical_axis='y')
 fig.suptitle(f'State {state}/20; display subset of Gaussian centers; red = exported predicted axis');fig.tight_layout();fig.savefig(out/f'gaussians_state{state:02d}.png',dpi=150);plt.close(fig)
mesh_reason='Base mesh is open; self-intersections also not fully verified. No mesh repair applied.' if not all(x['watertight'] for x in audits) else 'Self-intersections not yet independently verified.'
summary={'scope':'Official real-world supplied sample diagnostic; not PARIS45135 benchmark','states':21,'opacity_threshold':threshold,'proxy_definition':'Symmetric opacity-weighted nearest-center Gaussian kernel: sum_i alpha_i alpha_NN(i) exp(-d_i²/[2(sigma_i²+sigma_NN(i)²)]) / sum_i alpha_i over both directions; sigma=mean activated scales. All retained centers used; not all-pairs Gaussian kernel.','distance_definition':'Exact minimum Euclidean center distance over all opacity-retained cross-part centers; sphere gaps only among nearest-center pairs, not global minimum sphere distance or mesh penetration.','gaussian_serialization_rotation_undone':undo.tolist(),'coordinate_evidence':'Original export applies inverse(R) to both geometry and joints, then Gaussian.save_ply applies R again. Evaluation undoes only that final serialization R; no GT/ICP fitting.','mesh_collision':{'status':'unavailable','reason':mesh_reason,'intersection_volume':None,'penetration_depth':None},'GT_metrics':{'PSNR':None,'SSIM':None,'LPIPS':None,'chamfer_distance':None,'axis_direction_error':None,'pivot_error':None,'joint_type_accuracy':None,'motion_range_error':None,'reason':'Official real-world sample has no verified GT geometry/joints or calibrated multiview target.'},'predicted_joints':[n['joint'] for n in nodes],'mesh_audit':audits,'numeric_check_real_subsets_passed':True,'elapsed_seconds':time.monotonic()-started}
(out/'metrics.json').write_text(json.dumps(summary,indent=2,allow_nan=False));print('ANALYSIS COMPLETE',out,flush=True)
