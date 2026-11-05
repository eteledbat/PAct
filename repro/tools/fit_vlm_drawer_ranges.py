"""Post-geometry prismatic range experiment: closure/support/observed-q.

All features and optimization use frozen predicted geometry and source RGB/masks.
GT is prohibited until ALL interval candidates are sealed by the controller.
"""
import argparse,copy,datetime,json,os,pathlib,subprocess,sys
import numpy as np,trimesh
from scipy.optimize import minimize, minimize_scalar
from scipy.spatial import cKDTree
from collision_refinement_common import sha256,transforms,load_asset
from multicase_closed_common import REPO,write_json

BASE=REPO/'repro/outputs/vlm_component_geometry_20261029_v1'
POSE=REPO/'repro/outputs/observed_drawer_pose_20261029_v1'
CASES=['20411','31249','45194']
DEFAULT=dict(overlap_fraction=.20,front_weight=10.,rear_weight=4.,support_weight=4.,collision_weight=1.,span_prior_weight=.05,minimum_span_fraction=.10,collision_radius_fraction=.005,body_sample_count=32768,moving_sample_count=8192,q_grid_count=241,trajectory_samples=21,optimization_grid_count=81)

def read(p):return json.loads(pathlib.Path(p).read_text())
def weighted_quantile(x,w,q):
 x=np.asarray(x);w=np.asarray(w);ok=np.isfinite(x)&np.isfinite(w)&(w>0);x=x[ok];w=w[ok]
 if not len(x):raise ValueError('No area-weighted plane evidence')
 order=np.argsort(x,kind='stable');x=x[order];w=w[order];return float(np.interp(q,np.cumsum(w)/w.sum(),x))
def basis_for(direction):
 d=np.asarray(direction,float);d=d/np.linalg.norm(d);sign=1. if d@np.array([0,0,1.])>=0 else -1.;outward=sign*d;up=np.array([0.,1.,0.]);up-=outward*np.dot(up,outward)
 if np.linalg.norm(up)<1e-6:raise ValueError('Prismatic drawer direction is vertical, no front-facing drawer prior')
 up/=np.linalg.norm(up);side=np.cross(up,outward);return np.column_stack([side,up,outward]),sign

def descendants(nodes,gid):
 ids={gid}
 while True:
  extra=ids|{g['id'] for g in nodes if g['parent'] in ids}
  if extra==ids:return ids
  ids=extra

def geometry_features(drawer,body,direction):
 B,sign=basis_for(direction);v=np.array(drawer.vertices)@B;dc=np.asarray(drawer.triangles_center)@B;dn=np.asarray(drawer.face_normals)@B;da=np.asarray(drawer.area_faces);aligned=abs(dn[:,2])>=.8
 if da[aligned].sum()<.001*drawer.area:raise ValueError('Insufficient front/back planar drawer evidence')
 front=weighted_quantile(dc[aligned,2],da[aligned],.90);rear=weighted_quantile(dc[aligned,2],da[aligned],.10);depth=front-rear
 if depth<1e-4:raise ValueError('Degenerate drawer depth')
 panel=aligned&(dc[:,2]>=front-.08*depth);foot=np.array([[weighted_quantile(dc[panel,k],da[panel],q) for k in [0,1]] for q in [.02,.98]]);extent=np.maximum(foot[1]-foot[0],.01);margin=.1*extent
 bc=np.asarray(body.triangles_center)@B;bn=np.asarray(body.face_normals)@B;ba=np.asarray(body.area_faces);local=np.all((bc[:,:2]>=foot[0]-margin)&(bc[:,:2]<=foot[1]+margin),axis=1)&(abs(bn[:,2])>=.8);fallback=False
 if local.sum()<8 or ba[local].sum()<1e-5:
  local=np.all((bc[:,:2]>=foot[0]-.3*extent)&(bc[:,:2]<=foot[1]+.3*extent),axis=1)&(abs(bn[:,2])>=.7);fallback=True
 if not local.any():raise ValueError('No local predicted cabinet front/back evidence')
 bodyfront=weighted_quantile(bc[local,2],ba[local],.90);bodyback=weighted_quantile(bc[local,2],ba[local],.10)
 if bodyfront-bodyback<.10*depth:raise ValueError('Cabinet front/back depth evidence degenerate')
 return dict(basis=B.tolist(),axis_to_outward_sign=sign,drawer_front=front,drawer_rear=rear,drawer_depth=depth,cabinet_front=bodyfront,cabinet_back=bodyback,front_alignment_q=bodyfront-front,front_panel_footprint=foot.tolist(),cabinet_local_faces=int(local.sum()),cabinet_local_area=float(ba[local].sum()),expanded_local_evidence=fallback,plane_evidence='Area-weighted10/90% of predicted faces with front/back normals; local drawer-front footprint; no GT planes or manual regions')

def losses(endpoints,feature,options,qobs,collision_grid,collision_values,mode='full'):
 lo,hi=map(float,endpoints);depth=feature['drawer_depth'];front=feature['drawer_front'];rear=feature['drawer_rear'];bf=feature['cabinet_front'];bb=feature['cabinet_back'];span=hi-lo;overlap=max(0.,min(front+hi,bf)-max(rear+hi,bb));retained=overlap/depth;native_span=feature['native_outward_range'][1]-feature['native_outward_range'][0]
 close=((front+lo-bf)/depth)**2;rear_penalty=(max(0.,bb-(rear+lo))/depth)**2;support=(retained-options['overlap_fraction'])**2;positions=np.linspace(lo,hi,options['trajectory_samples']);collision=float(np.mean(np.interp(positions,collision_grid,collision_values)));prior=((span-native_span)/depth)**2
 value=options['front_weight']*close+options['rear_weight']*rear_penalty+options['span_prior_weight']*prior
 if mode!='close_only':value+=options['support_weight']*support
 if mode not in ['close_only','close_support']:value+=options['collision_weight']*collision
 # Hard observed containment is enforced in the search, not replaced by a soft loss.
 terms=dict(front_alignment=close,rear_clearance=rear_penalty,support_target=support,retained_depth_fraction=retained,mean_Gaussian_nearness_proxy=collision,native_span_prior=prior,observed_contained=bool(lo-1e-9<=qobs<=hi+1e-9),span=span)
 return float(value),terms

def solve_interval(feature,options,qobs,grid,proxy,mode='full',enforce_observation=True):
 depth=feature['drawer_depth'];a=feature['front_alignment_q'];old=feature['native_outward_range'];lower=min(a-.3*depth,old[0]-.3*depth,qobs-.3*depth);upper=max(a+1.5*depth,old[1]+.3*depth,qobs+.3*depth);minspan=options['minimum_span_fraction']*depth;lo_upper=min(qobs,upper-minspan) if enforce_observation else upper-minspan;hi_lower=max(qobs,lower+minspan) if enforce_observation else lower+minspan;bounds=[(lower,lo_upper),(hi_lower,upper)];best=None
 for lo in np.linspace(*bounds[0],options['optimization_grid_count']):
  for hi in np.linspace(*bounds[1],options['optimization_grid_count']):
   if hi-lo<minspan:continue
   val,_=losses((lo,hi),feature,options,qobs,grid,proxy,mode)
   if best is None or val<best[0]:best=(val,np.array([lo,hi]))
 objective=lambda z:losses(z,feature,options,qobs,grid,proxy,mode)[0];sol=minimize(objective,best[1],method='SLSQP',bounds=bounds,constraints=[dict(type='ineq',fun=lambda z:z[1]-z[0]-minspan)],options=dict(maxiter=120,ftol=1e-11));candidate=sol.x
 if candidate[1]-candidate[0]>=minspan-1e-8 and objective(candidate)<best[0]:best=(objective(candidate),candidate)
 value,terms=losses(best[1],feature,options,qobs,grid,proxy,mode);return dict(outward_range=best[1].tolist(),objective=value,terms=terms,hard_observed_containment=enforce_observation,optimizer_success=bool(sol.success),optimizer_message=str(sol.message),search_bounds=[list(b) for b in bounds])

def guard():
 denied=[];reads=[]
 def hook(event,args):
  if event=='open' and isinstance(args[0],(str,bytes)):
   name=os.fsdecode(args[0]);bad=any(s in name for s in ['/gt/','/gt_mesh/','joint_info','joint_evaluation','mobility_v2','part_info.json','/depth/','/closed_multicase_'])
   if '/FreeArt3D/' in name and name.endswith('_pure.png') and not name.endswith('/05_pure.png'):bad=True
   if '/images/' in name and name.endswith('.png') and not name.endswith('/000000.png'):bad=True
   if bad:denied.append(name);raise PermissionError('GT/extra image forbidden during range fit: '+name)
   if name.endswith(('.json','.npz','.npy','.pt','.png','.ply')):reads.append(name)
 sys.addaudithook(hook);return denied,reads


def load_frozen_model(source,kind):
 import torch
 from fit_observed_drawer_pose import PoseModel
 m=PoseModel(read(source/'config.json'),read(source/'motion_groups.json'),source);checkpoint='observed_pose_geometry.pt' if kind=='observed_pose_geometry' else 'checkpoint.pt';m.load_state_dict(torch.load(source/checkpoint,map_location='cuda',weights_only=False)['model']);m.requires_grad_(False);m.set_pose({});return m

def estimate_observation(m,targets,source,kind):
 if kind=='observed_pose_geometry':return read(source/'pose_estimation.json')['observed_pose'],dict(source='Previously sealed single-input observed pose; no re-estimation or geometry update')
 pose={};trace=[]
 import torch
 with torch.no_grad():
  for sweep in range(2):
   for g in targets:
    lo,hi=sorted(g['joint']['range'])
    def obj(q):pose[g['id']]=float(q);m.set_pose(pose);return float(m.losses()[0])
    grid=np.linspace(lo,hi,49);values=[obj(q) for q in grid];j=int(np.argmin(values));left=grid[max(0,j-1)];right=grid[min(48,j+1)];sol=minimize_scalar(obj,bounds=(left,right),method='bounded',options=dict(xatol=1e-5,maxiter=32));_,q=min((values[j],grid[j]),(float(sol.fun),float(sol.x)));pose[g['id']]=float(q);m.set_pose(pose);trace.append(dict(sweep=sweep,group=g['id'],selected_q=float(q)))
 return {str(k):v for k,v in pose.items()},dict(source='Single original RGB/mask, frozen optimized geometry and frozen camera;49-point interval grid+local bounded refinement,2 coordinate sweeps',trace=trace)


def fit_case(batch,case):
 denied,reads=guard();protocol=read(batch/'protocol.json');entry=protocol['sources'][case];source=pathlib.Path(entry['root']);root=batch/case;root.mkdir(exist_ok=False);obj=read(source/'optimized_asset/object.json');nodes=obj['diffuse_tree'];targets=[g for g in nodes if g['joint']['type']=='prismatic' and g.get('semantic_owner') is not None];targets=[g for g in targets if g['parent'] not in {v['id'] for v in nodes if v.get('semantic_owner')==g.get('semantic_owner')}];meshes={g['id']:trimesh.load(source/'optimized_asset'/g['plys'][0],process=False,force='mesh') for g in nodes};body=trimesh.util.concatenate([meshes[g['id']] for g in nodes if g.get('semantic_owner')==1]);D=float(np.linalg.norm(np.ptp(np.vstack([mesh.bounds for mesh in meshes.values()]),axis=0)));m=load_frozen_model(source,entry['kind']);qobs,audit=estimate_observation(m,targets,source,entry['kind']);write_json(root/'observed_pose.json',dict(pose=qobs,**audit));m=None
 from plyfile import PlyData
 undo=np.array([[1,0,0],[0,0,1],[0,-1,0]],float);rng=np.random.default_rng(42);clouds={}
 for g in nodes:
  data=PlyData.read(source/'optimized_asset'/g['gs'][0])['vertex'].data;alpha=1/(1+np.exp(-data['opacity'].astype(float)));ids=np.flatnonzero(alpha>=.05);xyz=np.column_stack([data[k] for k in ['x','y','z']]).astype(float)@undo.T;clouds[g['id']]=xyz[ids]
 bodycloud=np.concatenate([clouds[g['id']] for g in nodes if g.get('semantic_owner')==1]);bodycloud=bodycloud[rng.choice(len(bodycloud),min(DEFAULT['body_sample_count'],len(bodycloud)),replace=False)];bodytree=cKDTree(bodycloud);results=[];candidates={name:copy.deepcopy(nodes) for name in ['native','close_only','close_support','without_observation','full','support_10pct','support_35pct']}
 for g in targets:
  owned=descendants(nodes,g['id']);drawer=trimesh.util.concatenate([meshes[i] for i in sorted(owned)]);f=geometry_features(drawer,body,g['joint']['axis']['direction']);B=np.array(f['basis']);sign=f['axis_to_outward_sign'];original=np.array(g['joint']['range']);f['native_outward_range']=sorted((original*sign).tolist());qo=sign*float(qobs.get(str(g['id']),0.));depth=f['drawer_depth'];a=f['front_alignment_q'];lower=min(a-.3*depth,f['native_outward_range'][0]-.3*depth,qo-.3*depth);upper=max(a+1.5*depth,f['native_outward_range'][1]+.3*depth,qo+.3*depth);qgrid=np.linspace(lower,upper,DEFAULT['q_grid_count']);moving=np.concatenate([clouds[i] for i in sorted(owned)]);moving=moving[rng.choice(len(moving),min(DEFAULT['moving_sample_count'],len(moving)),replace=False)];outward=B[:,2];eps=DEFAULT['collision_radius_fraction']*D;proxy=[]
  for q in qgrid:
   distance=bodytree.query(moving+q*outward,workers=4)[0];proxy.append(float(np.exp(-distance**2/(2*eps**2)).mean()))
  proxy=np.array(proxy);oldloss,oldterms=losses(f['native_outward_range'],f,DEFAULT,qo,qgrid,proxy);row=dict(group=g['id'],semantic_instance=g.get('semantic_owner'),geometry_features=f,observed_outward_q=qo,native=dict(outward_range=f['native_outward_range'],objective=oldloss,terms=oldterms),variants={})
  variants=[('close_only','close_only',True,.2),('close_support','close_support',True,.2),('without_observation','full',False,.2),('full','full',True,.2),('support_10pct','full',True,.1),('support_35pct','full',True,.35)]
  for name,mode,contain,fraction in variants:
   opts=dict(DEFAULT,overlap_fraction=fraction);fit=solve_interval(f,opts,qo,qgrid,proxy,mode,contain);fit['overlap_fraction']=fraction;fit['joint_range']=(sign*np.array(fit['outward_range'])).tolist();row['variants'][name]=fit;target=next(v for v in candidates[name] if v['id']==g['id']);target['joint']['range']=fit['joint_range']
  results.append(row);np.savez_compressed(root/f'group_{g["id"]}_collision_grid.npz',q=qgrid,proxy=proxy);print('RANGE',case,g['id'],'native',original.tolist(),'full',row['variants']['full']['joint_range'],'qobs',qo,flush=True)
 for name,tree in candidates.items():
  for before,after in zip(nodes,tree):
   x=copy.deepcopy(before);y=copy.deepcopy(after);x['joint'].pop('range');y['joint'].pop('range');assert x==y
  exported=copy.deepcopy(tree)
  for g in exported:g['plys']=[str((source/'optimized_asset'/p).resolve()) for p in g['plys']];g['gs']=[str((source/'optimized_asset'/p).resolve()) for p in g['gs']]
  folder=root/name;folder.mkdir();write_json(folder/'object.json',dict(diffuse_tree=exported,range_policy='Only primary semantic drawer intervals changed; frozen geometry/axes/parents',source_asset=str(source/'optimized_asset/object.json')))
 write_json(root/'range_fit.json',dict(case=case,source=str(source),source_kind=entry['kind'],source_asset_sha256=sha256(source/'optimized_asset/object.json'),frozen_geometry=True,frozen_axes=True,source_qobs=qobs,rows=results,unresolved_prismatic_groups=[g['id'] for g in nodes if g['joint']['type']=='prismatic' and g.get('semantic_owner') is None],GT_forbidden_reads=denied,training_reads=sorted(set(reads)),collision_note='Sampled Gaussian-center nearness proxy, not signed solid penetration or proof of collision-free motion',support_note='Generic retained drawer-depth prior, default20%; not observed rail stops;10/35% sensitivity predeclared'))
 assert not denied;write_json(root/'FIT_SEAL.json',dict(state='complete',range_fit_sha256=sha256(root/'range_fit.json'),candidate_assets={name:sha256(root/name/'object.json') for name in candidates},source_geometry_unchanged=True));print('SEALED',case,flush=True)


def initialize(batch):
 batch.mkdir(exist_ok=False);sources={case:dict(root=str(POSE/'20411/vlm' if case=='20411' else BASE/case),kind='observed_pose_geometry' if case=='20411' else 'fixed_q0_geometry') for case in CASES};protocol=dict(cases=CASES,sources=sources,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),options=DEFAULT,algorithm='Fixed globally before GT evaluation: area-weighted front/back predicted planes,20% drawer-depth engagement target,Gaussian nearness proxy at21 trajectory samples,hard observed containment,weak native-span regularizer and10%-depth noncollapse',targets='Only primary semantic prismatic drawer groups; retain unresolved native-moving fragments and all door joints unchanged',variants=['native','close_only','close_support','without_observation','full','support_10pct','support_35pct'],geometry_policy='Use frozen VLM optimized geometry: latest observed-pose fit for20411; existing fixed-q0 VLM fit for31249/45194. No shape/camera/axis update.',GT_policy='All case interval candidates sealed before GT evaluator. Only original RGB/mask to estimate observations; no additional closed RGB, GT axes/ranges/meshes in fitting.',limitations='Single-image geometry+support prior estimates feasible range, not verified physical stops. Body surfaces may be open; collision term is only a nearness proxy.',code_sha256=sha256(__file__))
 write_json(batch/'protocol.json',protocol);protected={}
 for entry in sources.values():
  source=pathlib.Path(entry['root'])
  for p in source.rglob('*'):
   if p.is_file():protected[str(p)]=sha256(p)
 write_json(batch/'protected_sources.json',protected);snapshot=batch/'source_snapshot';snapshot.mkdir();import shutil;shutil.copyfile(__file__,snapshot/pathlib.Path(__file__).name)


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--case',choices=CASES);a=p.parse_args();batch=a.root.resolve()
 if a.case:fit_case(batch,a.case);return
 if not batch.exists():initialize(batch)
 for case in read(batch/'protocol.json')['cases']:
  if (batch/case/'FIT_SEAL.json').exists():continue
  with (batch/(case+'_fit.log')).open('w') as log:r=subprocess.run([sys.executable,__file__,'--root',str(batch),'--case',case],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4'})
  if r.returncode:raise RuntimeError(case+' failed')
 protected=read(batch/'protected_sources.json');assert all(sha256(p)==h for p,h in protected.items());write_json(batch/'ALL_FITS_SEAL.json',dict(state='complete',cases=CASES,case_seals={c:sha256(batch/c/'FIT_SEAL.json') for c in CASES},source_files_unchanged=True));print('ALL RANGES SEALED BEFORE GT',flush=True)
if __name__=='__main__':main()
