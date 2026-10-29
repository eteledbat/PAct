"""Reusable semantic/attachment motion rules. No dataset IDs or GT parameters.

Names are hypotheses; articulated carrying handles require separate attachment
and component evidence. Door/drawer grips stay attached to their moving parent.
"""
import copy
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from collision_refinement_common import transforms


def unit(v):
 v=np.asarray(v,float);n=np.linalg.norm(v)
 if n<1e-10:raise ValueError('Insufficient direction evidence')
 return v/n


def canonical_axis(direction,snap_degrees=12.):
 direction=unit(direction);k=int(np.argmax(abs(direction)))
 if direction[k]<0:direction=-direction
 angle=float(np.degrees(np.arccos(np.clip(abs(direction[k]),0,1))))
 if angle<=snap_degrees:
  direction=np.eye(3)[k]
 return direction,dict(snap_threshold_degrees=snap_degrees,nearest_axis_angle_degrees=angle,snapped=angle<=snap_degrees)


def fit_attachment_axis(attachments,mask_attachment=None,mask_weight=2.,snap_degrees=12.):
 """Axis from separated physical attachments; pivot fit in perpendicular plane.

 Along-axis origin coordinate is unidentifiable, retained at attachment mean.
 """
 points=np.asarray(attachments,float)
 if len(points)<2:raise ValueError('At least two separated attachments required')
 center=points.mean(0);_,sv,V=np.linalg.svd(points-center,full_matrices=False)
 if sv[0]<1e-6:raise ValueError('Coincident attachments cannot define rotation axis')
 direction,snap=canonical_axis(V[0],snap_degrees)
 P=np.eye(3)-np.outer(direction,direction)
 target=points.sum(0);denom=float(len(points))
 if mask_attachment is not None:target+=mask_weight*np.asarray(mask_attachment);denom+=mask_weight
 pivot=P@(target/denom)+direction*np.dot(direction,center)
 residual=float(np.sqrt(np.mean(np.sum(((points-pivot)@P)**2,axis=1))))
 return pivot,direction,dict(**snap,geometry_weight=1.,mask_weight=mask_weight if mask_attachment is not None else 0.,attachment_perpendicular_RMS=residual)


def semantic_handle_role(name,parent_role,attachments,object_diagonal):
 """A handle label alone never establishes an independent articulation."""
 name=str(name).lower();parent_role=str(parent_role).lower();points=np.asarray(attachments,float)
 if parent_role in ['door','drawer','sliding_drawer','side_door','upward_door']:
  return 'fixed_grip','Handle follows its door/drawer parent'
 if len(points)>=2 and np.linalg.norm(np.ptp(points,axis=0))>.10*object_diagonal:
  return 'articulated_handle','Two separated supports on static parent'
 return 'uncertain','Insufficient independent attachment evidence; retain original motion'


def radial_basis(vertices,pivot,direction):
 direction=unit(direction);P=np.eye(3)-np.outer(direction,direction);points=(np.asarray(vertices)-pivot)@P
 _,_,V=np.linalg.svd(points[::max(1,len(points)//5000)],full_matrices=False);radial=unit(P@V[0])
 if radial@points.mean(0)<0:radial=-radial
 B=np.column_stack([radial,direction,np.cross(radial,direction)])
 length=max(float(np.quantile((np.asarray(vertices)-pivot)@radial,.99)),.05)
 return B,length


def outward_sign(direction,free_radius,outward):
 derivative=np.cross(unit(direction),np.asarray(free_radius))
 return 1. if np.dot(derivative,outward)>=0 else -1.


def top_hinged_door(vertices,body_vertices,native_joint,minimum_span=1e-6):
 """Upper attachment line, horizontal edge axis, free edge lifts outwards.

 Coordinates are canonical Y-up/front+Z. Initial observed pose remains q=0.
 """
 v=np.asarray(vertices,float);body=np.asarray(body_vertices,float)
 high=np.quantile(v[:,1],.98);band=v[v[:,1]>=high-.02*max(np.ptp(v[:,1]),1e-3)]
 # Fit horizontal direction from upper edge rather than forcing global X.
 edge=band[:,[0,2]]-np.median(band[:,[0,2]],axis=0);_,_,V=np.linalg.svd(edge[::max(1,len(edge)//5000)],full_matrices=False)
 direction=np.array([V[0,0],0.,V[0,1]]);direction,snap=canonical_axis(direction)
 distances=cKDTree(body).query(band,workers=4)[0];near=band[distances<=np.quantile(distances,.5)]
 pivot=np.median(near,axis=0)
 # The hinge line runs through the panel mid-plane, not its front skin.
 # Otherwise a thick closed panel can initially move DOWN at a tiny stroke.
 centered=v-np.median(v,axis=0);_,_,plane_axes=np.linalg.svd(centered[::max(1,len(v)//5000)],full_matrices=False)
 normal=plane_axes[-1];normal=unit(normal-direction*np.dot(normal,direction));pivot-=normal*np.dot(normal,pivot-np.median(v,axis=0))
 B,length=radial_basis(v,pivot,direction)
 free=v.mean(0)-pivot;sign=outward_sign(direction,free,np.array([0.,0.,1.]))
 original_span=abs(float(np.diff(native_joint['range'])[0]));assumed=native_joint['type'] not in ['revolute','continuous'] or original_span<minimum_span
 span=90. if assumed else original_span
 moved=Rotation.from_rotvec(direction*sign*np.deg2rad(min(90.,span)) ).apply(free)
 return dict(type='revolute',axis=dict(direction=direction.tolist(),origin=pivot.tolist()),range=[0.,sign*span]),B,length,dict(rule='upper-edge outward lifting door',attachment_edge='upper',axis_fit=snap,span_assumed=assumed,free_edge_height_change=float(moved[1]-free[1]),free_edge_front_change=float(moved[2]-free[2]),median_upper_edge_body_distance=float(np.median(distances)),geometry_changed=False)


def resolve_parented_grips(groups):
 """Apply fixed-grip semantics using parent relation, preserving visible ID."""
 result=copy.deepcopy(groups);byid={g['id']:g for g in result};audit=[]
 for g in result:
  name=g.get('name','').lower();parent=byid.get(g.get('parent'))
  if not parent or not any(word in name for word in ['handle','grip']):continue
  parent_name=parent.get('name','').lower();role='drawer' if 'drawer' in parent_name else 'door' if 'door' in parent_name else parent.get('motion_role','')
  decision,reason=semantic_handle_role(name,role,[],1.)
  if decision=='fixed_grip':
   g.update(kind='fixed',motion_role='fixed_grip',observed_q=0.,joint=dict(type='fixed',range=[0.,0.],axis=dict(origin=[0.,0.,0.],direction=[0.,0.,0.])))
   audit.append(dict(group=g['id'],parent=g['parent'],decision=decision,reason=reason))
 return result,audit


def connected_handle_contacts(vertices,static_vertices,object_diagonal):
 """Two contact clusters for a complete U-shaped connected component."""
 v=np.asarray(vertices,float);tree=cKDTree(static_vertices);distance=tree.query(v,workers=4)[0];close=v[distance<=min(np.quantile(distance,.20),.035*object_diagonal)]
 if len(close)<8:raise ValueError('Insufficient contact samples')
 first=close[np.argmin(close[:,0])];second=close[np.argmax(np.linalg.norm(close-first,axis=1))];first=close[np.argmax(np.linalg.norm(close-second,axis=1))];centers=np.array([first,second])
 for _ in range(8):
  ids=np.linalg.norm(close[:,None]-centers[None],axis=2).argmin(1)
  if not (np.any(ids==0) and np.any(ids==1)):raise ValueError('Contact clusters collapse')
  centers=np.array([np.median(close[ids==i],0) for i in [0,1]])
 if np.linalg.norm(centers[1]-centers[0])<.10*object_diagonal:raise ValueError('Contacts insufficiently separated')
 return centers


def paired_handle_component_axis(components, object_diagonal, static_vertices=None):
 """Axis through body-facing support ENDPOINTS, never support-bar centers.

 A truncated generated support may miss the body slightly. Both endpoint
 choices must still be close and unambiguous; the better actual contact anchors
 the perpendicular axis position. Centers only identify the pair of bars.
 """
 D=float(object_diagonal)
 if not np.isfinite(D) or D<=0:raise ValueError('Invalid object diagonal')
 if static_vertices is None:raise ValueError('Static-body evidence required for handle attachments')
 body=np.asarray(static_vertices,float)
 if body.ndim!=2 or body.shape[1]!=3 or len(body)<8 or not np.isfinite(body).all():
  raise ValueError('Insufficient static-body attachment evidence')
 tree=cKDTree(body);candidates=[];rejected=[]
 for index,vertices in enumerate(components):
  v=np.asarray(vertices,float)
  if len(v)<64:continue
  span=np.ptp(v,axis=0);long=int(np.argmax(span));other=np.delete(span,long)
  if span[long]<.12*D or span[long]<2.5*max(float(other.max()),1e-8):continue
  center=np.median(v,axis=0);_,_,V=np.linalg.svd(v[::max(1,len(v)//4000)]-center,full_matrices=False)
  axial=(v-center)@V[0];low,high=np.quantile(axial,[.03,.97])
  ends=np.stack([np.median(v[axial<=low],axis=0),np.median(v[axial>=high],axis=0)])
  distances=tree.query(ends,workers=4)[0];selected=int(np.argmin(distances));best=float(distances[selected]);alternative=float(distances[1-selected])
  if best>.08*D or alternative-best<=max(.01*D,.25*best):
   rejected.append(dict(index=index,endpoint_body_distances=distances.tolist(),reason='Endpoint too far from body or attachment end ambiguous'));continue
  candidates.append(dict(index=index,count=len(v),center=center,long_axis=long,span=span,
   attachment=ends[selected],endpoints=ends,endpoint_body_distances=distances,
   selected_endpoint=selected,body_distance=best))
 pairs=[]
 for i,a in enumerate(candidates):
  for b in candidates[i+1:]:
   if a['long_axis']!=b['long_axis']:continue
   delta=b['center']-a['center'];distance=float(np.linalg.norm(delta))
   if distance<.15*D or abs(delta[a['long_axis']])>.3*distance:continue
   if np.linalg.norm(b['attachment']-a['attachment'])<.10*D:continue
   pairs.append((min(a['count'],b['count']),distance,a,b))
 if not pairs:raise ValueError('No two support bars with unambiguous body-facing attachment ends')
 _,_,a,b=max(pairs,key=lambda row:(row[0],row[1]))
 attachments=np.stack([a['attachment'],b['attachment']]);_,direction,audit=fit_attachment_axis(attachments)
 # A support with a missing lower fragment must not drag a well-observed
 # contact to the middle of the gap. Distance weights use generated body only.
 distances=np.array([a['body_distance'],b['body_distance']]);weights=1/(distances+.005*D)**2;weights/=weights.sum()
 pivot=np.average(attachments,axis=0,weights=weights)
 P=np.eye(3)-np.outer(direction,direction)
 residual=float(np.sqrt(np.mean(np.sum(((attachments-pivot)@P)**2,axis=1))))
 audit.update(rule='two body-facing support endpoints, contact-weighted axis position',
  support_component_indices=[a['index'],b['index']],
  support_centers=np.stack([a['center'],b['center']]).tolist(),
  attachment_points=attachments.tolist(),attachment_weights=weights.tolist(),
  attachment_body_distances=distances.tolist(),attachment_perpendicular_RMS=residual,
  endpoint_evidence=[dict(component=x['index'],endpoints=x['endpoints'].tolist(),
   endpoint_body_distances=x['endpoint_body_distances'].tolist(),selected_endpoint=x['selected_endpoint']) for x in [a,b]],
  support_separation=float(np.linalg.norm(attachments[1]-attachments[0])),
  attachment_policy='Nearest unambiguous endpoint within8% object diagonal; near-contact weights; never bar centers',
  approximate_body_gap=bool(np.any(distances>.035*D)),rejected_components=rejected)
 return pivot,direction,audit


def carrying_handle_outward_sign(vertices,pivot,direction,body_center,object_diagonal):
 """Choose the signed swing that moves a handle's upper free end outward.

 Axis direction alone is unsigned. For a pair of side-mounted carrying
 handles, applying the same sign to both makes both swing to the same side.
 The side and free-end geometry determine opposite signs without a case ID.
 """
 v=np.asarray(vertices,float);pivot=np.asarray(pivot,float);d=unit(direction)
 high=v[v[:,1]>=np.quantile(v[:,1],.85)]
 free=np.median(high,axis=0)-pivot
 free-=np.dot(free,d)*d
 side=np.median(v,axis=0)-np.asarray(body_center,float)
 side-=np.dot(side,d)*d
 D=float(object_diagonal)
 if np.linalg.norm(free)<.04*D or np.linalg.norm(side)<.08*D:
  raise ValueError('Handle free end or outward side is ambiguous')
 derivative=np.cross(d,free);score=float(np.dot(derivative,side))
 if abs(score)<.1*np.linalg.norm(derivative)*np.linalg.norm(side):
  raise ValueError('Handle swing direction is geometrically ambiguous')
 return (1. if score>0 else -1.),dict(rule='upper free end swings away from body center',
   free_end_radius=free.tolist(),outward_side=side.tolist(),signed_outward_score=score)


def project(m,points):
 R=m.R.cpu().numpy();s=float(m.log_global_scale.exp());shift=(m.translation_basis@m.screen_translation).detach().cpu().numpy();ext=m.extrinsics.cpu().numpy();intr=m.intrinsics.cpu().numpy();world=s*(np.asarray(points)@R.T)+shift;cam=world@ext[:3,:3].T+ext[:3,3];uv=cam[:,:2]/cam[:,2:3];return (uv*np.array([intr[0,0],intr[1,1]])+intr[:2,2])*800,cam[:,2]


def unproject(m,pixel,z):
 ext=m.extrinsics.cpu().numpy();intr=m.intrinsics.cpu().numpy();R=m.R.cpu().numpy();s=float(m.log_global_scale.exp());shift=(m.translation_basis@m.screen_translation).detach().cpu().numpy();ray=np.linalg.solve(intr,np.r_[np.asarray(pixel)/800,1.]);wo=-ext[:3,:3].T@ext[:3,3];wd=ext[:3,:3].T@ray;o=(wo-shift)@R/s;d=wd@R/s
 if abs(d[2])<1e-8:raise ValueError('Camera ray parallel to generated front depth plane')
 return o+(z-o[2])/d[2]*d


def mask_endpoint(mask,near_pixel):
 yy,xx=np.where(mask);xy=np.c_[xx,yy].astype(float)
 if len(xy)<16:raise ValueError('Insufficient visible mask support')
 _,_,V=np.linalg.svd(xy-xy.mean(0),full_matrices=False);u=(xy-xy.mean(0))@V[0];low=np.median(xy[u<=np.quantile(u,.03)],0);high=np.median(xy[u>=np.quantile(u,.97)],0)
 return min([low,high],key=lambda p:np.linalg.norm(p-near_pixel))


def handle_semantic_candidate(name,native_name):
 name=str(name).lower();native_name=str(native_name).lower()
 if any(word in name for word in ['handle','grip','提手','拉手']):return True
 if any(word in name for word in ['door','drawer','shelf','tray','wheel','knob']):return False
 return native_name=='handle'


def carrying_context_conflicts(groups,candidates):
 """Body-only regrouping must preserve unrelated active mechanisms.

 Mixed objects retain their existing family prior until local regrouping is
 supported; a carrying-handle proposal cannot freeze another door or drawer.
 """
 candidates=set(candidates)
 return sorted({g.get('semantic_owner',g['id']) for g in groups
                if g['joint']['type']!='fixed'
                and g.get('semantic_owner',g['id']) not in candidates})


def discover_handles(model,semantic_ids,owner_names):
 """Evidence-gated component grouping; no case IDs or fixed owner count.

 Preserves every native component; non-handle components are fixed to body only
 when a carrying-handle structure is supported. Otherwise returns no override.
 """
 from multicase_closed_common import component_labels
 meshes={p['node']['id']:p['mesh'] for p in model.asset['parts']};by_source={}
 for p,(xyz,_) in zip(model.parts,model.geometry(original=True)):
  nid=p['group']['source_node'];ids=model.membership[p['group']['membership_prefix']+'_gaussians'];by_source.setdefault(nid,[]).append((ids,xyz.cpu().numpy()))
 bounds=np.concatenate([mesh.vertices[::101] for mesh in meshes.values()]);D=float(np.linalg.norm(np.ptp(bounds,axis=0)))
 attached_owners={g['id'] for g in model.groups if g.get('motion_role')=='fixed_grip'}
 candidates=[]
 for owner in sorted(int(x) for x in np.unique(semantic_ids) if x and x!=10):
  if owner in attached_owners:continue
  yy,xx=np.where(semantic_ids==owner);xy=np.c_[xx,yy];sv=np.linalg.svd(xy-xy.mean(0),compute_uv=False)
  elongated=sv[0]/max(sv[1],1e-8)>3.
  name=owner_names.get(str(owner),owner_names.get(owner,''))
  group=next((g for g in model.groups if g['id']==owner),None)
  native=next((n.get('name','') for n in model.asset['nodes'] if group and n['id']==group['source_node']),'')
  # Shape validates a semantic hypothesis; a thin door alone is never a handle.
  if handle_semantic_candidate(name,native):candidates.append(owner)
 if not candidates:return None,dict(accepted=False,reason='No handle-like semantic or image evidence')
 conflicts=carrying_context_conflicts(model.groups,candidates)
 if conflicts:return None,dict(accepted=False,reason='Mixed active mechanisms require local handle regrouping; retain existing motions',candidates=candidates,unrelated_active_owners=conflicts)
 centers={owner:np.median(np.c_[np.where(semantic_ids==owner)[1],np.where(semantic_ids==owner)[0]],0) for owner in candidates};rows=[];cache={}
 for nid,mesh in meshes.items():
  labels=component_labels(mesh);parts=by_source[nid];xyz=np.empty((sum(len(ids) for ids,_ in parts),3))
  for ids,x in parts:xyz[ids]=x
  glab=labels[cKDTree(mesh.vertices).query(xyz,workers=4)[1]];cache[nid]=(labels,glab)
  for cid in np.unique(labels):
   v=np.asarray(mesh.vertices)[labels==cid];x=v[::max(1,len(v)//4000)];_,_,V=np.linalg.svd(x-x.mean(0),full_matrices=False);extent=np.ptp(x@V.T,axis=0);extent=np.sort(extent)[::-1];thin=extent[0]>3*max(extent[1],1e-6) and extent[0]>.12*D;owner=10;votes={}
   native=next(n for n in model.asset['nodes'] if n['id']==nid)
   native_handle=native.get('name')=='handle' and native['joint']['type'] in ['revolute','continuous']
   if thin or native_handle:
    uv,depth=project(model,x);pix=np.rint(uv).astype(int);ok=(depth>0)&(pix[:,0]>=0)&(pix[:,0]<800)&(pix[:,1]>=0)&(pix[:,1]<800);seen=semantic_ids[pix[ok,1],pix[ok,0]];votes={o:int((seen==o).sum()) for o in candidates};ordered=sorted(candidates,key=lambda o:votes[o],reverse=True);best=ordered[0];second=votes[ordered[1]] if len(ordered)>1 else 0
    if votes[best]>max(.1*len(seen),2*second):owner=best
    else:owner=min(candidates,key=lambda o:np.linalg.norm(np.median(uv,axis=0)-centers[o]))
   rows.append(dict(source=nid,component=int(cid),owner=owner,thin_support=bool(thin),mask_votes=votes,center=v.mean(0).tolist()))
 supported=[o for o in candidates if sum(r['owner']==o for r in rows)>=1]
 if len(supported)!=len(candidates):return None,dict(accepted=False,reason='Carrying handle requires component evidence for every candidate',candidates=candidates,supported=supported,component_evidence=rows)
 static=np.concatenate([np.asarray(meshes[r['source']].vertices)[cache[r['source']][0]==r['component']] for r in rows if r['owner']==10]);tree=cKDTree(static);axes={}
 for owner in supported:
  selected=[r for r in rows if r['owner']==owner];attachments=[]
  for r in selected:
   v=np.asarray(meshes[r['source']].vertices)[cache[r['source']][0]==r['component']];_,_,V=np.linalg.svd(v[::max(1,len(v)//4000)]-v.mean(0),full_matrices=False);u=(v-v.mean(0))@V[0];ends=[np.median(v[u<=np.quantile(u,.03)],0),np.median(v[u>=np.quantile(u,.97)],0)];attachments.append(min(ends,key=lambda a:tree.query(a)[0]))
  if len(selected)==1:
   native=next(n for n in model.asset['nodes'] if n['id']==selected[0]['source'])
   if native.get('name')!='handle':return None,dict(accepted=False,reason='Single thin door/panel is not sufficient handle evidence',component_evidence=rows)
   r=selected[0];v=np.asarray(meshes[r['source']].vertices)[cache[r['source']][0]==r['component']]
   try:attachments=list(connected_handle_contacts(v,static,D))
   except ValueError as error:return None,dict(accepted=False,reason=str(error),component_evidence=rows)
  role,reason=semantic_handle_role('handle','fixed_body',attachments,D)
  if role!='articulated_handle':return None,dict(accepted=False,reason=reason)
  depths=[float(project(model,np.array([a]))[1][0]) for a in attachments];front=int(np.argmin(depths));uv,_=project(model,np.array([attachments[front]]));pixel=mask_endpoint(semantic_ids==owner,uv[0]);observed=unproject(model,pixel,float(attachments[front][2]));pivot,direction,audit=fit_attachment_axis(attachments,observed)
  axes[owner]=dict(origin=pivot,direction=direction,attachments=attachments,mask_pixel=pixel,audit=audit,front_source=selected[min(front,len(selected)-1)]['source'])
 return dict(rows=rows,cache=cache,axes=axes,meshes=meshes,diagonal=D),dict(accepted=True,candidates=supported,reason='Independent supports plus instance-mask evidence',component_evidence=rows)
