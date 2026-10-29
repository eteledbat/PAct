"""Automatic semantic constraints for axes; no case recipes or GT parameters.

The reference-to-native ID link is checked against projected native geometry.
Only verified correspondences can change an exported joint. Ambiguous merged or
misassigned parts retain their native motion and are explicitly reported.
"""
import argparse
import copy
import json
import pathlib
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import binary_dilation, minimum_filter
from scipy.optimize import linear_sum_assignment
from auto_part_semantics import write_json, sha256
from semantic_motion_rules import top_hinged_door, canonical_axis, connected_handle_contacts, fit_attachment_axis, outward_sign


def cyclic_parents(parts):
    lookup={p['instance_id']:p.get('parent_instance_id') for p in parts}
    bad=set()
    for owner in lookup:
        trail=[];current=owner
        while current in lookup and current not in trail:
            trail.append(current);current=lookup[current]
        if current in trail:bad.update(trail[trail.index(current):])
    return bad


def route_part(part, parts):
    """Separate a semantic observation from a warranted mechanical decision."""
    semantic=part['semantic'];role=part['motion_role'];lookup={p['instance_id']:p for p in parts}
    result=dict(instance_id=part['instance_id'],semantic=semantic,requested_role=role,decision='retain_native',family='unknown',candidate_families=[],reason='Insufficient semantic/mechanical evidence')
    if part['semantic_confidence']=='low' or semantic in {'unknown','other'}:return result
    if part['instance_id'] in cyclic_parents(parts):
        result['reason']='Predicted parent cycle';return result
    if semantic=='base':
        result.update(decision='apply',family='fixed',reason='Recognized stationary main body');return result
    if semantic=='drawer':
        result.update(decision='apply',family='slider_front',candidate_families=['slider_front'],reason='Recognized pull-out drawer; canonical front-axis prior');return result
    if semantic=='door':
        result['candidate_families']=['hinge_side','hinge_upper','hinge_lower','slider_horizontal']
        edges={'side_door':{'left','right'},'upward_door':{'upper'},'drop_door':{'lower'}}
        if role in edges and part['motion_confidence']=='high' and part['hinge_visible'] and part['attachment_edge'] in edges[role]:
            result.update(decision='apply',family={'side_door':'hinge_side','upward_door':'hinge_upper','drop_door':'hinge_lower'}[role],reason='Door semantic plus explicit visible attachment evidence')
        elif role=='sliding_door':result['reason']='Sliding-door label lacks a verified rail direction'
        else:result['reason']='Door recognized, but attachment/motion subtype is ambiguous in the image'
        return result
    if semantic=='handle':
        parent=lookup.get(part.get('parent_instance_id'))
        if role=='fixed_grip' and parent and parent['semantic'] in {'door','drawer'} and parent['semantic_confidence']!='low':
            result.update(decision='apply',family='fixed_grip',reason='Grip mounted on recognized door/drawer',parent_instance_id=parent['instance_id'])
        elif role=='articulated_handle' and part['motion_confidence']=='high':
            result.update(decision='check_geometry',family='articulated_handle',reason='Carrying-handle hypothesis requires two separated geometric supports')
        else:
            result.update(candidate_families=['fixed_grip','articulated_handle'],reason='Handle recognized; independent articulation not established')
        return result
    if semantic in {'wheel','knob'}:
        result.update(candidate_families=['revolute','continuous'],reason='Rotational family proposed, but attachment axis not yet verified');return result
    if semantic in {'shelf','tray'}:
        result.update(candidate_families=['fixed','prismatic'],reason='Semantic category does not establish fixed versus sliding motion')
    return result


def side_hinge(vertices,body_vertices,edge,native):
    v=np.asarray(vertices);body=np.asarray(body_vertices);w=np.ptp(v,axis=0)
    if w[1]<1e-5:raise ValueError('Degenerate door height')
    y=np.quantile(v[:,1],[.1,.9]);xz=v[:,[0,2]];_,_,V=np.linalg.svd(xz-xz.mean(0),full_matrices=False);across=V[0]
    if across[0]<0:across=-across
    coordinate=(xz-xz.mean(0))@across;q=.01 if edge=='left' else .99
    boundary=xz.mean(0)+np.quantile(coordinate,q)*across
    origin=np.array([boundary[0],np.median(v[:,1]),boundary[1]]);direction=np.array([0.,1.,0.])
    span=abs(float(np.diff(native['range'])[0])) if native['type'] in {'revolute','continuous'} else 90.
    assumed=native['type'] not in {'revolute','continuous'} or span<1e-6
    if assumed:span=90.
    sign=outward_sign(direction,v.mean(0)-origin,np.array([0.,0.,1.]))
    return dict(type='revolute',axis=dict(origin=origin.tolist(),direction=direction.tolist()),range=[0.,sign*span]),dict(span_assumed=assumed,attachment_edge=edge)


def lower_hinge(vertices,body_vertices,native):
    # Mirror the tested top-hinge estimator vertically, then transform the
    # rotation's axial vector with det(M)*M, not with M alone.
    M=np.diag([1.,-1.,1.]);j,_,_,audit=top_hinged_door(np.asarray(vertices)@M,np.asarray(body_vertices)@M,native)
    j['axis']['origin']=(M@np.asarray(j['axis']['origin'])).tolist()
    j['axis']['direction']=(-M@np.asarray(j['axis']['direction'])).tolist()
    audit.update(attachment_edge='lower',rule='lower-edge outward drop door',free_edge_height_change=-audit['free_edge_height_change'])
    return j,audit


def project_vertices(vertices,registration_root):
    """Reuse only q=0 RGB/silhouette registration; no manual semantic recipes."""
    import torch
    cfg=json.loads((registration_root/'config.json').read_text())
    obs=np.load(registration_root/'observation.npz');ext=obs['extrinsics'];intr=obs['intrinsics'];R=np.asarray(cfg['world_R'])
    reg=torch.load(registration_root/'registration.pt',map_location='cpu',weights_only=False)
    s=float(reg['log_global_scale'].exp());shift=ext[:3,:3].T[:,:2]@reg['screen_translation'].numpy()
    world=s*(vertices@R.T)+shift;cam=world@ext[:3,:3].T+ext[:3,3]
    uv=(cam[:,:2]/np.maximum(cam[:,2:3],1e-8))*np.array([intr[0,0],intr[1,1]])+intr[:2,2]
    return uv,cam[:,2]


def geometry_correspondence(asset, ids, registration_root, native_link):
    """Visibility-filtered projection precision is a conservative rejection gate.

    It is not a learned instance matching model; tiny/off-camera parts are left
    unresolved. A wrong generated component cannot inherit an input label solely
    because the PAct array index happens to match.
    """
    height,width=ids.shape;cloud=np.concatenate([p['mesh'].vertices for p in asset['parts']]);uv,z=project_vertices(cloud,registration_root)
    pixels=np.rint(uv*np.array([width,height])).astype(int);ok=(z>0)&(pixels[:,0]>=0)&(pixels[:,0]<width)&(pixels[:,1]>=0)&(pixels[:,1]<height)
    front=np.full(height*width,np.inf);flat=pixels[ok,1]*width+pixels[ok,0];np.minimum.at(front,flat,z[ok]);front=minimum_filter(front.reshape(height,width),size=3)
    offset=0;results={};source_evidence={}
    for p in asset['parts']:
        n=len(p['mesh'].vertices);pix=pixels[offset:offset+n];depth=z[offset:offset+n];valid=ok[offset:offset+n].copy();offset+=n
        loc=np.flatnonzero(valid)
        valid[loc]=depth[loc]<=front[pix[loc,1],pix[loc,0]]+.015*asset['diagonal']
        seen=ids[pix[valid,1],pix[valid,0]];seen=seen[seen>0]
        votes={int(owner):int((seen==owner).sum()) for owner in np.unique(ids) if owner>0}
        total=max(len(seen),1)
        source_evidence[p['node']['id']]=dict(votes=votes,total=total,count=len(seen))
    # Match without reading predicted classes, joint types, axes or case recipes.
    owners=sorted(native_link);sources=sorted(source_evidence)
    score=np.array([[source_evidence[source]['votes'].get(owner,0)/source_evidence[source]['total'] for source in sources] for owner in owners])
    ri,ci=linear_sum_assignment(-score)
    assignment={owners[i]:sources[j] for i,j in zip(ri,ci)}
    for owner in owners:
        source=assignment.get(owner,native_link[owner]);e=source_evidence[source];precision=e['votes'].get(owner,0)/e['total'];competing=[count/e['total'] for key,count in e['votes'].items() if key!=owner]
        accepted=owner in assignment and e['count']>=32 and precision>=.55 and (not competing or precision>=max(competing)+.15)
        results[owner]=dict(source_node=source,original_input_index_node=native_link[owner],matching='type/axis-blind projected-geometry Hungarian assignment',visible_vertex_count=e['count'],target_projection_precision=precision,votes=e['votes'],verified=bool(accepted))
    return results


def apply_case(out,registration_root):
    from collision_refinement_common import load_asset
    parts=json.loads((out/'semantics.json').read_text())['parts'];inputs=json.loads((out/'inputs.json').read_text());asset=load_asset(inputs['source_run']);byid={p['node']['id']:p for p in asset['parts']}
    ids=np.load(out/'instance_ids.npy');native_link={int(k):int(v) for k,v in inputs['input_label_to_native_node'].items()}
    correspondence=geometry_correspondence(asset,ids,registration_root,native_link)
    native_link={owner:row['source_node'] for owner,row in correspondence.items()}
    body_instances=[p['instance_id'] for p in parts if p['semantic']=='base' and p['semantic_confidence']!='low' and correspondence.get(p['instance_id'],{}).get('verified')]
    body_ids=[native_link[x] for x in body_instances]
    # A static reference instance can span multiple generated pieces. Include
    # the native base hypothesis as well; never choose a body by case ID.
    body_ids=sorted(set(body_ids+[p['node']['id'] for p in asset['parts'] if p['node']['name']=='base']))
    body=np.concatenate([byid[x]['mesh'].vertices for x in body_ids]) if body_ids else None
    obj=copy.deepcopy(asset['object']);nodes={n['id']:n for n in obj['diffuse_tree']};audit=[]
    for part in parts:
        request=route_part(part,parts);owner=part['instance_id'];source=native_link.get(owner);request.update(native_node_id=source,correspondence=correspondence.get(owner),applied=False)
        if source not in nodes:
            request['reason']='Missing native node correspondence';audit.append(request);continue
        node=nodes[source];original=copy.deepcopy(node['joint']);request['native_semantic']=node['name'];request['original_joint']=original
        if request['decision']=='retain_native':audit.append(request);continue
        if not correspondence[owner]['verified']:
            request['reason']='Reference semantic recognized, but native geometry/instance correspondence is not verified';audit.append(request);continue
        v=np.asarray(byid[source]['mesh'].vertices);family=request['family'];extra={}
        try:
            if family in {'fixed','fixed_grip'}:
                parent=-1
                if family=='fixed_grip':
                    parent_owner=request['parent_instance_id'];parent=native_link.get(parent_owner)
                    if parent not in nodes or parent==source or not correspondence.get(parent_owner,{}).get('verified'):raise ValueError('Unverified grip-parent correspondence')
                    ancestors=set();current=parent
                    while current in nodes and current not in ancestors:
                        if current==source:raise ValueError('Native parent cycle would be introduced')
                        ancestors.add(current);current=nodes[current].get('parent',-1)
                node['parent']=parent
                joint=dict(type='fixed',axis=dict(origin=[0.,0.,0.],direction=[0.,0.,0.]),range=[0.,0.])
            elif family=='slider_front':
                span=abs(float(np.diff(original['range'])[0])) if original['type']=='prismatic' else .25*asset['diagonal']
                joint=dict(type='prismatic',axis=dict(origin=v.mean(0).tolist(),direction=[0.,0.,1.]),range=[0.,span]);extra['span_assumed']=original['type']!='prismatic'
            elif body is None:raise ValueError('No static-body geometry available')
            elif source in body_ids:raise ValueError('Moving-part request conflicts with static-body geometry')
            elif family=='hinge_upper':joint,_,_,extra=top_hinged_door(v,body,original)
            elif family=='hinge_lower':joint,extra=lower_hinge(v,body,original)
            elif family=='hinge_side':joint,extra=side_hinge(v,body,part['attachment_edge'],original)
            elif family=='articulated_handle':
                contacts=connected_handle_contacts(v,body,asset['diagonal']);pivot,direction,extra=fit_attachment_axis(contacts)
                span=abs(float(np.diff(original['range'])[0])) if original['type'] in {'revolute','continuous'} else 90.
                extra['span_assumed']=original['type'] not in {'revolute','continuous'} or span<1e-6
                if extra['span_assumed']:span=90.
                sign=outward_sign(direction,v.mean(0)-pivot,pivot-body.mean(0));joint=dict(type='revolute',axis=dict(origin=pivot.tolist(),direction=direction.tolist()),range=[0.,sign*span])
            else:raise ValueError('Axis solver not implemented for proposed family')
            node['joint']=joint;node['name']=part['semantic'];request.update(applied=True,selected_joint=joint,geometry_evidence=extra)
        except (ValueError,np.linalg.LinAlgError) as exc:request['reason']=str(exc)
        audit.append(request)
    # Export absolute references to immutable original geometry; never rewrite it.
    for node in obj['diffuse_tree']:
        for key in ['plys','gs']:
            if key in node:node[key]=[str((asset['object_path'].parent/p).resolve()) for p in node[key]]
    write_json(out/'axis_requests.json',dict(parts=audit,geometry_changed=False,manual_semantics_used=False,canonical_frame='Y up; +Z front',registration_source=str(registration_root),registration_reuse='Only existing original-RGB/silhouette alignment, no semantic/geometry-fit parameters',native_geometry_preserved=True))
    export=out/'axis_only_export';export.mkdir(exist_ok=True);write_json(export/'object.json',obj)
    write_json(out/'axis_only_export_manifest.json',dict(object=str(export/'object.json'),original_object_sha256=sha256(asset['object_path']),export_sha256=sha256(export/'object.json'),geometry_files=[dict(path=str(p['mesh_path']),sha256=p['mesh_sha256']) for p in asset['parts']],applied_joint_count=sum(x['applied'] for x in audit)))
    print(out.name,'axes',sum(x['applied'] for x in audit),'/',len(audit),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=pathlib.Path);p.add_argument('--registration-root',required=True,type=pathlib.Path);a=p.parse_args()
    assert json.loads((a.root/'RECOGNITION_SEAL.json').read_text())['state']=='complete'
    for case in json.loads((a.root/'protocol.json').read_text())['cases']:apply_case(a.root/case,a.registration_root/case)


if __name__=='__main__':main()
