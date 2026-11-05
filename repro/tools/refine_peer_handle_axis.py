"""Generic generated-only fallback for an unresolved handle beside a supported peer.

Requires a second independently recognized handle with a two-contact axis and
enough near-body vertices on the unresolved handle. No case IDs or GT joints.
"""
import argparse
import copy
import json
import pathlib
import sys

import numpy as np
from scipy.spatial import cKDTree

REPO=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'repro/tools'))
from auto_part_semantics import sha256,write_json
from collision_refinement_common import load_asset


def read(path):return json.loads(pathlib.Path(path).read_text())


def apply_case(root):
    assignment=read(root/'component_assignment.json');manifest=read(root/'motion_groups.json')
    groups=manifest['groups'];by_owner={}
    for g in groups:
        if g.get('semantic_owner') is not None:by_owner.setdefault(g['semantic_owner'],[]).append(g)
    peer=[a for a in assignment['assignments'] if a.get('semantic')=='handle'
          and a.get('family')=='articulated_handle'
          and a.get('axis_evidence',{}).get('rule')=='two separated body contact clusters']
    unresolved=[a for a in assignment['assignments'] if a.get('semantic')=='handle'
                and a.get('family')=='handle_native_unresolved']
    if not peer or not unresolved:return []
    asset=load_asset(manifest['source_run']);mesh={p['node']['id']:p['mesh'] for p in asset['parts']}
    membership=np.load(root/'membership.npz')
    def vertices(owner):
        return np.concatenate([np.asarray(mesh[g['source_node']].vertices)[membership[g['membership_prefix']+'_vertices']]
                               for g in by_owner[owner]])
    semantics=read(manifest['semantic_input'])['parts']
    bodies=[p['instance_id'] for p in semantics if p['semantic']=='base']
    if len(bodies)!=1 or bodies[0] not in by_owner:return []
    tree=cKDTree(vertices(bodies[0]));changes=[]
    for target in unresolved:
        owner=target['instance_id'];v=vertices(owner);distance=tree.query(v)[0]
        threshold=max(.02*asset['diagonal'],3*float(distance.min()))
        contacts=v[distance<=threshold]
        if len(contacts)<30:continue
        # A peer demonstrates the mechanism and supplies the mechanical axis
        # direction. The pivot still comes from this handle's own attachment.
        candidates=[p for p in peer if p['instance_id']!=owner]
        if not candidates:continue
        chosen=min(candidates,key=lambda p:np.linalg.norm(np.asarray(p['joint']['axis']['origin'])-np.median(contacts,axis=0)))
        direction=np.asarray(chosen['joint']['axis']['direction'],float)
        if np.linalg.norm(direction)<.9:continue
        pivot=np.median(contacts,axis=0)
        original=copy.deepcopy(target['joint'])
        joint=copy.deepcopy(original);joint['type']='revolute';joint['axis']={'origin':pivot.tolist(),'direction':direction.tolist()}
        target['joint']=joint;target['family']='articulated_handle_peer_contact'
        target['axis_evidence']=dict(rule='Supported articulated handle peer direction plus own near-body attachment contact',
            peer_instance_id=chosen['instance_id'],contact_vertices=len(contacts),contact_threshold=threshold,
            original_native_joint=original,GT_numeric_joint_inputs=False)
        primary=[g for g in by_owner[owner] if g['parent'] not in {h['id'] for h in by_owner[owner]}]
        assert len(primary)==1
        primary[0]['joint']=copy.deepcopy(joint);primary[0]['kind']='articulated_handle_peer_contact'
        changes.append(dict(instance_id=owner,peer_instance_id=chosen['instance_id'],pivot=pivot.tolist(),
                            direction=direction.tolist(),contact_vertices=len(contacts)))
    if changes:
        write_json(root/'component_assignment.json',assignment)
        write_json(root/'motion_groups.json',manifest)
        cfg=read(root/'config.json');cfg['prior_manifest_sha256']=sha256(root/'motion_groups.json');write_json(root/'config.json',cfg)
        write_json(root/'peer_handle_fallback.json',dict(changes=changes,source='automatic post-v0.1.1 generated-geometry fallback'))
    return changes


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);a=p.parse_args()
    root=a.root.resolve();cases=read(root/'protocol.json')['cases']
    summary={}
    for case in cases:
        summary[case]=apply_case(root/case)
        print('PEER HANDLE FALLBACK',case,len(summary[case]),flush=True)
    write_json(root/'peer_handle_fallback_summary.json',summary)


if __name__=='__main__':main()
