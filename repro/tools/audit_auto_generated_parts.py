"""Audit automatic reference semantics against original generated door joints.

This audit reads neither human semantic annotations nor GT joints.  It uses
only sealed VLM labels, original PAct node names/joints, and the already frozen
projection correspondence evidence.
"""
import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    root = a.root.resolve()
    assert read(root/'FINAL_SEAL.json')
    cases = read(root/'protocol.json')['cases']
    result = {}
    total_doors = verified = verified_hinges = 0
    for case in cases:
        out = root/case
        semantic = {x['instance_id']:x for x in read(out/'semantics.json')['parts']}
        requests = read(out/'axis_requests.json')['parts']
        inputs = read(out/'inputs.json')
        run = Path(inputs['source_run'])
        status = read(run/'status.json')
        original_path = Path(status['object'])
        if not original_path.is_absolute():original_path=run/original_path
        nodes = {n['id']:n for n in read(original_path)['diffuse_tree']}
        doors = [r for r in requests if semantic[r['instance_id']]['semantic']=='door']
        door_ids = {r['instance_id'] for r in doors}
        rows=[]
        for r in doors:
            owner=r['instance_id'];source=r['native_node_id'];e=r['correspondence'];node=nodes[source]
            votes={int(k):int(v) for k,v in e['votes'].items()}
            door_votes={k:v for k,v in votes.items() if k in door_ids}
            total=sum(door_votes.values())
            mixed=sorted(k for k,v in door_votes.items() if total and v/total>=.25 and v>=32)
            hinge=node['joint']['type'] in {'revolute','continuous'}
            row=dict(instance_id=owner,automatic_semantic='door',motion_role=semantic[owner]['motion_role'],
                     source_node=source,original_node_name=node['name'],original_joint_type=node['joint']['type'],
                     original_parent=node['parent'],original_axis=node['joint']['axis'],
                     original_range=node['joint']['range'],correspondence_verified=e['verified'],
                     target_projection_precision=e['target_projection_precision'],
                     source_projection_door_votes=door_votes,
                     source_projects_to_multiple_door_masks=len(mixed)>1,
                     mixed_door_instances=mixed if len(mixed)>1 else [],
                     independent_verified_hinge=bool(e['verified'] and hinge),
                     auto_axis_applied=r['applied'],routing_reason=r['reason'])
            rows.append(row)
        unique_verified_hinge_nodes={x['source_node'] for x in rows if x['independent_verified_hinge']}
        safe=bool(rows) and len(unique_verified_hinge_nodes)==len(rows) and all(x['independent_verified_hinge'] and not x['source_projects_to_multiple_door_masks'] for x in rows)
        result[case]=dict(door_count=len(rows),verified_count=sum(x['correspondence_verified'] for x in rows),
                          verified_hinge_count=sum(x['independent_verified_hinge'] for x in rows),
                          all_doors_independently_verified_hinges=safe,doors=rows)
        total_doors+=len(rows);verified+=result[case]['verified_count'];verified_hinges+=result[case]['verified_hinge_count']
    summary=dict(case_count=len(cases),automatic_door_mask_count=total_doors,
                 door_to_generated_node_verified=verified,
                 verified_door_with_independent_hinge=verified_hinges,
                 multi_door_cases=sum(x['door_count']>1 for x in result.values()),
                 multi_door_cases_all_independently_verified=sum(x['door_count']>1 and x['all_doors_independently_verified_hinges'] for x in result.values()),
                 policy='Conservative evidence audit. A verified hinge requires one mask-to-generated-node projection match and a revolute/continuous joint; a mixed source node cannot certify one door per joint. No GT or human semantics used.')
    path=root/'GENERATED_PART_DOOR_AXIS_AUDIT.json'
    path.write_text(json.dumps(dict(summary=summary,cases=result),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
