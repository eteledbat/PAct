"""Match frozen input semantic instances to generated connected components, then set joints.

Only source RGB instance masks, frozen automatic labels, original PAct geometry and
the RGB/silhouette registration are used. GT is never opened by this module.
The original meshes and Gaussians are referenced, not rewritten.
"""
import argparse
import copy
import json
import pathlib

import numpy as np
from plyfile import PlyData
from scipy.ndimage import minimum_filter
from scipy.spatial import cKDTree

from collision_refinement_common import load_asset, sha256
from multicase_closed_common import component_labels, write_json
from semantic_axis_router import lower_hinge, project_vertices, side_hinge
from semantic_motion_rules import carrying_handle_outward_sign, connected_handle_contacts, fit_attachment_axis, outward_sign, paired_handle_component_axis, top_hinged_door
from semantic_motion_prior_45194 import UNDO


def read(path):
    return json.loads(path.read_text())


def fixed():
    return dict(type='fixed', range=[0., 0.], axis=dict(origin=[0., 0., 0.], direction=[0., 0., 0.]))


def gs_xyz(path):
    data = PlyData.read(path)['vertex'].data
    return np.column_stack([data[k] for k in ('x', 'y', 'z')]).astype(np.float32) @ UNDO.T


def projection_votes(asset, ids, registration):
    """Component votes from visible Gaussian centers; mesh vertices break weak ties."""
    h, w = ids.shape
    records = []
    for part in asset['parts']:
        mesh = part['mesh']
        labels = component_labels(mesh)
        path = asset['object_path'].parent / part['node']['gs'][0]
        xyz = gs_xyz(path)
        glabels = labels[cKDTree(mesh.vertices).query(xyz, workers=4)[1]]
        uv, depth = project_vertices(xyz, registration)
        pix = np.rint(uv * [w, h]).astype(np.int32)
        valid = (depth > 0) & (pix[:, 0] >= 0) & (pix[:, 0] < w) & (pix[:, 1] >= 0) & (pix[:, 1] < h)
        records.append(dict(nid=part['node']['id'], part=part, labels=labels,
                            glabels=glabels, xyz=xyz, pixels=pix, depth=depth,
                            valid=valid, gs_path=path))
    front = np.full(h*w, np.inf, np.float32)
    for r in records:
        p = r['pixels'][r['valid']]
        np.minimum.at(front, p[:, 1]*w+p[:, 0], r['depth'][r['valid']])
    front = minimum_filter(front.reshape(h, w), size=3)
    for r in records:
        p = r['pixels']
        visible = r['valid'].copy()
        loc = np.flatnonzero(visible)
        visible[loc] = r['depth'][loc] <= front[p[loc, 1], p[loc, 0]] + .015*asset['diagonal']
        r['visible'] = visible
    return records


def choose_owner(votes, native, semantics, count, component_center, body_center):
    ordered = sorted(votes, key=lambda x: votes[x], reverse=True)
    best = ordered[0] if ordered else 0
    lead = votes.get(best, 0)
    second = votes.get(ordered[1], 0) if len(ordered) > 1 else 0
    total = sum(votes.values())
    confident = count >= 20 and total >= 20 and lead/total >= .55 and lead-second >= .12*total
    if confident:
        return best, 'visible_projected_Gaussian_majority', lead/max(total, 1)
    # A generated moving panel can be occluded by its body in the source view.
    # Accept the runner-up mask only when it is a matching movable category,
    # it explains a substantial fraction of visible pixels, and the winner is
    # the static body. This is still projection evidence, never index matching.
    base = next((i for i, p in semantics.items() if p['semantic'] == 'base'), 0)
    if best == base and len(ordered) > 1 and count >= 40:
        second_id = ordered[1]
        second_semantic = semantics[second_id]['semantic']
        native_name = native['name'].lower()
        compatible = (second_semantic == 'door' and 'door' in native_name) or (second_semantic == 'drawer' and 'drawer' in native_name)
        if compatible and votes[second_id]/max(total, 1) >= .35:
            return second_id, 'occluded_moving_panel_second_vote', votes[second_id]/max(total, 1)
    # Static body is a safe default for weak or occluded small fragments.
    if native['joint']['type'] == 'fixed' or native['name'] == 'base':
        return base, 'ambiguous_static_source', lead/max(total, 1)
    return 0, 'unresolved_retains_native_motion', lead/max(total, 1)


def infer_door_family(part, vertices, whole_bounds):
    role = part['motion_role']
    edge = part['attachment_edge']
    ext = np.ptp(vertices, axis=0)
    # Aspect proposes a horizontal-hinge family, but does not select its edge.
    # The user's rule uses a VISIBLE PULL's height relative to THIS door first.
    # Side doors remain on the existing side-hinge path.
    horizontal = ext[1] < .68*max(ext[0], 1e-6)
    if horizontal:
        pull = part.get('pull_evidence', {})
        if pull.get('valid') and pull.get('visible') and pull.get('confidence') == 'high':
            if pull.get('position') == 'lower':
                return 'hinge_upper', 'reference_lower_pull_upper_hinge'
            if pull.get('position') == 'upper':
                return 'hinge_lower', 'reference_upper_pull_lower_hinge'
        middle = np.mean(whole_bounds[:, 1])
        return ('hinge_upper' if np.median(vertices[:, 1]) > middle else 'hinge_lower'), 'cabinet_position_fallback_no_unambiguous_pull'
    if role == 'upward_door' and part['motion_confidence'] == 'high' and edge == 'upper':
        return 'hinge_upper', 'visible_upper_attachment'
    if role == 'drop_door' and part['motion_confidence'] == 'high' and edge == 'lower':
        return 'hinge_lower', 'visible_lower_attachment'
    if role == 'sliding_door' and part['motion_confidence'] == 'high':
        return 'slider_horizontal', 'high_confidence_sliding_semantic'
    return 'hinge_side', 'door_panel_side_hinge_prior'


def fit_axis(owner, fragments, semantics, body, asset, whole_bounds):
    part = semantics[owner]
    v = np.concatenate([f['vertices'] for f in fragments])
    native_candidates = [f['native'] for f in fragments]
    semantic = part.get('effective_semantic', part['semantic'])
    compatible = ('prismatic',) if semantic == 'drawer' else ('revolute', 'continuous')
    native = next((j for j in native_candidates if j['type'] in compatible), native_candidates[0])
    if semantic == 'base':
        return fixed(), 'fixed_body', {'rule': 'recognized main body'}
    if semantic == 'drawer':
        span = abs(float(np.diff(native['range'])[0])) if native['type'] == 'prismatic' else .25*asset['diagonal']
        j = dict(type='prismatic', range=[0., span], axis=dict(origin=v.mean(0).tolist(), direction=[0., 0., 1.]))
        return j, 'slider_front', {'span_assumed': native['type'] != 'prismatic'}
    if semantic == 'door':
        family, evidence = infer_door_family(part, v, whole_bounds)
        if family == 'hinge_upper':
            j, _, _, audit = top_hinged_door(v, body, native)
        elif family == 'hinge_lower':
            j, audit = lower_hinge(v, body, native)
        elif family == 'slider_horizontal':
            span = abs(float(np.diff(native['range'])[0])) if native['type'] == 'prismatic' else .25*asset['diagonal']
            j = dict(type='prismatic', range=[0., span], axis=dict(origin=v.mean(0).tolist(), direction=[1., 0., 0.]))
            audit = {'span_assumed': native['type'] != 'prismatic'}
        else:
            edge = part['attachment_edge'] if part['attachment_edge'] in ('left', 'right') and part['motion_confidence'] == 'high' else ('left' if v[:, 0].mean() < np.median(body[:, 0]) else 'right')
            j, audit = side_hinge(v, body, edge, native)
            # Preserve an already plausible generated hinge. In some objects
            # the front/side frame is inset: the generic outer-edge heuristic
            # can move a correct PAct axis away from its actual body contact.
            if native['type'] in ('revolute', 'continuous'):
                nd = np.asarray(native['axis']['direction'], float)
                no = np.asarray(native['axis']['origin'], float)
                inferred = np.asarray(j['axis']['origin'], float)
                if np.linalg.norm(nd) > 1e-8 and abs(nd[1])/np.linalg.norm(nd) >= .985:
                    ncontact = float(cKDTree(body).query(no)[0])
                    icontact = float(cKDTree(body).query(inferred)[0])
                    leaf_line = float(cKDTree(v[:, [0, 2]]).query(no[[0, 2]])[0])
                    if (ncontact <= .025*asset['diagonal'] and
                        ncontact < .5*icontact and leaf_line <= .03*asset['diagonal']):
                        j = copy.deepcopy(native)
                        audit.update(native_axis_preserved=True,
                                     native_body_contact_distance=ncontact,
                                     inferred_body_contact_distance=icontact,
                                     native_leaf_axis_distance=leaf_line)
        audit['family_evidence'] = evidence
        audit['pull_evidence'] = part.get('pull_evidence')
        audit['pull_priority_used'] = evidence.startswith('reference_') and '_pull_' in evidence
        audit['cabinet_position_fallback_used'] = evidence == 'cabinet_position_fallback_no_unambiguous_pull'
        audit['selection_policy'] = 'Horizontal doors: lower pull -> upper hinge; upper pull -> lower hinge; absent/uncertain/middle -> cabinet-position fallback. Side doors retain existing logic.'
        return j, family, audit
    if semantic == 'handle':
        # A grip on a door/drawer follows the parent. A carrying handle may
        # articulate only with two separated geometric attachment contacts.
        parent = part.get('parent_instance_id')
        if part['motion_role'] == 'fixed_grip' and parent in semantics and semantics[parent]['semantic'] in ('door', 'drawer'):
            return fixed(), 'fixed_grip', {'parent_instance_id': parent}
        try:
            try:
                contacts = connected_handle_contacts(v, body, asset['diagonal'])
                pivot, direction, audit = fit_attachment_axis(contacts)
                audit['rule'] = 'two separated body contact clusters'
            except ValueError as contact_error:
                pivot, direction, audit = paired_handle_component_axis(
                    [f['vertices'] for f in fragments], asset['diagonal'], static_vertices=body)
                audit['body_contact_rule_failure'] = str(contact_error)
                contacts = np.asarray(audit['attachment_points'])
            span = abs(float(np.diff(native['range'])[0])) if native['type'] in ('revolute', 'continuous') else 90.
            try:
                sign, sign_audit = carrying_handle_outward_sign(
                    v, pivot, direction, body.mean(0), asset['diagonal'])
                audit['direction_evidence'] = sign_audit
            except ValueError as sign_error:
                sign = outward_sign(direction, v.mean(0)-pivot, pivot-body.mean(0))
                audit['direction_evidence'] = dict(rule='legacy_outward_fallback',reason=str(sign_error))
            j = dict(type='revolute', range=[0., sign*span], axis=dict(origin=pivot.tolist(), direction=direction.tolist()))
            audit['two_contact_support'] = contacts.tolist()
            return j, 'articulated_handle', audit
        except ValueError as exc:
            return copy.deepcopy(native), 'handle_native_unresolved', {'reason': str(exc)}
    return copy.deepcopy(native), 'native_unresolved_semantic', {'reason': 'No supported semantic motion rule'}


def prepare(auto_root, reg_root, output, case):
    auto, registration = auto_root/case, reg_root/case
    asset = load_asset(read(auto/'inputs.json')['source_run'])
    ids = np.load(auto/'instance_ids.npy')
    semantics = {p['instance_id']: p for p in read(auto/'semantics.json')['parts']}
    records = projection_votes(asset, ids, registration)
    output.mkdir(parents=True, exist_ok=False)
    fragments = []
    for r in records:
        nid = r['nid']
        mesh = r['part']['mesh']
        native = r['part']['node']
        for cid in np.unique(r['labels']):
            vi = np.flatnonzero(r['labels'] == cid)
            face_select = (r['labels'][mesh.faces[:, 0]] == cid)
            assert np.all(r['labels'][mesh.faces[face_select]] == cid)
            fi = np.flatnonzero(face_select)
            gi = np.flatnonzero(r['glabels'] == cid)
            visible = gi[r['visible'][gi]]
            pix = r['pixels'][visible]
            values = ids[pix[:, 1], pix[:, 0]] if len(pix) else np.empty(0, int)
            votes = {i: int(np.sum(values == i)) for i in semantics}
            owner, why, confidence = choose_owner(votes, native, semantics, len(visible),
                                                   mesh.vertices[vi].mean(0), None)
            fragments.append(dict(source_node=nid, component=int(cid), owner=int(owner),
                                  match_rule=why, match_confidence=float(confidence), votes=votes,
                                  visible_gaussians=len(visible), vertices=np.asarray(mesh.vertices)[vi],
                                  vi=vi, fi=fi, gi=gi, native=copy.deepcopy(native['joint'])))
    # A component must be counted once even if several source nodes reproduce
    # one semantic leaf. All matching fragments then share that leaf's joint.
    body_id = next((i for i, p in semantics.items() if p['semantic'] == 'base'), None)
    body_fragments = [f['vertices'] for f in fragments if f['owner'] == body_id]
    if not body_fragments:
        body_fragments = [p['mesh'].vertices for p in asset['parts'] if p['node']['joint']['type'] == 'fixed']
    body = np.concatenate(body_fragments)
    whole_bounds = np.array([np.min(body, axis=0), np.max(body, axis=0)])
    assignments = []
    for owner in sorted(semantics):
        selected = [f for f in fragments if f['owner'] == owner]
        if not selected:
            assignments.append(dict(instance_id=owner, semantic=semantics[owner]['semantic'], supported=False))
            continue
        effective = semantics[owner]['semantic']
        category_evidence = 'input_VLM'
        if effective == 'unknown' and selected:
            names = [next(r['part']['node']['name'] for r in records if r['nid'] == f['source_node']) for f in selected]
            if all('door' in name.lower() for name in names):
                effective, category_evidence = 'door', 'input_mask_projection_plus_generated_native_door'
            elif all('drawer' in name.lower() for name in names):
                effective, category_evidence = 'drawer', 'input_mask_projection_plus_generated_native_drawer'
        semantics[owner]['effective_semantic'] = effective
        try:
            joint, family, audit = fit_axis(owner, selected, semantics, body, asset, whole_bounds)
        except (ValueError, np.linalg.LinAlgError) as exc:
            joint, family, audit = copy.deepcopy(selected[0]['native']), 'native_axis_fallback', {'reason': str(exc)}
        assignments.append(dict(instance_id=owner, semantic=effective, input_semantic=semantics[owner]['semantic'],
                                category_evidence=category_evidence, supported=True,
                                family=family, joint=joint, axis_evidence=audit,
                                components=[dict(source_node=f['source_node'], component=f['component']) for f in selected]))
    # One group per connected fragment; fixed child fragments inherit the
    # transform of the primary semantic group, giving each instance one axis.
    groups, arrays = [], {}
    owner_primary = {}
    for assignment in assignments:
        if not assignment['supported']:
            continue
        own = assignment['instance_id']
        selected = sorted([f for f in fragments if f['owner'] == own], key=lambda f: len(f['gi']), reverse=True)
        owner_primary[own] = 10 + len(groups)
        for index, f in enumerate(selected):
            gid = 10 + len(groups)
            prefix = f'g{gid}'
            for key, idx in [('vertices', f['vi']), ('faces', f['fi']), ('gaussians', f['gi'])]:
                arrays[f'{prefix}_{key}'] = idx.astype(np.int64)
            primary = index == 0
            groups.append(dict(id=gid, name=f"{assignment['semantic']}_{own}_{index}",
                               semantic_owner=own, source_node=f['source_node'], component=f['component'],
                               kind=assignment['family'] if primary else 'fixed_child',
                               parent=-1 if primary else owner_primary[own],
                               shape_owner=None, box=None, edge=None, membership_prefix=prefix,
                               counts=dict(vertices=len(f['vi']), faces=len(f['fi']), gaussians=len(f['gi'])),
                               joint=copy.deepcopy(assignment['joint'] if primary else fixed()),
                               observed_q=0., deformation_basis=np.eye(3).tolist(),
                               deformation_anchor=f['vertices'].mean(0).tolist(), length=1.))
    for f in [f for f in fragments if f['owner'] == 0]:
        gid = 10 + len(groups)
        prefix = f'g{gid}'
        for key, idx in [('vertices', f['vi']), ('faces', f['fi']), ('gaussians', f['gi'])]:
            arrays[f'{prefix}_{key}'] = idx.astype(np.int64)
        groups.append(dict(id=gid, name=f"unresolved_{f['source_node']}_{f['component']}",
                           semantic_owner=None, source_node=f['source_node'], component=f['component'],
                           kind='native_unresolved', parent=-1, shape_owner=None, box=None, edge=None,
                           membership_prefix=prefix,
                           counts=dict(vertices=len(f['vi']), faces=len(f['fi']), gaussians=len(f['gi'])),
                           joint=copy.deepcopy(f['native']), observed_q=0.,
                           deformation_basis=np.eye(3).tolist(),
                           deformation_anchor=f['vertices'].mean(0).tolist(), length=1.))
    # Parent semantic instance is applied only after all primary IDs are known.
    for g in groups:
        own = g['semantic_owner']
        if own is None or g['id'] != owner_primary[own]:
            continue
        if own == body_id:
            g['parent'] = -1
            continue
        parent_owner = semantics[own].get('parent_instance_id')
        if g['kind'] == 'fixed_grip' and parent_owner in owner_primary and parent_owner != own:
            g['parent'] = owner_primary[parent_owner]
        else:
            g['parent'] = owner_primary.get(body_id, -1)
    for r in records:
        nid = r['nid']
        selected = [g for g in groups if g['source_node'] == nid]
        for key, count in [('vertices', len(r['labels'])), ('faces', len(r['part']['mesh'].faces)), ('gaussians', len(r['glabels']))]:
            index = np.concatenate([arrays[g['membership_prefix']+'_'+key] for g in selected])
            assert np.array_equal(np.sort(index), np.arange(count)), (case, nid, key)
    np.savez_compressed(output/'membership.npz', **arrays)
    sources = {str(r['nid']): dict(mesh=str(r['part']['mesh_path']), gs=str(r['gs_path'])) for r in records}
    files = [dict(path=str(p), sha256=sha256(p)) for v in sources.values() for p in v.values()]
    manifest = dict(schema='auto_component_axis_prior_v1', case=case, source_run=str(asset['run']),
                    source_object=str(asset['object_path']), source_object_sha256=sha256(asset['object_path']),
                    semantic_input=str(auto/'semantics.json'), semantic_input_sha256=sha256(auto/'semantics.json'),
                    instance_mask=str(auto/'instance_ids.npy'), instance_mask_sha256=sha256(auto/'instance_ids.npy'),
                    sources=sources, groups=groups, geometry_files=files, pairs=[],
                    membership_sha256=sha256(output/'membership.npz'),
                    selection='Visible projected Gaussian centers matched to frozen input semantic instance masks; connected meshes form independent physical fragments; no manual case recipes or GT axes',
                    geometry_changed=False)
    write_json(output/'motion_groups.json', manifest)
    write_json(output/'component_assignment.json', dict(case=case, assignments=assignments,
        fragments=[{k: v for k, v in f.items() if k not in ('vertices', 'vi', 'fi', 'gi', 'native')} for f in fragments],
        original_source_nodes=len(asset['nodes']), semantic_instances=len(semantics),
        independent_semantic_joints=sum(a['supported'] and a['joint']['type'] != 'fixed' for a in assignments)))
    # Configuration is only for shared render camera/source scene; no GT
    # evaluation is read or copied here.
    cfg = read(registration/'config.json')
    source_observation = np.load(registration/'observation.npz')
    # The shared Gaussian model only needs the original RGB/camera arrays for
    # rendering. Remove the older experiment's manually annotated labels so
    # no manual semantic pixels enter this automatic output at all.
    clean_labels = np.zeros_like(source_observation['alpha'], dtype=np.int32)
    np.savez_compressed(output/'observation.npz',
        rgb=source_observation['rgb'], alpha=source_observation['alpha'],
        labels=clean_labels, semantic=np.ones((*clean_labels.shape, 3), np.float32),
        extrinsics=source_observation['extrinsics'], intrinsics=source_observation['intrinsics'])
    cfg['prior_root'] = str(output)
    cfg['prior_manifest_sha256'] = sha256(output/'motion_groups.json')
    cfg['observations_sha256'] = sha256(output/'observation.npz')
    cfg['palette'] = {str(g['id']): [.5, .5, .5] for g in groups}
    write_json(output/'config.json', cfg)
    print('PREPARED', case, 'components', len(fragments), 'instance joints',
          sum(a['supported'] and a['joint']['type'] != 'fixed' for a in assignments), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--auto-root', type=pathlib.Path, required=True)
    parser.add_argument('--registration-root', type=pathlib.Path, required=True)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    parser.add_argument('--case')
    args = parser.parse_args()
    cases = [args.case] if args.case else read(args.auto_root/'protocol.json')['cases']
    args.output.mkdir(parents=True, exist_ok=True)
    if not (args.output/'protocol.json').exists():
        write_json(args.output/'protocol.json', dict(cases=cases, automatic_semantics=str(args.auto_root),
                   registration=str(args.registration_root), GT_used_during_preparation=False,
                   geometry_policy='Immutable source meshes and Gaussians; only component membership and joints change'))
    for case in cases:
        if not (args.output/case/'motion_groups.json').exists():
            prepare(args.auto_root, args.registration_root, args.output/case, case)


if __name__ == '__main__':
    main()
