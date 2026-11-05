"""One reference-pose convention shared by fitting, evaluation and playback.

The historical pipeline fitted a closed pose per joint but rendered each asset
over its *native* range, so for many cases the fitted closed state was never
shown.  This module records, for every selected automatic candidate:

  q_closed        fitted observed closed pose (optimization / CD evaluation)
  native_range    the asset's declared [lo, hi] joint range
  q_open_native   the native open extreme (hi)
  playback        [q_closed, q_open_native] -- always contains the fitted state

It also flags every joint whose fitted closed pose lies outside the native
range, and every joint whose native travel is too small to define an opening
direction.  Those flags are the joints where the second observed state, not a
rule, has to decide the motion structure.

Nothing here reads GT 3D geometry or GT joints, and nothing rewrites a sealed
fit.  The output is a sidecar convention record.
"""
import argparse
import hashlib
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from auto_part_semantics import write_json
from fit_auto_closed_geometry import CASES, read

EPS_REVOLUTE = 1.0       # degrees; below this the native range cannot define a direction
EPS_PRISMATIC = 1e-5     # model units
DEFAULT_SELECTION = 'closed_auto_geometry_selected_20261104'


def load_pose(root):
    """Return (pose dict, source description) for a selected candidate root."""
    auto = root / 'auto_closed_fit.json'
    seal = root / 'fit_seal.json'
    if auto.exists():
        fit = read(auto)
        return fit['pose'], 'auto_closed_fit.json'
    if seal.exists():
        fit = read(seal)
        return fit['observed_pose'], 'fit_seal.json'
    raise FileNotFoundError(f'No fitted pose in {root}')


def within_native(q, lo, hi, joint_type, tol=0.0):
    """Whether q is inside [min(lo,hi), max(lo,hi)]; revolute also mod 360."""
    a, b = min(lo, hi) - tol, max(lo, hi) + tol
    if a <= q <= b:
        return True
    if joint_type in ('revolute', 'continuous'):
        period = 360.0
        k = round((q - a) / period)
        return a <= q - k * period <= b
    return False


def joint_record(group, q_closed):
    j = group['joint']
    kind = j['type']
    lo, hi = (float(x) for x in j['range'])
    travel = hi - lo
    eps = EPS_REVOLUTE if kind in ('revolute', 'continuous') else EPS_PRISMATIC
    rec = {
        'group': group['id'],
        'name': group.get('name'),
        'type': kind,
        'native_range': [lo, hi],
        'q_closed': q_closed,
        'q_open_native': hi,
        'native_travel': travel,
        'native_travel_near_zero': abs(travel) < eps,
    }
    if q_closed is None:
        rec.update(
            status='unresolved',
            closed_pose_within_native=False,
            playback=None,
            q_closed_outside_native=None,
            open_direction_ambiguous=True,
        )
        return rec
    inside = within_native(q_closed, lo, hi, kind, tol=eps)
    outside = not inside
    rec.update(
        status='resolved',
        closed_pose_within_native=inside,
        closed_pose_within_native_mod360=inside,
        q_closed_outside_native=outside,
        playback=[q_closed, hi],
        playback_span=hi - q_closed,
        playback_contains_fitted_closed=True,
        open_direction_ambiguous=bool(outside or abs(travel) < eps),
    )
    # Candidate open endpoints, recorded but not selected by this module.
    if outside:
        rec['candidate_open_endpoints'] = {
            'native_open': hi,
            'mirrored_open': q_closed + travel,
            'span_third_state_required': True,
        }
    return rec


def build_case(outputs, selection_root, case):
    selection = {r['case']: r for r in read(selection_root / 'SELECTION.json')['cases']}
    chosen = selection[case]
    root = pathlib.Path(chosen['selected_root'])
    pose, pose_source = load_pose(root)
    manifest = read(root / 'motion_groups.json')
    records = [joint_record(g, pose.get(str(g['id'])))
               for g in manifest['groups'] if g['joint']['type'] != 'fixed']
    return {
        'case': case,
        'selected_candidate': chosen['selected'],
        'selected_root': str(root),
        'checkpoint_sha256': chosen['checkpoint_sha256'],
        'pose_source': pose_source,
        'pose_source_sha256': None,
        'convention': {
            'q_closed': 'fitted observed closed pose; used for geometry fitting and CD evaluation',
            'playback': '[q_closed, q_open_native]; video must pass through the fitted closed state',
            'open_direction': 'not resolved from one image; second state required when flagged',
        },
        'joints': records,
        'flags': {
            'joints': len(records),
            'resolved': sum(r['status'] == 'resolved' for r in records),
            'closed_outside_native': sum(bool(r.get('q_closed_outside_native')) for r in records),
            'native_travel_near_zero': sum(bool(r['native_travel_near_zero']) for r in records),
            'open_direction_ambiguous': sum(bool(r['open_direction_ambiguous']) for r in records),
        },
    }


def markdown(rows):
    lines = [
        '# Reference-pose convention audit',
        '',
        'Built from the selected automatic candidates at '
        '`closed_auto_geometry_selected_20261104`. No GT 3D mesh or GT joint was read.',
        '',
        'Convention: geometry is fitted at `q_closed` (the observed closed state) and '
        'CD is evaluated there; video playback spans `[q_closed, q_open_native]`, so the '
        'fitted closed state is always inside the played range. When '
        '`q_closed` lies outside the native range, or native travel is near zero, the '
        'opening direction is flagged ambiguous and must be decided by the second '
        'observed state, not by a rule.',
        '',
        '| Case | Candidate | Joint | Type | Native range | q_closed | Outside native | Near-zero travel |',
        '|---|---|---|---|---:|---:|---|---|',
    ]
    for row in rows:
        for j in row['joints']:
            q = 'n/a' if j['q_closed'] is None else f"{j['q_closed']:.3f}"
            lines.append(
                f"| {row['case']} | {row['selected_candidate']} | {j['name']} ({j['group']}) | "
                f"{j['type']} | [{j['native_range'][0]:.3f}, {j['native_range'][1]:.3f}] | "
                f"{q} | {'yes' if j.get('q_closed_outside_native') else 'no'} | "
                f"{'yes' if j['native_travel_near_zero'] else 'no'} |"
            )
    totals = {k: sum(r['flags'][k] for r in rows)
              for k in ('joints', 'resolved', 'closed_outside_native',
                        'native_travel_near_zero', 'open_direction_ambiguous')}
    lines += [
        '',
        f"Totals: {totals['joints']} movable joints, {totals['resolved']} resolved, "
        f"{totals['closed_outside_native']} with the fitted closed pose outside the native range, "
        f"{totals['native_travel_near_zero']} with near-zero native travel, "
        f"{totals['open_direction_ambiguous']} requiring second-state evidence.",
        '',
        'This record changes no sealed artifact. It defines the pose contract the '
        'unified baseline must satisfy before any method comparison is meaningful.',
    ]
    return '\n'.join(lines) + '\n'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--outputs', type=pathlib.Path, required=True)
    p.add_argument('--selection-root', type=pathlib.Path)
    p.add_argument('--out', type=pathlib.Path)
    a = p.parse_args()
    outputs = a.outputs.resolve()
    selection_root = (a.selection_root or outputs / DEFAULT_SELECTION).resolve()
    out = (a.out or outputs / 'pose_convention_20261105').resolve()
    out.mkdir(parents=True, exist_ok=False)
    rows = [build_case(outputs, selection_root, case) for case in CASES]
    for row in rows:
        root = pathlib.Path(row['selected_root'])
        row['pose_source_sha256'] = hashlib.sha256(
            (root / row['pose_source']).read_bytes()).hexdigest()
    write_json(out / 'pose_convention.json', {'cases': rows, 'selection_root': str(selection_root)})
    (out / 'POSE_CONVENTION.md').write_text(markdown(rows), encoding='utf-8')
    print('Wrote', out / 'pose_convention.json')


if __name__ == '__main__':
    main()
