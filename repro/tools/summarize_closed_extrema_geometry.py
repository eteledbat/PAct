"""Create auditable all-case report from sealed closed-reference fits and videos."""
import argparse,csv,json,pathlib,statistics,sys
REPO=pathlib.Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO/'repro/tools'))
from auto_part_semantics import sha256,write_json
from fit_closed_extrema_geometry import CASES

def read(p):return json.loads(p.read_text())
def mean(v):return sum(v)/len(v)
def fmt(v):return "—" if v is None else f"{v:.1f}"
def summarize(root):
    rows=[]
    for case in CASES:
        d=read(root/case/'metrics.json');video=read(root/case/'video_results.json');fit=read(root/case/'fit_seal.json')
        row=dict(case=case,closed_frame=fit['closed_frame'],video=video['video'],
                 origin_3d_IoU=d['surface_voxel_3d_IoU']['origin']['surface_voxel_IoU_r1_mean'],
                 axis_3d_IoU=d['surface_voxel_3d_IoU']['axis_only']['surface_voxel_IoU_r1_mean'],
                 closed_geometry_3d_IoU=d['surface_voxel_3d_IoU']['closed_geometry']['surface_voxel_IoU_r1_mean'],
                 axis_closed_pose_3d_IoU=d['calibrated_closed_3d_IoU']['axis_only']['surface_voxel_IoU_r1'],
                 geometry_closed_pose_3d_IoU=d['calibrated_closed_3d_IoU']['closed_geometry']['surface_voxel_IoU_r1'],
                 origin_closed_2d_IoU=d['closed_2d_silhouette_IoU']['origin_at_native_endpoint'],
                 axis_closed_2d_IoU=d['closed_2d_silhouette_IoU']['axis_at_checked_pose'],
                 geometry_closed_2d_IoU=d['closed_2d_silhouette_IoU']['closed_geometry_at_checked_pose'],
                 axis_part_mIoU=d['closed_part_mean_IoU']['axis_at_checked_pose'],
                 geometry_part_mIoU=d['closed_part_mean_IoU']['closed_geometry_at_checked_pose'],
                 GT_joints=d['axis']['origin']['GT_joint_count'],
                 origin_axis_correct=d['axis']['origin']['axis_correct_count'],
                 axis_axis_correct=d['axis']['axis_only']['axis_correct_count'],
                 geometry_axis_correct=d['axis']['closed_geometry']['axis_correct_count'],
                 origin_type_coverage=d['axis']['origin']['type_correct_GT_coverage'],
                 axis_type_coverage=d['axis']['axis_only']['type_correct_GT_coverage'],
                 geometry_type_coverage=d['axis']['closed_geometry']['type_correct_GT_coverage'],
                 axis_direction_error_deg=d['axis']['axis_only']['conditional_mean_angle_degrees'],
                 axis_pivot_error_percent_D=d['axis']['axis_only']['conditional_mean_pivot_percent_D'])
        rows.append(row)
    with (root/'metrics_all_cases.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    keys=['origin_3d_IoU','axis_3d_IoU','closed_geometry_3d_IoU','axis_closed_pose_3d_IoU',
          'geometry_closed_pose_3d_IoU','origin_closed_2d_IoU','axis_closed_2d_IoU','geometry_closed_2d_IoU',
          'axis_part_mIoU','geometry_part_mIoU']
    aggregate={k:mean([r[k] for r in rows]) for k in keys}
    for k in ['GT_joints','origin_axis_correct','axis_axis_correct','geometry_axis_correct']:aggregate[k]=sum(r[k] for r in rows)
    write_json(root/'metrics_all_cases.json',dict(cases=rows,aggregate=aggregate,code_sha256=sha256(__file__)))
    lines=['# Closed travel-reference image geometry fit: all ten cases','',
      'Base: published PAct v0.1.1 tag `bdf55de84df403bf1093eefb55b5b775382c5caf`; 100481 uses published handle-axis repair, the other nine use the published single-image ten-case output. Existing Origin and automatic-axis artifacts are reused. The established all-leaves-closed RGB/mask and its known camera calibrate per-part observed pose; the published 240-step shape optimizer then fits geometry. Axis type, direction, pivot, parent, and native ranges are byte-identical to the reused axis result.','',
      'All ten selected fully closed stills are shown in [closed reference contact sheet](closed_reference_contact_sheet.png).', '',
      'The video columns are GT | Origin | automatic axis | closed-frame geometry. The first three columns come from the published four-column video; the last is rendered with the same two virtual cameras. Playback uses original native ranges, so the GT source-to-endpoint timing need not coincide with the chosen closed reference frame.','',
      '| Case | Closed frame | 3D IoU Origin | Axis | Closed geometry | Closed-pose 3D Axis→Geometry | Closed 2D Axis→Geometry | Axis correct Origin→Axis→Geometry | Direction ° | Pivot %D | Video |',
      '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in rows:
        lines.append(f"| {r['case']} | {r['closed_frame']} | {r['origin_3d_IoU']:.3f} | {r['axis_3d_IoU']:.3f} | {r['closed_geometry_3d_IoU']:.3f} | {r['axis_closed_pose_3d_IoU']:.3f}→{r['geometry_closed_pose_3d_IoU']:.3f} | {r['axis_closed_2d_IoU']:.3f}→{r['geometry_closed_2d_IoU']:.3f} | {r['origin_axis_correct']}→{r['axis_axis_correct']}→{r['geometry_axis_correct']}/{r['GT_joints']} | {fmt(r['axis_direction_error_deg'])} | {fmt(r['axis_pivot_error_percent_D'])} | [{r['case']} video]({r['video']}) |")
    a=aggregate
    lines += ['',f"Macro mean 3D surface-voxel IoU (r1): Origin {a['origin_3d_IoU']:.3f}, axis {a['axis_3d_IoU']:.3f}, closed geometry {a['closed_geometry_3d_IoU']:.3f}. At the image-estimated closed pose: axis {a['axis_closed_pose_3d_IoU']:.3f} → geometry {a['geometry_closed_pose_3d_IoU']:.3f}.",
      f"Macro mean closed-image 2D silhouette IoU: Origin {a['origin_closed_2d_IoU']:.3f}, axis {a['axis_closed_2d_IoU']:.3f}, geometry {a['geometry_closed_2d_IoU']:.3f}. Axis correct: {a['origin_axis_correct']} → {a['axis_axis_correct']} → {a['geometry_axis_correct']} / {a['GT_joints']} GT joints.",
      '', '3D IoU is of sampled surface shells at a shared voxel pitch equal to 1/96 of GT fixed-body diagonal, dilated one six-connected voxel. Source and GT meshes are open; this is not solid-volume IoU. Axis correct requires type match, unsigned direction error ≤5°, and for revolute joints pivot-line distance ≤5% of GT fixed-body diagonal. A prismatic axis has no uniquely observable pivot point. Closed-pose 3D scores use the pose estimated from the closed 2D mask for both axis and geometry. Per-endpoint 3D scores instead use each model’s unchanged native range, as in the video. GT numeric joints and 3D meshes are used only for evaluation, except that reference meshes are projected to obtain the 2D instance masks used in fitting.',
      '', 'Detailed per-part axis errors, IoU at both extreme frames, closed pose, and input hashes are in each case `metrics.json`, `fit_seal.json`, and `position_check.json`.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(aggregate,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);a=p.parse_args();summarize(a.root.resolve())
