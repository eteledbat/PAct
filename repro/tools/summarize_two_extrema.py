"""Aggregate ten-case two-extrema metrics beside the sealed closed-image baseline."""
import argparse
import csv
import json
import pathlib
import statistics
import sys

REPO=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'repro/tools'))
from auto_part_semantics import write_json

OLD=REPO/'repro/outputs/closed_multicase_20261028_v1'
VERSIONS=[('origin','original'),('axis','prior'),('axis_geometry','optimized')]


def read(path):return json.loads(pathlib.Path(path).read_text())
def avg(values):
    valid=[x for x in values if x is not None]
    return float(statistics.mean(valid)) if valid else None


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);a=p.parse_args()
    root=a.root.resolve();cases=read(root/'protocol.json')['cases']
    assert all((root/c/'evaluation.json').exists() and (root/c/'video_results.json').exists() for c in cases)
    records=[];aggregate={}
    for case in cases:
        new=read(root/case/'evaluation.json');old=read(OLD/case/'evaluation.json')
        for name,old_name in VERSIONS:
            entry=new['methods'][name];joint=entry['axis'];held=entry['heldout_mean']
            old_joint=old['joint_methods'][old_name];old_held=old['unfitted_open_mean'].get(old_name)
            row=dict(case=case,method=name,endpoint_frames='-'.join(map(str,new['endpoint_frames'])),
                endpoint_IoU_mean=avg([v['image']['IoU'] for v in entry['endpoints']]),
                endpoint_part_mIoU_mean=avg([v['part_mean_IoU'] for v in entry['endpoints']]),
                heldout_IoU=held['IoU'] if held else None,heldout_PSNR=held['PSNR'] if held else None,
                surface_CD_percent_D=entry['surface_reconstruction']['CD_percent_D'],
                surface_Fscore_5pct_D=entry['surface_reconstruction']['Fscore_5pct_D'],
                six_state_CD_percent_D=avg([x['CD_percent_D'] for x in new['six_state_surface'][name]]),
                GT_joint_count=joint['GT_joint_count'],matched_joint_count=sum(x['matched'] for x in joint['rows']),
                type_correct_count=sum(x['type_correct'] for x in joint['rows']),
                axis_correct_count=joint['axis_correct_count'],axis_correct_rate=joint['axis_correct_rate'],
                axis_angle_degrees=joint['conditional_mean_angle_degrees'],
                pivot_percent_D=joint['conditional_mean_pivot_percent_D'],
                old_closed_fit_IoU=old['fit_image_metrics'][old_name]['IoU'],
                old_heldout_IoU=old_held['IoU'] if old_held else None,
                old_surface_CD_percent_D=old['surface_reconstruction'][old_name]['CD_percent_D'],
                old_axis_correct_rate=old_joint['axis_correct_rate'],
                old_type_correct_count=sum(x['type_correct'] for x in old_joint['rows']))
            records.append(row)
    for name,_ in VERSIONS:
        rows=[x for x in records if x['method']==name]
        aggregate[name]=dict(cases=len(rows),endpoint_IoU_macro=avg([x['endpoint_IoU_mean'] for x in rows]),
            endpoint_part_mIoU_macro=avg([x['endpoint_part_mIoU_mean'] for x in rows]),
            heldout_IoU_macro=avg([x['heldout_IoU'] for x in rows]),heldout_PSNR_macro=avg([x['heldout_PSNR'] for x in rows]),
            surface_CD_percent_D_macro=avg([x['surface_CD_percent_D'] for x in rows]),
            six_state_CD_percent_D_macro=avg([x['six_state_CD_percent_D'] for x in rows]),
            GT_joint_count=sum(x['GT_joint_count'] for x in rows),
            type_correct_count=sum(x['type_correct_count'] for x in rows),
            axis_correct_count=sum(x['axis_correct_count'] for x in rows),
            axis_correct_rate_micro=sum(x['axis_correct_count'] for x in rows)/max(sum(x['GT_joint_count'] for x in rows),1),
            old_heldout_IoU_macro=avg([x['old_heldout_IoU'] for x in rows]),
            old_surface_CD_percent_D_macro=avg([x['old_surface_CD_percent_D'] for x in rows]),
            old_axis_correct_rate_micro=sum(int(round(x['old_axis_correct_rate']*x['GT_joint_count'])) for x in rows)/max(sum(x['GT_joint_count'] for x in rows),1))
    # A later manual audit found no predicted fixed-body owner in 9016. Its
    # original static-body alignment used a moving door and cannot support
    # aligned 3D/axis metrics. Preserve the raw ten-case rows, and publish a
    # qualified nine-case subset for those metric families.
    valid_alignment={}
    for name,_ in VERSIONS:
        rows=[x for x in records if x['method']==name and x['case']!='9016']
        valid_alignment[name]=dict(cases=len(rows),GT_joint_count=sum(x['GT_joint_count'] for x in rows),
            axis_correct_count=sum(x['axis_correct_count'] for x in rows),
            surface_CD_percent_D_macro=avg([x['surface_CD_percent_D'] for x in rows]),
            six_state_CD_percent_D_macro=avg([x['six_state_CD_percent_D'] for x in rows]))
    with (root/'metrics.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    write_json(root/'aggregate.json',dict(cases=cases,summary=aggregate,rows=records,
        audited_static_alignment_excluding_9016=valid_alignment,
        baseline_root=str(OLD),video_files={case:read(root/case/'video_results.json')['video'] for case in cases},
        comparability='Same metric definitions and static-body alignment where applicable. New endpoint training IoU differs in pose/image from old closed-image training IoU; the columns are descriptive, not a controlled input-matched delta. Heldout state0 cameras and GT axis metrics provide closer comparisons.'))
    lines=['# Two-motion-extrema automatic semantic reconstruction',
        '',f'Published base: `v0.1.1-handle-attachment-axis` (`{read(root/"protocol.json")["release_commit"]}`).',
        '', 'Input: two observed frames per case with the largest normalized multi-joint-state separation; 20411 uses frames 00/05. The published single-image PAct generator is rerun from the first selected frame, and its new Origin is registered to that image. Qwen3-VL-4B runs locally on both endpoint RGBs and oracle projected 2D part masks. Semantic labels are transferred to the newly generated Origin by stable instance ID, then v0.1.1 priors set separate component axes. Two endpoint images fit each joint travel and one shared geometry.',
        '', 'The instance masks are GT mesh projections, so they provide oracle part boundaries. GT numeric joints are used to choose frames and project masks, never directly as fitted axes/ranges. This is a two-image supervised experiment, not a RGB-only segmentation benchmark.',
        '', '| Version | Endpoint IoU | Endpoint part mIoU | Heldout IoU | Source CD %D | Six-state CD %D | Correct axes / GT | Old heldout IoU | Old CD %D |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(x):return '—' if x is None else f'{x:.3f}'
    for key,label in [('origin','Origin'),('axis','VLM axis + observed travel'),('axis_geometry','VLM axis + two-frame geometry')]:
        x=aggregate[key];lines.append(f'| {label} | {fmt(x["endpoint_IoU_macro"])} | {fmt(x["endpoint_part_mIoU_macro"])} | {fmt(x["heldout_IoU_macro"])} | {fmt(x["surface_CD_percent_D_macro"])} | {fmt(x["six_state_CD_percent_D_macro"])} | {x["axis_correct_count"]}/{x["GT_joint_count"]} | {fmt(x["old_heldout_IoU_macro"])} | {fmt(x["old_surface_CD_percent_D_macro"])} |')
    lines += ['', 'Endpoint IoU is a training-image diagnostic; the old closed-image IoU uses a different target image. Heldout cameras and GT joint metrics provide closer comparisons. The six-state trajectories also cover different frame ranges. A single global extreme pair may not reach every independently moving part\'s own minimum and maximum.',
        '', '**Semantic audit correction:** 9016 has no generated fixed-body group. Its axis-pivot and aligned 3D scores in the raw ten-case table above used a moving door as the static alignment anchor and must not be interpreted as valid. Excluding 9016, the qualified nine-case axis counts are Origin 6/22 and axis/geometry 10/22; source CD %D is 3.681 / 3.681 / 3.716 and six-state CD %D is 3.642 / 3.685 / 3.489. Input and held-out 2D scores still cover all ten cases. See [manual semantic audit](SEMANTIC_MANUAL_AUDIT.md).',
        '', '| Case | Frames | Origin axis | New axis | New geometry heldout IoU | Four-version video |',
        '|---|---|---:|---:|---:|---|']
    for case in cases:
        a=next(x for x in records if x['case']==case and x['method']=='origin');b=next(x for x in records if x['case']==case and x['method']=='axis_geometry')
        lines.append(f'| {case} | {a["endpoint_frames"]} | {a["axis_correct_count"]}/{a["GT_joint_count"]} | {b["axis_correct_count"]}/{b["GT_joint_count"]} | {fmt(b["heldout_IoU"])} | [{case}]({case}/{case}_GT_Origin_VLMAxis_TwoFrameGeo.mp4) |')
    lines += ['', 'Observed failure modes: 100481 has two independent predicted handle axes and both directions reach 0° error, but pivot offsets are 16.02% and 8.46% of the GT body diagonal, above the 5% correctness threshold. New Origin has only one independently matchable moving group for the two GT joints in 101808 and 10655, and three for four GT joints in 31249. Case 45194 matches all four GT joints correctly but also predicts one extra moving fragment; its second endpoint silhouette IoU after geometry fit is 0.603. These cases remain in the aggregate.',
        '', 'The held-out view score is unavailable for 20411 because its reference sequence has no additional static-state cameras. All per-case scores, exact axes and pivots, endpoint hashes, VLM outputs, masks and videos are retained in this directory. `metrics.csv` and `aggregate.json` contain numeric comparisons.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('REPORT',root/'REPORT.md',flush=True)


if __name__=='__main__':main()
