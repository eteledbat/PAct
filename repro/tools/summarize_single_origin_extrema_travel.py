"""Summarize travel-only fitting on existing published single-image assets."""
import argparse
import csv
import json
import pathlib
import statistics
import sys

REPO=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'repro/tools'))
from auto_part_semantics import write_json


def read(path):return json.loads(pathlib.Path(path).read_text())
def mean(values):return statistics.mean(values)


def main(root):
    cases=read(root/'protocol.json')['cases'];rows=[]
    for case in cases:
        fit=read(root/case/'travel_fit.json');video=read(root/case/'five_video_results.json')
        assert video['frames']==84 and video['all_frames_decoded']
        old,new=fit['before'],fit['after']
        rows.append(dict(case=case,frames='-'.join(map(str,fit['frames'])),matched_joints=len(fit['matched_movable_groups']),
            native_endpoint_IoU=mean(x['image']['IoU'] for x in old),
            optimized_endpoint_IoU=mean(x['image']['IoU'] for x in new),
            native_endpoint_part_mIoU=mean(x['part_mean_IoU'] for x in old),
            optimized_endpoint_part_mIoU=mean(x['part_mean_IoU'] for x in new),
            video=video['video'],source_four_panel=fit['source_four_panel']))
    with (root/'metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={key:mean(row[key] for row in rows) for key in ['native_endpoint_IoU','optimized_endpoint_IoU',
        'native_endpoint_part_mIoU','optimized_endpoint_part_mIoU']}
    write_json(root/'aggregate.json',dict(cases=cases,summary=summary,rows=rows,
        metric_scope='Two selected input frames only. These are fitting diagnostics, not held-out reconstruction metrics.',
        video_scope='First four columns are decoded from the original published single-image videos; fifth column changes numeric joint range only.'))
    lines=['# 单图 PAct 产物的双极限帧行程优化','',
        '最新 GitHub 标签 `v0.1.1-handle-attachment-axis` 对应的 100481 单图修复产物：`repro/outputs/handle_attachment_four_versions_20261030_v1/100481`。其余九个 case 的已发布单图四组产物位于 `repro/outputs/github_v010_multicase_20261030_v1/<case>`，对应 v0.1.0 十例批次。v0.1.1 没有重新生成九例 Origin。',
        '', '本实验从上述每个已保存的单图模型和原四列视频出发。两张选定的运动极限帧仅用来估计各已匹配关节的起止数值；轴类型、轴位置和方向、部件归属、单图几何与相机均不变。第五列渲染原单图优化几何和新的行程。',
        '', f'10 例、两张训练帧的平均轮廓 IoU：原行程 **{summary["native_endpoint_IoU"]:.3f}** → 双帧行程 **{summary["optimized_endpoint_IoU"]:.3f}**；部件 mIoU：**{summary["native_endpoint_part_mIoU"]:.3f}** → **{summary["optimized_endpoint_part_mIoU"]:.3f}**。这些是输入帧诊断，不能当作未见视角的提升。',
        '', '| Case | 极限帧 | 原行程 IoU | 新行程 IoU | 原部件 mIoU | 新部件 mIoU | 五列视频 |',
        '|---|---:|---:|---:|---:|---:|---|']
    for row in rows:
        case=row['case'];lines.append(f'| {case} | {row["frames"]} | {row["native_endpoint_IoU"]:.3f} | {row["optimized_endpoint_IoU"]:.3f} | {row["native_endpoint_part_mIoU"]:.3f} | {row["optimized_endpoint_part_mIoU"]:.3f} | [{case}]({case}/{row["video"]}) |')
    lines += ['', '五列顺序：GT｜Origin｜VLM 轴｜单图姿态＋几何｜同一单图几何＋双极限帧行程。原四列来自现有视频；新列使用相同的两个视角。原 GT 列保留旧视频的 source→旧终点时间轴，新列使用本次选定的两张极限帧；两列的 `t` 是各自行程比例，不能逐帧解释为同一真实姿态。',
        '', '极限帧是按已知 GT 多关节轨迹挑选的全局最远两帧；拟合使用该两帧 RGB 与 GT 网格投影得到的 2D 实例 mask，未读入 GT 数值关节参数。多活动部件可能并未在这同一对帧中分别达到自己的极值。旧语义和部件匹配沿用发布产物，未应用事后人工审查的修正。',
        '', '`metrics.csv` 给出逐例数值；`travel_fit.json` 与 `travel_motion_groups.json` 保存搜索轨迹和唯一改变的关节区间；`five_video_results.json` 保存原视频和合成视频哈希。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('REPORT',root/'REPORT.md')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);a=p.parse_args()
    main(a.root.resolve())
