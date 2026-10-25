"""Axis-only analysis: parameter displacement versus actual GT error improvement."""
import argparse
import json
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO=pathlib.Path(__file__).resolve().parents[2]


def angle(a,b):
    a,b=np.array(a),np.array(b);n=np.linalg.norm(a)*np.linalg.norm(b)
    return None if n<1e-12 else float(np.degrees(np.arccos(np.clip(abs(a@b)/n,0,1))))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=pathlib.Path(a.root).resolve();cfg=json.loads((root/'config.json').read_text());output={};lines=['# 仅分析加入运动碰撞 Loss 对轴的影响','','A是不加碰撞Loss的同预算细化，B是加入碰撞Loss；两组初始化、图像注册、可训练参数和最终步数相同。位置是预测pivot到GT无限轴线的垂距，单位为GT静态部件对角线百分比；方向误差忽略轴正负。此处只评价轴，不将图像质量作为结论依据。','','| 实验 / GT部件 | 位置误差 A → B（% GT D） | 方向误差 A → B（°） | 原点实际移动（% GT D） | 方向实际改变（°） |','| --- | --- | --- | --- | --- |']
    def fmt(x):return 'N/A' if x is None else f'{x:.3f}'
    for experiment in cfg['experiments']:
        name=experiment['name'];results=json.loads((root/f'results_{name}.json').read_text());D=results['A']['geometry']['GT_static_D'];rows=[];valid=[j for j in results['A']['joints'] if j['type_correct']]
        fig,axes=plt.subplots(2,len(valid),figsize=(4.4*len(valid),8),squeeze=False)
        for joint,other,original in zip(results['A']['joints'],results['B']['joints'],results['R0']['joints']):
            g=np.array(joint['GT_axis_direction_world']);go=np.array(joint['GT_axis_origin_world']);ao=np.array(joint['predicted_axis_origin_world']);bo=np.array(other['predicted_axis_origin_world']);ad=np.array(joint['predicted_axis_direction_world']);bd=np.array(other['predicted_axis_direction_world']);revolute=joint['type_correct'] and joint['GT_type']=='revolute'
            row={'GT_part':joint['GT_part'],'predicted_node_id':joint['node_id'],'GT_type':joint['GT_type'],'predicted_type':joint['type'],'type_correct':joint['type_correct'],
                 'R0_axis_error_degrees':original['axis_degrees'],'A_axis_error_degrees':joint['axis_degrees'],'B_axis_error_degrees':other['axis_degrees'],
                 'R0_pivot_error_percent_GT_D':original['pivot_percent_GT_static_D'],'A_pivot_error_percent_GT_D':joint['pivot_percent_GT_static_D'],'B_pivot_error_percent_GT_D':other['pivot_percent_GT_static_D'],
                 'actual_origin_displacement_world':(bo-ao).tolist() if revolute else None,
                 'actual_origin_displacement_percent_GT_D':float(100*np.linalg.norm(bo-ao)/D) if revolute else None,
                 'actual_direction_change_degrees':angle(ad,bd) if joint['type_correct'] else None,
                 'A_origin_world':ao.tolist(),'B_origin_world':bo.tolist(),'A_direction_world':ad.tolist(),'B_direction_world':bd.tolist()}
            rows.append(row)
            if not joint['type_correct']:
                lines.append(f"| {name} / {joint['GT_part']} | 类型错误：{joint['type']} | 不计有效轴误差 | N/A | N/A |")
                continue
            lines.append(f"| {name} / {joint['GT_part']} | {fmt(row['A_pivot_error_percent_GT_D'])} → {fmt(row['B_pivot_error_percent_GT_D'])} | {fmt(row['A_axis_error_degrees'])} → {fmt(row['B_axis_error_degrees'])} | {fmt(row['actual_origin_displacement_percent_GT_D'])} | {fmt(row['actual_direction_change_degrees'])} |")
            col=next(i for i,x in enumerate(valid) if x['GT_part']==joint['GT_part']);u=np.cross(g,np.eye(3)[np.argmin(abs(g))]);u/=np.linalg.norm(u);v=np.cross(g,u);basis=np.stack([u,v],axis=1)
            ax=axes[0,col]
            if revolute:
                pa=100*(ao-go)@basis/D;pb=100*(bo-go)@basis/D;ax.scatter(0,0,c='black',marker='*',s=110,label='GT axis');ax.scatter(*pa,c='#2166ac',label='A: no collision');ax.scatter(*pb,c='#b2182b',label='B: + collision');ax.annotate('',xy=pb,xytext=pa,arrowprops={'arrowstyle':'->','color':'#b2182b'});ax.set(xlabel='GT-normal-plane coordinate u (% D)',ylabel='GT-normal-plane coordinate v (% D)',title=f"GT part {joint['GT_part']}: axis position\nerror {fmt(row['A_pivot_error_percent_GT_D'])} -> {fmt(row['B_pivot_error_percent_GT_D'])} % D");ax.axis('equal');ax.grid(alpha=.25);ax.legend(fontsize=8)
            else:ax.text(.5,.5,'Prismatic joint:\npivot / origin is not identifiable',ha='center',va='center',transform=ax.transAxes);ax.set_axis_off()
            def direction_plane(d):
                d=d/np.linalg.norm(d)
                if d@g<0:d=-d
                return np.degrees(np.arctan2(d@basis,d@g))
            qa,qb=direction_plane(ad),direction_plane(bd);ax=axes[1,col];ax.scatter(0,0,c='black',marker='*',s=110,label='GT direction');ax.scatter(*qa,c='#2166ac',label='A');ax.scatter(*qb,c='#b2182b',label='B');ax.annotate('',xy=qb,xytext=qa,arrowprops={'arrowstyle':'->','color':'#b2182b'});ax.set(xlabel='Direction tilt toward u (degrees)',ylabel='Direction tilt toward v (degrees)',title=f"Direction error {fmt(row['A_axis_error_degrees'])} -> {fmt(row['B_axis_error_degrees'])} deg");ax.axis('equal');ax.grid(alpha=.25);ax.legend(fontsize=8)
        fig.suptitle(name+' | axis-only comparison, shared alignment; arrows A to B');fig.tight_layout(rect=(0,0,1,.95));directory=root/f'figures_{name}';fig.savefig(directory/'axis_only_changes.png',dpi=180);fig.savefig(directory/'axis_only_changes.pdf');plt.close(fig);output[name]=rows
    lines+=['','图中位置投影到垂直GT轴的平面，以GT轴为原点；方向图画相对GT方向的两个倾角分量。箭头表示A到B。原点移动幅度和方向改变幅度不等于GT误差改善幅度。','','8961两条pivot误差均小幅下降，第一条轴方向改善、第二条方向变差。45194的revolute pivot几乎没有改善；一个prismatic方向近乎不变，另一个明显变差，另一扇门被预测为fixed，不能计有效轴误差。','','8961原始R0的导出轴经官方坐标轴吸附后方向误差为0°；A/B均偏离了这一方向。因此“B某条轴比A好”不等于“方向优于原始PAct”。本轮位置采用共同图像注册，与旧GT静态AABB对齐报告的绝对数值不可混用。']
    for name in output:lines+=['',f'[{name}：轴位置与方向变化图]({root/f"figures_{name}"/"axis_only_changes.png"})。']
    (root/'axis_only_changes.json').write_text(json.dumps(output,indent=2));(REPO/'repro/evidence/axis-only-ablation-changes.json').write_text(json.dumps(output,indent=2));(REPO/'repro/AXIS_ONLY_ABLATION_RESULTS.md').write_text('\n'.join(lines)+'\n');print('Axis-only report and plots completed.')


if __name__=='__main__':main()
