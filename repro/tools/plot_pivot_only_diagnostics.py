"""Measured pivot comparison and actual hinge-plane cuts, no re-registration."""
import argparse,json,pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np
import trimesh
from collision_refinement_common import load_asset,candidate_nodes,sha256

p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=pathlib.Path(a.root);config=json.loads((root/'experiment.json').read_text());case,run=next(iter(config['cases'].items()));asset=load_asset(run)
out=root/'diagnostics';out.mkdir(exist_ok=False)
metrics={m:json.loads((root/f'eval_{case}_{m}/metrics.json').read_text()) for m in ['original','position']};tri={m:json.loads((root/f'mesh_{case}_{m}/metrics.json').read_text()) for m in metrics}
base=json.loads((asset['run']/'joint_evaluation/joint_errors.json').read_text());prov=json.loads((asset['run']/'input_provenance.json').read_text());scene=pathlib.Path(prov['source_data_root'])
R,s,t=np.array(base['R_export_to_gtworld']),base['scale'],np.array(base['translation']);D=base['GT_static_diagonal']
parts={part['node']['id']:part for part in asset['parts']};base_id=next(m['pact_node_id'] for m in prov['part_correspondence'] if m['gt_mesh_index']==0)
nodes={m:{n['id']:n for n in candidate_nodes(asset,None if m=='original' else root/f'{case}_position/candidate.json')} for m in metrics}
fig,axes=plt.subplots(1,3,figsize=(17,4.5));colors={'original':'#1872bb','position':'#ad2b9b'}
for method in metrics:
    a=metrics[method];axes[0].plot([r['t'] for r in a['states']],[r['mean_pair_proxy'] for r in a['states']],'.-',ms=3,c=colors[method],label=method)
    b=tri[method];axes[1].plot([r['t'] for r in b['states']],[sum(p['line_length_over_diagonal'] for p in r['pairs']) for r in b['states']],'.-',c=colors[method],label=method)
    joints=a['GT']['joints'];xs=np.arange(len(joints));axes[2].bar(xs+(-.18 if method=='original' else .18),[r['pivot_to_GT_axis_percent_static_diagonal'] for r in joints],.36,color=colors[method],label=method)
axes[0].set(title='41-state Gaussian overlap proxy',xlabel='Native motion progress',ylabel='Nearest-center kernel')
axes[1].set(title='21-state full-triangle intersections',xlabel='Native motion progress',ylabel='Intersection length / asset diagonal')
axes[2].set(title='GT pivot error (fixed global frame)',xlabel='GT moving part',ylabel='% GT static diagonal',xticks=np.arange(len(joints)),xticklabels=[str(r['gt_mobility_id']) for r in joints])
for ax in axes:ax.legend();ax.grid(alpha=.2)
fig.suptitle(f'{case}: pivot-only, frozen geometry/direction/range; native motion <10deg is a limitation');fig.tight_layout();fig.savefig(out/'pivot_and_collision.png',dpi=180);fig.savefig(out/'pivot_and_collision.pdf');plt.close(fig)
fig,axes=plt.subplots(len(base['joints']),2,figsize=(13,6*len(base['joints'])),squeeze=False);metadata=[]
for row,joint in enumerate(base['joints']):
    pid=joint['predicted_node_id'];gi=joint['gt_mesh_index'];gd=np.array(joint['GT_direction_world']);gd/=np.linalg.norm(gd);go=np.array(joint['GT_origin_world'])
    gtpart=trimesh.load(scene/f'gt/part_{gi}.ply',process=False,force='mesh');gtbody=trimesh.load(scene/'gt/part_0.ply',process=False,force='mesh')
    plane_origin=go+gd*np.dot(gtpart.bounds.mean(0)-go,gd);reference=np.eye(3)[np.argmin(abs(gd))];u=reference-gd*np.dot(reference,gd);u/=np.linalg.norm(u);v=np.cross(gd,u);basis=np.stack([u,v],axis=1)
    sections=[]
    for label,mesh,color,style in [('GT body',gtbody,'#444444','--'),('GT moving',gtpart,'#23934a','--')]:
        segments=trimesh.intersections.mesh_plane(mesh,gd,plane_origin);sections.append((label,(segments-plane_origin)@basis,color,style))
    for label,part in [('PAct body',parts[base_id]),('PAct moving (shared)',parts[pid])]:
        mesh=trimesh.Trimesh(s*(part['mesh'].vertices@R.T)+t,part['mesh'].faces,process=False);segments=trimesh.intersections.mesh_plane(mesh,gd,plane_origin);sections.append((label,(segments-plane_origin)@basis,'#b0b6be','-'))
    markers={'GT':np.zeros(2)}
    for method in nodes:
        j=nodes[method][pid]['joint'];d=R@np.array(j['axis']['direction']);origin=s*(R@np.array(j['axis']['origin']))+t
        assert abs(d@gd)>.99
        on_plane=origin+d*((plane_origin-origin)@gd)/(d@gd);markers[method]=(on_plane-plane_origin)@basis
    points=np.concatenate([seg.reshape(-1,2) for _,seg,_,_ in sections if len(seg)]);center=(points.min(0)+points.max(0))/2;rad=.56*np.ptp(points,axis=0).max()
    for column,zoom in enumerate([False,True]):
        ax=axes[row,column]
        for label,segments,color,style in sections:
            if len(segments):ax.add_collection(LineCollection(segments,colors=color,linestyles=style,linewidths=1,alpha=.8))
        ax.scatter(*markers['GT'],c='red',s=65,marker='x',label='GT axis',zorder=8);ax.scatter(*markers['original'],c=colors['original'],s=55,marker='o',label='Original axis',zorder=8);ax.scatter(*markers['position'],c=colors['position'],s=85,marker='+',label='Refined axis',zorder=9)
        ax.annotate('',xy=markers['position'],xytext=markers['original'],arrowprops={'arrowstyle':'->','color':colors['position'],'lw':1.3})
        cc=np.zeros(2) if zoom else center;rr=.10*D if zoom else rad
        ax.set(xlim=(cc[0]-rr,cc[0]+rr),ylim=(cc[1]-rr,cc[1]+rr),xlabel='Plane u (GT units)',ylabel='Plane v (GT units)',title=f'GT part{gi}: '+('hinge zoom' if zoom else 'same hinge-normal section'));ax.set_aspect('equal');ax.grid(alpha=.2);ax.legend(fontsize=8)
    metadata.append({'GT_part':gi,'predicted_node':pid,'plane_origin_GT_world':plane_origin.tolist(),'plane_normal':gd.tolist(),'basis':basis.tolist(),'axis_intersections_plane_coordinates':{k:v.tolist() for k,v in markers.items()}})
fig.suptitle(f'{case}: actual GT/PAct mesh cuts and axis positions; geometry is shared before/after\nCommon primary alignment; no per-part fitting; open contours preserved');fig.tight_layout(rect=(0,0,1,.95));fig.savefig(out/'GT_pivot_sections.png',dpi=180);fig.savefig(out/'GT_pivot_sections.pdf');plt.close(fig)
(out/'figure_info.json').write_text(json.dumps({'case':case,'hinge_planes':metadata,'candidate_sha256':sha256(root/f'{case}_position/candidate.json'),'geometry_frozen':True,'GT_used_for_plotting_only':True},indent=2))
print(out,flush=True)
