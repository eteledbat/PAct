"""Scientific curves, frozen geometry and GT direction overlays after sealing."""
import argparse,json,pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from collision_refinement_common import load_asset,candidate_nodes,surface
from evaluate_freeart_partnet_case import gt_parts

p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
root=pathlib.Path(a.root);config=json.loads((root/'experiment.json').read_text());assert len(config['cases'])==1
name,run=next(iter(config['cases'].items()));asset=load_asset(run)
gtreport=json.loads((root/'GT_geometry_and_motion.json').read_text());base=gtreport['alignment']
methods=['original',*config['variants']];colors={'original':'#2166ac','direction':'#8e44ad'}
out=root/'figures';out.mkdir(exist_ok=False)
gs={m:json.loads((root/f'eval_{name}_{m}/metrics.json').read_text()) for m in methods}
tri={m:json.loads((root/f'mesh_{name}_{m}/metrics.json').read_text()) for m in methods}
fig,axes=plt.subplots(1,3,figsize=(17,4.5))
for method in methods:
    axes[0].plot([r['t'] for r in gs[method]['states']],[r['mean_pair_proxy'] for r in gs[method]['states']],'.-',ms=3,label=method,c=colors[method])
    rows=tri[method]['states']
    axes[1].plot([r['t'] for r in rows],[sum(p['line_length_over_diagonal'] for p in r['pairs']) for r in rows],'.-',label=method,c=colors[method])
    axes[1].plot([r['t'] for r in rows],[sum(p['transverse_line_length_over_diagonal'] for p in r['pairs']) for r in rows],'--',lw=1,c=colors[method])
    axes[2].plot([r['t'] for r in gs[method]['states']],[min(p['min_heldout_source_to_full_target_center_distance'] for p in r['pairs'])/asset['diagonal'] for r in gs[method]['states']],'.-',ms=3,label=method,c=colors[method])
for ax,title,ylabel in zip(axes,['Gaussian overlap proxy (41 states)','Full-triangle intersections (21 states)','Held-out center separation'],['Opacity-weighted nearest-center kernel','Intersection line length / asset diagonal','Minimum sampled distance / asset diagonal']):
    ax.set(xlabel='Predicted motion progress t',ylabel=ylabel,title=title);ax.grid(alpha=.25);ax.legend()
axes[1].text(.02,.96,'solid: all intersections\ndashed: transverse (>5 deg)',transform=axes[1].transAxes,va='top',fontsize=8)
fig.suptitle(f'{name}: frozen geometry/type/range; proxies are NOT solid overlap volume');fig.tight_layout();fig.savefig(out/'collision_curves.png',dpi=180);plt.close(fig)
history=json.loads((root/f'{name}_direction/history.json').read_text())
fig,ax=plt.subplots(figsize=(7,4));ax.plot([r['step'] for r in history],[r['gaussian_proxy'] for r in history]);ax.set(xlabel='Optimization step',ylabel='Training Gaussian overlap proxy',title='Fixed final step200; no GT selection');ax.grid(alpha=.25);fig.tight_layout();fig.savefig(out/'training_curve.png',dpi=160);plt.close(fig)
provenance=json.loads((asset['run']/'input_provenance.json').read_text());gt,delta=gt_parts(pathlib.Path(provenance['source_data_root']))
R,s,t=np.array(base['R_export_to_gtworld']),base['scale'],np.array(base['translation']);D=base['GT_static_diagonal']
mapping={v['gt_mesh_index']:v['pact_node_id'] for v in provenance['part_correspondence']}
nodes={m:{n['id']:n for n in candidate_nodes(asset,None if m=='original' else root/f'{name}_{m}/candidate.json')} for m in methods}
pred={p['node']['id']:p['mesh'] for p in asset['parts']}
center=gt[0][0].bounds.mean(0);rad=.60*D
sets=[(surface(gt[0][0],7000,1),'grey','GT fixed'),(surface(gt[0][1],5000,2),'green','GT closed drawer'),
      (s*(surface(pred[mapping[0]],7000,3)@R.T)+t,'lightblue','PAct fixed'),(s*(surface(pred[mapping[1]],5000,4)@R.T)+t,'orange','PAct rest drawer')]
fig=plt.figure(figsize=(16,5))
for i,azim in enumerate([25,125,225]):
    ax=fig.add_subplot(1,3,i+1,projection='3d')
    for points,color,label in sets:ax.scatter(*points.T,s=.28,c=color,alpha=.16,label=label,rasterized=True)
    # A common display anchor is intentional: a prismatic axis has no pivot.
    anchor=gt[0][1].bounds.mean(0)
    for label,d,color in [('GT',delta,'red'),*[(m,R@np.array(nodes[m][mapping[1]]['joint']['axis']['direction']),colors[m]) for m in methods]]:
        d=np.array(d,float);d/=np.linalg.norm(d)
        if d@delta<0:d=-d
        line=anchor+np.array([-.4,.4])[:,None]*D*d
        ax.plot(*line.T,c=color,lw=2,label=label+' direction')
    ax.set(xlim=(center[0]-rad,center[0]+rad),ylim=(center[1]-rad,center[1]+rad),zlim=(center[2]-rad,center[2]+rad),xlabel='GT x',ylabel='y up',zlabel='z front')
    ax.view_init(elev=15,azim=azim,vertical_axis='y');ax.set_box_aspect([1,1,1])
    if i==0:ax.legend(fontsize=6)
fig.suptitle(f'{name}: GT / original / refined direction; shared DISPLAY anchor, prismatic pivot is undefined');fig.tight_layout();fig.savefig(out/'GT_direction_overlay.png',dpi=180);plt.close(fig)
(out/'figure_provenance.json').write_text(json.dumps({'geometry_repair':False,'GT_optimization':False,'axis_anchor':'Shared display anchor only; prismatic pivot is unidentifiable.','figure_sampling':'Deterministic area-weighted samples for geometry scatter; collision curves from full targets / full triangles.'},indent=2))
print(out)
