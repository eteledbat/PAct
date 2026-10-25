"""Observed-pose mesh-plane cuts in the SAME world planes, with GT axes."""
import argparse
import json
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import trimesh


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--experiment');a=p.parse_args();root=pathlib.Path(a.root);cfg=json.loads((root/'config.json').read_text())
    for experiment in cfg['experiments']:
        if a.experiment and experiment['name']!=a.experiment:continue
        name,case=experiment['name'],experiment['case'];results=json.loads((root/f'results_{name}.json').read_text());manifest=json.loads((root/'data'/case/'manifest.json').read_text());gtroot=pathlib.Path(manifest['source_data_root'])/'gt'
        joint=next((j for j in results['R0']['joints'] if j['GT_type']=='revolute'),None)
        if joint is None:continue
        gi=joint['GT_part'];target=[trimesh.load(gtroot/f'part_{k}.ply',process=False,force='mesh') for k in [0,gi]]
        meshes={m:[trimesh.load(root/f'eval_{name}_{m}/observed_world/part_{k}.ply',process=False,force='mesh') for k in [0,gi]] for m in ['R0','A','B']}
        zlo,zhi=target[1].bounds[:,2];heights=zlo+(zhi-zlo)*np.array([.25,.5,.75]);fig,axes=plt.subplots(1,3,figsize=(15,5));records=[]
        bounds=np.array([m.bounds[:,:2] for m in target]);lo=bounds[:,0].min(0);hi=bounds[:,1].max(0);pad=.12*np.linalg.norm(hi-lo)
        for ax,z in zip(axes,heights):
            for label,items,color,style in [('GT',target,'#222222','--'),('R0',meshes['R0'],'#999999','-'),('A',meshes['A'],'#2166ac','-'),('B',meshes['B'],'#b2182b','-')]:
                for index,mesh in enumerate(items):
                    segments=trimesh.intersections.mesh_plane(mesh,[0,0,1],[0,0,float(z)])
                    if len(segments):
                        assert np.max(np.abs(segments[:,:,2]-z))<1e-5
                        ax.add_collection(LineCollection(segments[:,:,:2],colors=color,linestyles=style,linewidths=.9 if index else .5,label=label if index else None))
                    records.append({'label':label,'part':index,'world_z':float(z),'segments':len(segments)})
            for method,color,marker in [('R0','#999999','o'),('A','#2166ac','+'),('B','#b2182b','x')]:
                j=next(j for j in results[method]['joints'] if j['GT_part']==gi);o=np.array(j['predicted_axis_origin_world']);d=np.array(j['predicted_axis_direction_world'])
                if abs(d[2])>1e-8:
                    q=o+(z-o[2])/d[2]*d;ax.scatter(q[0],q[1],color=color,marker=marker,s=70,label=method+' axis')
            o=np.array(joint['GT_axis_origin_world']);d=np.array(joint['GT_axis_direction_world'])
            if abs(d[2])>1e-8:q=o+(z-o[2])/d[2]*d;ax.scatter(q[0],q[1],color='red',marker='*',s=110,label='GT axis')
            ax.set(xlim=(lo[0]-pad,hi[0]+pad),ylim=(lo[1]-pad,hi[1]+pad),xlabel='World x',ylabel='World y',title=f'Observed pose z={z:.3f}');ax.set_aspect('equal');ax.grid(alpha=.2)
        handles,labels=axes[0].get_legend_handles_labels();bylabel=dict(zip(labels,handles));fig.legend(bylabel.values(),bylabel.keys(),loc='lower center',ncol=8,fontsize=8);fig.suptitle(name+f' GT part {gi}: actual mesh cuts, fixed camera-derived alignment; no per-part GT fitting');fig.tight_layout(rect=(0,.08,1,.94));out=root/f'figures_{name}';fig.savefig(out/'GT_sections.png',dpi=180);fig.savefig(out/'GT_sections.pdf');plt.close(fig);(out/'GT_sections.json').write_text(json.dumps(records,indent=2))


if __name__=='__main__':main()
