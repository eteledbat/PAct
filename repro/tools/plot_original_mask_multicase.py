"""Standalone figures for the sealed batch comparison; no model operations."""
import argparse,csv,json,pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=pathlib.Path(a.root);rows=list(csv.DictReader((root/'old_new_comparison.csv').open()));names=[r['case'] for r in rows]
fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
fields=[('old_shared_input_IoU','new_shared_input_IoU',100,'Original-input silhouette IoU (%)'),('old_shared_heldout_IoU','new_shared_heldout_IoU',100,'Held-out silhouette IoU (%) | same frozen registration'),('old_CD_percent_D','new_CD_percent_D',1,'Surface Chamfer (% static GT diagonal; lower is better)'),('old_axis_correct','new_axis_correct',1,'Complete correct independent axes (counts)')]
x=np.arange(len(rows));width=.37
for ax,(old,new,scale,label) in zip(axes.flat,fields):
 oa=np.array([float(r[old])*scale if r[old] else np.nan for r in rows]);nb=np.array([float(r[new])*scale if r[new] else np.nan for r in rows]);ax.bar(x-width/2,oa,width,label='Old closed-image fit',color='#8d96a6');ax.bar(x+width/2,nb,width,label='New original-mask fit',color='#2475b0');ax.set_xticks(x,names,rotation=40,ha='right');ax.set_title(label,fontsize=10);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
 if old=='old_axis_correct':
  ax.plot(x,[int(r['GT_joints']) for r in rows],'.',color='black',label='GT joint count');ax.set_yticks(range(5))
axes[0,0].legend(fontsize=8);axes[1,1].legend(fontsize=8);fig.suptitle('Original-mask semantic refinement vs previous closed-image refinement',fontsize=13);fig.savefig(root/'old_new_metrics.png',dpi=160);fig.savefig(root/'old_new_metrics.pdf');plt.close(fig)
print('Saved old_new_metrics.png/.pdf')
