"""Visualize saved high-proxy regions; no new physical or GT labels."""
import pathlib,json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=pathlib.Path(__file__).resolve().parents[2];run=pathlib.Path((ROOT/'repro/current_run.txt').read_text().strip());out=run/'motion_analysis';rows=[]
for state in [0,10,20]:
 fig=plt.figure(figsize=(13,5))
 for view,azim in enumerate([25,125]):
  ax=fig.add_subplot(1,2,view+1,projection='3d')
  for idx in [0,1]:
   p=np.load(out/f'hotspot_state{state:02d}_part{idx}.npz');x=p['xyz'];q=p['proxy'];keep=q>=.1
   if view==0:
    rows.append({'state':state,'part':idx,'definition':'Saved top0.5%/max3000 nearest-kernel candidates, further filtered q>=0.1; not full hotspot area or physical collision','retained_points':int(keep.sum()),'bbox':np.array([x[keep].min(0),x[keep].max(0)]).tolist() if keep.any() else None,'max_kernel':float(q.max())})
   if keep.any():ax.scatter(*x[keep].T,s=2,c=['#3784be','#e88b31'][idx],label=f'part{idx}')
  ax.set(xlim=(-.65,.65),ylim=(-.6,.6),zlim=(-.6,1.3),xlabel='x',ylabel='y (up)',zlabel='z');ax.set_box_aspect([1,1,1.4]);ax.view_init(elev=15,azim=azim,vertical_axis='y');ax.legend()
 fig.suptitle(f'State {state}/20: high nearest-kernel candidates, NOT verified penetration');fig.tight_layout();fig.savefig(out/f'hotspot_state{state:02d}.png',dpi=150);plt.close(fig)
(out/'hotspot_summary.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
