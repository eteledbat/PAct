"""Exact mesh/plane sections with shared GT planes, poses and existing alignment.

Open contours stay open. No occupancy filling, geometry repair, re-registration
or GT-guided candidate selection. Endpoint-to-plane/triangle-edge checks validate
that plotted lines are actual surface intersections in the recorded GT frame.
"""
import argparse,datetime,json,pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np
import trimesh
from collision_refinement_common import load_asset,candidate_nodes,moved,transforms,sha256
from evaluate_freeart_partnet_case import gt_parts

p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args()
root=pathlib.Path(a.root).resolve();out=pathlib.Path(a.out).resolve();out.mkdir(exist_ok=False)
config=json.loads((root/'experiment.json').read_text());name,run=next(iter(config['cases'].items()));assert name=='20411'
asset=load_asset(run);provenance=json.loads((asset['run']/'input_provenance.json').read_text())
source=pathlib.Path(provenance['source_data_root']);gt,delta=gt_parts(source)
report=json.loads((root/'GT_geometry_and_motion.json').read_text());base=report['alignment']
R,s,translation=np.array(base['R_export_to_gtworld']),base['scale'],np.array(base['translation']);D=base['GT_static_diagonal']
reg=report['static_ICP_sensitivity'];rr,tt=np.array(reg['rotation']),np.array(reg['translation'])
methods=['GT','original','direction'];times=[0.,.5,1.];nodes={m:candidate_nodes(asset,None if m=='original' else root/f'{name}_direction/candidate.json') for m in methods[1:]}
mapping={r['gt_mesh_index']:r['pact_node_id'] for r in provenance['part_correspondence']};byid={p['node']['id']:p for p in asset['parts']}
lo,hi=gt[0][1].bounds;center=(lo+hi)/2
planes=[{'name':'YZ_'+label,'normal_axis':0,'value':float(center[0]+f*.5*(hi[0]-lo[0])),'plot_axes':[2,1],
         'label':label+' sagittal'} for label,f in [('left',-.95),('center',0.),('right',.95)]]
planes += [{'name':'XZ_'+label,'normal_axis':1,'value':float(lo[1]+f*(hi[1]-lo[1])),'plot_axes':[0,2],
            'label':label+' horizontal'} for label,f in [('lower',.15),('middle',.5),('upper',.95)]]
meshes={};all_moving=[]
for alignment in ['fixed','static_icp']:
    for method in methods:
        for state,t in enumerate(times):
            for part in [0,1]:
                if method=='GT':
                    m=gt[0][part];vertices=m.vertices+(t*delta if part else 0)
                else:
                    m=byid[mapping[part]]['mesh'];vertices=s*(moved(m.vertices,transforms(nodes[method],t)[mapping[part]])@R.T)+translation
                    if alignment=='static_icp':vertices=vertices@rr.T+tt
                mesh=trimesh.Trimesh(vertices,m.faces,process=False)
                meshes[alignment,method,state,part]=mesh
                if part:all_moving.append(vertices)
cloud=np.concatenate(all_moving);padding=.045*D
world_limits=np.array([cloud.min(0)-padding,cloud.max(0)+padding])
cache={};arrays={};checks=[]
for alignment in ['fixed','static_icp']:
    for method in methods:
        for state,t in enumerate(times):
            for plane in planes:
                axis=plane['normal_axis'];normal=np.eye(3)[axis];origin=normal*plane['value']
                for part in [0,1]:
                    key=(alignment,method,state,plane['name'],part)
                    mesh=meshes[alignment,method,state,part]
                    if method=='direction' and part==0:
                        segments=cache[alignment,'original',state,plane['name'],part]
                    else:
                        segments,faces=trimesh.intersections.mesh_plane(mesh,normal,origin,return_faces=True)
                        if len(segments):
                            assert np.isfinite(segments).all()
                            plane_error=float(abs(segments[:,:,axis]-plane['value']).max());edge_error=0.
                            tri=mesh.triangles[faces]
                            for endpoint in [0,1]:
                                # Plane-cut endpoints must lie on an original edge.
                                # Segment projection remains stable for the GT's very
                                # thin/degenerate faces, unlike barycentric closest-point.
                                errors=[]
                                for v0,v1 in [(0,1),(1,2),(2,0)]:
                                    edge=tri[:,v1]-tri[:,v0];den=(edge*edge).sum(1)
                                    parameter=np.divide(((segments[:,endpoint]-tri[:,v0])*edge).sum(1),den,out=np.zeros_like(den),where=den>0)
                                    closest=tri[:,v0]+np.clip(parameter,0,1)[:,None]*edge
                                    errors.append(np.linalg.norm(closest-segments[:,endpoint],axis=1))
                                distances=np.min(errors,axis=0);assert np.isfinite(distances).all()
                                edge_error=max(edge_error,float(distances.max()))
                            assert plane_error<1e-7*D and edge_error<1e-7*D,(key,plane_error,edge_error)
                        else:plane_error=edge_error=0.
                        checks.append({'alignment':alignment,'method':method,'state':t,'plane':plane['name'],'part':part,'segments':len(segments),
                                       'max_endpoint_plane_error':plane_error,'max_endpoint_original_triangle_edge_error':edge_error})
                    cache[key]=segments
                    arrays[f'{alignment}__{method}__state{state}__{plane["name"]}__part{part}']=segments
            print('SECTIONED',alignment,method,t,flush=True)
for alignment in ['fixed','static_icp']:
    for plane in planes:
        for part in [0,1]:assert np.array_equal(cache[alignment,'original',0,plane['name'],part],cache[alignment,'direction',0,plane['name'],part])
styles={('GT',0):('#424242','--',1.15),('GT',1):('#23934a','--',1.3),('original',0):('#adb5bd','-',1.0),('original',1):('#1872bb','-',1.0),('direction',1):('#ad2b9b','-',1.0)}
legend=[Line2D([0],[0],color=c,ls=ls,lw=lw,label=label) for label,(c,ls,lw) in zip(['GT body','GT drawer','PAct body (shared)','PAct drawer','Refined drawer'],styles.values())]
labels=['x lateral','y up','z front']

def panel(ax,alignment,state,plane,selected):
    dims=plane['plot_axes']
    # Draw prediction first, then GT dashes so overlap is still legible.
    for method,part in [('original',0),('original',1),('direction',1),('GT',0),('GT',1)]:
        if method not in selected:continue
        color,ls,lw=styles[method,part]
        seg=cache[alignment,method,state,plane['name'],part]
        if len(seg):ax.add_collection(LineCollection(seg[:,:,dims],colors=color,linestyles=ls,linewidths=lw,alpha=.90))
    if selected==['direction']:
        seg=cache[alignment,'direction',state,plane['name'],0]
        if len(seg):ax.add_collection(LineCollection(seg[:,:,dims],colors='#adb5bd',linewidths=1))
    ax.set(xlim=world_limits[:,dims[0]],ylim=world_limits[:,dims[1]],xlabel=labels[dims[0]],ylabel=labels[dims[1]])
    ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.18);ax.tick_params(labelsize=8)

figures=[]
for alignment in ['fixed','static_icp']:
    for view,chosen in [('YZ_center',[planes[1]]),('YZ_walls',planes[:3]),('XZ_heights',planes[3:])]:
        fig,axes=plt.subplots(len(chosen),3,figsize=(15,4.5*len(chosen)),squeeze=False)
        for row,plane in enumerate(chosen):
            for state,t in enumerate(times):
                ax=axes[row,state];panel(ax,alignment,state,plane,methods)
                ax.set_title(f'{plane["label"]}: {labels[plane["normal_axis"]]}={plane["value"]:.5f} | t={t:.1f}')
        fig.legend(handles=legend,loc='lower center',ncol=5,fontsize=9)
        fig.suptitle(f'20411 exact mesh-plane sections | {alignment} alignment | closed / half / open\nGT dashed; shared world planes and axes; open contours preserved',fontsize=12)
        fig.tight_layout(rect=(0,.055,1,.94));stem=f'{alignment}_{view}_overlay'
        fig.savefig(out/(stem+'.png'),dpi=180);fig.savefig(out/(stem+'.pdf'));plt.close(fig);figures.append(stem+'.png')
    fig,axes=plt.subplots(3,3,figsize=(15,12))
    for row,method in enumerate(methods):
        for state,t in enumerate(times):
            panel(axes[row,state],alignment,state,planes[1],[method]);axes[row,state].set_title(f'{method} | central YZ, t={t:.1f}')
    fig.suptitle(f'20411 central section: GT / original / refined | {alignment}\nAll panels share the SAME plane and axis limits; surfaces are not filled',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.95));stem=f'{alignment}_YZ_center_separate';fig.savefig(out/(stem+'.png'),dpi=180);fig.savefig(out/(stem+'.pdf'));plt.close(fig);figures.append(stem+'.png')
np.savez_compressed(out/'raw_world_intersection_segments.npz',**arrays)
assert all(sha256(p['mesh_path'])==p['mesh_sha256'] for p in asset['parts'])
manifest={'case':name,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'root':str(root),'source_sha256':sha256(__file__),
          'source_mesh_SHA_unchanged':True,'candidate_sha256':sha256(root/f'{name}_direction/candidate.json'),
          'GT_files_sha256':base['GT_files_sha256'],'planes':planes,'states':times,'world_limits':world_limits.tolist(),'figures':figures,
          'pose_convention':'GT00 closed to GT05 open; t=.5 uses linear translation verified against all six original GT meshes. Predictions retain original ranges.',
          'alignment':'Reuse stored fixed canonical rotation/static-AABB scale/translation. static_icp is a separately labeled sensitivity using exactly the existing global static registration, no new fitting.',
          'GT_geometry_status':'Open/nonmanifold surfaces: contours may be open; no hole repair, occupancy fill or collision-volume claim.',
          'plane_rule':'GT closed drawer x-center and +/-95% half-width; y=15%,50%,95% of GT closed drawer height; unchanged across methods, poses and alignments.',
          'validation':{'checks':checks,'original_refined_closed_sections_bitwise_equal':True}}
(out/'sections_manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False))
(root/'current_sections.txt').write_text(str(out)+'\n')
print('COMPLETE',out,flush=True)
