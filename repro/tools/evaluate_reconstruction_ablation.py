"""Independent image, observed-pose mesh, joint, and motion evaluation after sealing."""
import argparse
import copy
import json
import pathlib
import time

import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
from PIL import Image,ImageDraw
from plyfile import PlyData,PlyElement
from scipy.spatial import cKDTree
from skimage.metrics import structural_similarity

from collision_refinement_common import load_asset,sha256,surface
from reconstruction_ablation_common import Reconstruction,UNDO_GS,R_EXPORT_TO_WORLD


def metric(a,b,D):
    da=cKDTree(b).query(a,workers=4)[0];db=cKDTree(a).query(b,workers=4)[0]
    return {'CD_percent_GT_static_D':float(50*(da.mean()+db.mean())/D),'RMS_percent_GT_static_D':float(100*np.sqrt((np.mean(da**2)+np.mean(db**2))/2)/D)}


def angle(a,b):
    a,b=np.asarray(a),np.asarray(b);return float(np.degrees(np.arccos(np.clip(abs(a@b)/(np.linalg.norm(a)*np.linalg.norm(b)),0,1))))


def export_asset(model,directory,mode):
    directory.mkdir();(directory/'ply').mkdir();(directory/'gs').mkdir();obj=model.joints.exported(model.asset)
    if mode=='signed90_assumption':
        for n in obj['diffuse_tree']:
            if n['joint']['type'] in ['revolute','continuous']:n['joint']['range']=[0.,float(np.sign(n['joint']['range'][1] or 1)*90.)]
    geometry=model.deformed()
    for i,(part,node) in enumerate(zip(model.parts,obj['diffuse_tree'])):
        mesh=model.deformed_mesh(i);mesh.export(directory/'ply'/f"part_{node['id']}.ply")
        v=PlyData.read(part['gs_path'])['vertex'].data.copy();xyz=geometry[i].detach().cpu().numpy()@UNDO_GS
        for k,key in enumerate(['x','y','z']):v[key]=xyz[:,k]
        PlyData([PlyElement.describe(v,'vertex')]).write(directory/'gs'/f"part_{node['id']}.ply")
        node['plys']=[f"ply/part_{node['id']}.ply"];node['gs']=[f"gs/part_{node['id']}.ply"]
        node['aabb']={'min':mesh.bounds[0].tolist(),'max':mesh.bounds[1].tolist(),'center':mesh.bounds.mean(0).tolist(),'size':np.ptp(mesh.bounds,axis=0).tolist()}
    obj['ablation_motion_protocol']=mode;obj['observation_scale']=float(model.log_scale.exp());obj['observation_translation']=model.translation.detach().cpu().tolist();obj['observation_input_q']=model.input_q.detach().cpu().tolist()
    (directory/'object.json').write_text(json.dumps(obj,indent=2));return obj


def image_metrics(model,network):
    result={};pictures=[]
    for split,indices in [('train',range(6)),('heldout',range(6,12))]:
        rows=[]
        for index in indices:
            pred,seg=model.render(index);target=model.data['rgb'][index];pred=pred.clamp(0,1);mask=model.labels[index]>0
            mse=float((pred-target).square().mean());fg_mse=float((pred[:,mask]-target[:,mask]).square().mean());a=pred.permute(1,2,0).cpu().numpy();b=target.permute(1,2,0).cpu().numpy()
            row={'view_index':index,'PSNR':float(-10*np.log10(max(mse,1e-12))),'foreground_PSNR':float(-10*np.log10(max(fg_mse,1e-12))),
                 'SSIM':float(structural_similarity(a,b,data_range=1.,channel_axis=2)),
                 'semantic_palette_MSE':float((seg-model.target_labels[index]).square().mean()),
                 'LPIPS_vgg':float(network(2*pred[None]-1,2*target[None]-1).item()) if network is not None else None}
            rows.append(row)
            if split=='heldout':pictures.append(a)
        result[split]={'views':rows,'mean':{k:float(np.mean([r[k] for r in rows])) if rows[0][k] is not None else None for k in rows[0] if k!='view_index'}}
    return result,pictures


def geometry_joints(model,manifest):
    src=pathlib.Path(manifest['source_data_root']);targets={x['gt_mesh_index']:trimesh.load(src/f"gt/part_{x['gt_mesh_index']}.ply",process=False,force='mesh') for x in manifest['part_correspondence']}
    D=float(np.linalg.norm(np.ptp(targets[0].vertices,axis=0)));world,_=model.joints.world_q(model.input_q[None]);scale=float(model.log_scale.exp());shift=model.translation.cpu().numpy();pred={};rows=[];byid={p['node']['id']:i for i,p in enumerate(model.parts)}
    for mapping in manifest['part_correspondence']:
        gi,pid=mapping['gt_mesh_index'],mapping['pact_node_id'];mesh=model.deformed_mesh(byid[pid]);tr=world[pid][0].cpu().numpy();mesh.vertices=scale*((mesh.vertices@tr[:3,:3].T+tr[:3,3])@R_EXPORT_TO_WORLD.T)+shift;pred[gi]=mesh
        rows.append({'GT_part':gi,'node_id':pid,**metric(surface(mesh,16000,100+gi),surface(targets[gi],16000,200+gi),D),
                     'surface_area_relative_to_original':float(model.deformed_mesh(byid[pid]).area/model.parts[byid[pid]]['mesh'].area)})
    def samples(meshes,seed):
        areas=np.array([m.area for m in meshes]);counts=np.maximum(100,np.round(24000*areas/areas.sum()).astype(int));return np.concatenate([surface(m,int(n),seed+i) for i,(m,n) in enumerate(zip(meshes,counts))])
    groups={}
    for group,ids in [('fixed',[0]),('moving',sorted(k for k in targets if k)),('whole',sorted(targets))]:groups[group]=metric(samples([pred[k] for k in ids],1000),samples([targets[k] for k in ids],2000),D)
    baseline=json.loads((model.asset['run']/'joint_evaluation/joint_errors.json').read_text());nodes=model.joints.exported(model.asset)['diffuse_tree'];nodes_byid={n['id']:n for n in nodes};joints=[]
    for old in baseline['joints']:
        node=nodes_byid[old['predicted_node_id']];joint=node['joint'];parent=world[node.get('parent',-1)][0].cpu().numpy();direction=R_EXPORT_TO_WORLD@parent[:3,:3]@np.array(joint['axis']['direction']);origin=scale*(R_EXPORT_TO_WORLD@(parent[:3,:3]@np.array(joint['axis']['origin'])+parent[:3,3]))+shift;correct=joint['type']==old['GT_type'];gt_dir=np.array(old['GT_direction_world']);gt_origin=np.array(old['GT_origin_world'])
        joints.append({'GT_part':old['gt_mesh_index'],'node_id':node['id'],'GT_type':old['GT_type'],'type':joint['type'],'type_correct':correct,
                       'axis_degrees':angle(direction,gt_dir) if correct else None,
                       'pivot_percent_GT_static_D':float(100*np.linalg.norm(np.cross(origin-gt_origin,gt_dir))/D) if correct and joint['type']=='revolute' else None,
                       'predicted_axis_origin_world':origin.tolist(),'predicted_axis_direction_world':direction.tolist(),'GT_axis_origin_world':gt_origin.tolist(),'GT_axis_direction_world':gt_dir.tolist()})
    return {'parts':rows,'groups':groups,'GT_static_D':D,'alignment':'Shared image-only similarity fixed before A/B; observed poses rendered by each candidate own joint+geometry, no GT alignment or per-part fitting during scoring.'},joints,pred,targets


def motion_metrics(model,mode):
    indices=[]
    for part in model.parts:
        retained=part['retained'].cpu().numpy();training=part['source'].cpu().numpy();remaining=np.setdiff1d(retained,training);rng=np.random.default_rng(2026+int(part['node']['id']));selected=np.sort(rng.choice(remaining,min(4096,len(remaining)),replace=False));assert not np.intersect1d(selected,training).size;indices.append(torch.as_tensor(selected,device='cuda'))
    geometry=model.deformed();rows=[]
    for name,qs in model.motion_q(mode,41):
        values=model.collision(geometry,qs,indices).cpu().numpy();assert np.isfinite(values).all();rows.append({'path':name,'states':values.tolist(),'mean':float(values.mean()),'mean_excluding_initial':float(values[1:].mean()),'max':float(values.max()),'initial':float(values[0]),'interleaved_heldout_mean':float(values[1::2].mean())})
    return {'paths':rows,'mean_across_paths':float(np.mean([x['mean'] for x in rows])),'worst_state':float(max(x['max'] for x in rows)),
            'sampling':'41 states/path; seed2026 up to4096 held-out source centers/part, no optimizer source centers; all retained targets. No sampled state means continuous collision proof.',
            'solid_volume':None,'penetration_depth':None,'solid_status':'unavailable: mesh solid validity not established'}


def mesh_motion(export_dir,D,out):
    from partitioned_triangle_motion import PartitionedTrianglePairs
    asset=load_asset(export_dir);asset['diagonal']=D;checker=PartitionedTrianglePairs(asset,partition_size=12000);rows=[]
    for state,t in enumerate(np.linspace(0,1,21)):
        values,_=checker.state(asset['nodes'],float(t),False);rows.append({'state':state,'t':float(t),'pairs':values,'sum_line_over_fixed_D':float(sum(p['line_length_over_diagonal'] for p in values)),'sum_transverse_line_over_fixed_D':float(sum(p['transverse_line_length_over_diagonal'] for p in values))})
        (out/'triangle_states.json').write_text(json.dumps(rows,indent=2));print('TRIANGLE',out.name,state,rows[-1]['sum_line_over_fixed_D'],flush=True)
    return {'states':rows,'mean_line_over_fixed_D':float(np.mean([s['sum_line_over_fixed_D'] for s in rows])),'mean_transverse_line_over_fixed_D':float(np.mean([s['sum_transverse_line_over_fixed_D'] for s in rows])),
            'protocol':'21 synchronized motion states, all deformed triangles, no decimation/repair. Same INITIAL asset diagonal for A/B/R0 and tiny-segment cutoff. Includes legitimate contact and surface tessellation effects; not volume or depth.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--experiment');a=p.parse_args();root=pathlib.Path(a.root).resolve();cfg=json.loads((root/'config.json').read_text());torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    if a.experiment:
        assert all((root/f'{a.experiment}_{m}_train/seal.json').exists() for m in ['A','B'])
    else:assert json.loads((root/'controller_status.json').read_text())['state']=='optimization_complete'
    network=None;lpips_reason=None
    try:
        assert (pathlib.Path('/root/.cache/torch/hub/checkpoints/vgg16-397923af.pth')).is_file(),'No local VGG16 backbone'
        import lpips
        network=lpips.LPIPS(net='vgg').cuda().eval()
    except Exception as exc:lpips_reason=f'{type(exc).__name__}: {exc}'
    for experiment in cfg['experiments']:
        if a.experiment and experiment['name']!=a.experiment:continue
        case,name=experiment['case'],experiment['name'];manifest=json.loads((root/'data'/case/'manifest.json').read_text());data=np.load(root/'data'/case/'observations.npz');all_rows={};pictures={};models={}
        for method in ['R0','A','B']:
            out=root/f'eval_{name}_{method}';out.mkdir(exist_ok=True)
            if (out/'metrics.json').exists():all_rows[method]=json.loads((out/'metrics.json').read_text());continue
            started=time.monotonic();model=Reconstruction(cfg['cases'][case],cfg,data)
            source=root/f'prealign_{case}' if method=='R0' else root/f'{name}_{method}_train'
            seal=json.loads((source/'seal.json').read_text());assert seal['checkpoint_sha256']==sha256(source/'checkpoint.pt') and not seal['GT_reads_blocked'];saved=torch.load(source/'checkpoint.pt',map_location='cuda',weights_only=False);model.load_state_dict(saved['model']);model.eval().requires_grad_(False)
            with torch.no_grad():
                images,pics=image_metrics(model,network);geometry,joints,pred,gt=geometry_joints(model,manifest);motion=motion_metrics(model,experiment['range'])
                observed=out/'observed_world';observed.mkdir(exist_ok=True)
                for gi,mesh in pred.items():mesh.export(observed/f'part_{gi}.ply')
                export=out/'export';obj=export_asset(model,export,experiment['range']) if not export.exists() else json.loads((export/'object.json').read_text())
                movie=[];qs=dict(model.motion_q(experiment['range'],21))['synchronized']
                for state in range(21):
                    _,seg=model.render(0,qs=qs[state:state+1]);array=(255*seg.clamp(0,1).permute(1,2,0).cpu().numpy()).astype(np.uint8);pic=Image.fromarray(array);ImageDraw.Draw(pic).text((4,4),f'{method} t={state/20:.2f}',fill='black');movie.append(np.array(pic))
                    if state in [0,10,20]:pic.save(out/f'motion_state{state:02d}.png')
                imageio.mimsave(out/'motion.mp4',movie,fps=10)
            for i,pic in enumerate(pics):Image.fromarray((255*pic).astype(np.uint8)).save(out/f'heldout_{i}.png')
            triangles=mesh_motion(export,model.D,out)
            field=torch.stack(list(model.geometry)).detach().norm(dim=-1).cpu().numpy();result={'experiment':experiment,'method':method,'source':str(source),'checkpoint_sha256':seal['checkpoint_sha256'],
                    'images':images,'LPIPS_unavailable_reason':lpips_reason,'geometry':geometry,'joints':joints,'motion':motion,'mesh_motion':triangles,
                    'deformation':{'max_percent_initial_D':float(100*field.max()/model.D),'mean_percent_initial_D':float(100*field.mean()/model.D),'control_points_at_bound':int((field>=.999*model.bound).sum())},
                    'elapsed_seconds':time.monotonic()-started,'sources_unchanged':seal['original_files_unchanged'],
                    'GT_usage':{'oracle_2D_GT_masks_in_optimizer':True,'GT_mesh_used_in_2D_mask_preparation':True,'GT_mesh_read_by_optimizer':False,'GT_joint_values_in_optimizer':False,'GT_3D_loss_targets_in_optimizer':False},'geometry_open_to_optimization':method!='R0'}
            (out/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False));all_rows[method]=result;print('EVALUATED',name,method,flush=True)
            del model;torch.cuda.empty_cache()
        # Actual sampled image comparison, common camera, and independent motion curves.
        directory=root/f'figures_{name}';directory.mkdir(exist_ok=True)
        panels=[]
        for i in [0,2,4]:
            target=(255*data['rgb'][6+i]).astype(np.uint8);images=[target,*[np.array(Image.open(root/f'eval_{name}_{m}/heldout_{i}.png')) for m in ['R0','A','B']]];panels.append(np.concatenate(images,1))
        pic=Image.fromarray(np.concatenate(panels,0));draw=ImageDraw.Draw(pic)
        for col,label in enumerate(['Input held-out camera','R0: original geometry','A: reconstruction only','B: +motion collision']):draw.text((col*cfg['resolution']+4,4),label,fill='black')
        pic.save(directory/'heldout_reconstruction.png')
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for method,color in [('R0','gray'),('A','#2166ac'),('B','#b2182b')]:
            paths=all_rows[method]['motion']['paths'];sync=next(p for p in paths if p['path']=='synchronized');axes[0].plot(np.linspace(0,1,41),sync['states'],label=method,color=color)
            mesh=all_rows[method]['mesh_motion']['states'];axes[1].plot([s['t'] for s in mesh],[s['sum_line_over_fixed_D'] for s in mesh],label=method,color=color)
        axes[0].set_ylabel('Gaussian proxy (not volume)');axes[1].set_ylabel('Mesh surface intersection length / fixed D')
        for ax in axes:ax.set_xlabel('Fraction of declared motion range');ax.legend();ax.grid(alpha=.25)
        fig.suptitle(name+' | synchronized path; individual paths retained in JSON');fig.tight_layout();fig.savefig(directory/'motion_comparison.png',dpi=180);fig.savefig(directory/'motion_comparison.pdf');plt.close(fig)
        readers=[imageio.get_reader(root/f'eval_{name}_{m}/motion.mp4') for m in ['R0','A','B']];frames=[np.concatenate([r.get_data(i) for r in readers],1) for i in range(21)];imageio.mimsave(directory/'R0_A_B_motion.mp4',frames,fps=10)
        for r in readers:r.close()
        (root/f'results_{name}.json').write_text(json.dumps(all_rows,indent=2));print('EXPERIMENT EVALUATED',name,flush=True)


if __name__=='__main__':main()
