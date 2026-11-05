"""RGB/silhouette-only camera registration for newly generated Origin assets."""
import argparse
import json
import pathlib
import sys

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData

REPO=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'repro/tools'))
from auto_part_semantics import sha256,write_json
from collision_refinement_common import load_asset
from multicase_closed_common import Model,R_WORLD
from extract_motion_extrema import R_GLB


def read(path):return json.loads(pathlib.Path(path).read_text())


def prepare(batch,case):
    root=batch/'registration'/case
    if (root/'config.json').exists():return
    root.mkdir(parents=True)
    generated=read(batch/'origin_runs'/f'{case}.json')
    asset=load_asset(generated['run'])
    first=read(batch/'extremes'/case/'manifest.json')['endpoints'][0]
    image=Image.open(first['rgb']).convert('RGBA').resize((256,256),Image.Resampling.LANCZOS)
    rgba=np.asarray(image).astype(np.float32)/255
    alpha=rgba[:,:,3];rgb=rgba[:,:,:3]*alpha[:,:,None]+1-alpha[:,:,None]
    groups=[];arrays={};sources={};geometry=[]
    for part in asset['parts']:
        native=part['node'];nid=native['id'];gid=10+nid;mesh=part['mesh']
        path=asset['object_path'].parent/native['gs'][0]
        count=len(PlyData.read(path)['vertex'].data)
        prefix=f'g{gid}'
        arrays[prefix+'_vertices']=np.arange(len(mesh.vertices),dtype=np.int64)
        arrays[prefix+'_faces']=np.arange(len(mesh.faces),dtype=np.int64)
        arrays[prefix+'_gaussians']=np.arange(count,dtype=np.int64)
        sources[str(nid)]=dict(mesh=str(part['mesh_path']),gs=str(path))
        geometry.extend([dict(path=str(p),sha256=sha256(p)) for p in [part['mesh_path'],path]])
        groups.append(dict(id=gid,name=native['name'],source_node=nid,kind='fixed',parent=-1,
            membership_prefix=prefix,shape_owner=None,box=None,observed_q=0.,
            deformation_basis=np.eye(3).tolist(),deformation_anchor=np.asarray(mesh.vertices).mean(0).tolist(),length=1.,
            joint=dict(type='fixed',range=[0.,0.],axis=dict(origin=[0.,0.,0.],direction=[0.,0.,0.]))))
    np.savez_compressed(root/'membership.npz',**arrays)
    manifest=dict(source_run=str(asset['run']),source_object=str(asset['object_path']),sources=sources,
                  groups=groups,pairs=[],geometry_files=geometry,
                  selection='All native Origin parts, identity articulation; RGB/silhouette registration only')
    write_json(root/'motion_groups.json',manifest)
    np.savez_compressed(root/'observation.npz',rgb=rgb,alpha=alpha,labels=np.zeros((256,256),np.int32),
        semantic=np.ones((256,256,3),np.float32),extrinsics=np.array(first['extrinsics']),
        intrinsics=np.array(first['intrinsics']))
    config=dict(case=case,source_image=first['rgb'],source_scene=read(batch/'extremes'/case/'manifest.json')['source_scene'],
        source_camera='native selected endpoint camera',reference_frame=first['frame'],
        world_R=(R_GLB if case=='20411' else R_WORLD).tolist(),resolution=256,
        palette={str(g['id']):[.5,.5,.5] for g in groups},
        prior_root=str(root),prior_manifest_sha256=sha256(root/'motion_groups.json'),
        observations_sha256=sha256(root/'observation.npz'))
    write_json(root/'config.json',config)
    print('REGISTER PREPARED',case,flush=True)


def fit(batch,case):
    root=batch/'registration'/case
    if (root/'registration.pt').exists():return
    torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42)
    model=Model(read(root/'config.json'),read(root/'motion_groups.json'),root)
    model.log_shape.requires_grad_(False);model.vertical_shift.requires_grad_(False)
    opt=torch.optim.Adam([{'params':[model.log_global_scale],'lr':.008},
                          {'params':[model.screen_translation],'lr':.006}])
    history=[]
    for step in range(100):
        opt.zero_grad(set_to_none=True);loss,values=model.losses(align=True)
        assert torch.isfinite(loss);loss.backward();opt.step();model.project()
        history.append(dict(step=step+1,loss=float(loss.detach()),**values))
    torch.save({name:getattr(model,name).detach().cpu() for name in ['log_global_scale','screen_translation']},root/'registration.pt')
    with torch.no_grad():
        rgb=model.render().clamp(0,1).permute(1,2,0).cpu().numpy()
    Image.fromarray((rgb*255).astype(np.uint8)).save(root/'registered_native_source.png')
    write_json(root/'registration_results.json',dict(case=case,history=history,
        scale=float(model.log_global_scale.exp()),translation=model.screen_translation.detach().cpu().tolist(),
        registration_sha256=sha256(root/'registration.pt'),
        policy='100-step Adam RGB+2silhouette on newly generated Origin q0; no GT numeric joint, manual class, or second endpoint in camera fit'))
    print('REGISTERED',case,round(history[-1]['loss'],4),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True)
    p.add_argument('--case');p.add_argument('--stage',choices=['prepare','fit','all'],default='all')
    a=p.parse_args();batch=a.root.resolve()
    cases=[a.case] if a.case else read(REPO/'repro/outputs/closed_multicase_20261028_v1/protocol.json')['cases']
    if a.stage in ('prepare','all'):
        for case in cases:prepare(batch,case)
    if a.stage in ('fit','all'):
        for case in cases:fit(batch,case)


if __name__=='__main__':main()
