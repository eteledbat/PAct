"""Reassign rigid motion groups without modifying any source mesh/Gaussian bytes.

Case-specific priors: two separate leaves, outer vertical hinges, two parallel
drawer rails, and a contained drawer insert attached to its enclosing drawer.
No GT joints, GT meshes, or video tracking are loaded for this trial.
"""
import argparse
import copy
import json
import pathlib
import sys
import time

import numpy as np
from plyfile import PlyData
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from collision_refinement_common import load_asset, moved, sha256, transforms

UNDO = np.array([[1,0,0],[0,0,1],[0,-1,0]],float)


def xyz_from_gs(path):
    v = PlyData.read(path)['vertex'].data
    return np.column_stack([v[k] for k in ['x','y','z']]).astype(np.float64) @ UNDO.T


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def build(run,root):
    root.mkdir(parents=True,exist_ok=False)
    asset = load_asset(run)
    parts = {p['node']['id']:p for p in asset['parts']}
    assert set(parts)=={0,1,2,3,4}
    mesh = parts[0]['mesh']
    edges = mesh.edges_unique
    _, labels = connected_components(coo_matrix((np.ones(len(edges)),(edges[:,0],edges[:,1])),
                                     shape=(len(mesh.vertices),len(mesh.vertices))),directed=False)
    counts = np.bincount(labels)
    major = np.flatnonzero(counts > .2*len(mesh.vertices))
    assert len(major)==2 and len(counts)==2,'This case needs exactly two disconnected door leaves'
    major = sorted(major,key=lambda i:float(mesh.vertices[labels==i,0].mean()))
    assert (mesh.vertices[labels==major[0],0]<0).all()
    assert (mesh.vertices[labels==major[1],0]>0).all()
    face_labels = labels[mesh.faces]
    assert (face_labels==face_labels[:,:1]).all()
    paths = {}
    for part in asset['parts']:
        n = part['node']
        paths[n['id']] = {'mesh':str(part['mesh_path']),
                          'gs':str(asset['object_path'].parent/n['gs'][0])}
    frozen = [{'path':path,'sha256':sha256(path)} for item in paths.values() for path in item.values()]
    all_xyz = {nid:xyz_from_gs(item['gs']) for nid,item in paths.items()}
    gs_door_labels = labels[cKDTree(mesh.vertices).query(all_xyz[0],workers=4)[1]]
    arrays,groups,hinges = {},[],[]
    def add(gid,name,source,parent,joint,vertices=None,faces=None,gaussians=None):
        original = parts[source]['mesh']
        vertices = np.arange(len(original.vertices)) if vertices is None else vertices
        faces = np.arange(len(original.faces)) if faces is None else faces
        gaussians = np.arange(len(all_xyz[source])) if gaussians is None else gaussians
        for kind,idx in [('vertices',vertices),('faces',faces),('gaussians',gaussians)]:
            arrays[f'g{gid}_{kind}'] = np.asarray(idx,np.int64)
        groups.append({'id':gid,'name':name,'source_node':source,'parent':parent,'joint':joint,
                       'membership_prefix':f'g{gid}',
                       'counts':{'vertices':len(vertices),'faces':len(faces),'gaussians':len(gaussians)}})
    def joint(typ,direction=(0,0,0),origin=(0,0,0),limits=(0,0)):
        return {'type':typ,'range':list(limits),
                'axis':{'direction':list(direction),'origin':list(origin)}}
    add(10,'cabinet_body',1,-1,joint('fixed'))
    angle = abs(parts[0]['node']['joint']['range'][1]-parts[0]['node']['joint']['range'][0])
    assert angle>1
    base_tree = cKDTree(parts[1]['mesh'].vertices)
    for cid,gid,side in zip(major,[11,12],[-1,1]):
        points = np.asarray(mesh.vertices)[labels==cid]
        horizontal = points[:,[0,2]]
        center = horizontal.mean(0)
        eig,vectors = np.linalg.eigh(np.cov(horizontal.T))
        d = vectors[:,-1]
        d *= np.sign(d[0])
        projection = (horizontal-center)@d
        candidates = []
        for fraction in [.01,.99]:
            xz = center+np.quantile(projection,fraction)*d
            y = np.linspace(np.quantile(points[:,1],.1),np.quantile(points[:,1],.9),31)
            line = np.c_[np.full_like(y,xz[0]),y,np.full_like(y,xz[1])]
            candidates.append({'xz':xz.tolist(),'median_distance_to_body':float(np.median(base_tree.query(line)[0]))})
        # Outer side hinge prior; body distance is a separately reported check.
        chosen = max(candidates,key=lambda c:side*c['xz'][0])
        pivot = [chosen['xz'][0],float(np.median(points[:,1])),chosen['xz'][1]]
        add(gid,'left_door' if side<0 else 'right_door',0,10,
            joint('revolute',(0,1,0),pivot,(0,side*angle)),
            np.flatnonzero(labels==cid),np.flatnonzero(face_labels[:,0]==cid),
            np.flatnonzero(gs_door_labels==cid))
        hinges.append({'group_id':gid,'source_component_id':int(cid),'horizontal_PCA_eigenvalues':eig.tolist(),
                       'edge_candidates':candidates,'selected_pivot':pivot,
                       'selection':'Outer endpoint of horizontal principal direction, 1/99 percentiles; Y is a gauge',
                       'parent_connection_distance':chosen['median_distance_to_body']})
    for nid,gid,name in [(2,13,'left_drawer'),(3,14,'right_drawer')]:
        span = abs(np.diff(parts[nid]['node']['joint']['range'])[0])
        add(gid,name,nid,10,joint('prismatic',(0,0,1),(0,0,0),(0,float(span))))
    bounds = parts[2]['mesh'].bounds
    insert = parts[4]['mesh'].bounds
    assert (insert[0]>=bounds[0]).all() and (insert[1]<=bounds[1]).all()
    add(15,'left_drawer_insert',4,13,joint('fixed'))
    np.savez_compressed(root/'membership.npz',**arrays)
    manifest = {'schema':'frozen_geometry_rigid_motion_groups_v1',
                'source_run':str(asset['run']),'source_object':str(asset['object_path']),
                'source_object_sha256':sha256(asset['object_path']),
                'sources':paths,'geometry_files':frozen,'groups':groups,'hinge_estimation':hinges,
                'membership_file':'membership.npz','membership_sha256':sha256(root/'membership.npz'),
                'world_up':[0,1,0],'gravity_unit':[0,-1,0],
                'prior':'Two independent vertical outer hinges; two independent parallel front-facing sliders; insert rigidly attached to containing left drawer; cabinet fixed',
                'case_specific_structure':'Visual user prior plus two disconnected source door components and one fully contained insert; not claimed to be an automatic semantic classifier',
                'geometry_policy':'Source files immutable. Membership indices only; all source vertices/faces/Gaussians used exactly once. No shape, rest pose, opacity, scale, or color edits.',
                'range_policy':'Hinge magnitudes use original predicted door sweep, opposite signs per outer hinge; each drawer keeps original absolute travel. No GT ranges.',
                'GT_used':False,'method':'Explicit motion prior and regrouping; no neural retraining or physics optimization',
                'tool_sha256':sha256(__file__)}
    write_json(root/'motion_groups.json',manifest)
    checks = verify(root)
    write_json(root/'verification.json',checks)
    print(json.dumps({'root':str(root),'groups':groups,'verification':checks},indent=2),flush=True)


def verify(root):
    manifest = json.loads((root/'motion_groups.json').read_text())
    membership = np.load(root/'membership.npz')
    assert sha256(root/'membership.npz')==manifest['membership_sha256']
    assert all(sha256(x['path'])==x['sha256'] for x in manifest['geometry_files'])
    assert sha256(manifest['source_object'])==manifest['source_object_sha256']
    asset = load_asset(manifest['source_run'])
    groups = manifest['groups']
    byid = {g['id']:g for g in groups}
    coverage = []
    for part in asset['parts']:
        nid = part['node']['id']
        source_groups = [g for g in groups if g['source_node']==nid]
        for kind,total in [('vertices',len(part['mesh'].vertices)),('faces',len(part['mesh'].faces)),
                           ('gaussians',len(xyz_from_gs(manifest['sources'][str(nid)]['gs'])))]:
            indices = np.concatenate([membership[g['membership_prefix']+'_'+kind] for g in source_groups])
            assert np.array_equal(np.sort(indices),np.arange(total)),('Non-bijective coverage',nid,kind)
            coverage.append({'source_node':nid,'kind':kind,'count':total,'used_once':True})
        for g in source_groups:
            verts = membership[g['membership_prefix']+'_vertices']
            faces = part['mesh'].faces[membership[g['membership_prefix']+'_faces']]
            assert np.isin(faces,verts).all(),'Face crosses motion groups'
    identity_error,rigid_error,independence_error,height_error = 0.,0.,0.,0.
    all_points = {}
    for part in asset['parts']:
        all_points[part['node']['id']] = np.asarray(part['mesh'].vertices)
    initial = transforms(groups,0.)
    for g in groups:
        idx = membership[g['membership_prefix']+'_vertices']
        sample = all_points[g['source_node']][idx][::max(1,len(idx)//257)]
        identity_error = max(identity_error,float(np.max(abs(moved(sample,initial[g['id']])-sample))))
        for t in np.linspace(0,1,41):
            tr = transforms(groups,t)[g['id']]
            posed = moved(sample,tr)
            before = np.linalg.norm(sample-sample[0],axis=1)
            after = np.linalg.norm(posed-posed[0],axis=1)
            rigid_error = max(rigid_error,float(np.max(abs(before-after))))
            height_error = max(height_error,float(np.max(abs(posed[:,1]-sample[:,1]))))
    for active in [11,12,13,14]:
        single = copy.deepcopy(groups)
        for g in single:
            if g['id']!=active:g['joint']['range']=[0.,0.]
        tr = transforms(single,1.)
        for gid in [11,12,13,14]:
            if gid!=active:independence_error = max(independence_error,float(np.max(abs(tr[gid]-np.eye(4)))))
        if active==13:assert np.array_equal(tr[13],tr[15]),'Insert fails to follow left drawer'
    assert identity_error==0 and rigid_error<1e-12 and independence_error==0 and height_error==0
    assert byid[11]['joint']['axis']['origin'][0]<0<byid[12]['joint']['axis']['origin'][0]
    assert byid[11]['joint']['axis']['origin']!=byid[12]['joint']['axis']['origin']
    assert [g['joint']['type'] for g in groups].count('revolute')==2
    assert [g['joint']['type'] for g in groups].count('prismatic')==2
    return {'source_bytes_unchanged':True,'coverage':coverage,'rest_pose_max_error':identity_error,
            'rigid_distance_max_error':rigid_error,'independent_joint_max_leak':independence_error,
            'vertical_motion_max_error':height_error,'two_distinct_hinges':True,
            'insert_follows_left_drawer':True,'GT_metrics':None,
            'GT_metric_note':'Old input-mask node correspondence is not valid for these regrouped leaves; no GT accuracy claimed.'}


def render(root):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    from scipy.spatial.transform import Rotation
    import torch
    from render_gravity_motion_trial import Gaussians, quaternion_product, tensor
    repo = pathlib.Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(repo))
    from modules.pact.renderers.gaussian_render import GaussianRenderer
    import utils3d
    manifest = json.loads((root/'motion_groups.json').read_text())
    verify(root)
    membership = np.load(root/'membership.npz')
    asset = load_asset(manifest['source_run'])
    nodes = manifest['groups']
    originals = asset['nodes']
    torch.set_num_threads(4)
    gaussians = []
    source_counts = []
    offset = 0
    for part in asset['parts']:
        nid = part['node']['id']
        path = manifest['sources'][str(nid)]['gs']
        v = PlyData.read(path)['vertex'].data
        xyz = xyz_from_gs(path)
        scales = np.exp(np.column_stack([v[k] for k in ['scale_0','scale_1','scale_2']]).astype(np.float32))
        alpha = (1/(1+np.exp(-v['opacity'].astype(np.float32))))[:,None]
        q = np.column_stack([v[k] for k in ['rot_0','rot_1','rot_2','rot_3']])
        rotation = Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix()
        q = Rotation.from_matrix(UNDO[None]@rotation).as_quat()[:,[3,0,1,2]].copy()
        colors = (.28209479177387814*np.column_stack([v[k] for k in ['f_dc_0','f_dc_1','f_dc_2']])+.5).clip(0,1)
        group_ids = np.empty(len(v),np.int32)
        for g in nodes:
            if g['source_node']==nid:
                group_ids[membership[g['membership_prefix']+'_gaussians']] = g['id']
        gaussians.append({'source':nid,'xyz':xyz,'scales':scales,'q':q,'alpha':alpha,'colors':colors,'groups':group_ids})
        source_counts.append((nid,offset,offset+len(v)))
        offset += len(v)
    # Preserve original source and Gaussian order, including at identical depths.
    arrays = {k:tensor(np.concatenate([p[k] for p in gaussians])) for k in ['xyz','scales','q','alpha','colors']}
    group_labels = np.concatenate([p['groups'] for p in gaussians])
    group_indices = {g['id']:torch.as_tensor(np.flatnonzero(group_labels==g['id']),device='cuda') for g in nodes}
    original_indices = {nid:torch.arange(lo,hi,device='cuda') for nid,lo,hi in source_counts}
    renderer = GaussianRenderer({'resolution':512,'near':.05,'far':20.,'bg_color':(1.,1.,1.)})
    clouds = []
    for t in np.linspace(0,1,21):
        for typ,tree in [('original',originals),('prior',nodes)]:
            tr = transforms(tree,t)
            for part in asset['parts']:
                src = part['node']['id']
                if typ=='original':
                    pts=np.asarray(part['mesh'].vertices)[::71]
                    clouds.append(moved(pts,tr[src]))
                else:
                    for g in nodes:
                        if g['source_node']==src:
                            pts=np.asarray(part['mesh'].vertices)[membership[g['membership_prefix']+'_vertices']][::71]
                            clouds.append(moved(pts,tr[g['id']]))
    cloud=np.concatenate(clouds)
    center=(cloud.min(0)+cloud.max(0))/2
    radius=1.65*np.linalg.norm(np.ptp(cloud,axis=0))
    intr=utils3d.torch.intrinsics_from_fov_xy(tensor(np.deg2rad(42.)),tensor(np.deg2rad(42.)))
    cameras=[]
    for yaw in [.38,-.72]:
        eye=center+radius*np.array([np.sin(yaw)*np.cos(.22),np.sin(.22),np.cos(yaw)*np.cos(.22)])
        cameras.append(utils3d.torch.extrinsics_look_at(tensor(eye),tensor(center),tensor([0,1,0])))
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    palette={10:[.6,.6,.6],11:[.9,.25,.15],12:[.2,.4,.95],13:[.2,.75,.4],14:[.8,.4,.8],15:[.2,.75,.4]}
    segmentation=tensor(np.array([palette[int(g)] for g in group_labels]))

    def pose(tree,t,active=None):
        posed=copy.deepcopy(tree)
        for g in posed:
            lo,hi=g['joint']['range']
            q=lo+(hi-lo)*(t if active is None or g['id']==active else 0.)
            g['joint']['range']=[q,q]
        return posed

    @torch.no_grad()
    def frame(method,t,camera,active=None,seg=False):
        tree=originals if method=='original' else nodes
        tr=transforms(pose(tree,t,active),1.)
        indices=original_indices if method=='original' else group_indices
        xyz=arrays['xyz'].clone()
        rotations=arrays['q'].clone()
        for gid,ids in indices.items():
            T=tr[gid]
            xyz[ids]=arrays['xyz'][ids]@tensor(T[:3,:3]).T+tensor(T[:3,3])
            q=Rotation.from_matrix(T[:3,:3]).as_quat()[[3,0,1,2]].copy()
            rotations[ids]=quaternion_product(tensor(q),arrays['q'][ids])
        g=Gaussians(xyz,arrays['scales'],rotations,arrays['alpha'])
        rgb=renderer.render(g,cameras[camera],intr,colors_overwrite=segmentation if seg else arrays['colors'])['color']
        pic=Image.fromarray((rgb.clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype('uint8'))
        draw=ImageDraw.Draw(pic)
        draw.rectangle((0,0,511,56),fill='white')
        label=f'{method} | t={t:.2f}' if active is None else f'{next(g["name"] for g in nodes if g["id"]==active)} | t={t:.2f}'
        draw.text((12,8),label,fill='black',font=font)
        draw.text((12,31),'45194 | unchanged shape and appearance',fill='#555555',font=font)
        return np.asarray(pic)
    started=time.monotonic()
    files=[]
    video=root/'45194_semantic_prior_reconstruction.mp4'
    comparison=root/'45194_original_vs_semantic_prior.mp4'
    files.extend([video,comparison])
    seq=np.r_[np.zeros(10),np.linspace(0,1,41),np.ones(10),np.linspace(1,0,41),np.zeros(10)]
    with imageio.get_writer(video,fps=20,codec='libx264',quality=8) as writer, imageio.get_writer(comparison,fps=20,codec='libx264',quality=8) as cmp:
        for i,t in enumerate(seq):
            a=[frame('original',float(t),c) for c in [0,1]]
            b=[frame('semantic_prior',float(t),c) for c in [0,1]]
            writer.append_data(np.concatenate(b,axis=1))
            panel=np.concatenate([np.concatenate([a[c],b[c]],axis=1) for c in [0,1]],axis=0)
            cmp.append_data(panel)
            if i in [10,30,50]:Image.fromarray(panel).save(root/f'comparison_t{t:.2f}.png')
            if i%30==0:print('RENDER combined',i,flush=True)
    independent=root/'45194_doors_drawers_independent.mp4'
    files.append(independent)
    with imageio.get_writer(independent,fps=15,codec='libx264',quality=8) as writer:
        for gid in [11,12,13,14]:
            for t in np.r_[np.linspace(0,1,26),np.ones(6),np.linspace(1,0,26)]:
                panel=np.concatenate([frame('semantic_prior',float(t),c,active=gid) for c in [0,1]],axis=1)
                writer.append_data(panel)
                if t==1.:Image.fromarray(panel).save(root/f'independent_g{gid}.png')
            print('RENDER independent',gid,flush=True)
    Image.fromarray(np.concatenate([frame('semantic_prior',0.,0,seg=True),frame('semantic_prior',1.,0,seg=True)],axis=1)).save(root/'motion_groups_preview.png')
    checks=verify(root)
    video_checks=[]
    for path in files:
        with imageio.get_reader(path) as reader:
            count=reader.count_frames()
            meta=reader.get_meta_data()
            reader.get_data(0);reader.get_data(count-1)
        video_checks.append({'file':path.name,'frames':count,'fps':meta['fps'],'size':list(meta['size']),'sha256':sha256(path)})
    checks['videos']=video_checks
    checks['elapsed_render_seconds']=time.monotonic()-started
    write_json(root/'verification.json',checks)
    write_json(root/'render_info.json',{'shared_camera_center':center.tolist(),'shared_camera_radius':float(radius),
              'renderer':'Original PAct GaussianRenderer, all source splats in original order, fixed DC colors',
              'geometry_bytes_unchanged':True,'render_seconds':checks['elapsed_render_seconds'],
              'gpu_peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20,'videos':video_checks})
    print('COMPLETE',root,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True)
    p.add_argument('--run')
    p.add_argument('--render',action='store_true')
    a=p.parse_args()
    root=pathlib.Path(a.root).resolve()
    if a.run:build(a.run,root)
    if a.render:render(root)
