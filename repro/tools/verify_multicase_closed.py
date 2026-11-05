"""Audit exported geometry and source immutability for all completed cases."""
import argparse
import json
import pathlib

import numpy as np
from plyfile import PlyData
from scipy.spatial.transform import Rotation
import trimesh

from collision_refinement_common import sha256
from semantic_motion_prior_45194 import UNDO


def verify_case(root):
    cfg=json.loads((root/'config.json').read_text());manifest=json.loads((root/'motion_groups.json').read_text());seal=json.loads((root/'fit_results.json').read_text())
    assert sha256(root/'checkpoint.pt')==seal['checkpoint_sha256']
    assert sha256(root/'motion_groups.json')==seal['manifest_sha256']
    assert sha256(root/'optimized_parts.npz')==seal['optimized_parts_sha256']
    assert all(sha256(x['path'])==x['sha256'] for x in manifest['geometry_files'])
    idx=np.load(root/'membership.npz');data=np.load(root/'optimized_parts.npz')
    native=root/'optimized_asset';nodes=json.loads((native/'object.json').read_text())['diffuse_tree']
    source_mesh={};source_gs={};checks=[]
    exceptions={'x','y','z','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3'}
    for old,new in zip(manifest['groups'],nodes):
        for key in ['id','parent','joint']:assert old[key]==new[key],('Motion changed',cfg['case'],old['id'],key)
        source=old['source_node'];prefix=old['membership_prefix']
        if source not in source_mesh:source_mesh[source]=trimesh.load(manifest['sources'][str(source)]['mesh'],process=False,force='mesh')
        m=source_mesh[source];vi=idx[prefix+'_vertices'];fi=idx[prefix+'_faces'];lut=np.full(len(m.vertices),-1,np.int64);lut[vi]=np.arange(len(vi))
        exported=trimesh.load(native/new['plys'][0],process=False,force='mesh')
        assert np.array_equal(exported.faces,lut[m.faces[fi]])
        vertex_error=float(np.max(abs(np.asarray(exported.vertices)-data[prefix+'_mesh_vertices'])))
        assert vertex_error<1e-7
        if source not in source_gs:source_gs[source]=PlyData.read(manifest['sources'][str(source)]['gs'])['vertex'].data
        original=source_gs[source][idx[prefix+'_gaussians']];v=PlyData.read(native/new['gs'][0])['vertex'].data
        assert len(v)==len(original)
        for key in v.dtype.names:
            if key not in exceptions:assert np.array_equal(v[key],original[key]),('Appearance changed',old['id'],key)
        xyz=np.column_stack([v[k] for k in ['x','y','z']]).astype(np.float64)@UNDO.T
        point_error=float(np.max(abs(xyz-data[prefix+'_xyz'])));assert point_error<1e-7
        scale=np.exp(np.column_stack([v[k] for k in ['scale_0','scale_1','scale_2']]).astype(np.float64))
        q=np.column_stack([v[k] for k in ['rot_0','rot_1','rot_2','rot_3']]);r=UNDO[None]@Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix()
        c=(r*scale[:,None,:]**2)@r.transpose(0,2,1);target=data[prefix+'_covariance']
        cov_error=float(np.max(np.linalg.norm(c-target,axis=(1,2))/np.linalg.norm(target,axis=(1,2)).clip(1e-20)))
        assert cov_error<5e-6
        checks.append({'group':old['id'],'mesh_vertices':len(vi),'mesh_faces':len(fi),'gaussians':len(v),
                       'vertex_export_error':vertex_error,'point_export_error':point_error,'covariance_relative_error':cov_error,
                       'appearance_unchanged':True,'motion_fields_unchanged':True})
    render=json.loads((root/'render_results.json').read_text())
    assert all(sha256(root/v['file'])==v['sha256'] for v in render['videos'])
    result={'case':cfg['case'],'original_source_files_unchanged':True,'all_parts_checked':checks,
            'motion_metadata_frozen_during_image_fit':True,'video_count':len(render['videos']),
            'max_rigid_error':render['max_rigid_error'],'forbidden_GT_reads_in_fit':seal['GT_forbidden_reads']}
    (root/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print('VERIFIED',cfg['case'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=pathlib.Path(a.root)
    protocol=json.loads((root/'protocol.json').read_text());states={}
    for case in protocol['cases']:
        try:verify_case(root/case);states[case]='complete'
        except Exception as exc:states[case]='failed: '+repr(exc);print(case,states[case],flush=True)
    (root/'verification_status.json').write_text(json.dumps(states,indent=2)+'\n')
