"""Validate actual exported mesh/GS bytes, motion metadata and fixed-step fit."""
import argparse
import json
import pathlib

import numpy as np
from plyfile import PlyData
from scipy.spatial.transform import Rotation
import trimesh

from collision_refinement_common import sha256
from semantic_motion_prior_45194 import UNDO


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
    root=pathlib.Path(a.root).resolve()
    cfg=json.loads((root/'config.json').read_text())
    fit=json.loads((root/'fit_results.json').read_text())
    prior=pathlib.Path(cfg['prior_root'])
    manifest=json.loads((prior/'motion_groups.json').read_text())
    membership=np.load(prior/'membership.npz')
    optimized=np.load(root/'optimized_parts.npz')
    native=root/'optimized_asset'
    nodes=json.loads((native/'object.json').read_text())['diffuse_tree']
    assert sha256(root/'optimized_parts.npz')==fit['optimized_parts_sha256']
    assert sha256(root/'checkpoint.pt')==fit['checkpoint_sha256']
    assert sha256(prior/'motion_groups.json')==cfg['prior_manifest_sha256']
    assert sha256(root/'observation.npz')==cfg['observation_sha256']
    assert sha256(cfg['source_image'])==cfg['source_image_sha256']
    assert all(sha256(x['path'])==x['sha256'] for x in manifest['geometry_files'])
    records={};rows=[]
    protected_exceptions={'x','y','z','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3'}
    for old,new in zip(manifest['groups'],nodes):
        for key in ['id','parent','name','joint']:assert new[key]==old[key],('Motion field changed',old['id'],key)
        gid=old['id'];prefix=f'g{gid}'
        mesh=trimesh.load(native/new['plys'][0],process=False,force='mesh')
        assert np.array_equal(mesh.faces,optimized[prefix+'_mesh_faces'])
        mesh_error=float(np.max(abs(np.asarray(mesh.vertices)-optimized[prefix+'_mesh_vertices'])))
        assert mesh_error<1e-7
        v=PlyData.read(native/new['gs'][0])['vertex'].data
        xyz=np.column_stack([v[k] for k in ['x','y','z']]).astype(np.float64)@UNDO.T
        xyz_error=float(np.max(abs(xyz-optimized[prefix+'_xyz'])))
        assert xyz_error<1e-7
        scales=np.exp(np.column_stack([v[k] for k in ['scale_0','scale_1','scale_2']]).astype(np.float64))
        q=np.column_stack([v[k] for k in ['rot_0','rot_1','rot_2','rot_3']])
        R=UNDO[None]@Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix()
        cov=(R*scales[:,None,:]**2)@R.transpose(0,2,1)
        target=optimized[prefix+'_covariance']
        relative=np.linalg.norm(cov-target,axis=(1,2))/np.linalg.norm(target,axis=(1,2)).clip(1e-20)
        assert relative.max()<5e-6
        nid=old['source_node']
        if nid not in records:records[nid]=PlyData.read(manifest['sources'][str(nid)]['gs'])['vertex'].data
        original=records[nid][membership[prefix+'_gaussians']]
        for key in v.dtype.names:
            if key not in protected_exceptions:assert np.array_equal(v[key],original[key]),('Appearance modified',gid,key)
        rows.append({'group_id':gid,'mesh_vertex_export_max_error':mesh_error,
                     'Gaussian_center_export_max_error':xyz_error,
                     'Gaussian_covariance_export_max_relative_error':float(relative.max()),
                     'mesh_faces_identical':True,'point_counts_identical':len(v)==old['counts']['gaussians'],
                     'color_opacity_and_non_geometry_fields_identical':True,
                     'motion_fields_identical':True})
    result={'all_original_files_unchanged':True,'single_reference_frame':cfg['reference_frame'],
            'part_checks':rows,'metrics':{'before':fit['before'],'after':fit['after']},
            'evidence_limit':'Metrics measure only the one fitted closed RGB frame. No held-out view/motion or calibrated 3D geometry evaluation.',
            'forbidden_GT_depth_reads':fit['forbidden_reads'],
            'render_verification':json.loads((root/'render_verification.json').read_text()),
            'checkpoint_step_policy':'100 image-registration steps, then fixed final 240 geometry steps; manual method refinements retained as separate runs'}
    (root/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'original_unchanged':True,'part_checks':rows,'metrics':result['metrics']},indent=2))


if __name__=='__main__':main()
