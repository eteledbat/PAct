"""Bounded self-intersection checks; timeouts stay unknown, never become False."""
import argparse, datetime, json, os, pathlib, subprocess, sys, time

p=argparse.ArgumentParser();p.add_argument('--run');p.add_argument('--out');p.add_argument('--worker');p.add_argument('--timeout',type=float,default=45)
a=p.parse_args()
if a.worker:
    # Open3D/TBB may otherwise occupy the entire server despite OMP settings.
    if hasattr(os,'sched_getaffinity'):
        os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
    import numpy as np, trimesh, open3d as o3d
    mesh=trimesh.load(a.worker,process=False,force='mesh')
    v,inv=np.unique(mesh.vertices,axis=0,return_inverse=True)
    faces=inv[mesh.faces]
    assert np.array_equal(v[faces],mesh.vertices[mesh.faces])
    m=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(faces.astype(np.int32)))
    result={'self_intersecting':bool(m.is_self_intersecting()),'method':'Open3D exact-position welded original triangles; no surface changes'}
    print(json.dumps(result),flush=True)
    sys.exit(0)
from collision_refinement_common import load_asset,sha256
asset=load_asset(a.run);rows=[]
for part in asset['parts']:
    start=time.monotonic()
    env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
    # Open3D shared EGL libraries only, not a Python import from the other env.
    env['LD_LIBRARY_PATH']='/root/autodl-tmp/conda_envs/part2gs/lib/python3.10/site-packages/open3d'+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    row={'node_id':part['node']['id'],'mesh':str(part['mesh_path']),'sha256':sha256(part['mesh_path'])}
    try:
        process=subprocess.run([sys.executable,__file__,'--worker',str(part['mesh_path'])],env=env,capture_output=True,text=True,timeout=a.timeout)
        if process.returncode:row.update(status='unavailable',reason=process.stderr[-2000:],self_intersecting=None)
        else:row.update(status='measured',**json.loads(process.stdout.strip().splitlines()[-1]))
    except subprocess.TimeoutExpired:
        row.update(status='unavailable_timeout',self_intersecting=None,reason=f'Self-intersection worker exceeded {a.timeout}s; undefined is not collision-free.')
    row['elapsed_seconds']=time.monotonic()-start;rows.append(row)
    result={'run':str(asset['run']),'parts':rows,'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'geometry_repaired':False}
    pathlib.Path(a.out).write_text(json.dumps(result,indent=2));print(json.dumps(row),flush=True)
