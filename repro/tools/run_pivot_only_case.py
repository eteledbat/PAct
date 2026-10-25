"""Fixed-geometry, fixed-direction pivot-only test, including native-range limits."""
import argparse,concurrent.futures,datetime,json,os,pathlib,subprocess,sys,time
REPO=pathlib.Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=pathlib.Path(a.root).resolve();config=json.loads((root/'experiment.json').read_text());assert len(config['cases'])==1 and config['variants']==['position']
case,run=next(iter(config['cases'].items()));start=time.monotonic();env=os.environ.copy();env.update({'PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4'})
def status(state,**kw):
    path=root/'controller_status.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps({'state':state,'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-start,**kw},indent=2));tmp.replace(path)
def job(label,tool,args,marker):
    if marker.exists():print('REUSE',label,flush=True);return
    command=[sys.executable,str(REPO/'repro/tools'/tool),*map(str,args)];(root/f'{label}_command.json').write_text(json.dumps(command,indent=2));beg=time.monotonic()
    with (root/f'{label}.log').open('ab') as log:subprocess.run(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    assert marker.exists();(root/f'{label}_timing.json').write_text(json.dumps({'seconds':time.monotonic()-beg},indent=2));print('COMPLETE',label,flush=True)
try:
    status('gradient_and_GPU_smoke')
    job('gradient','test_collision_axis_gradients.py',['--run',run,'--out',root/'gradient_validation.json'],root/'gradient_validation.json')
    smoke=root/'smoke_position';job('smoke','refine_collision_axes.py',['--run',run,'--out',smoke,'--variant','position','--steps','5','--samples','2048',*(['--resume'] if (smoke/'checkpoint.pt').exists() else [])],smoke/'prediction_seal.json')
    status('formal_pivot_only')
    trial=root/f'{case}_position';job('formal','refine_collision_axes.py',['--run',run,'--out',trial,'--variant','position','--steps','200','--samples','2048',*(['--resume'] if (trial/'checkpoint.pt').exists() else [])],trial/'prediction_seal.json')
    status('independent_validation')
    tasks=[]
    for method in ['original','position']:
        tasks.append((f'Gaussian_{method}','evaluate_refined_axes.py',['--root',root,'--case',case,'--method',method],root/f'eval_{case}_{method}/metrics.json'))
        directory=root/f'mesh_{case}_{method}'
        tasks.append((f'triangles_{method}','inspect_triangle_motion.py',['--run',run,'--out',directory,'--states','21','--partition-size','12000',*(['--candidate',trial/'candidate.json'] if method=='position' else []),*(['--resume'] if (directory/'protocol.json').exists() else [])],directory/'metrics.json'))
    tasks.append(('render','render_refinement_comparison.py',['--root',root,'--case',case],root/f'previews_{case}/render_info.json'))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(job,*task) for task in tasks]
        for future in futures:future.result()
    job('hotspots','plot_intersection_hotspots.py',['--root',root,'--case',case],root/f'hotspots_{case}/regions.json')
    status('complete')
except BaseException as exc:status('failed',error=f'{type(exc).__name__}: {exc}');raise
