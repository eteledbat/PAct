"""Restartable tmux controller for a single frozen-geometry PAct comparison."""
import argparse, concurrent.futures, datetime, json, os, pathlib, subprocess, sys, time

REPO=pathlib.Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
root=pathlib.Path(a.root).resolve();config=json.loads((root/'experiment.json').read_text())
assert len(config['cases'])==1 and config['variants']==['direction']
name,run=next(iter(config['cases'].items()))
env=os.environ.copy();env.update({'PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4'})
started=time.monotonic()

def status(state,**kw):
    path=root/'controller_status.json';tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps({'state':state,'elapsed_seconds':time.monotonic()-started,'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw},indent=2));tmp.replace(path)

def job(label,tool,argv,marker):
    marker=pathlib.Path(marker)
    if marker.exists():
        print('REUSE COMPLETE STAGE',label,flush=True);return
    command=[sys.executable,str(REPO/'repro/tools'/tool),*map(str,argv)]
    (root/(label+'_command.json')).write_text(json.dumps(command,indent=2))
    begin=time.monotonic()
    with (root/(label+'.log')).open('ab') as log:
        subprocess.run(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    assert marker.exists(),f'{label} did not produce {marker}'
    (root/(label+'_timing.json')).write_text(json.dumps({'seconds':time.monotonic()-begin,'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2))
    print('COMPLETE',label,flush=True)

try:
    status('gradient_validation')
    job('gradient','test_collision_axis_gradients.py',['--run',run,'--out',root/'gradient_validation.json'],root/'gradient_validation.json')
    status('GPU_smoke')
    smoke=root/'smoke_direction'
    job('smoke','refine_collision_axes.py',['--run',run,'--out',smoke,'--variant','direction','--steps','5','--samples','2048',*(['--resume'] if (smoke/'checkpoint.pt').exists() else [])],smoke/'prediction_seal.json')
    status('formal_refinement')
    trial=root/f'{name}_direction'
    job('formal','refine_collision_axes.py',['--run',run,'--out',trial,'--variant','direction','--steps',config['steps'],'--samples',config['samples'],*(['--resume'] if (trial/'checkpoint.pt').exists() else [])],trial/'prediction_seal.json')
    status('GT_evaluation_after_prediction_seal')
    job('GT','evaluate_freeart_partnet_case.py',['--root',root],root/'GT_geometry_and_motion.json')
    status('independent_collision_validation_and_previews')
    tasks=[]
    for method in ['original','direction']:
        tasks.append((f'gaussian_{method}','evaluate_refined_axes.py',['--root',root,'--case',name,'--method',method],root/f'eval_{name}_{method}/metrics.json'))
        meshout=root/f'mesh_{name}_{method}'
        tasks.append((f'triangle_{method}','inspect_triangle_motion.py',['--run',run,'--out',meshout,'--states','21',*(['--partition-size',config['triangle_partition_size']] if config.get('triangle_partition_size') else []),*(['--candidate',trial/'candidate.json'] if method!='original' else []),*(['--resume'] if (meshout/'protocol.json').exists() else [])],meshout/'metrics.json'))
    tasks.append(('render','render_refinement_comparison.py',['--root',root,'--case',name],root/f'previews_{name}/render_info.json'))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        results=[executor.submit(job,*task) for task in tasks]
        for result in results:result.result()
    job('hotspots','plot_intersection_hotspots.py',['--root',root,'--case',name],root/f'hotspots_{name}/regions.json')
    status('complete')
except BaseException as exc:
    status('failed',error=f'{type(exc).__name__}: {exc}')
    raise
