"""Small, fixed-seed diagnostic case screen; all failures and candidates retained."""
import argparse,datetime,json,os,pathlib,subprocess,sys,time
REPO=pathlib.Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=pathlib.Path(a.root).resolve();config=json.loads((root/'screen_protocol.json').read_text())
env=os.environ.copy();env.update({'PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4'})
results={}
def save(state,**kw):
    path=root/'status.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps({'state':state,'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'results':results,**kw},indent=2));tmp.replace(path)
def job(scene,label,tool,args):
    command=[sys.executable,str(REPO/'repro/tools'/tool),*map(str,args)]
    (root/f'{scene}_{label}_command.json').write_text(json.dumps(command,indent=2));start=time.monotonic()
    with (root/f'{scene}_{label}.log').open('ab') as log:subprocess.run(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    (root/f'{scene}_{label}_timing.json').write_text(json.dumps({'seconds':time.monotonic()-start},indent=2))
for scene in config['new_cases']:
    try:
        save('running',scene=scene,stage='input_validation')
        inputs=REPO/f'repro/inputs/video_gt_{scene}_frame000000'
        if not (inputs/'input_provenance.json').exists():job(scene,'prepare','prepare_video_gt_case.py',['--scene',scene])
        save('running',scene=scene,stage='pretrained_inference')
        pointer=REPO/f'repro/current_gt_{scene}.txt'
        run=pathlib.Path(pointer.read_text().strip()) if pointer.exists() else None
        if run is None or not (run/'status.json').exists() or json.loads((run/'status.json').read_text())['state']!='complete':
            job(scene,'inference','run_gt_case.py',['--scene',scene]);run=pathlib.Path(pointer.read_text().strip())
        save('running',scene=scene,stage='joint_evaluation')
        if not (run/'joint_evaluation/joint_errors.json').exists():job(scene,'joints','evaluate_gt_joints.py',['--run',run])
        if not (run/'joint_evaluation/geometry_registration_sensitivity.json').exists():job(scene,'ICP','evaluate_registration_sensitivity.py',['--run',run])
        results[scene]={'state':'complete','run':str(run)};save('running',scene=scene,stage='case_complete')
    except BaseException as exc:
        results[scene]={'state':'failed','error':f'{type(exc).__name__}: {exc}'};save('running',scene=scene,stage='case_failed')
save('complete');print(json.dumps(results,indent=2),flush=True)
