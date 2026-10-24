"""Restartable artifact/verification completion after the single-case controller."""
import argparse,datetime,json,os,pathlib,subprocess,sys
import imageio.v2 as imageio
from PIL import Image,ImageDraw

REPO=pathlib.Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=pathlib.Path(a.root).resolve()
config=json.loads((root/'experiment.json').read_text());assert len(config['cases'])==1
name,run=next(iter(config['cases'].items()));run=pathlib.Path(run)
assert json.loads((root/'controller_status.json').read_text())['state']=='complete'
env=os.environ.copy();env.update({'PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'})
commands=[]
def state(value,**kw):
    (root/'artifact_status.json').write_text(json.dumps({'state':value,'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw},indent=2))
def job(tool,argv,marker=None):
    if marker is not None and marker.exists():print('REUSE',marker,flush=True);return
    command=[sys.executable,str(REPO/'repro/tools'/tool),*map(str,argv)];commands.append(command)
    (root/'artifact_commands.json').write_text(json.dumps(commands,indent=2))
    with (root/'artifact_completion.log').open('ab') as log:subprocess.run(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
try:
    state('finishing')
    montage=root/'original_textured_animation_samples.png'
    if not montage.exists():
        video,=run.rglob('00__articulation_animation.mp4');reader=imageio.get_reader(str(video));frames=[f for f in reader];reader.close()
        h,w=frames[0].shape[:2];panel=Image.new('RGB',(3*w,h+30),'white');draw=ImageDraw.Draw(panel)
        for i,index in enumerate([0,len(frames)//2,len(frames)-1]):panel.paste(Image.fromarray(frames[index]),(i*w,30));draw.text((i*w+10,5),f'Original PAct saved video frame {index}',fill='black')
        panel.save(montage)
    for directory,marker in [('GT_mesh_previews','render_info.json'),('figures','figure_provenance.json')]:
        folder=root/directory
        if folder.exists() and not (folder/marker).exists():folder.rename(root/(directory+'_incomplete_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')))
    job('audit_exported_mesh_solids.py',['--run',run,'--out',root/'mesh_self_intersections.json','--timeout','45'],root/'mesh_self_intersections.json')
    job('render_partnet_gt_comparison.py',['--root',root],root/'GT_mesh_previews/render_info.json')
    job('plot_partnet_refinement_results.py',['--root',root],root/'figures/figure_provenance.json')
    state('validating')
    job('validate_single_case_refinement.py',['--root',root])
    job('build_partnet_refinement_report.py',['--root',root])
    state('complete',integrity_passed=True,report=str(REPO/f'repro/PACT_{name}_RESULTS.md'))
except BaseException as exc:
    state('failed',error=f'{type(exc).__name__}: {exc}');raise
