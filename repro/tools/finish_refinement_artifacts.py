"""Wait for sealed predictions/validation, then render and build the factual report."""
import json
import os
import pathlib
import subprocess
import sys
import time

repo = pathlib.Path(__file__).resolve().parents[2]
root = pathlib.Path((repo / 'repro/current_refinement.txt').read_text().strip())
config = json.loads((root / 'experiment.json').read_text())
env = os.environ.copy()
env.update(PYTHONDONTWRITEBYTECODE='1', XDG_CACHE_HOME=str(repo / 'repro/cache'),
           OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')


def status(state, **kwargs):
    tmp = root / 'artifact_status.tmp'
    tmp.write_text(json.dumps({'state': state, **kwargs}, indent=2))
    tmp.replace(root / 'artifact_status.json')


def run(script, *args):
    with (root / 'artifact-build.log').open('ab') as log:
        subprocess.run([sys.executable, '-u', str(repo / 'repro/tools' / script), *args], env=env, cwd=repo,
                       stdout=log, stderr=subprocess.STDOUT, check=True)


def finished(path):
    return path.exists() and json.loads(path.read_text()).get('state') == 'complete'


try:
    for name in config['cases']:
        status('waiting_for_candidates', case=name)
        while not all(finished(root / f'{name}_{method}/status.json') for method in ['position', 'direction', 'both']):
            time.sleep(10)
        status('rendering', case=name)
        if not (root / f'previews_{name}/render_info.json').exists():
            run('render_refinement_comparison.py', '--root', str(root), '--case', name)
        if name == '31249' and not (root / 'GT_axis_previews').exists():
            run('plot_refined_gt_axes.py', '--root', str(root))
    for name in config['cases']:
        status('waiting_for_mesh_validation', case=name)
        while not (root / f'mesh_{name}_both/metrics.json').exists():
            time.sleep(10)
        if not (root / f'hotspots_{name}').exists():
            run('plot_intersection_hotspots.py', '--root', str(root), '--case', name)
    status('waiting_for_all_validation')
    while not finished(root / 'validation_status.json'):
        time.sleep(10)
    run('build_collision_refinement_report.py')
    status('complete', report=str(repo / 'repro/COLLISION_REFINEMENT_RESULTS.md'))
except BaseException as exc:
    status('failed', error=str(exc))
    raise
