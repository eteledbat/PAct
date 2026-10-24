"""Wait for sealed candidates, then independently evaluate each; no optimization."""
import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

parser = argparse.ArgumentParser()
parser.add_argument('--root', required=True)
args = parser.parse_args()
root = pathlib.Path(args.root).resolve()
repo = pathlib.Path(__file__).resolve().parents[2]
config = json.loads((root / 'experiment.json').read_text())
env = os.environ.copy()
env.update(PYTHONDONTWRITEBYTECODE='1', XDG_CACHE_HOME=str(repo / 'repro/cache'),
           OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')


def evaluate(name, run, method):
    with (root / f'validation_{name}_{method}.log').open('ab') as log:
        subprocess.run([sys.executable, '-u', str(repo / 'repro/tools/evaluate_refined_axes.py'),
                        '--root', str(root), '--case', name, '--method', method], env=env, cwd=repo,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
        directory = root / f'mesh_{name}_{method}'
        if not (directory / 'metrics.json').exists():
            command = [sys.executable, '-u', str(repo / 'repro/tools/inspect_triangle_motion.py'),
                       '--run', run, '--out', str(directory), '--candidate', str(root / f'{name}_{method}/candidate.json'), '--states', '21']
            if directory.exists():
                command.append('--resume')
            subprocess.run(command, env=env, cwd=repo, stdout=log, stderr=subprocess.STDOUT, check=True)
    return name + '_' + method


jobs = {(name, method): run for name, run in config['cases'].items() for method in ['position', 'direction', 'both']}
completed, running = [], {}
try:
    with ThreadPoolExecutor(max_workers=2) as pool:
        while jobs or running:
            for key in list(jobs):
                name, method = key
                state = root / f'{name}_{method}/status.json'
                if len(running) >= 2:
                    break
                if state.exists() and json.loads(state.read_text())['state'] == 'complete':
                    running[key] = pool.submit(evaluate, name, jobs.pop(key), method)
                    print('VALIDATE', key, flush=True)
            for key, future in list(running.items()):
                if future.done():
                    completed.append(future.result())
                    del running[key]
                    print('VALIDATED', key, flush=True)
            tmp = root / 'validation_status.tmp'
            tmp.write_text(json.dumps({'state': 'running' if jobs or running else 'complete',
                                       'completed': completed, 'running': list(running), 'waiting': list(jobs)}, indent=2))
            tmp.replace(root / 'validation_status.json')
            if jobs or running:
                time.sleep(10)
except BaseException as exc:
    (root / 'validation_status.json').write_text(json.dumps({'state': 'failed', 'error': str(exc), 'completed': completed}, indent=2))
    raise
