"""Restartable driver for six fixed final-step refinements, with bounded concurrency."""
import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

parser = argparse.ArgumentParser()
parser.add_argument('--root', required=True)
parser.add_argument('--jobs', type=int, default=1)
args = parser.parse_args()
root = pathlib.Path(args.root).resolve()
config = json.loads((root / 'experiment.json').read_text())
repo = pathlib.Path(__file__).resolve().parents[2]
started = time.monotonic()
env = os.environ.copy()
env.update(PYTHONDONTWRITEBYTECODE='1', XDG_CACHE_HOME=str(repo / 'repro/cache'),
           OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1')


def status(state, **kwargs):
    tmp = root / 'status.tmp'
    tmp.write_text(json.dumps({'state': state, 'elapsed_seconds': time.monotonic() - started, **kwargs}, indent=2))
    tmp.replace(root / 'status.json')


def active_process(output):
    for path in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
        try:
            tokens = path.read_bytes().decode().split('\0')
            if '--out' in tokens and tokens[tokens.index('--out') + 1] == str(output) and any('refine_collision_axes.py' in t for t in tokens):
                return int(path.parent.name)
        except (OSError, UnicodeDecodeError, IndexError):
            pass
    return None


def run_one(name, run, variant):
    output = root / f'{name}_{variant}'
    # Adopt an already running worker when restarting just the driver.
    pid = active_process(output)
    if pid:
        print('ADOPT', name, variant, 'pid', pid, flush=True)
        while active_process(output):
            time.sleep(10)
    complete = output / 'prediction_seal.json'
    if complete.is_file():
        assert json.loads((output / 'status.json').read_text())['state'] == 'complete'
        print('ALREADY COMPLETE', output, flush=True)
        return name + '_' + variant
    argv = [sys.executable, '-u', str(repo / 'repro/tools/refine_collision_axes.py'), '--run', run,
            '--out', str(output), '--variant', variant, '--steps', str(config['steps']), '--samples', str(config['samples'])]
    if output.exists():
        argv.append('--resume')
    print('START', name, variant, flush=True)
    with (root / f'{name}_{variant}.log').open('ab') as log:
        subprocess.run(argv, env=env, cwd=repo, stdout=log, stderr=subprocess.STDOUT, check=True)
    print('COMPLETE', name, variant, flush=True)
    return name + '_' + variant


try:
    assert 1 <= args.jobs <= 3, 'At most three verified small-memory workers'
    completed = []
    status('running', jobs=args.jobs)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_one, name, run, variant) for name, run in config['cases'].items()
                   for variant in ['position', 'direction', 'both']]
        for future in as_completed(futures):
            completed.append(future.result())
            status('running', jobs=args.jobs, completed=completed)
    status('complete', completed_refinements=len(completed), completed=completed)
except BaseException as exc:
    status('failed', error=f'{type(exc).__name__}: {exc}')
    raise
