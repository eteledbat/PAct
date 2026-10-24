"""Sequential, restartable driver for six fixed final-step refinements."""
import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('--root', required=True)
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


try:
    for name, run in config['cases'].items():
        for variant in ['position', 'direction', 'both']:
            output = root / f'{name}_{variant}'
            complete = output / 'prediction_seal.json'
            if complete.is_file():
                assert json.loads((output / 'status.json').read_text())['state'] == 'complete'
                print('ALREADY COMPLETE', output, flush=True)
                continue
            argv = [sys.executable, '-u', str(repo / 'repro/tools/refine_collision_axes.py'), '--run', run,
                    '--out', str(output), '--variant', variant, '--steps', str(config['steps']), '--samples', str(config['samples'])]
            if output.exists():
                argv.append('--resume')
            status('running', case=name, variant=variant, output=str(output))
            print('START', name, variant, flush=True)
            with (root / f'{name}_{variant}.log').open('ab') as log:
                subprocess.run(argv, env=env, cwd=repo, stdout=log, stderr=subprocess.STDOUT, check=True)
            print('COMPLETE', name, variant, flush=True)
    status('complete', completed_refinements=6)
except BaseException as exc:
    status('failed', error=f'{type(exc).__name__}: {exc}')
    raise
