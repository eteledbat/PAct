"""Read-only progress monitor, usable independently of Codex."""
import argparse
import json
import pathlib
import time

parser = argparse.ArgumentParser()
parser.add_argument('--root')
parser.add_argument('--watch', action='store_true')
args = parser.parse_args()
repo = pathlib.Path(__file__).resolve().parents[2]
root = pathlib.Path(args.root or (repo / 'repro/current_refinement.txt').read_text().strip())
config = json.loads((root / 'experiment.json').read_text())
while True:
    print('Experiment:', root, flush=True)
    for name in config['cases']:
        for variant in config.get('variants', ['position', 'direction', 'both']):
            path = root / f'{name}_{variant}/status.json'
            if path.exists():
                data = json.loads(path.read_text())
                print(name, variant, data['state'], f'{data["step"]}/{data["total_steps"]}',
                      data.get('error', ''), flush=True)
            else:
                print(name, variant, 'waiting', flush=True)
    validation = root / 'validation_status.json'
    if validation.exists():
        print('Validation:', validation.read_text(), flush=True)
    artifacts = root / 'artifact_status.json'
    if artifacts.exists():
        print('Artifacts:', artifacts.read_text(), flush=True)
    if not args.watch:
        break
    time.sleep(20)
