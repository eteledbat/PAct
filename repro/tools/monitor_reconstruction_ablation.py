"""Read-only standalone monitor; no Codex process required."""
import argparse
import json
import pathlib
import time

p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--watch',action='store_true');a=p.parse_args();root=pathlib.Path(a.root)
while True:
    for path in [root/'controller_status.json',root/'evaluation_status.json',*sorted(root.glob('*/status.json'))]:
        if not path.exists():continue
        d=json.loads(path.read_text());print(path.parent.name,path.name,d.get('state'),f"{d.get('step','')}/{d.get('total_steps','')}",d.get('job',''),d.get('error',''),flush=True)
    if not a.watch:break
    time.sleep(20)
