"""Read-only independent monitor; does not require an active Codex session."""
import argparse
import json
import pathlib
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--root');p.add_argument('--watch',action='store_true');a=p.parse_args()
    repo=pathlib.Path(__file__).resolve().parents[2];root=pathlib.Path(a.root or (repo/'repro/current_frozen_joint_routes.txt').read_text().strip())
    while True:
        print('\n'+str(root),flush=True)
        for f in [root/'controller_status.json',root/'evaluation_status.json',*sorted(root.glob('features/*/status.json')),*sorted(root.glob('*/status.json'))]:
            if not f.exists():continue
            try:
                s=json.loads(f.read_text());detail=' '.join(str(s[k]) for k in ['route','case','method'] if k in s);print(f.parent.name,f.name,s.get('state'),s.get('step',''),s.get('job',''),detail,s.get('error',''),flush=True)
            except (ValueError,OSError):continue
        if not a.watch:break
        time.sleep(20)


if __name__=='__main__':main()
