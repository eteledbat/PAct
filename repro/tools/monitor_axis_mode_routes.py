"""Independent recursive read-only progress monitor for the two ablations."""
import argparse
import json
import pathlib
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--root');p.add_argument('--watch',action='store_true');a=p.parse_args();repo=pathlib.Path(__file__).resolve().parents[2];root=pathlib.Path(a.root or (repo/'repro/current_axis_mode_ablation.txt').read_text().strip())
    while True:
        print(str(root),flush=True)
        files=[root/'controller_status.json',root/'evaluation_status.json',*sorted(root.glob('*/controller_status.json')),*sorted(root.glob('*/*/status.json'))]
        for file in files:
            if not file.exists():continue
            try:
                d=json.loads(file.read_text());print(str(file.relative_to(root)),d.get('state'),d.get('step',''),d.get('mode',''),d.get('job',''),d.get('case',''),d.get('method',''),d.get('error',''),flush=True)
            except (ValueError,OSError):continue
        if not a.watch:break
        time.sleep(20)


if __name__=='__main__':main()
