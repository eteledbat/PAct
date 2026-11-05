"""Turn extracted endpoint RGBs and oracle 2D masks into a fresh VLM inventory."""
import argparse
import json
import os
import pathlib
import sys

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
import cv2
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from auto_part_semantics import write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=pathlib.Path, required=True)
    a = p.parse_args()
    root = a.root.resolve()
    inventory = json.loads((REPO/'repro/outputs/original_mask_multicase_20261029_v1/input_inventory.json').read_text())
    cases = json.loads((REPO/'repro/outputs/closed_multicase_20261028_v1/protocol.json').read_text())['cases']
    out = []
    for case in cases:
        old = next(i for i in inventory if i['case'] == case)
        manifest = json.loads((root/'extremes'/case/'manifest.json').read_text())
        first, second = manifest['endpoints']
        exr = root/'extremes'/case/'endpoint_0_mask.exr'
        labels = np.load(first['mask']).astype(np.float32)
        assert cv2.imwrite(str(exr), np.repeat(labels[:, :, None], 3, axis=2))
        out.append(dict(case=case, run=old['run'], provenance=old['provenance'],
                        mask=str(exr), rgb=first['rgb'], second_rgb=second['rgb'],
                        second_mask=second['mask'], source_origin_rgb=old['rgb']))
    write_json(root/'two_extrema_inventory.json', out)
    print('INVENTORY', len(out), root/'two_extrema_inventory.json', flush=True)


if __name__ == '__main__': main()
