"""Inspect the saved original video and animation parameters without changing motion."""
import argparse
import json
import pathlib

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

parser = argparse.ArgumentParser()
parser.add_argument('--run', required=True)
args = parser.parse_args()
run = pathlib.Path(args.run).resolve()
assert json.loads((run / 'status.json').read_text())['state'] == 'complete'
video, = run.rglob('00__articulation_animation.mp4')
reader = imageio.get_reader(str(video))
first = last = None
count = 0
try:
    # Stream frames: imageio FFmpeg reports infinite length, so list(reader) is unsafe.
    for frame in reader:
        if first is None:
            first = frame.copy()
        last = frame
        count += 1
finally:
    reader.close()
assert first is not None and count > 1
delta = np.abs(last.astype(float) - first.astype(float))
trace = json.loads((run / 'joint_trace/animation_after.json').read_text())
assert len(trace) == 1
report = {
    'video': str(video), 'frames': count,
    'mean_absolute_RGB_difference_first_last': float(delta.mean()),
    'pixels_with_any_channel_change_over_5_percent': float((delta.max(2) > 5).mean() * 100),
    'difference_note': 'Decoded original video, includes codec effects; image difference is not a joint/geometry metric.',
    'animation_joints': [{'node_id': n['id'], 'name': n['name'], **n['joint']} for n in trace[0]],
    'motion_note': 'Original _local_transform uses linear interpolation of the predicted range in degrees; fixed parts remain static. No range override or GT motion injected.'}
out = run / 'joint_evaluation/animation_check'
out.mkdir(exist_ok=True)
height, width = first.shape[:2]
panel = Image.new('RGB', (width * 3, height + 38), 'white')
draw = ImageDraw.Draw(panel)
for i, (label, frame) in enumerate([
        ('First frame', first), ('Last frame', last),
        ('Absolute difference x8', np.clip(delta * 8, 0, 255).astype('uint8'))]):
    panel.paste(Image.fromarray(frame), (width * i, 30))
    draw.text((width * i + 10, 8), label, fill='black')
panel.save(out / 'original_first_last_difference.png')
(out / 'video_probe.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
