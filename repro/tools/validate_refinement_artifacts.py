"""Verify the sealed candidates, unchanged geometry, measurements, and report links."""
import ast
import json
import pathlib
import re

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from collision_refinement_common import sha256

repo = pathlib.Path(__file__).resolve().parents[2]
root = pathlib.Path((repo / 'repro/current_refinement.txt').read_text().strip())
summary = json.loads((repo / 'repro/evidence/collision-refinement-results.json').read_text())
assert summary['gradient_check']['passed'] and summary['resume_check']['passed']
assert summary['nn_backend_check']['passed_distance_tolerance']
source_files = 0
for path, expected in summary['sources'].items():
    assert sha256(path) == expected, path
    ast.parse(pathlib.Path(path).read_text())
    source_files += 1
unique_files = {}
trial_count = metric_states = surface_states = 0
for name, case in summary['cases'].items():
    baseline_object = next(pathlib.Path(summary['config']['cases'][name]).rglob('object.json'))
    original = {n['id']: n for n in json.loads(baseline_object.read_text())['diffuse_tree']}
    original_metric = json.loads((root / f'eval_{name}_original/metrics.json').read_text())
    original_mesh = json.loads((root / f'mesh_{name}_original/metrics.json').read_text())
    for method, entry in case['methods'].items():
        for path, expected in entry['evidence'].items():
            assert sha256(path) == expected
        metric = json.loads((root / f'eval_{name}_{method}/metrics.json').read_text())
        mesh = json.loads((root / f'mesh_{name}_{method}/metrics.json').read_text())
        assert len(metric['states']) == 41 and len(mesh['states']) == 21
        metric_states += len(metric['states'])
        surface_states += len(mesh['states'])
        assert metric['solid_collision']['volume'] is None and metric['solid_collision']['penetration_depth'] is None
        assert np.isclose(metric['initial_state_mean_pair_proxy'], original_metric['initial_state_mean_pair_proxy'], atol=1e-12, rtol=1e-7)
        initial = sum(p['line_length_over_diagonal'] for p in mesh['states'][0]['pairs'])
        expected = sum(p['line_length_over_diagonal'] for p in original_mesh['states'][0]['pairs'])
        assert np.isclose(initial, expected, atol=1e-12)
        for change in metric['axis_changes']:
            assert change['direction_change_degrees'] <= 5.001
            assert change['origin_shift_percent_asset_diagonal'] is None or change['origin_shift_percent_asset_diagonal'] <= 5.001
        if method == 'original':
            continue
        trial_count += 1
        directory = root / f'{name}_{method}'
        seal = json.loads((directory / 'prediction_seal.json').read_text())
        assert sha256(directory / 'candidate.json') == seal['candidate_sha256']
        assert not seal['GT_reads_blocked'] and seal['frozen_files_unchanged']
        protocol = json.loads((directory / 'protocol.json').read_text())
        assert sha256(baseline_object) == protocol['original_object_sha256']
        for path, expected in protocol['source_sha256'].items():
            assert sha256(path) == expected, 'Executed optimizer source changed: ' + path
        for file in protocol['frozen_files']:
            for kind in ['mesh', 'gs']:
                unique_files[file[kind]] = file[kind + '_sha256']
        history = json.loads((directory / 'history.json').read_text())
        assert [h['step'] for h in history] == list(range(1, 201))
        assert all(np.isfinite(h['loss']) and np.isfinite(h['gaussian_proxy']) for h in history)
        candidate = json.loads((directory / 'candidate.json').read_text())
        for node in candidate['diffuse_tree']:
            before = original[node['id']]
            for field in ['name', 'parent', 'children', 'plys', 'gs']:
                assert node[field] == before[field]
            for field in ['range', 'type']:
                assert node['joint'][field] == before['joint'][field]
            if method == 'position':
                assert node['joint']['axis']['direction'] == before['joint']['axis']['direction']
            if method == 'direction' or node['joint']['type'] not in ['revolute', 'continuous']:
                assert node['joint']['axis']['origin'] == before['joint']['axis']['origin']
            for rel in node['plys'] + node['gs']:
                assert (directory / rel).is_file()
    preview = root / f'previews_{name}'
    image = np.asarray(Image.open(preview / 'mesh_state00_comparison.png'))
    for row in [0, 384]:
        reference = image[row + 32:row + 384, :384]
        for col in [1, 2, 3]:
            assert np.array_equal(reference, image[row + 32:row + 384, col * 384:(col + 1) * 384]), 'Rest geometry render changed'
    reader = imageio.get_reader(str(preview / 'original_position_direction_both.mp4'))
    assert reader.count_frames() == 21
    reader.close()
for path, expected in unique_files.items():
    assert sha256(path) == expected, path
report = repo / 'repro/COLLISION_REFINEMENT_RESULTS.md'
links = re.findall(r'\]\(([^)]+)\)', report.read_text())
for link in links:
    if not link.startswith(('https://', 'http://', '#')):
        assert (report.parent / link).is_file(), link
assert trial_count == 6 and metric_states == 8 * 41 and surface_states == 8 * 21
result = {'passed': True, 'sealed_200_step_trials': trial_count, 'optimizer_steps': trial_count * 200,
          'measured_Gaussian_states_including_originals': metric_states,
          'measured_full_triangle_states_including_originals': surface_states,
          'unique_frozen_mesh_and_Gaussian_files_verified': len(unique_files),
          'source_files_verified_and_syntax_parsed': source_files, 'report_links_verified': len(links),
          'rest_render_body_pixels_identical_across_methods': True,
          'both_comparison_videos_have_21_frames': True, 'solid_undefined_kept_null': True,
          'report_sha256': sha256(report)}
(repo / 'repro/evidence/collision-refinement-integrity.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
