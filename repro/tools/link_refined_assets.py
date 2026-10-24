"""Make sealed candidates viewable through relative original geometry symlinks."""
import json
import pathlib

from collision_refinement_common import sha256

repo = pathlib.Path(__file__).resolve().parents[2]
root = pathlib.Path((repo / 'repro/current_refinement.txt').read_text().strip())
config = json.loads((root / 'experiment.json').read_text())
links = []
for name in config['cases']:
    for method in ['position', 'direction', 'both']:
        directory = root / f'{name}_{method}'
        candidate = directory / 'candidate.json'
        seal = json.loads((directory / 'prediction_seal.json').read_text())
        assert sha256(candidate) == seal['candidate_sha256']
        obj = json.loads(candidate.read_text())
        source = pathlib.Path(obj['geometry_source_object'])
        assert source.resolve().is_relative_to(repo / 'repro/outputs')
        for kind in ['ply', 'gs']:
            target = source.parent / kind
            assert target.is_dir()
            destination = directory / kind
            if destination.exists() or destination.is_symlink():
                assert destination.is_symlink() and destination.resolve() == target.resolve()
            else:
                destination.symlink_to(target, target_is_directory=True)
            links.append({'link': str(destination), 'target': str(target)})
        for node in obj['diffuse_tree']:
            for key in ['plys', 'gs']:
                for rel in node[key]:
                    assert (directory / rel).is_file()
        assert sha256(candidate) == seal['candidate_sha256']
(root / 'geometry_links.json').write_text(json.dumps({'no_mesh_copies_or_modifications': True, 'candidate_hashes_unchanged': True, 'links': links}, indent=2))
print('Linked original geometry to6sealed candidates')
