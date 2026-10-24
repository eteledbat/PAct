"""Independent full-triangle VTK intersections, not solid overlap or penetration depth."""
import argparse
import json
import pathlib
import time

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy

from collision_refinement_common import candidate_nodes, load_asset, sha256, transforms


def polydata(mesh):
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(np.asarray(mesh.vertices, np.float64), deep=True))
    triangles = np.column_stack([np.full(len(mesh.faces), 3, dtype=np.int64), mesh.faces]).ravel()
    cells = vtk.vtkCellArray()
    cells.SetCells(len(mesh.faces), numpy_to_vtkIdTypeArray(triangles, deep=True))
    data = vtk.vtkPolyData()
    data.SetPoints(points)
    data.SetPolys(cells)
    return data


class TrianglePairs:
    def __init__(self, asset, selected=None, leaf_cells=2):
        self.asset = asset
        self.inputs = [polydata(p['mesh']) for p in asset['parts']]
        self.filters = {}
        for i in range(len(self.inputs)):
            for j in range(i + 1, len(self.inputs)):
                ids = [asset['nodes'][i]['id'], asset['nodes'][j]['id']]
                if selected is not None and ids != selected:
                    continue
                f = vtk.vtkCollisionDetectionFilter()
                matrices = [vtk.vtkMatrix4x4(), vtk.vtkMatrix4x4()]
                for k, idx in enumerate([i, j]):
                    matrices[k].Identity()
                    f.SetInputData(k, self.inputs[idx])
                    f.SetMatrix(k, matrices[k])
                f.SetCollisionModeToAllContacts()
                f.SetBoxTolerance(0.)
                f.SetCellTolerance(0.)
                f.SetNumberOfCellsPerNode(leaf_cells)
                f.SetGenerateScalars(0)
                self.filters[i, j] = (f, matrices)

    def state(self, nodes, t, save_segments=False):
        world = transforms(nodes, float(t))
        rows, snapshots = [], []
        D = self.asset['diagonal']
        for (i, j), (f, matrices) in self.filters.items():
            ids = [self.asset['nodes'][k]['id'] for k in [i, j]]
            for mat, nid in zip(matrices, ids):
                tr = world[nid]
                for r in range(4):
                    for c in range(4):
                        mat.SetElement(r, c, float(tr[r, c]))
                mat.Modified()
            f.Update()
            output = f.GetContactsOutput()
            total, transverse = 0., 0.
            nonzero = 0
            if output.GetNumberOfLines():
                xyz = vtk_to_numpy(output.GetPoints().GetData())
                lines = vtk_to_numpy(output.GetLines().GetData()).reshape(-1, 3)
                assert np.all(lines[:, 0] == 2)
                segments = xyz[lines[:, 1:]]
                length = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
                valid = length > 1e-6 * D
                total = float(length[valid].sum())
                nonzero = int(valid.sum())
                ci = vtk_to_numpy(f.GetContactCells(0))
                cj = vtk_to_numpy(f.GetContactCells(1))
                assert len(ci) == len(cj) == len(length)
                ni = self.asset['parts'][i]['mesh'].face_normals[ci] @ world[ids[0]][:3, :3].T
                nj = self.asset['parts'][j]['mesh'].face_normals[cj] @ world[ids[1]][:3, :3].T
                crossing = valid & (abs((ni * nj).sum(1)) < np.cos(np.deg2rad(5)))
                transverse = float(length[crossing].sum())
                if save_segments:
                    select = np.flatnonzero(crossing)
                    if len(select) > 5000:
                        select = select[np.argsort(length[select])[-5000:]]
                    snapshots.append({'ids': ids, 'segments': segments[select].astype(float).tolist()})
            rows.append({'ids': ids, 'contacting_triangle_pairs': int(f.GetNumberOfContacts()),
                         'nonzero_segments': nonzero, 'line_length_over_diagonal': total / D,
                         'transverse_line_length_over_diagonal': transverse / D})
        return rows, snapshots


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--candidate')
    p.add_argument('--states', type=int, default=21)
    p.add_argument('--pair', nargs=2, type=int)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--partition-size', type=int, default=0, help='Optional conservative spatial partition of all original triangles; zero preserves legacy evaluator')
    a = p.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=a.resume)
    asset = load_asset(a.run)
    nodes = candidate_nodes(asset, a.candidate)
    protocol = {'run': str(asset['run']), 'states': a.states, 'selected_pair': a.pair,
                'prediction_sha256': sha256(a.candidate or asset['object_path'])}
    if a.partition_size:
        protocol['partition_size'] = a.partition_size
        protocol['evaluation_source_sha256'] = {str(q): sha256(q) for q in [pathlib.Path(__file__),pathlib.Path(__file__).with_name('partitioned_triangle_motion.py')]}
    if a.resume:
        assert json.loads((out / 'protocol.json').read_text()) == protocol
    else:
        (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    if a.partition_size:
        from partitioned_triangle_motion import PartitionedTrianglePairs
        checker = PartitionedTrianglePairs(asset, a.pair, a.partition_size)
    else:
        checker = TrianglePairs(asset, a.pair)
    started = time.monotonic()
    rows = json.loads((out / 'states.json').read_text()) if a.resume and (out / 'states.json').exists() else []
    for state, t in enumerate(np.linspace(0, 1, a.states)):
        if state < len(rows):
            continue
        pairs, segments = checker.state(nodes, t, state in [0, a.states // 2, a.states - 1])
        rows.append({'state': state, 't': float(t), 'pairs': pairs})
        if segments:
            (out / f'segments_{state:02d}.json').write_text(json.dumps(segments))
        (out / 'states.json').write_text(json.dumps(rows, indent=2, allow_nan=False))
        print('TRIANGLES', state, 'line/D', sum(p['line_length_over_diagonal'] for p in pairs), flush=True)
    result = {'states': rows, 'method': 'VTK full original triangles, AllContacts, zero tolerances; segments <=1e-6*diagonal excluded',
              'spatial_partition_size': a.partition_size, 'geometry_decimated': False,
              'interpretation': 'Surface intersections include valid contact and inherited geometry errors. Transverse excludes face normal angles <=5deg but is still not a solid collision measure.',
              'mean_total_line_length_over_diagonal': float(np.mean([sum(p['line_length_over_diagonal'] for p in row['pairs']) for row in rows])),
              'mean_transverse_line_length_over_diagonal': float(np.mean([sum(p['transverse_line_length_over_diagonal'] for p in row['pairs']) for row in rows])),
              'solid_overlap_volume': None, 'solid_penetration_depth': None, 'elapsed_seconds': time.monotonic() - started}
    (out / 'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print('COMPLETE', out, flush=True)


if __name__ == '__main__':
    main()
