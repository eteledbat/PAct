"""Numerical checks on actual exported geometry, not experimental improvement claims."""
import argparse
import json
import pathlib

import numpy as np
import torch

from collision_refinement_common import load_asset, transforms
from refine_collision_axes import AxisParameters, GaussianObjective

parser = argparse.ArgumentParser()
parser.add_argument('--run', required=True)
parser.add_argument('--out', required=True)
args = parser.parse_args()
torch.set_num_threads(4)
asset = load_asset(args.run, gaussian_count=2048, seed=0)
params = AxisParameters(asset, 'both')
times = torch.linspace(0, 1, 21, device='cuda')
world = params.world(times)
for index in [0, 7, 20]:
    expected = transforms(asset['nodes'], index / 20)
    for nid in expected:
        assert np.allclose(world[nid][index].detach().cpu().numpy(), expected[nid], atol=2e-6)
objective = GaussianObjective(asset, params, times, freeze_correspondences=True)
loss = objective()
loss.backward()
checks = []
for name, param, epsilon in [('direction', params.theta, 2e-4), ('pivot', params.phi, 2e-5 * asset['diagonal'])]:
    if name == 'pivot' and not bool(params.pivot_mask.any()):
        assert param.grad is None or torch.count_nonzero(param.grad) == 0
        checks.append({'parameter': name, 'status': 'not_applicable', 'reason': 'All active joints are translational: origin does not affect kinematics.'})
        continue
    grad = param.grad.detach().clone()
    flat = int(grad.abs().argmax())
    analytic = float(grad.flatten()[flat])
    assert abs(analytic) > 1e-10, (name, 'missing effective gradient')
    with torch.no_grad():
        param.flatten()[flat] += epsilon
        plus = float(objective())
        param.flatten()[flat] -= 2 * epsilon
        minus = float(objective())
        param.flatten()[flat] += epsilon
    numeric = (plus - minus) / (2 * epsilon)
    # Validate the derivative of the selected-neighbor branch. Production refreshes
    # NN every call; that discrete selection has no derivative at correspondence swaps.
    relative = abs(numeric - analytic) / max(abs(numeric), abs(analytic), 1e-8)
    checks.append({'parameter': name, 'analytic': analytic, 'finite_difference': numeric, 'relative_error': relative})
    assert relative < .08, checks[-1]
at_zero = params.world(times[:1])
with torch.no_grad():
    params.theta.fill_(.01)
    params.phi.fill_(.01 * asset['diagonal'])
after = params.world(times[:1])
assert all(torch.allclose(at_zero[n], after[n], atol=1e-7) for n in at_zero)
result = {'passed': True, 'run': str(asset['run']), 'scope': 'Real-asset kinematic and gradient validation, not improvement metrics',
          'numpy_torch_transform_agreement': True, 'zero_state_invariant_under_axis_refinement': True,
          'gradient_scope': 'Fixed NN branch, production recomputes NN and is piecewise smooth',
          'gradient_checks': checks}
pathlib.Path(args.out).write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
