"""Kinematic checks for observed-pose fitting independent of GT or a renderer."""
import copy,types,unittest
import numpy as np
from fit_observed_drawer_pose import PoseModel

def group(gid,parent,kind='fixed',direction=(0,0,1),limits=(0,.4)):
 return dict(id=gid,parent=parent,joint=dict(type=kind,range=list(limits),axis=dict(origin=[0,0,0],direction=list(direction))))

class ObservedPoseTests(unittest.TestCase):
 def test_drawer_pose_moves_descendants_without_other_drawer(self):
  groups=[group(10,-1),group(11,10,'prismatic'),group(12,11),group(13,10,'prismatic')];m=types.SimpleNamespace(groups=groups);PoseModel.set_pose(m,{'11':.28});np.testing.assert_allclose(m.closed_transforms[11][:3,3],[0,0,.28]);np.testing.assert_allclose(m.closed_transforms[12][:3,3],[0,0,.28]);np.testing.assert_allclose(m.closed_transforms[13],np.eye(4))
 def test_pose_does_not_edit_axes_ranges_or_parents(self):
  groups=[group(10,-1),group(11,10,'prismatic')];before=copy.deepcopy(groups);m=types.SimpleNamespace(groups=groups);PoseModel.set_pose(m,{'11':.28});self.assertEqual(groups,before)
 def test_reset_pose_restores_canonical_reference(self):
  m=types.SimpleNamespace(groups=[group(10,-1),group(11,10,'prismatic',direction=(0,0,2))]);PoseModel.set_pose(m,{11:.28});np.testing.assert_allclose(m.closed_transforms[11][:3,3],[0,0,.28]);PoseModel.set_pose(m,{});np.testing.assert_allclose(m.closed_transforms[11],np.eye(4))
if __name__=='__main__':unittest.main()
