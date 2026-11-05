import copy,unittest
import numpy as np,trimesh
from fit_vlm_drawer_ranges import DEFAULT,weighted_quantile,basis_for,geometry_features,solve_interval,losses

class DrawerRangeTests(unittest.TestCase):
 def feature(self):return dict(drawer_front=1.,drawer_rear=0.,drawer_depth=1.,cabinet_front=1.,cabinet_back=0.,front_alignment_q=0.,native_outward_range=[0.,2.])
 def test_weighted_quantile_uses_area(self):self.assertAlmostEqual(weighted_quantile([1,2,3],[1,1,1],2/3),2.)
 def test_negative_axis_preserves_outward_convention(self):
  B,sign=basis_for([0,0,-2]);self.assertEqual(sign,-1.);np.testing.assert_allclose(B[:,2],[0,0,1]);np.testing.assert_allclose(B.T@B,np.eye(3))
 def test_plane_features_translate_invariant(self):
  drawer=trimesh.creation.box(extents=[1,1,1]);body=trimesh.creation.box(extents=[1,1,2]);a=geometry_features(drawer,body,[0,0,1]);drawer.apply_translation([3,4,5]);body.apply_translation([3,4,5]);b=geometry_features(drawer,body,[0,0,1]);self.assertAlmostEqual(a['front_alignment_q'],b['front_alignment_q']);self.assertAlmostEqual(a['drawer_depth'],b['drawer_depth'])
 def test_observation_is_hard_constraint(self):
  f=self.feature();before=copy.deepcopy(f);q=np.linspace(-1,3,241);fit=solve_interval(f,DEFAULT,1.2,q,np.zeros_like(q));lo,hi=fit['outward_range'];self.assertLessEqual(lo,1.2+1e-9);self.assertGreaterEqual(hi,1.2-1e-9);self.assertGreaterEqual(hi-lo,.1-1e-8);self.assertEqual(f,before)
 def test_without_observation_does_not_treat_it_as_endpoint(self):
  q=np.linspace(-1,3,241);fit=solve_interval(self.feature(),DEFAULT,1.2,q,np.zeros_like(q),enforce_observation=False);self.assertLess(fit['outward_range'][1],1.2)
 def test_no_collision_does_not_collapse_interval(self):
  q=np.linspace(-1,3,241);fit=solve_interval(self.feature(),DEFAULT,.5,q,np.zeros_like(q));lo,hi=fit['outward_range'];self.assertLess(abs(lo),.01);self.assertGreater(hi-lo,.5);self.assertLess(hi-lo,1.)
 def test_front_alignment_is_minimal_at_flush(self):
  f=self.feature();q=np.linspace(-1,3,241);a,ta=losses([0,.8],f,DEFAULT,.5,q,np.zeros_like(q));b,tb=losses([.2,1.],f,DEFAULT,.5,q,np.zeros_like(q));self.assertEqual(ta['front_alignment'],0.);self.assertGreater(tb['front_alignment'],0.)
if __name__=='__main__':unittest.main()
