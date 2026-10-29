"""Geometric contract tests for reusable rules, independent of dataset IDs."""
import pathlib,unittest
import numpy as np
from scipy.spatial.transform import Rotation
from semantic_motion_rules import fit_attachment_axis,semantic_handle_role,top_hinged_door,resolve_parented_grips,connected_handle_contacts,paired_handle_component_axis,carrying_handle_outward_sign,handle_semantic_candidate,carrying_context_conflicts

class MotionRuleTests(unittest.TestCase):
 def test_handle_axis_is_not_forced_to_front(self):
  for direction in [np.array([0.,0.,1.]),np.array([1.,0.,0.]),np.array([1.,2.,3.])/np.sqrt(14)]:
   center=np.array([.2,.4,.1]);points=center+np.array([-1.,1.])[:,None]*direction
   pivot,d,audit=fit_attachment_axis(points,center,snap_degrees=0.)
   self.assertAlmostEqual(abs(d@direction),1.,places=12);self.assertLess(np.linalg.norm(np.cross(pivot-center,d)),1e-12)
 def test_door_and_drawer_grips_follow_parent(self):
  for role in ['door','drawer','upward_door']:
   result,_=semantic_handle_role('handle',role,[[-1,0,0],[1,0,0]],3.)
   self.assertEqual(result,'fixed_grip')
 def test_one_attachment_does_not_create_hinge(self):
  role,_=semantic_handle_role('handle','fixed_body',[[0,0,0]],1.)
  self.assertEqual(role,'uncertain')
  with self.assertRaises(ValueError):fit_attachment_axis([[0,0,0]])
 def test_top_hinged_door_lifts_up_and_out(self):
  v=np.array([[x,y,z] for x in [-1.,1.] for y in [0.,2.] for z in [-.01,.01]])
  body=np.array([[x,2.,z] for x in np.linspace(-1,1,20) for z in [-.01,.01]])
  j,B,length,a=top_hinged_door(v,body,dict(type='revolute',range=[0.,80.]))
  self.assertEqual(j['type'],'revolute');self.assertGreater(a['free_edge_height_change'],0.);self.assertGreater(a['free_edge_front_change'],0.)
  self.assertGreater(j['axis']['origin'][1],1.9);self.assertAlmostEqual(abs(np.linalg.det(B)),1.,places=10)
 def test_upward_direction_tracks_object_heading(self):
  v=np.array([[x,y,z] for x in [-1.,1.] for y in [0.,2.] for z in [-.01,.01]])
  body=np.array([[x,2.,z] for x in np.linspace(-1,1,20) for z in [-.01,.01]])
  R=Rotation.from_euler('y',27,degrees=True).as_matrix();j,_,_,_=top_hinged_door(v@R.T,body@R.T,dict(type='revolute',range=[0.,60.]))
  d=np.array(j['axis']['direction']);self.assertGreater(abs(d@(R@np.array([1.,0,0]))),.999)
 def test_fixed_grip_rule_changes_actual_tree(self):
  groups=[dict(id=4,name='drawer',parent=-1),dict(id=8,name='handle',parent=4,joint=dict(type='revolute'))]
  new,audit=resolve_parented_grips(groups)
  self.assertEqual(new[1]['joint']['type'],'fixed');self.assertEqual(new[1]['parent'],4);self.assertEqual(len(audit),1)
 def test_connected_u_has_two_supports(self):
  v=np.array([[x,y,z] for x in [-1.,1.] for y in np.linspace(0,2,40) for z in [-.02,.02]]+[[x,2.,0.] for x in np.linspace(-1,1,40)])
  body=np.array([[x,0.,z] for x in np.linspace(-1.1,1.1,40) for z in [-.05,.05]])
  points=connected_handle_contacts(v,body,3.);self.assertGreater(np.linalg.norm(points[0]-points[1]),1.8)
 def test_disconnected_carrying_handle_uses_two_long_support_bars(self):
  bars=[np.array([[x,y,z] for x in [-.02,.02] for y in np.linspace(0,.42,40) for z in [zc-.01,zc+.01]]) for zc in [-.31,.31]]
  pivot,direction,audit=paired_handle_component_axis(bars,1.4)
  self.assertGreater(abs(direction[2]),.999)
  self.assertLess(abs(pivot[1]-.21),.01)
  self.assertEqual(audit['support_component_indices'],[0,1])
 def test_single_support_bar_cannot_create_handle_axis(self):
  bar=np.array([[x,y,z] for x in [-.02,.02] for y in np.linspace(0,.42,40) for z in [-.01,.01]])
  with self.assertRaises(ValueError):paired_handle_component_axis([bar],1.4)
 def test_left_and_right_carrying_handles_swing_outward_oppositely(self):
  signs=[]
  for x in [-.25,.25]:
   bar=np.array([[x+dx,y,z] for dx in [-.02,.02] for y in np.linspace(0,.5,40) for z in [-.3,.3]])
   sign,_=carrying_handle_outward_sign(bar,[x,.25,0.],[0,0,1],[0,0,0],1.4)
   signs.append(sign)
  self.assertEqual(signs,[1.,-1.])
 def test_semantic_doors_are_not_handles(self):
  for name in ['left_door','right_drawer','upper_shelf']:
   self.assertFalse(handle_semantic_candidate(name,'door'))
 def test_model_or_declared_handle_can_propose_rule(self):
  self.assertTrue(handle_semantic_candidate('left_handle','door'))
  self.assertTrue(handle_semantic_candidate('unknown','handle'))
 def test_independent_handle_cannot_freeze_another_door(self):
  groups=[dict(id=1,joint=dict(type='revolute')),dict(id=2,joint=dict(type='prismatic'))]
  self.assertEqual(carrying_context_conflicts(groups,[1]),[2])
 def test_handle_context_allows_fixed_fragments(self):
  groups=[dict(id=1,joint=dict(type='revolute')),dict(id=2,joint=dict(type='fixed')),dict(id=3,semantic_owner=1,joint=dict(type='fixed'))]
  self.assertEqual(carrying_context_conflicts(groups,[1]),[])
 def test_reusable_engine_has_no_dataset_branches(self):
  source=(pathlib.Path(__file__).parent/'semantic_motion_rules.py').read_text()
  for token in ['100481','103015','root.name','config[\'case\']']:self.assertNotIn(token,source)

if __name__=='__main__':unittest.main()
