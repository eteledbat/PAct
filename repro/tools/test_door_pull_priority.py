"""Regression checks: visible pull evidence outranks cabinet-height fallback."""
import unittest
import numpy as np
from auto_component_axis_prior import infer_door_family

class DoorPullPriorityTests(unittest.TestCase):
 def panel(self,height):return np.array([[x,y+height,z] for x in [-1.,1.] for y in [-.2,.2] for z in [-.01,.01]])
 def part(self,position='lower',**changes):
  pull=dict(valid=True,visible=True,confidence='high',position=position);pull.update(changes)
  return dict(motion_role='side_door',attachment_edge='left',motion_confidence='high',pull_evidence=pull)
 def test_lower_pull_on_low_panel_selects_upper_hinge(self):
  family,why=infer_door_family(self.part(),self.panel(-2.),np.array([[-2.,-3.,-1.],[2.,3.,1.]]));self.assertEqual(family,'hinge_upper');self.assertEqual(why,'reference_lower_pull_upper_hinge')
 def test_upper_pull_on_high_panel_selects_lower_hinge(self):
  family,_=infer_door_family(self.part('upper'),self.panel(2.),np.array([[-2.,-3.,-1.],[2.,3.,1.]]));self.assertEqual(family,'hinge_lower')
 def test_height_cannot_override_valid_pull(self):
  for height in [-2.,2.]:self.assertEqual(infer_door_family(self.part(),self.panel(height),np.array([[-2.,-3.,-1.],[2.,3.,1.]]))[0],'hinge_upper')
 def test_uncertain_or_middle_pull_uses_requested_position_fallback(self):
  for p in [self.part('middle'),self.part(valid=False),self.part(confidence='low'),self.part(visible=False)]:
   family,why=infer_door_family(p,self.panel(-2.),np.array([[-2.,-3.,-1.],[2.,3.,1.]]));self.assertEqual(family,'hinge_lower');self.assertTrue(why.startswith('cabinet_position_fallback'))
 def test_missing_pull_keeps_position_fallback(self):
  p=self.part();p.pop('pull_evidence');self.assertEqual(infer_door_family(p,self.panel(2.),np.array([[-2.,-3.,-1.],[2.,3.,1.]]))[0],'hinge_upper')
 def test_tall_side_door_is_not_changed_by_low_pull(self):
  v=np.array([[x,y,z] for x in [-.2,.2] for y in [-1.,1.] for z in [-.01,.01]]);self.assertEqual(infer_door_family(self.part(),v,np.array([[-2.,-3.,-1.],[2.,3.,1.]]))[0],'hinge_side')

if __name__=='__main__':unittest.main()
