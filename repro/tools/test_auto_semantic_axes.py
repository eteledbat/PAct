"""Safety and ambiguity contracts for semantic-only articulation selection."""
import json
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
from auto_part_semantics import parse_result,unknown_result
from semantic_axis_router import route_part,cyclic_parents,lower_hinge,geometry_correspondence
from recognize_semantics_two_stage import canonical_category

def observed(owner,semantic,role='unknown',**kwargs):
    p=unknown_result(owner,'test visual observation')
    p.update(semantic=semantic,semantic_confidence='high',motion_role=role)
    p.update(kwargs)
    return p

class AutomaticSemanticTests(unittest.TestCase):
    def test_window_synonyms_use_existing_door_ontology(self):
        for name in ['window','window_panel','shutter','door']:self.assertEqual(canonical_category(name),'door')
        self.assertEqual(canonical_category('body'),'base')

    def test_unrecognized_category_is_not_guessed(self):
        with self.assertRaises(ValueError):canonical_category('wooden board')

    def test_closed_door_label_does_not_force_rotation(self):
        door=observed(1,'door')
        r=route_part(door,[door])
        self.assertEqual(r['decision'],'retain_native')
        self.assertIn('slider_horizontal',r['candidate_families'])

    def test_upward_hinge_requires_explicit_visible_evidence(self):
        door=observed(1,'door','upward_door',motion_confidence='high',attachment_edge='upper')
        self.assertEqual(route_part(door,[door])['decision'],'retain_native')
        door['hinge_visible']=True
        self.assertEqual(route_part(door,[door])['family'],'hinge_upper')

    def test_fixed_grip_keeps_door_parent(self):
        door=observed(2,'door');handle=observed(7,'handle','fixed_grip',parent_instance_id=2)
        r=route_part(handle,[door,handle])
        self.assertEqual(r['family'],'fixed_grip');self.assertEqual(r['parent_instance_id'],2)

    def test_carrying_handle_still_needs_geometric_contacts(self):
        handle=observed(3,'handle','articulated_handle',motion_confidence='high')
        self.assertEqual(route_part(handle,[handle])['decision'],'check_geometry')

    def test_low_confidence_drawer_does_not_override_native(self):
        drawer=observed(5,'drawer',semantic_confidence='low')
        self.assertEqual(route_part(drawer,[drawer])['decision'],'retain_native')
        drawer['semantic_confidence']='medium'
        self.assertEqual(route_part(drawer,[drawer])['family'],'slider_front')

    def test_parent_cycle_is_rejected(self):
        a=observed(1,'handle','fixed_grip',parent_instance_id=2);b=observed(2,'door',parent_instance_id=1)
        self.assertEqual(cyclic_parents([a,b]),{1,2})
        self.assertEqual(route_part(a,[a,b])['decision'],'retain_native')

    def test_invented_instance_or_parent_is_rejected(self):
        row=unknown_result(2,'visible evidence')
        with self.assertRaises(ValueError):parse_result(json.dumps(row),1,[1,2])
        row.update(instance_id=1,parent_instance_id=99)
        with self.assertRaises(ValueError):parse_result(json.dumps(row),1,[1,2])

    def test_lower_hinge_really_moves_out_and_down(self):
        v=np.array([[x,y,z] for x in [-1.,1.] for y in [0.,2.] for z in [-.01,.01]])
        body=np.array([[x,0.,z] for x in np.linspace(-1,1,20) for z in [-.01,.01]])
        j,a=lower_hinge(v,body,dict(type='revolute',range=[0.,60.]))
        direction=np.array(j['axis']['direction']);pivot=np.array(j['axis']['origin']);free=v.mean(0)-pivot
        delta=Rotation.from_rotvec(direction*np.deg2rad(j['range'][1])).apply(free)-free
        self.assertGreater(delta[2],0);self.assertLess(delta[1],0);self.assertLess(a['free_edge_height_change'],0)

    def test_swapped_native_indices_are_rematched_from_projection(self):
        mesh=SimpleNamespace(vertices=np.zeros((64,3)))
        asset=dict(parts=[dict(node=dict(id=i),mesh=mesh) for i in [0,1]],diagonal=1.)
        labels=np.array([[1,1,2,2]]*4)
        uv=np.r_[np.tile([.75,.5],(64,1)),np.tile([0.,.5],(64,1))]
        with patch('semantic_axis_router.project_vertices',return_value=(uv,np.ones(128))):
            result=geometry_correspondence(asset,labels,None,{1:0,2:1})
        self.assertEqual(result[1]['source_node'],1);self.assertTrue(result[1]['verified'])
        self.assertEqual(result[2]['source_node'],0);self.assertTrue(result[2]['verified'])

    def test_shared_geometry_without_a_clear_instance_is_rejected(self):
        mesh=SimpleNamespace(vertices=np.zeros((64,3)))
        asset=dict(parts=[dict(node=dict(id=0),mesh=mesh)],diagonal=1.)
        labels=np.array([[1,1,2,2]]*4)
        uv=np.r_[np.tile([.75,.5],(32,1)),np.tile([0.,.5],(32,1))]
        with patch('semantic_axis_router.project_vertices',return_value=(uv,np.ones(64))):
            result=geometry_correspondence(asset,labels,None,{1:0,2:0})
        self.assertFalse(result[1]['verified']);self.assertFalse(result[2]['verified'])

if __name__=='__main__':unittest.main()
