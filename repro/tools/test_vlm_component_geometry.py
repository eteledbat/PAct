"""Contracts for replacing manual shape ownership with automatic instances."""
import unittest
import numpy as np
from fit_vlm_component_geometry import automatic_owners,semantic_target

class AutomaticGeometryTests(unittest.TestCase):
 def test_shared_native_node_can_be_two_independent_semantic_doors(self):
  groups=[dict(id=10,parent=-1,semantic_owner=1,source_node=1),dict(id=11,parent=10,semantic_owner=1,source_node=3),dict(id=12,parent=10,semantic_owner=2,source_node=0),dict(id=13,parent=12,semantic_owner=2,source_node=2),dict(id=14,parent=10,semantic_owner=3,source_node=0)]
  self.assertEqual(automatic_owners(groups),{1:10,2:12,3:14})
 def test_ambiguous_two_roots_are_rejected(self):
  with self.assertRaises(ValueError):automatic_owners([dict(id=10,parent=-1,semantic_owner=1),dict(id=11,parent=-1,semantic_owner=1)])
 def test_unsupported_instance_is_ignored_not_background(self):
  ids=np.array([[0,1],[2,3]],np.int32);labels=semantic_target(ids,{1:10,2:12},2);self.assertTrue(np.array_equal(labels,np.array([[0,10],[12,-1]])))
 def test_supported_labels_preserve_original_mask_boundaries(self):
  ids=np.array([[0,1,1],[2,2,3],[3,0,3]],np.int32);mapping={1:10,2:15,3:24};labels=semantic_target(ids,mapping,3);back=np.zeros_like(ids)
  for source,dest in mapping.items():back[labels==dest]=source
  self.assertTrue(np.array_equal(back,ids))
 def test_unresolved_fragments_do_not_become_semantic_instances(self):
  self.assertEqual(automatic_owners([dict(id=10,parent=-1,semantic_owner=1),dict(id=11,parent=-1,semantic_owner=None)]),{1:10})

if __name__=='__main__':unittest.main()
