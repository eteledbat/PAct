"""Check semantic ownership conservation, frozen stages, and output integrity."""
import argparse,json,pathlib,sys
import numpy as np
import torch
from semantic_parts_ablation import SemanticModel,read,BASE
from multicase_closed_common import tensor,write_json
from collision_refinement_common import sha256,transforms
p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=pathlib.Path(a.root).resolve();proto=read(root/'protocol.json');checks={}
protected=read(root/'baseline_preservation.json');assert all(sha256(pathlib.Path(p))==digest for p,digest in protected.items())
for case in proto['cases']:
 out=root/case;cfg=read(out/'config.json');manifest=read(out/'motion_groups.json');fit=read(out/'fit_results.json');evaluation=read(out/'evaluation.json');before=read(BASE/case/'evaluation.json')
 assert not fit['GT_forbidden_reads'] and not read(out/'motion_prior_search.json')['GT_forbidden_reads']
 for key,name in [('checkpoint_sha256','checkpoint.pt'),('optimized_parts_sha256','optimized_parts.npz'),('config_sha256','config.json'),('manifest_sha256','motion_groups.json')]:assert sha256(out/name)==fit[key]
 assert all(sha256(pathlib.Path(f['path']))==f['sha256'] for f in manifest['geometry_files'])
 assert evaluation['alignment']==before['alignment']
 for metric in ['IoU','PSNR','foreground_PSNR','foreground_L1']:
  assert abs(evaluation['fit_image_metrics']['original'][metric]-before['fit_image_metrics']['original'][metric])<1e-5
 for metric in ['IoU','PSNR','foreground_PSNR','foreground_L1']:
  assert abs(evaluation['unfitted_open_mean']['original'][metric]-before['unfitted_open_mean']['original'][metric])<1e-5
 assert evaluation['joint_methods']['prior']==evaluation['joint_methods']['optimized'] or all(evaluation['joint_methods']['prior'][key]==evaluation['joint_methods']['optimized'][key] for key in ['axis_correct_count','conditional_mean_angle_degrees','conditional_mean_pivot_percent_D'])
 model=SemanticModel(cfg,manifest,out);model.load_state_dict(torch.load(out/'checkpoint.pt',map_location='cuda',weights_only=False)['model']);model.requires_grad_(False)
 source={part['node']['id']:part['mesh'] for part in model.asset['parts']};partitions=[];error=0.;rigid=0.
 for nid,mesh in source.items():
  groups=[g for g in model.groups if g['source_node']==nid]
  for suffix,n in [('vertices',len(mesh.vertices)),('faces',len(mesh.faces))]:
   ids=np.concatenate([model.membership[g['membership_prefix']+'_'+suffix] for g in groups]);assert np.array_equal(np.sort(ids),np.arange(n))
  ids=np.concatenate([model.membership[g['membership_prefix']+'_gaussians'] for g in groups]);assert np.array_equal(np.sort(ids),np.arange(len(ids)))
  partitions.append(dict(source=nid,vertices=len(mesh.vertices),faces=len(mesh.faces),gaussians=len(ids)))
 for part in model.parts:
  g=part['group'];vertices=np.asarray(source[g['source_node']].vertices)[model.membership[g['membership_prefix']+'_vertices']]
  restored=part['vertex_local']@part['B'].cpu().numpy().T+part['anchor'].cpu().numpy();error=max(error,float(abs(restored-vertices).max()))
 assert error<1e-6
 for t in np.linspace(0,1,21):
  tr=transforms(model.groups,t)
  for g in model.groups:
   if g['joint']['type']=='fixed' and g['parent']!=-1:
    assert np.allclose(tr[g['id']],tr[g['parent']],atol=1e-12)
  for part in model.parts:
   with torch.no_grad():xyz,_=model.deform(model.indices[part['group']['id']])
   points=xyz[::997].cpu().numpy().astype(float);T=tr[part['group']['id']];moved=points@T[:3,:3].T+T[:3,3];rigid=max(rigid,float(abs(np.linalg.norm(points-points[0],axis=1)-np.linalg.norm(moved-moved[0],axis=1)).max()))
 assert rigid<1e-12
 videos=[]
 for file in ['render_results.json','gt_video_results.json','semantic_video_results.json']:
  data=read(out/file)
  for v in data['videos']:
   assert sha256(out/v['file'])==v['sha256'];assert v['frames']==84
   videos.append(v)
 assert len(videos)==11
 checks[case]=dict(source_partitions=partitions,max_rest_vertex_reconstruction_error=error,max_rigid_length_error=rigid,fixed_children_follow_parent=True,original_geometry_and_origin_metrics_unchanged=True,old_and_new_static_GT_alignment_identical=True,GT_forbidden_reads=[],video_count=len(videos),videos=videos,
  old_geometry_correspondence=before['correspondence'],semantic_geometry_correspondence=evaluation['correspondence'])
 del model
write_json(root/'VERIFICATION.json',dict(cases=checks,baseline_all_protected_hashes_match=True,protocol_sha256=sha256(root/'protocol.json'),source_sha256=sha256(__file__)))
print('VERIFIED',list(checks),flush=True)
