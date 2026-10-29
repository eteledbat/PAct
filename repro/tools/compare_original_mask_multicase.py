"""Post-seal comparison only: native old scores and same-registration old/new.
No parameter optimization, GT fitting, or checkpoint selection in this module.
"""
import argparse,json,pathlib,copy,sys,types
import imageio.v2 as imageio
import numpy as np,torch
from PIL import Image,ImageDraw
from original_mask_multicase import CASES,BASE,REPO,OriginalMaskModel,load_model,mask_scores,read,FONT,COLORS
from multicase_closed_common import Model,tensor,image_metrics,write_json
from collision_refinement_common import sha256

@torch.no_grad()
def score_at(model,geometry,target,mask,ext,intr):
 rgb=model.render(geometry,extrinsics=ext,intrinsics=intr,reference=True);alpha=model.render(geometry,'silhouette',extrinsics=ext,intrinsics=intr,reference=True)
 return image_metrics(rgb,alpha,target,mask),rgb

def predicted_mask(model,geometry):
 probabilities=OriginalMaskModel.probabilities(model,geometry,reference=True).cpu().numpy();indices=np.concatenate([(1-probabilities.sum(0)).clip(0,1)[None],probabilities]).argmax(0);ids=np.array([0,*model.visible_owners])[indices];rgb=np.full((*ids.shape,3),255,np.uint8)
 for gid in model.visible_owners:rgb[ids==gid]=(255*np.array(COLORS[gid])).astype(np.uint8)
 return Image.fromarray(rgb)

def compare_case(root):
 assert all((root.parent/c/'fit_results.json').exists() for c in CASES)
 torch.set_num_threads(4);case=root.name;new=load_model(root,checkpoint=True);new.requires_grad_(False);oldroot=BASE/case;old=Model(read(oldroot/'config.json'),read(oldroot/'motion_groups.json'),oldroot);old.load_state_dict(torch.load(oldroot/'checkpoint.pt',map_location='cuda',weights_only=False)['model']);old.requires_grad_(False)
 old.visible_owners=new.visible_owners;old.owner_map={g['id']:(g['shape_owner'] if g['shape_owner'] is not None else g['id']) for g in old.groups};old.targets=new.targets;old_native_translation_basis=old.translation_basis.clone();old.translation_basis=new.translation_basis;old.extrinsics=new.extrinsics;old.intrinsics=new.intrinsics;assert torch.allclose(old.R,new.R);old.probabilities=types.MethodType(OriginalMaskModel.probabilities,old);old.closed_transforms={g['id']:np.eye(4) for g in old.groups}
 # This copies the already frozen new camera-plane scale/translation, and never
 # optimizes any old/new model. All metrics below use the same camera transform.
 with torch.no_grad():
  old.log_global_scale.copy_(new.log_global_scale);old.screen_translation.copy_(new.screen_translation)
 native=read(oldroot/'evaluation.json');evaluation=read(root/'evaluation.json');geometries={'Origin':(new,new.geometry(original=True)),'Old motion prior':(old,old.geometry(original=True)),'Old closed-image fit':(old,old.geometry()),'New motion prior':(new,new.geometry(original=True)),'New original-mask fit':(new,new.geometry())};scores={};pics=[]
 with torch.no_grad():
  for title,(model,geometry) in geometries.items():
   scores[title],rgb=score_at(model,geometry,new.rgb,new.alpha,new.extrinsics,new.intrinsics);scores[title]['part_mask']=mask_scores(model,geometry);pics.append(Image.fromarray((rgb.clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8)))
  assert abs(scores['Origin']['IoU']-scores['Old motion prior']['IoU'])<.002,'Unchanged source geometry failed common registration sanity check'
  if case!='100481':assert abs(scores['Old motion prior']['part_mask']['mean_IoU']-scores['New motion prior']['part_mask']['mean_IoU'])<.002,'Unchanged9-case physical grouping failed same-camera part-mask sanity check'
  # Native/published old-registration score at ORIGINAL conditioning q0, to show
  # registration sensitivity separately from the shared-registration comparison.
  ck=torch.load(oldroot/'checkpoint.pt',map_location='cuda',weights_only=False)['model']
  old.translation_basis=old_native_translation_basis;old.log_global_scale.copy_(ck['log_global_scale']);old.screen_translation.copy_(ck['screen_translation'])
  old_original_input_native,_=score_at(old,old.geometry(),new.rgb,new.alpha,new.extrinsics,new.intrinsics)
  old.translation_basis=new.translation_basis;old.log_global_scale.copy_(new.log_global_scale);old.screen_translation.copy_(new.screen_translation)
 novel=[];novel_panels=[]
 if case!='20411':
  cfg=new.config;scene=pathlib.Path(cfg['source_scene']);camera=read(scene/'transforms.json')
  for index in [25,50,75,100,125,149]:
   frame=camera['frames'][index];c2w=np.array(frame['transform_matrix']);c2w[:3,1:3]*=-1;ext=tensor(np.linalg.inv(c2w));image=np.array(Image.open(scene/frame['file_path']).convert('RGBA').resize((256,256),Image.Resampling.LANCZOS)).astype(np.float32)/255.;target=tensor(image[:,:,:3]*image[:,:,3:4]+1-image[:,:,3:4]).permute(2,0,1);mask=tensor(image[:,:,3]);row={'frame':index};panels=[Image.fromarray((target.permute(1,2,0).cpu().numpy()*255).astype(np.uint8))]
   with torch.no_grad():
    for title,(model,geometry) in geometries.items():
     row[title],rgb=score_at(model,geometry,target,mask,ext,new.intrinsics);panels.append(Image.fromarray((rgb.clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8)))
   novel.append(row)
   if index in [25,75,125]:novel_panels.append(panels)
 novelmean={title:{k:float(np.mean([r[title][k] for r in novel])) for k in ['IoU','PSNR','foreground_PSNR','foreground_L1']} for title in geometries} if novel else {}
 canvas=Image.new('RGB',(256*6,296),'white');d=ImageDraw.Draw(canvas);titles=['Input RGB',*geometries]
 targetpic=Image.fromarray((new.rgb.clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8))
 for j,(title,pic) in enumerate(zip(titles,[targetpic,*pics])):canvas.paste(pic,(j*256,40));d.text((j*256+3,5),title,font=FONT,fill='black')
 canvas.save(root/'old_new_original_input_comparison.png')
 if novel_panels:
  canvas=Image.new('RGB',(1536,40+256*len(novel_panels)),'white');d=ImageDraw.Draw(canvas)
  for j,title in enumerate(['Held-out RGB',*geometries]):d.text((j*256+3,5),title,font=FONT,fill='black')
  for row,panels in enumerate(novel_panels):
   for j,pic in enumerate(panels):canvas.paste(pic,(j*256,40+row*256))
  canvas.save(root/'old_new_shared_camera_unfitted_views.png')
 with torch.no_grad():
  maskpics=[Image.open(root/'approved_semantic_target.png').convert('RGB'),predicted_mask(old,old.geometry(original=True)),predicted_mask(old,old.geometry()),predicted_mask(new,new.geometry(original=True)),predicted_mask(new,new.geometry())]
 canvas=Image.new('RGB',(1280,296),'white');d=ImageDraw.Draw(canvas)
 for j,(title,pic) in enumerate(zip(['Exact original target','Old prior grouping','Old fit grouping','New motion prior','New geometry fit'],maskpics)):canvas.paste(pic,(j*256,40));d.text((j*256+3,5),title,font=FONT,fill='black')
 canvas.save(root/'old_new_part_mask_comparison.png')
 # Compare sealed old/new four-way videos with their own camera/phase policies
 # visibly labelled. First shared viewpoint only; the main new video has2 views.
 oldvideo=oldroot/f'{case}_four_versions_with_GT.mp4';newvideo=root/f'{case}_four_versions_with_GT.mp4';out=root/f'{case}_old_vs_original_mask_four_versions.mp4'
 with imageio.get_reader(oldvideo) as a,imageio.get_reader(newvideo) as b,imageio.get_writer(out,fps=16,codec='libx264',quality=8) as writer:
  assert a.count_frames()==b.count_frames()==84
  for i in range(84):
   img=Image.new('RGB',(1536,832),'white');d=ImageDraw.Draw(img);d.text((8,5),'OLD: closed-image fit | native old camera/phase | '+case,font=FONT,fill='black');d.text((8,421),'NEW: original-mask geometry fit | new shared camera/phase | '+case,font=FONT,fill='black');img.paste(Image.fromarray(a.get_data(i)[:384]),(0,32));img.paste(Image.fromarray(b.get_data(i)[:384]),(0,448));writer.append_data(np.array(img))
   if i in [6,22,38]:img.save(root/f'old_vs_original_mask_t{i:02d}.png')
 result=dict(case=case,old_published_evaluation=native,new_evaluation=evaluation,common_registration=dict(policy='Frozen new original-input RGB-only scale, translation coordinates AND camera-plane basis copied to old model; no refit. Same input camera and6 held-out reference-pose views. Old/new source geometry share canonical coordinates.',training_frame_metrics=scores,held_out_views=novel,held_out_mean=novelmean,registered_scale=float(new.log_global_scale.exp()),screen_translation=new.screen_translation.cpu().tolist(),translation_basis=new.translation_basis.cpu().tolist(),world_translation=(new.translation_basis@new.screen_translation).cpu().tolist()),old_fit_original_input_with_native_old_registration=old_original_input_native,part_mask_warning='Training-mask diagnostic; original/old stages are semantically recolored by their corrected physical ownership, NOT raw PAct semantic accuracy.100481 old physical grouping is knowingly incorrect.',video_comparison=dict(file=out.name,sha256=sha256(out),frames=84,warning='Each row retains its experiment camera and phase. Not a matched-pose pixel comparison; use static common-registration images/metrics for controlled comparison.'),old_checkpoint_sha256=sha256(oldroot/'checkpoint.pt'),new_checkpoint_sha256=sha256(root/'checkpoint.pt'))
 write_json(root/'baseline_comparison.json',result);print('COMPARED',case,json.dumps({title:v['IoU'] for title,v in novelmean.items()}),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--case',required=True);a=p.parse_args();compare_case(pathlib.Path(a.root)/a.case)
