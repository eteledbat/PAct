"""Locate door pull grips in unchanged reference masks, without inferring hinges.

Only original reference RGB, original instance masks and frozen category labels
are read. No generated joint type, family, GT, or cabinet-position prompt.
"""
import argparse,copy,datetime,json,os,pathlib,shutil,sys,time,hashlib
import numpy as np
from PIL import Image,ImageOps,ImageDraw,ImageFont
from auto_part_semantics import sha256,write_json
from recognize_semantics_two_stage import first_json
REPO=pathlib.Path(__file__).resolve().parents[2]
PROMPT='''Locate a user-operated PULL HANDLE or PULL KNOB attached to the TARGET DOOR PANEL.
Image 1 shows the isolated target door assembly. All non-white pixels belong to this target, including attached accessories. Image 2 shows the SAME crop with surrounding context. Image 3 is the whole original object, only for context.
Look only for a graspable pull/knob on this target, not a hinge, corner bracket, decorative groove, shelf edge, latch, or a handle on an adjacent door. A small round knob counts. Do NOT infer a handle from furniture stereotypes. If the pull is absent, occluded or not distinguishable, say visible false and confidence low.
Describe the visible pull, then give its bounding box IN IMAGE 1, using normalized coordinates from0 to1000: [left, top, right, bottom]. Do not give a box covering the entire panel. For multiple grips on this same door, give the box of their combined group only if their vertical location is consistent; otherwise visible false.
Do not infer hinges, motion, cabinet height, or opening direction. Report only visible pull location.
Return ONE JSON object: description (short visible evidence), visible (boolean), confidence (high/medium/low), bbox_1000 (4 integers, or null).'''

def guard():
 denied=[]
 def hook(event,args):
  if event=='open' and isinstance(args[0],(str,bytes)):
   path=os.fsdecode(args[0])
   if any(s in path for s in ['/gt/','/gt_mesh/','joint_evaluation','mobility_v2','part_info.json','/closed_multicase_','/general_semantic_priors_']):denied.append(path);raise PermissionError('Forbidden handle recognition input: '+path)
 sys.addaudithook(hook);return denied

def prepare(source,out):
 out.mkdir(exist_ok=False);protocol=json.loads((source/'protocol.json').read_text());protocol.update(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_semantics=str(source.resolve()),pull_recognition_prompt=PROMPT,source_semantics_unchanged=True,rule_priority='For horizontal-hinge candidate doors: visible lower pull -> upper hinge; visible upper pull -> lower hinge; absent/uncertain/middle pull -> cabinet-position fallback per user. Side-hinged doors retain side logic.',no_GT_during_recognition=True)
 write_json(out/'protocol.json',protocol);frozen={}
 for case in protocol['cases']:
  src=source/case;root=out/case;root.mkdir()
  for name in ['inputs.json','instance_ids.npy','reference.png','semantics.json']:
   shutil.copyfile(src/name,root/name);frozen[str(src/name)]=sha256(src/name)
  ids=np.load(root/'instance_ids.npy');rgb=Image.open(root/'reference.png').convert('RGB');array=np.array(rgb);sem=json.loads((root/'semantics.json').read_text())
  for part in sem['parts']:
   if part['semantic']!='door':continue
   owner=part['instance_id'];mask=ids==owner;yy,xx=np.where(mask);box=[int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1];isolated=np.full_like(array,255);isolated[mask]=array[mask];pic=ImageOps.contain(Image.fromarray(isolated).crop(box),(640,640),Image.Resampling.LANCZOS);context=ImageOps.contain(rgb.crop(box),(640,640),Image.Resampling.LANCZOS);pic.save(root/f'pull_target_{owner:03d}.png');context.save(root/f'pull_context_{owner:03d}.png');write_json(root/f'pull_crop_{owner:03d}.json',dict(original_bbox=box,resized_size=list(pic.size),original_mask_sha256=sha256(root/'instance_ids.npy')))
 write_json(out/'source_inputs_sha256.json',frozen);print('PREPARED',out,flush=True)

def parse_pull(raw,owner,root):
 obj=first_json(raw);visible=obj.get('visible');confidence=obj.get('confidence');box=obj.get('bbox_1000');description=obj.get('description')
 if type(visible)!=bool or confidence not in ['high','medium','low'] or not isinstance(description,str):raise ValueError('Invalid pull schema')
 result=dict(instance_id=owner,visible=visible,confidence=confidence,description=description,bbox_1000=box,relative_height=None,position='unknown',valid=False)
 if not visible:return result
 if not isinstance(box,list) or len(box)!=4 or not all(isinstance(n,(int,float)) and np.isfinite(n) for n in box):raise ValueError('Invalid grip box')
 x0,y0,x1,y1=map(float,box)
 if not(0<=x0<x1<=1000 and 0<=y0<y1<=1000) or (x1-x0)*(y1-y0)>.30*1e6:raise ValueError('Grip box is outside image or implausibly large')
 crop=json.loads((root/f'pull_crop_{owner:03d}.json').read_text());bb=crop['original_bbox'];cx=bb[0]+(x0+x1)/2000*(bb[2]-bb[0]);cy=bb[1]+(y0+y1)/2000*(bb[3]-bb[1]);ids=np.load(root/'instance_ids.npy');xi,yi=int(round(cx)),int(round(cy));on_target=0<=yi<ids.shape[0] and 0<=xi<ids.shape[1] and ids[yi,xi]==owner
 t=(y0+y1)/2000;position='upper' if t<=.35 else 'lower' if t>=.65 else 'middle';result.update(relative_height=t,position=position,center_original=[cx,cy],center_inside_original_target=bool(on_target),valid=bool(confidence=='high' and on_target),selection='Only high confidence, target-contained grips are used; top/bottom35%, middle ambiguous')
 return result

def recognize(root,model_path):
 forbidden=guard();import torch
 from transformers import AutoProcessor,AutoModelForImageTextToText
 torch.manual_seed(42);torch.set_num_threads(4)
 processor=AutoProcessor.from_pretrained(model_path,local_files_only=True,min_pixels=128*28*28,max_pixels=640*640)
 model=AutoModelForImageTextToText.from_pretrained(model_path,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='eager').to('cuda').eval();protocol=json.loads((root/'protocol.json').read_text());results=[]
 for case in protocol['cases']:
  dest=root/case;sem=json.loads((dest/'semantics.json').read_text())
  for part in sem['parts']:
   if part['semantic']!='door':continue
   owner=part['instance_id'];target=dest/f'pull_{owner:03d}.json'
   if target.exists():result=json.loads(target.read_text())
   else:
    paths=[dest/f'pull_target_{owner:03d}.png',dest/f'pull_context_{owner:03d}.png',dest/'reference.png'];images=[Image.open(p).convert('RGB') for p in paths];content=[{'type':'image'} for _ in images]+[{'type':'text','text':PROMPT}];text=processor.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True);batch=processor(text=[text],images=images,return_tensors='pt').to('cuda');start=time.monotonic()
    with torch.inference_mode():output=model.generate(**batch,max_new_tokens=220,do_sample=False)
    raw=processor.batch_decode(output[:,batch.input_ids.shape[1]:],skip_special_tokens=True)[0];rp=dest/f'raw_pull_{owner:03d}.txt';rp.write_text(raw+'\n')
    try:result=parse_pull(raw,owner,dest)
    except (ValueError,KeyError) as e:result=dict(instance_id=owner,visible=False,valid=False,position='unknown',confidence='low',validation_error=str(e),raw=raw)
    result.update(inference_seconds=time.monotonic()-start,model=str(model_path),prompt_sha256=hashlib.sha256(PROMPT.encode()).hexdigest(),image_sha256={p.name:sha256(p) for p in paths},raw_sha256=sha256(rp));write_json(target,result)
   part['pull_evidence']=result;results.append(dict(case=case,**result));print('PULL',case,owner,result.get('visible'),result.get('position'),result.get('confidence'),'valid',result.get('valid'),flush=True)
  sem['source']='frozen_automatic_categories_plus_reference_pull_localization';sem['pull_guided_policy']='User requested handle-position-first horizontal hinge inference; original masks unchanged';write_json(dest/'semantics.json',sem)
  # Original-mask visualization with automatic grip boxes, no selected mask edits.
  pic=Image.open(dest/'reference.png').convert('RGB');draw=ImageDraw.Draw(pic)
  for part in sem['parts']:
   e=part.get('pull_evidence',{});box=e.get('bbox_1000')
   if not box or not e.get('visible'):continue
   owner=part['instance_id'];bb=json.loads((dest/f'pull_crop_{owner:03d}.json').read_text())['original_bbox'];x0,y0,x1,y1=box;v=[bb[0]+x0/1000*(bb[2]-bb[0]),bb[1]+y0/1000*(bb[3]-bb[1]),bb[0]+x1/1000*(bb[2]-bb[0]),bb[1]+y1/1000*(bb[3]-bb[1])];draw.rectangle(v,outline='red' if e['valid'] else 'orange',width=3);draw.text((v[0],max(v[1]-18,0)),f"ID{owner}: {e['position']}",fill='red')
  pic.save(dest/'recognized_door_pulls.png')
 assert not forbidden
 write_json(root/'PULL_RECOGNITION_SEAL.json',dict(state='complete',results=results,GT_forbidden_reads=forbidden,source_tool_sha256=sha256(__file__),semantic_files={case:sha256(root/case/'semantics.json') for case in protocol['cases']},source_inputs_unchanged=all(sha256(p)==h for p,h in json.loads((root/'source_inputs_sha256.json').read_text()).items())))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',type=pathlib.Path);p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--stage',choices=['prepare','recognize'],required=True);p.add_argument('--model',type=pathlib.Path,default=REPO/'repro/cache/Qwen3-VL-4B-Instruct');a=p.parse_args();prepare(a.source,a.root) if a.stage=='prepare' else recognize(a.root,a.model)
