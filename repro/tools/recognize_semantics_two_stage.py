"""First classify an isolated component, then infer its mechanical role.

Semantic predictions are sealed before motion questions. No human case labels
or prior family recipes are read by either stage.
"""
import argparse
import hashlib
import json
import pathlib
import shutil
import time
import numpy as np
from PIL import Image,ImageOps
from auto_part_semantics import prepare_case,sha256,write_json,unknown_result,parse_result

REPO=pathlib.Path(__file__).resolve().parents[2]
SEMANTIC_PROMPT='''Classify the ENTIRE segmented assembly in Image 2. Image 1 is only whole-object context. If Images 3 and 4 are supplied, they show the same assembly and target in a second motion-extreme frame.
ALL non-white pixels in Image 2 belong to the ONE target, including attached accessories. Choose the assembly represented by ALL these pixels, not its most salient small subpart. A large panel with an attached grip is a panel assembly, not just a grip. A whole container/frame with internal structures is its body, not just one visible internal piece.
First describe the entire foreground: its overall shape and all major visible pieces. Then choose ONE category for that whole assembly.
Categories:
body = the main container, housing, carcass or frame of the object, including its integral walls, framing and internal fixed structures;
door = one closure panel/leaf, including any attached pull grips; window and shutter panels are included;
drawer = one pull-out storage compartment with its front and any attached grip;
handle = a grip/strap/carrying-handle assembly itself, when the larger mounting panel or container is NOT part of the isolated foreground;
wheel = rolling wheel assembly; knob = small control knob; shelf = one horizontal support board; tray = a shallow platform; unknown = insufficient evidence.
Return ONE JSON object in this key order: description, category, confidence. confidence is high, medium or low. Do not discuss motion, hinges, axes or parents.'''
RETRY_PROMPT='''Both images show ONLY ONE isolated target component at two articulation states. White is background, not part of the target. Classify ONLY the colored/non-white pixels. Do not classify the larger object to which this target is attached. A long strap or arch attached to a basket is a handle, not the basket body; a closure panel attached to a cabinet is a door, not the cabinet body. Categories: body, door, drawer, handle, wheel, knob, shelf, tray, unknown. Return one JSON object with description, category, confidence. Do not infer numeric motion parameters.'''
ROLE_PROMPT='''The target component was independently classified as {semantic}. Do not change that category here.
Image 1 shows this target in context and isolation. Image 2 shows the whole object with numbered instances. If Images 3 and 4 are supplied, they show the same target in the other motion-extreme frame. Use their visible displacement to infer mechanical role and attachment.
Target ID: {owner}. Recognized categories by ID: {categories}.
Determine its mechanical role only from visible evidence. Allowed roles: {roles}.
If motion/attachment is not visible, answer unknown rather than assuming a standard mechanism. Do not invent hinge hardware. A parent is a mounting part, not merely a nearby object.
Return ONE JSON object with: motion_role, motion_confidence (high/medium/low), parent_instance_id (valid other ID or null), hinge_visible (boolean), attachment_edge (left/right/upper/lower/unknown), evidence (one short sentence). Do not give numeric joint parameters.'''


def first_json(raw):
    decoder=json.JSONDecoder()
    for i,c in enumerate(raw):
        if c=='{':
            try:
                obj,_=decoder.raw_decode(raw[i:])
                if isinstance(obj,dict):return obj
            except json.JSONDecodeError:pass
    raise ValueError('No JSON object')


def canonical_category(value):
    """Map ordinary synonyms to the PAct ontology, without guessing a class."""
    name=' '.join(str(value).lower().replace('_',' ').replace('-',' ').split())
    aliases={'body':'base','base':'base','main body':'base','cabinet body':'base','housing':'base','frame':'base',
             'door':'door','window':'door','window panel':'door','shutter':'door','shutter panel':'door','cabinet door':'door','glass door':'door',
             'handle':'handle','grip':'handle','pull handle':'handle','carrying handle':'handle',
             'drawer':'drawer','wheel':'wheel','knob':'knob','shelf':'shelf','tray':'tray','unknown':'unknown'}
    if name not in aliases:raise ValueError('Unrecognized category vocabulary: '+name)
    return aliases[name]


def parse_category_response(raw,owner):
    obj=first_json(raw);category=canonical_category(obj['category'])
    confidence=obj.get('confidence')
    if isinstance(confidence,(int,float)) and not isinstance(confidence,bool) and 0<=confidence<=1:
        confidence='high' if confidence>=.8 else 'medium' if confidence>=.5 else 'low'
    if confidence not in {'high','medium','low'} or not isinstance(obj.get('description'),str):raise ValueError('Invalid category schema')
    return dict(instance_id=owner,semantic=category,semantic_confidence=confidence,description=obj['description'],raw_category=obj['category'],validation_error=None)


def focused_inputs(out):
    ids=np.load(out/'instance_ids.npy');rgb=Image.open(out/'reference.png').convert('RGB');array=np.asarray(rgb)
    for owner in sorted(int(x) for x in np.unique(ids) if x):
        yy,xx=np.where(ids==owner);pad=max(5,int(.07*max(np.ptp(xx),np.ptp(yy))));box=(max(0,int(xx.min())-pad),max(0,int(yy.min())-pad),min(rgb.width,int(xx.max())+pad+1),min(rgb.height,int(yy.max())+pad+1))
        isolated=np.full_like(array,255);isolated[ids==owner]=array[ids==owner]
        # contain scales up as well as down; tiny parts must not remain postage
        # stamps in a large white canvas.
        chip=ImageOps.contain(Image.fromarray(isolated).crop(box),(448,448),Image.Resampling.LANCZOS);canvas=Image.new('RGB',(448,448),'white');canvas.paste(chip,((448-chip.width)//2,(448-chip.height)//2));canvas.save(out/f'focused_{owner:03d}.png')


def second_extreme_inputs(out,item):
    if 'second_rgb' not in item:return
    from auto_part_semantics import write_json
    rgba=np.asarray(Image.open(item['second_rgb']).convert('RGBA')).astype(np.float32)/255
    rgb=(255*(rgba[:,:,:3]*rgba[:,:,3:4]+1-rgba[:,:,3:4])).round().astype(np.uint8)
    ids=np.load(item['second_mask'])
    assert ids.shape==rgb.shape[:2]
    first=np.load(out/'instance_ids.npy')
    assert set(np.unique(ids))==set(np.unique(first))
    Image.fromarray(rgb).save(out/'second_reference.png')
    palette=np.array([[233,72,61],[57,125,236],[45,178,97],[188,77,211],[235,166,39],[25,166,183]])
    overlay=rgb.copy()
    for j,owner in enumerate(int(x) for x in np.unique(ids) if x):
        mask=ids==owner;yy,xx=np.where(mask)
        assert len(xx)>0
        overlay[mask]=(.7*rgb[mask]+.3*palette[j%len(palette)]).astype(np.uint8)
        pad=max(5,int(.07*max(np.ptp(xx),np.ptp(yy))))
        box=(max(0,int(xx.min())-pad),max(0,int(yy.min())-pad),min(rgb.shape[1],int(xx.max())+pad+1),min(rgb.shape[0],int(yy.max())+pad+1))
        isolated=np.full_like(rgb,255);isolated[mask]=rgb[mask]
        chip=ImageOps.contain(Image.fromarray(isolated).crop(box),(448,448),Image.Resampling.LANCZOS)
        canvas=Image.new('RGB',(448,448),'white');canvas.paste(chip,((448-chip.width)//2,(448-chip.height)//2))
        canvas.save(out/f'focused_second_{owner:03d}.png')
        context=Image.new('RGB',(768,414),'white');context.paste(Image.fromarray(rgb).crop(box).resize((384,384)),(0,30));context.paste(chip.resize((384,384)),(384,30));context.save(out/f'instance_second_{owner:03d}.png')
    Image.fromarray(overlay).save(out/'second_numbered_masks.png')
    info=json.loads((out/'inputs.json').read_text());info.update(second_source_image=str(pathlib.Path(item['second_rgb']).resolve()),second_source_mask=str(pathlib.Path(item['second_mask']).resolve()),second_source_image_sha256=sha256(item['second_rgb']),second_source_mask_sha256=sha256(item['second_mask']))
    write_json(out/'inputs.json',info)


def run(root,model_path):
    import torch
    from transformers import AutoProcessor,AutoModelForImageTextToText
    torch.manual_seed(42);torch.set_num_threads(4)
    processor=AutoProcessor.from_pretrained(model_path,local_files_only=True,min_pixels=128*28*28,max_pixels=448*448)
    model=AutoModelForImageTextToText.from_pretrained(model_path,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='eager').to('cuda').eval()
    def query(paths,prompt,limit):
        images=[Image.open(p).convert('RGB') for p in paths];content=[{'type':'image'} for _ in images]+[{'type':'text','text':prompt}];text=processor.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True);batch=processor(text=[text],images=images,return_tensors='pt').to('cuda')
        with torch.inference_mode():output=model.generate(**batch,max_new_tokens=limit,do_sample=False)
        return processor.batch_decode(output[:,batch.input_ids.shape[1]:],skip_special_tokens=True)[0]
    cases=json.loads((root/'protocol.json').read_text())['cases']
    for case in cases:
        out=root/case;inputs=json.loads((out/'inputs.json').read_text())
        for owner in inputs['instance_ids']:
            dest=out/f'category_{owner:03d}.json'
            if dest.exists():continue
            started=time.monotonic();paths=[out/'reference.png',out/f'focused_{owner:03d}.png'];
            if (out/'second_reference.png').exists():paths += [out/'second_reference.png',out/f'focused_second_{owner:03d}.png']
            raw=query(paths,SEMANTIC_PROMPT,180);raw_path=out/f'raw_category_{owner:03d}.txt';raw_path.write_text(raw+'\n')
            try:
                result=parse_category_response(raw,owner)
            except (ValueError,KeyError,AssertionError) as exc:result=dict(instance_id=owner,semantic='unknown',semantic_confidence='low',description='Invalid category response',validation_error=type(exc).__name__)
            result.update(inference_seconds=time.monotonic()-started,raw_sha256=sha256(raw_path),image_sha256={p.name:sha256(p) for p in paths},prompt_sha256=hashlib.sha256(SEMANTIC_PROMPT.encode()).hexdigest());write_json(dest,result);print('CATEGORY',case,owner,result['semantic'],flush=True)
    for case in cases:
        out=root/case;inputs=json.loads((out/'inputs.json').read_text());items={owner:json.loads((out/f'category_{owner:03d}.json').read_text()) for owner in inputs['instance_ids']}
        bases=[owner for owner,item in items.items() if item['semantic']=='base']
        if len(bases)<=1:continue
        ids=np.load(out/'instance_ids.npy');primary=max(bases,key=lambda owner:int((ids==owner).sum()))
        for owner in bases:
            if owner==primary:continue
            paths=[out/f'focused_{owner:03d}.png']
            if (out/f'focused_second_{owner:03d}.png').exists():paths.append(out/f'focused_second_{owner:03d}.png')
            started=time.monotonic();raw_path=out/f'raw_category_retry_{owner:03d}.txt'
            if raw_path.exists():raw=raw_path.read_text()
            else:raw=query(paths,RETRY_PROMPT,160);raw_path.write_text(raw+'\n')
            try:retry=parse_category_response(raw,owner)
            except (ValueError,KeyError):retry=None
            previous=items[owner];previous['duplicate_body_retry']=dict(primary_body_id=primary,raw_sha256=sha256(raw_path),image_sha256={p.name:sha256(p) for p in paths},retry_result=retry,seconds=time.monotonic()-started)
            if retry is not None and retry['semantic']!='base':previous.update({k:retry[k] for k in ['semantic','semantic_confidence','description','raw_category','validation_error']})
            write_json(out/f'category_{owner:03d}.json',previous)
            print('RETRY CATEGORY',case,owner,previous['semantic'],flush=True)
    write_json(root/'CATEGORY_SEAL.json',dict(state='complete',files={str(p.relative_to(root)):sha256(p) for case in cases for p in (root/case).glob('category_*.json')}))
    for case in cases:
        out=root/case;inputs=json.loads((out/'inputs.json').read_text());categories=[json.loads((out/f'category_{owner:03d}.json').read_text()) for owner in inputs['instance_ids']];parts=[]
        for item in categories:
            owner=item['instance_id'];semantic=item['semantic'];dest=out/f'recognition_{owner:03d}.json'
            if dest.exists():
                previous=json.loads(dest.read_text())
                if previous.get('semantic')==semantic:parts.append(previous);continue
            result=unknown_result(owner,item['description']);result.update(semantic=semantic,semantic_confidence=item['semantic_confidence'],inference_seconds=item['inference_seconds'],category_sha256=sha256(out/f'category_{owner:03d}.json'),validation_error=item['validation_error'])
            if semantic in {'door','handle'} and item['semantic_confidence']!='low':
                roles=['side_door','upward_door','drop_door','sliding_door','unknown'] if semantic=='door' else ['fixed_grip','articulated_handle','unknown'];prompt=ROLE_PROMPT.format(semantic=semantic,owner=owner,categories={p['instance_id']:p['semantic'] for p in categories},roles=roles);paths=[out/f'instance_{owner:03d}.png',out/'numbered_masks.png'];
                if (out/'second_reference.png').exists():paths += [out/f'instance_second_{owner:03d}.png',out/'second_numbered_masks.png']
                started=time.monotonic();raw=query(paths,prompt,220);raw_path=out/f'raw_{owner:03d}.txt';raw_path.write_text(raw+'\n')
                try:
                    role=first_json(raw);candidate=dict(result,**role);candidate['semantic']=semantic;candidate['semantic_confidence']=item['semantic_confidence'];candidate['instance_id']=owner
                    if role.get('motion_role') not in roles:raise ValueError('Role outside category candidates')
                    validated=parse_result(json.dumps(candidate),owner,inputs['instance_ids']);result.update(validated)
                except (ValueError,KeyError) as exc:result['validation_error']=str(exc)
                result.update(inference_seconds=item['inference_seconds']+time.monotonic()-started,raw_sha256=sha256(raw_path),role_prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
            elif semantic in {'base','drawer','wheel','knob','shelf','tray'}:
                result.update(motion_role={'base':'fixed_body'}.get(semantic,semantic),motion_confidence=item['semantic_confidence'])
            write_json(dest,result);parts.append(result);print('ROLE',case,owner,result['semantic'],result['motion_role'],flush=True)
        write_json(out/'semantics.json',dict(parts=parts,source='local_vlm_two_stage_reference_masks',native_node_mapping=inputs['input_label_to_native_node'],manual_semantic_annotations_used=False,geometry_changed=False))
    write_json(root/'RECOGNITION_SEAL.json',dict(state='complete',category_seal_sha256=sha256(root/'CATEGORY_SEAL.json'),files={str(p.relative_to(root)):sha256(p) for c in cases for p in (root/c).glob('recognition_*.json')},GT_joint_inputs=False,manual_semantic_annotations_used=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--inventory',type=pathlib.Path);p.add_argument('--stage',choices=['prepare','recognize'],required=True);p.add_argument('--model',type=pathlib.Path,default=REPO/'repro/cache/Qwen3-VL-4B-Instruct');a=p.parse_args()
    if a.stage=='recognize':run(a.root,a.model);return
    items=json.loads(a.inventory.read_text());a.root.mkdir(exist_ok=False);model=json.loads((a.model/'pinned_download_metadata.json').read_text());write_json(a.root/'protocol.json',dict(cases=[i['case'] for i in items],model=model['model'],revision=model['revision'],source_sha256=sha256(__file__),semantic_prompt=SEMANTIC_PROMPT,role_prompt=ROLE_PROMPT,semantics_sealed_before_motion_questions=True,geometry_policy='No geometry change during semantic recognition',manual_semantics_used=False,observations='Two motion-extreme RGB frames and corresponding projected 2D part instance masks'))
    shutil.copyfile(__file__,a.root/pathlib.Path(__file__).name)
    for item in items:prepare_case(a.root/item['case'],item);focused_inputs(a.root/item['case']);second_extreme_inputs(a.root/item['case'],item)
    print('Prepared two-stage',len(items),'cases',flush=True)


if __name__=='__main__':main()
