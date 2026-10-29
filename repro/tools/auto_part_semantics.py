"""Recognize reference-mask instances without case-specific semantic recipes.

Uses a local, pinned vision-language model. Neither PAct weights nor geometry
are changed. Numeric joint parameters are deliberately excluded from prompts.
"""
import argparse
import datetime
import hashlib
import json
import os
import pathlib
import shutil
import time

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = pathlib.Path(__file__).resolve().parents[2]
SEMANTICS = {'base', 'door', 'drawer', 'handle', 'wheel', 'knob', 'shelf', 'tray', 'other', 'unknown'}
ROLES = {'fixed_body', 'fixed_grip', 'articulated_handle', 'side_door', 'upward_door', 'drop_door', 'sliding_door', 'drawer', 'wheel', 'knob', 'shelf', 'tray', 'unknown'}
CONFIDENCE = {'high', 'medium', 'low'}
PROMPT = '''Identify the EXACT TARGET component, not the whole object or nearby components.
Image 1: left is the target's local context with a red outline; right contains ONLY the target's visible pixels on white. The isolated right panel is the subject to classify. It can be a small handle even when the surrounding object is a cabinet.
Image 2 shows the whole object and an overlay with numbered masks, for context and parent IDs.
White missing regions are occlusion/background, not necessarily physical holes.
TARGET instance_id: {instance_id}. Valid instance IDs: {instance_ids}.
Identify its function from the images. Do not infer identity from ID, color, or order.
semantic must be one of base, door, drawer, handle, wheel, knob, shelf, tray, other, unknown.
base means the main body/frame/container of the object. A storage container is not automatically a drawer: drawer means a separate pull-out compartment within a larger body. Hinged shutters or glass window panels are door. handle includes both a fixed pull grip and a rotating carrying handle; distinguish their role.
motion_role must be one of fixed_body, fixed_grip, articulated_handle, side_door, upward_door, drop_door, sliding_door, drawer, wheel, knob, shelf, tray, unknown.
Do not assume every door rotates: a sliding door is possible. Door is a semantic category; hinge side and movement are separate hypotheses. If the hinge or motion is not visible, use motion_role unknown, hinge_visible false, attachment_edge unknown. Do not invent a hinge based only on furniture stereotypes.
parent_instance_id is an existing instance ID for the containing/mounting part, or null when unclear. A fixed grip on a door/drawer moves with that parent; a carrying handle rotates at supports on the body.
Return ONE JSON object, no markdown. Keys: instance_id, semantic, semantic_confidence (high/medium/low), motion_role, motion_confidence (high/medium/low), parent_instance_id, hinge_visible (true/false), attachment_edge (left/right/upper/lower/unknown), evidence (one short sentence describing visible evidence). Never output numeric axes, angles, or distances. Confidence is a qualitative self-assessment, not a calibrated probability.'''


def sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = pathlib.Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def parse_result(text, instance_id, valid_ids):
    """Reject invented IDs and invalid output; never silently assign a class."""
    decoder = json.JSONDecoder()
    obj = None
    for start, c in enumerate(text):
        if c == '{':
            try:
                value, _ = decoder.raw_decode(text[start:])
                if isinstance(value, dict):
                    obj = value
                    break
            except json.JSONDecodeError:
                pass
    if obj is None:
        raise ValueError('No valid JSON object in model response')
    if type(obj.get('instance_id')) is not int or obj['instance_id'] != instance_id:
        raise ValueError('Response instance ID differs from target')
    if obj.get('semantic') not in SEMANTICS or obj.get('motion_role') not in ROLES:
        raise ValueError('Unknown semantic/role vocabulary')
    if any(obj.get(k) not in CONFIDENCE for k in ['semantic_confidence', 'motion_confidence']):
        raise ValueError('Invalid qualitative confidence')
    parent = obj.get('parent_instance_id')
    if parent is not None and (type(parent) is not int or parent not in valid_ids or parent == instance_id):
        raise ValueError('Invalid or self parent ID')
    if type(obj.get('hinge_visible')) is not bool or obj.get('attachment_edge') not in {'left', 'right', 'upper', 'lower', 'unknown'}:
        raise ValueError('Invalid attachment evidence')
    if not isinstance(obj.get('evidence'), str):
        raise ValueError('Missing visual evidence')
    return {k: obj[k] for k in ['instance_id', 'semantic', 'semantic_confidence', 'motion_role', 'motion_confidence', 'parent_instance_id', 'hinge_visible', 'attachment_edge', 'evidence']}


def unknown_result(instance_id, reason):
    return dict(instance_id=instance_id, semantic='unknown', semantic_confidence='low', motion_role='unknown', motion_confidence='low', parent_instance_id=None, hinge_visible=False, attachment_edge='unknown', evidence=reason)


def prepare_case(out, item):
    import cv2
    out.mkdir()
    rgba = np.asarray(Image.open(item['rgb']).convert('RGBA')).astype(float) / 255.
    rgb = (255 * (rgba[:, :, :3] * rgba[:, :, 3:] + 1 - rgba[:, :, 3:])).round().astype(np.uint8)
    raw = cv2.imread(item['mask'], cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError('Cannot read original mask')
    raw = raw if raw.ndim == 2 else raw[:, :, 0]
    ids = raw.astype(np.int32)
    if ids.shape != rgb.shape[:2] or not np.array_equal(raw, ids):
        raise ValueError('Mask/image size or integer label mismatch')
    valid = sorted(int(x) for x in np.unique(ids) if x > 0)
    np.save(out/'instance_ids.npy', ids)
    Image.fromarray(rgb).save(out/'reference.png')
    colors = np.array([[233,72,61],[57,125,236],[45,178,97],[188,77,211],[235,166,39],[25,166,183],[165,111,78],[140,150,40]])
    overlay = rgb.copy()
    for i, owner in enumerate(valid):
        mask = ids == owner
        overlay[mask] = (.7 * rgb[mask] + .3 * colors[i % len(colors)]).astype(np.uint8)
    overlay = Image.fromarray(overlay)
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 23)
    for owner in valid:
        yy, xx = np.where(ids == owner)
        # Put the ID inside the visible region, even for rings/nonconvex masks.
        target = np.array([np.median(xx), np.median(yy)])
        i = np.argmin((xx-target[0])**2 + (yy-target[1])**2)
        x, y = int(xx[i]), int(yy[i])
        draw.rectangle((x-15,y-15,x+20,y+16), fill='white', outline='black')
        draw.text((x-9,y-13), str(owner), fill='black', font=font)
    overlay.save(out/'numbered_masks.png')
    full = Image.new('RGB', (1024, 512), 'white')
    full.paste(Image.fromarray(rgb).resize((512,512)), (0,0))
    full.paste(overlay.resize((512,512)), (512,0))
    full.save(out/'object_context.png')
    for owner in valid:
        mask = ids == owner
        yy, xx = np.where(mask)
        pad = max(15, int(.15 * max(np.ptp(xx), np.ptp(yy))))
        box = (max(0,int(xx.min())-pad),max(0,int(yy.min())-pad),min(rgb.shape[1],int(xx.max())+pad+1),min(rgb.shape[0],int(yy.max())+pad+1))
        contour = mask & ~cv2.erode(mask.astype(np.uint8), np.ones((3,3),np.uint8)).astype(bool)
        context = rgb.copy();context[contour] = [255,0,0]
        isolated = np.full_like(rgb,255);isolated[mask] = rgb[mask]
        view = Image.new('RGB',(768,414),'white');d=ImageDraw.Draw(view)
        d.text((8,5),f'TARGET {owner}: context / isolated visible pixels',fill='black',font=font)
        for j, pic in enumerate([context,isolated]):
            chip=Image.fromarray(pic).crop(box);chip.thumbnail((384,384))
            view.paste(chip,(384*j+(384-chip.width)//2,30+(384-chip.height)//2))
        view.save(out/f'instance_{owner:03d}.png')
    # Only ID correspondence is retained; no GT part names, joints or geometry.
    prov=json.loads(pathlib.Path(item['provenance']).read_text())
    correspondence={str(x['input_label']):x['pact_node_id'] for x in prov['part_correspondence']}
    metadata=dict(instance_ids=valid,source_image=str(pathlib.Path(item['rgb']).resolve()),source_mask=str(pathlib.Path(item['mask']).resolve()),source_run=item['run'],source_image_sha256=sha256(item['rgb']),source_mask_sha256=sha256(item['mask']),input_label_to_native_node=correspondence,mask_boundaries_changed=False)
    write_json(out/'inputs.json',metadata)


def recognize(root, model_path):
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    torch.manual_seed(42);torch.set_num_threads(4)
    processor=AutoProcessor.from_pretrained(model_path,local_files_only=True,min_pixels=128*28*28,max_pixels=448*448)
    model=AutoModelForImageTextToText.from_pretrained(model_path,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='eager').to('cuda').eval()
    protocol=json.loads((root/'protocol.json').read_text())
    for case in protocol['cases']:
        out=root/case;inputs=json.loads((out/'inputs.json').read_text());results=[]
        for owner in inputs['instance_ids']:
            result_path=out/f'recognition_{owner:03d}.json'
            if result_path.exists():
                results.append(json.loads(result_path.read_text()));continue
            prompt=PROMPT.format(instance_id=owner,instance_ids=inputs['instance_ids'])
            pictures=[out/f'instance_{owner:03d}.png',out/'object_context.png']
            images=[Image.open(p).convert('RGB') for p in pictures]
            messages=[{'role':'user','content':[{'type':'image','image':im} for im in images]+[{'type':'text','text':prompt}]}]
            text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            batch=processor(text=[text],images=images,padding=True,return_tensors='pt').to('cuda')
            started=time.monotonic()
            with torch.inference_mode():generated=model.generate(**batch,max_new_tokens=320,do_sample=False)
            raw=processor.batch_decode(generated[:,batch['input_ids'].shape[1]:],skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
            (out/f'raw_{owner:03d}.txt').write_text(raw+'\n')
            error=None
            try:result=parse_result(raw,owner,inputs['instance_ids'])
            except ValueError as exc:error=str(exc);result=unknown_result(owner,'Invalid structured response: '+error)
            result.update(inference_seconds=time.monotonic()-started,validation_error=error,raw_sha256=sha256(out/f'raw_{owner:03d}.txt'),prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),image_sha256={p.name:sha256(p) for p in pictures})
            write_json(result_path,result);results.append(result)
            print(case,owner,result['semantic'],result['motion_role'],round(result['inference_seconds'],2),flush=True)
        write_json(out/'semantics.json',dict(parts=results,source='local_vlm_reference_masks',native_node_mapping=inputs['input_label_to_native_node'],manual_semantic_annotations_used=False,geometry_changed=False))
    write_json(root/'RECOGNITION_SEAL.json',dict(state='complete',files={str(p.relative_to(root)):sha256(p) for c in protocol['cases'] for p in (root/c).glob('recognition_*.json')},GT_joint_inputs=False,manual_semantic_annotations_used=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=pathlib.Path);p.add_argument('--inventory',type=pathlib.Path);p.add_argument('--cases',nargs='+');p.add_argument('--stage',choices=['prepare','recognize'],required=True);p.add_argument('--model',type=pathlib.Path,default=REPO/'repro/cache/Qwen3-VL-4B-Instruct');a=p.parse_args()
    if a.stage=='prepare':
        items=json.loads(a.inventory.read_text());items=[i for i in items if not a.cases or i['case'] in a.cases];a.root.mkdir(exist_ok=False)
        pinned=a.model/'pinned_download_metadata.json'
        if not pinned.exists():raise ValueError('Missing pinned model metadata; run download_semantic_vlm.py --repo first')
        model_info=json.loads(pinned.read_text())
        protocol=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cases=[i['case'] for i in items],model=model_info['model'],revision=model_info['revision'],geometry_policy='Original PAct geometry unchanged. Automatic semantics used only by axis router.',observation_policy='Original RGB and existing instance masks only. No GT joints, no manual semantic labels, no closed/open target images.',confidence_policy='Ordinal model self-report, not calibrated probability',prompt=PROMPT,source_sha256=sha256(__file__))
        write_json(a.root/'protocol.json',protocol);shutil.copyfile(__file__,a.root/pathlib.Path(__file__).name)
        for item in items:prepare_case(a.root/item['case'],item)
        print('Prepared',len(items),'cases',flush=True)
    else:recognize(a.root,a.model)


if __name__=='__main__':main()
