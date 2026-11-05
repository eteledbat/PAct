import sys,pathlib,json,argparse,numpy as np,trimesh,torch
from PIL import Image,ImageDraw
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from add_gt_multicase_videos import GT,MeshExtractResult,MeshRenderer,tensor,read,write,Rotation
parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);args=parser.parse_args();root=pathlib.Path(args.root).resolve();torch.set_num_threads(4)
renderer=MeshRenderer({'resolution':256,'near':.05,'far':30.,'ssaa':2})
results={}
for case in read(root/'protocol.json')['cases']:
 cfg=read(root/case/'config.json');gt=GT(root/case);scene=pathlib.Path(cfg['source_scene']);records=[];panels=[]
 if case=='20411':
  from evaluate_freeart_partnet_case import gt_parts
  states,_=gt_parts(scene);frames=[0];cam=read(pathlib.Path(cfg['source_camera']));ext=np.array(cam['extrinsic']);intr=np.array(cam['intrinsic_normalized'])
 else:
  frames=[0,cfg['reference_frame']];cam=read(scene/'transforms.json');info=read(scene/'gt/part_info.json');intr=np.array([[cam['focal_x']/cam['w'],0,cam['cx']/cam['w']],[0,cam['focal_y']/cam['h'],cam['cy']/cam['h']],[0,0,1]])
 for frame in frames:
  if case=='20411':
   vertices=np.concatenate([p.vertices for p in states[0]])@np.array(cfg['world_R']).T;target=scene/'00_pure.png'
  else:
   f=cam['frames'][frame];c2w=np.array(f['transform_matrix']);c2w[:3,1:3]*=-1;ext=np.linalg.inv(c2w);target=scene/f['file_path'];vs=[]
   for i,part in enumerate(gt.parts):
    v=np.asarray(part.vertices)
    if i:
     r=gt.records[i-1];axis=np.array(r['axis_world']);seq=info[i-1]['part_move'];move=np.array(seq[str(frame)]);channel=np.argmax(abs(np.array(seq['0'])));q=move[channel]*np.sign(axis[channel])-r['mesh_frame0_value']
     if r['type']=='revolute':
      o=np.array(r['origin_world']);v=(v-o)@Rotation.from_rotvec(q*axis).as_matrix().T+o
     else:v=v+q*axis
    vs.append(v)
   vertices=np.concatenate(vs)
  with torch.inference_mode():
   rep=MeshExtractResult(tensor(vertices),gt.faces,gt.attrs);out=renderer.render(rep,tensor(ext),tensor(intr),return_types=['color','mask','normal']);mask=out['mask'].cpu().numpy()>.5
   rgb=(out['color']*(.65+.35*abs(2*out['normal'][2:3]-1))+1-out['mask'].unsqueeze(0)).clamp(0,1);pic=(rgb.permute(1,2,0).cpu().numpy()*255).astype('uint8')
  image=np.array(Image.open(target).convert('RGBA').resize((256,256)));targetmask=image[:,:,3]>127;iou=(mask&targetmask).sum()/(mask|targetmask).sum()
  a=image[:,:,3:].astype(float)/255;reference=(image[:,:,:3]*a+255*(1-a)).astype('uint8')
  panel=Image.fromarray(np.concatenate([reference,pic],axis=1));d=ImageDraw.Draw(panel);d.rectangle((0,0,511,22),fill='white');d.text((4,4),f'{case} actual frame {frame} | RGB vs GT | mask IoU {iou:.3f}',fill='black');panels.append(np.array(panel));records.append(dict(frame=frame,silhouette_IoU=float(iou),source=str(target)))
 Image.fromarray(np.concatenate(panels,axis=0)).save(root/case/'gt_native_camera_pose_audit.png');results[case]=records;print(case,records,flush=True)
write(root/'GT_NATIVE_CAMERA_POSE_AUDIT.json',dict(protocol='Actual GT geometry at source frame 0 and fitting reference frame; native dataset cameras, no alignment fit. This validates only GT pose/camera conventions, not prediction metrics.',cases=results))
