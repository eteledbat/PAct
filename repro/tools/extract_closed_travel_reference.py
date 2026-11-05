"""Extract the established fully closed reference frame and projected 2D mask."""
import argparse,pathlib,shutil,sys
import numpy as np
from PIL import Image
REPO=pathlib.Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO/'repro/tools'))
from auto_part_semantics import sha256,write_json
from fit_observed_drawer_pose import read
from extract_motion_extrema import video_states,video_meshes,freeart_meshes,camera,rasterize
OLD=REPO/'repro/outputs/closed_multicase_20261028_v1'

def extract(batch,case):
    old=OLD/case;cfg=read(old/'config.json');frame=int(cfg['reference_frame'])
    scene=pathlib.Path(cfg['source_scene']);src=pathlib.Path(cfg['source_image'])
    assert src.exists() and np.array_equal(np.asarray(Image.open(src).convert('RGBA')),np.asarray(Image.open(old/'closed_reference.png').convert('RGBA')))
    root=batch/'closed_references'/case;root.mkdir(parents=True,exist_ok=True)
    dst=root/f'closed_frame{frame:06d}.png';shutil.copyfile(src,dst)
    img=Image.open(dst).convert('RGBA');is_freeart=case=='20411'
    if is_freeart:meshes=freeart_meshes(scene,frame)
    else:
        _,infos=video_states(scene);meshes=video_meshes(scene,frame,infos)
    ext,intr=camera(scene,frame,is_freeart,img.size)
    labels=rasterize(meshes,ext,intr,img.size)
    mask=root/'closed_mask.npy';np.save(mask,labels)
    alpha=np.asarray(img)[:,:,3]>=128;visible=labels>0
    iou=float((alpha&visible).sum()/max((alpha|visible).sum(),1))
    assert iou>.975,(case,frame,iou)
    manifest=dict(case=case,source_scene=str(scene),endpoints=[dict(frame=frame,rgb=str(dst),rgb_sha256=sha256(dst),
        mask=str(mask),mask_sha256=sha256(mask),silhouette_iou=iou,extrinsics=ext.tolist(),intrinsics=intr.tolist(),
        instance_pixels={str(i):int((labels==i).sum()) for i in range(1,len(meshes)+1)})],
        closed_reference_source=str(old/'closed_reference.png'),closed_reference_sha256=sha256(old/'closed_reference.png'),
        extraction='Established all-leaves-closed still from closed_multicase; same camera/instance-mask projection code as the two-extrema travel references.',
        GT_use='Only 2D projected visible instance mask and selection of closed frame during fit; no GT numeric axes or 3D meshes passed to optimizer.')
    write_json(root/'manifest.json',manifest);print('CLOSED',case,frame,'mask IoU',round(iou,5),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--case',required=True);a=p.parse_args();extract(a.root.resolve(),a.case)
