"""Conservative spatial acceleration of the SAME full-triangle VTK checks.

Each triangle belongs to exactly one centroid-kd partition. Partition AABBs only
reject disjoint pairs; every remaining pair uses original VTK AllContacts and
zero tolerances. No decimation, triangle omission, hole filling or surrogate.
"""
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import trimesh
from inspect_triangle_motion import polydata
from collision_refinement_common import moved,transforms


class PartitionedTrianglePairs:
    def __init__(self,asset,selected=None,partition_size=12000):
        self.asset=asset;self.selected=selected;self.filters={};self.chunks=[]
        for part in asset['parts']:
            mesh=part['mesh'];centers=mesh.triangles_center;groups=[]
            def split(ids):
                if len(ids)<=partition_size:groups.append(ids);return
                c=centers[ids];axis=np.ptp(c,axis=0).argmax();order=np.argsort(c[:,axis],kind='stable');mid=len(ids)//2
                split(ids[order[:mid]]);split(ids[order[mid:]])
            split(np.arange(len(mesh.faces)))
            assert np.array_equal(np.sort(np.concatenate(groups)),np.arange(len(mesh.faces)))
            chunks=[]
            for ids in groups:
                faces=mesh.faces[ids];unique,inverse=np.unique(faces,return_inverse=True)
                sub=trimesh.Trimesh(mesh.vertices[unique],inverse.reshape(-1,3),process=False)
                assert np.array_equal(sub.vertices[sub.faces],mesh.vertices[faces])
                lo,hi=sub.bounds
                corners=np.array([[x,y,z] for x in [lo[0],hi[0]] for y in [lo[1],hi[1]] for z in [lo[2],hi[2]]])
                chunks.append({'poly':polydata(sub),'original_face_ids':ids,'corners':corners})
            self.chunks.append(chunks)

    def state(self,nodes,t,save_segments=False):
        world=transforms(nodes,float(t));D=self.asset['diagonal'];rows=[];snapshots=[]
        bboxes=[]
        for part,chunks in zip(self.asset['parts'],self.chunks):
            tr=world[part['node']['id']];box=[]
            for chunk in chunks:
                v=moved(chunk['corners'],tr);box.append([v.min(0),v.max(0)])
            bboxes.append(np.array(box))
        for i in range(len(self.chunks)):
            for j in range(i+1,len(self.chunks)):
                ids=[self.asset['parts'][k]['node']['id'] for k in [i,j]]
                if self.selected is not None and ids!=self.selected:continue
                counts=0;nonzero=0;total=0.;transverse=0.;saved=[]
                # Conservative expanded broadphase; final VTK tolerances stay zero.
                valid=np.all(bboxes[i][:,None,0]<=bboxes[j][None,:,1]+1e-12*D,axis=-1)&np.all(bboxes[j][None,:,0]<=bboxes[i][:,None,1]+1e-12*D,axis=-1)
                for ci,cj in np.argwhere(valid):
                    key=(i,j,int(ci),int(cj))
                    chunks=[self.chunks[i][ci],self.chunks[j][cj]]
                    if key not in self.filters:
                        f=vtk.vtkCollisionDetectionFilter();matrices=[vtk.vtkMatrix4x4(),vtk.vtkMatrix4x4()]
                        for k in [0,1]:matrices[k].Identity();f.SetInputData(k,chunks[k]['poly']);f.SetMatrix(k,matrices[k])
                        f.SetCollisionModeToAllContacts();f.SetBoxTolerance(0.);f.SetCellTolerance(0.);f.SetNumberOfCellsPerNode(2);f.SetGenerateScalars(0)
                        self.filters[key]=(f,matrices)
                    f,matrices=self.filters[key]
                    for mat,nid in zip(matrices,ids):
                        tr=world[nid]
                        for r in range(4):
                            for c in range(4):mat.SetElement(r,c,float(tr[r,c]))
                        mat.Modified()
                    f.Update();counts+=int(f.GetNumberOfContacts());output=f.GetContactsOutput()
                    if not output.GetNumberOfLines():continue
                    xyz=vtk_to_numpy(output.GetPoints().GetData());lines=vtk_to_numpy(output.GetLines().GetData()).reshape(-1,3)
                    assert np.all(lines[:,0]==2)
                    segments=xyz[lines[:,1:]];length=np.linalg.norm(segments[:,1]-segments[:,0],axis=1);keep=length>1e-6*D
                    total+=float(length[keep].sum());nonzero+=int(keep.sum())
                    fa=chunks[0]['original_face_ids'][vtk_to_numpy(f.GetContactCells(0))];fb=chunks[1]['original_face_ids'][vtk_to_numpy(f.GetContactCells(1))]
                    assert len(fa)==len(fb)==len(length)
                    na=self.asset['parts'][i]['mesh'].face_normals[fa]@world[ids[0]][:3,:3].T
                    nb=self.asset['parts'][j]['mesh'].face_normals[fb]@world[ids[1]][:3,:3].T
                    cross=keep&(abs((na*nb).sum(1))<np.cos(np.deg2rad(5)))
                    transverse+=float(length[cross].sum())
                    if save_segments and cross.any():saved.append(segments[cross])
                rows.append({'ids':ids,'contacting_triangle_pairs':counts,'nonzero_segments':nonzero,'line_length_over_diagonal':total/D,'transverse_line_length_over_diagonal':transverse/D})
                if saved:
                    seg=np.concatenate(saved)
                    if len(seg)>5000:seg=seg[np.argsort(np.linalg.norm(seg[:,1]-seg[:,0],axis=1))[-5000:]]
                    snapshots.append({'ids':ids,'segments':seg.astype(float).tolist()})
        return rows,snapshots
