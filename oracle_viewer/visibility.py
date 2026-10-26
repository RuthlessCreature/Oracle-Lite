"""CAD-derived viewing frames and conservative marker-ray visibility screening.
These are geometry tools for the local planner, not optical or metrology acceptance.
"""
import numpy as np


def frame(normal):
 n=np.asarray(normal,float);n/=np.linalg.norm(n)
 reference=np.array([1.,0,0]) if abs(n[0])<.9 else np.array([0.,1,0])
 u=reference-n*np.dot(reference,n);u/=np.linalg.norm(u)
 return np.array([u,np.cross(n,u),n])


def closest_surface(point,triangles):
 triangles=np.asarray(triangles,np.float64)
 p=np.asarray(point,float);a=triangles[:,0];ab=triangles[:,1]-a;ac=triangles[:,2]-a
 normals=np.cross(ab,ac);length=np.linalg.norm(normals,axis=1);valid=length>1e-9
 if not valid.any():return None
 t=triangles[valid];a=t[:,0];ab=t[:,1]-a;ac=t[:,2]-a;n=normals[valid]/length[valid,None]
 signed=np.einsum('ij,ij->i',p-a,n);q=p-signed[:,None]*n;ap=q-a
 aa=np.einsum('ij,ij->i',ab,ab);bb=np.einsum('ij,ij->i',ac,ac);cc=np.einsum('ij,ij->i',ab,ac)
 d=np.sum(np.cross(ab,ac)**2,axis=1);v=(bb*np.einsum('ij,ij->i',ap,ab)-cc*np.einsum('ij,ij->i',ap,ac))/d
 w=(aa*np.einsum('ij,ij->i',ap,ac)-cc*np.einsum('ij,ij->i',ap,ab))/d
 inside=(v>=-1e-7)&(w>=-1e-7)&(v+w<=1+1e-7)
 distances=np.where(inside,signed*signed,np.inf);points=q.copy()
 for start,end in [(0,1),(1,2),(2,0)]:
  edge=t[:,end]-t[:,start];den=np.einsum('ij,ij->i',edge,edge)
  f=np.clip(np.einsum('ij,ij->i',p-t[:,start],edge)/np.maximum(den,1e-20),0,1)
  candidate=t[:,start]+f[:,None]*edge;dist=np.sum((candidate-p)**2,axis=1);better=dist<distances
  points[better]=candidate[better];distances[better]=dist[better]
 i=int(np.argmin(distances));return dict(point=points[i].tolist(),normal=n[i].tolist(),distance_mm=float(np.sqrt(distances[i])))


class ViewGeometry:
 def __init__(self,mesh,zones=()):
  self.faces={};self.plane_normals={};self.cylinder_axes={};self.flat_view_normals={};self.nodes=[];self.zones=zones;rows=[]
  for f in (mesh or {}).get('faces',[]):
   if not f.get('indices'):continue
   positions=np.asarray(f['positions'],np.float32).reshape(-1,3);tri=positions[np.asarray(f['indices'],int).reshape(-1,3)]
   tri=tri[np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)>1e-9]
   if not len(tri):continue
   self.faces[f['id']]=tri;rows.append(tri)
   normals=np.cross(tri[:,1].astype(float)-tri[:,0],tri[:,2].astype(float)-tri[:,0]);lengths=np.linalg.norm(normals,axis=1)
   if f.get('surface')=='cylinder':
    _,singular,axes=np.linalg.svd(normals/lengths[:,None],full_matrices=False)
    if len(singular)==3 and singular[1]>.01 and singular[2]<singular[1]*.02:self.cylinder_axes[f['id']]=axes[-1].tolist()
   elif f.get('surface')!='plane':
    points=positions.astype(float);_,singular,axes=np.linalg.svd(points-points.mean(axis=0),full_matrices=False)
    if len(singular)==3 and singular[1]>.1 and singular[2]<singular[1]*.01:self.flat_view_normals[f['id']]=axes[-1].tolist()
   if f.get('surface')=='plane':
    normals=np.cross(tri[:,1].astype(float)-tri[:,0],tri[:,2].astype(float)-tri[:,0]);i=int(np.argmax(np.linalg.norm(normals,axis=1)));self.plane_normals[f['id']]=(normals[i]/np.linalg.norm(normals[i])).tolist()
  self.triangles=np.concatenate(rows) if rows else np.empty((0,3,3),np.float32)
  if len(self.triangles):
   self.lo=self.triangles.min(axis=1);self.hi=self.triangles.max(axis=1);self.centers=(self.lo+self.hi)/2
   self._build(np.arange(len(self.triangles)))
 def _build(self,ids):
  lo=self.lo[ids].min(axis=0);hi=self.hi[ids].max(axis=0);index=len(self.nodes);self.nodes.append(None)
  if len(ids)<=128:self.nodes[index]=(lo,hi,ids,None)
  else:
   axis=int(np.argmax(np.ptp(self.centers[ids],axis=0)));order=ids[np.argsort(self.centers[ids,axis],kind='stable')];mid=len(order)//2
   children=(self._build(order[:mid]),self._build(order[mid:]));self.nodes[index]=(lo,hi,None,children)
  return index
 @staticmethod
 def box_hit(origin,direction,distance,lo,hi):
  enter=0.;leave=distance
  for k in range(3):
   if abs(direction[k])<1e-12:
    if origin[k]<lo[k]-1e-6 or origin[k]>hi[k]+1e-6:return False
   else:
    a=(lo[k]-origin[k])/direction[k];b=(hi[k]-origin[k])/direction[k];enter=max(enter,min(a,b));leave=min(leave,max(a,b))
    if enter>leave+1e-6:return False
  return True
 def blocked(self,point,direction,distance):
  # Start on the referenced CAD surface and ignore only numerical self-intersection.
  d=np.asarray(direction,float);d/=np.linalg.norm(d);origin=np.asarray(point,float)+d*.002;length=distance-.002
  for z in self.zones:
   if self.box_hit(origin,d,length,np.asarray(z['min_mm']),np.asarray(z['max_mm'])):return 'fixture_keepout'
  if not self.nodes:return 'geometry_missing'
  stack=[0]
  while stack:
   lo,hi,ids,children=self.nodes[stack.pop()]
   if not self.box_hit(origin,d,length,lo,hi):continue
   if children:stack.extend(children);continue
   t=self.triangles[ids];e1=t[:,1]-t[:,0];e2=t[:,2]-t[:,0];h=np.cross(d,e2);det=np.einsum('ij,ij->i',e1,h)
   valid=np.abs(det)>1e-10;inv=np.divide(1.,det,out=np.zeros_like(det),where=valid);s=origin-t[:,0];u=np.einsum('ij,ij->i',s,h)*inv;q=np.cross(s,e1);v=q@d*inv;distance_hit=np.einsum('ij,ij->i',e2,q)*inv
   if np.any(valid&(u>=-1e-7)&(v>=-1e-7)&(u+v<=1+1e-7)&(distance_hit>1e-5)&(distance_hit<length)):return 'workpiece_occlusion'
  return None
 def locate(self,anchor,tolerance=.1):
  triangles=self.faces.get(anchor.get('face_id'))
  if triangles is None:return dict(status='unknown',reason='标记点缺少当前CAD面关联，无法确定观察方向')
  surface=closest_surface(anchor['position'],triangles)
  if not surface or surface['distance_mm']>tolerance:return dict(status='unknown',reason='标记点偏离关联CAD面，请先核对位置',distance_mm=surface['distance_mm'] if surface else None)
  if anchor['face_id'] in self.plane_normals:surface['normal']=self.plane_normals[anchor['face_id']]
  normals=[surface['normal']];basis='local_surface_normal'
  if anchor.get('vision_circle') and anchor['face_id'] in self.cylinder_axes:
   normals=[self.cylinder_axes[anchor['face_id']]];basis='cylinder_axis_for_vision_circle'
  elif anchor['face_id'] in self.flat_view_normals:
   normals.append(self.flat_view_normals[anchor['face_id']]);basis='local_normal_and_CAD_flat_patch_direction'
  return dict(status='located',face_id=anchor['face_id'],view_normals=normals,direction_basis=basis,**surface)
