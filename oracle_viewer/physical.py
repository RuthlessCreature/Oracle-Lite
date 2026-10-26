"""Display-mesh/envelope collision screening, deliberately not installation certification."""
import math
import numpy as np
from functools import lru_cache

def prepare_mesh(mesh):
 """Reuse face arrays and AABBs across all candidate installations."""
 if not mesh or '_screen_bounds' in mesh:return mesh
 faces=[];low=[];high=[]
 for face in mesh['faces']:
  points=np.asarray(face['positions'],dtype=float).reshape(-1,3)
  faces.append(dict(face,positions=points));low.append(points.min(axis=0));high.append(points.max(axis=0))
 return dict(mesh,faces=faces,_screen_bounds=(np.asarray(low).reshape(-1,3),np.asarray(high).reshape(-1,3)))

def solids(scene):
 n=np.asarray(scene['normal'],float);f=np.asarray(scene['lens_front_mm']);c=np.asarray(scene['camera_flange_mm']);d=np.asarray(scene['light_direction'],float);l=np.asarray(scene['light_face_mm'])
 rows=[dict(id='lens',shape='cylinder',origin=f,axis=n,height=scene['lens_length_mm'],outer=scene['lens_diameter_mm']/2,inner=0),dict(id='camera',shape='box',origin=c,axis=n,height=scene['camera_body_mm'][2],half=np.asarray(scene['camera_body_mm'][:2])/2)]
 if scene['light_shape']=='ring':rows.append(dict(id='light',shape='cylinder',origin=l,axis=-d,height=scene['light_height_mm'],outer=scene['light_outer_mm']/2,inner=scene['light_inner_mm']/2))
 else:rows.append(dict(id='light',shape='box',origin=l,axis=-d,height=scene['light_body_mm'][2],half=np.asarray(scene['light_body_mm'][:2])/2))
 for r in rows:
  from .visibility import frame
  r['basis']=frame(r['axis'])
  if r['id'] in ('camera','lens') and scene.get('basis_x'):
   r['basis']=np.array([scene['basis_x'],scene['basis_y'],scene['normal']],float)
 return rows

def inside(points,r,margin=0):
 q=(np.asarray(points)-r['origin'])@r['basis'].T
 z=(q[:,2]>margin)&(q[:,2]<r['height']-margin)
 if r['shape']=='box':return z & np.all(np.abs(q[:,:2])<r['half']-margin,axis=1)
 radial=np.linalg.norm(q[:,:2],axis=1)
 return z & (radial<r['outer']-margin) & (radial>r['inner']+margin)

@lru_cache(maxsize=1024)
def _sample_template(shape,height,outer,inner,half):
 z=np.linspace(.001,height-.001,17)
 if shape=='cylinder':
  angle=np.linspace(0,2*math.pi,65)[:-1];radii=[outer-.001] if not inner else [inner+.001,(inner+outer)/2,outer-.001]
  local=np.array([[rad*math.cos(a),rad*math.sin(a),h] for h in z for rad in radii for a in angle])
 else:local=np.array([[x,y,h] for h in z for x in np.linspace(-half[0]+.001,half[0]-.001,9) for y in np.linspace(-half[1]+.001,half[1]-.001,9)])
 local.setflags(write=False)
 return local

def samples(r):
 local=_sample_template(r['shape'],float(r['height']),float(r.get('outer',0)),float(r.get('inner',0)),tuple(r.get('half',())))
 return local@r['basis']+r['origin']

def triangles_hit_box(points,indices,r):
 """Triangle/OBB separating-axis test: 3 box + triangle normal + 9 edge axes."""
 local=(points-r['origin'])@r['basis'].T;local[:,2]-=r['height']/2
 half=np.r_[r['half'],r['height']/2]-.001
 if np.any(local.min(axis=0)>=half) or np.any(local.max(axis=0)<=-half):return False
 tri=local[np.asarray(indices,dtype=int).reshape(-1,3)]
 hit=np.all(tri.min(axis=1)<half,axis=1)&np.all(tri.max(axis=1)>-half,axis=1)
 tri=tri[hit]
 if not len(tri):return False
 edges=[tri[:,1]-tri[:,0],tri[:,2]-tri[:,1],tri[:,0]-tri[:,2]]
 axes=[np.cross(edges[0],edges[1])]+[np.cross(edge,axis) for edge in edges for axis in np.eye(3)]
 alive=np.ones(len(tri),dtype=bool)
 for axis in axes:
  projections=np.einsum('nij,nj->ni',tri,axis);extent=np.sum(np.abs(axis)*half,axis=1)
  alive&=(projections.min(axis=1)<=extent)&(projections.max(axis=1)>=-extent)
 return bool(np.any(alive))

def triangles_hit_cylinder(points,indices,r):
 """Triangle/finite annular-cylinder intersection, including crossing triangles.

 Clip to the axial slab, then test the projected convex polygon against the
 annulus. Unlike vertex sampling this detects a triangle crossing a ring wall
 with all three vertices outside the cylinder.
 """
 local=(np.asarray(points,float)-r['origin'])@r['basis'].T
 tris=local[np.asarray(indices,int).reshape(-1,3)]
 outer=r['outer']-.001;inner=max(0.,r['inner']+.001);height=r['height']
 mask=(tris[:,:,2].max(1)>.001)&(tris[:,:,2].min(1)<height-.001)
 mask&=np.all(tris[:,:,:2].min(1)<outer,axis=1)&np.all(tris[:,:,:2].max(1)>-outer,axis=1)
 for tri in tris[mask]:
  poly=list(tri)
  for z,sign in ((.001,1),(height-.001,-1)):
   clipped=[]
   for a,b in zip(poly,poly[1:]+poly[:1]):
    ai=sign*(a[2]-z)>=0;bi=sign*(b[2]-z)>=0
    if ai:clipped.append(a)
    if ai!=bi:clipped.append(a+(b-a)*((z-a[2])/(b[2]-a[2])))
   poly=clipped
   if not poly:break
  if not poly:continue
  xy=np.asarray(poly)[:,:2];max_rad=float(np.max(np.linalg.norm(xy,axis=1)))
  if max_rad<=inner:continue
  min_rad=float('inf');cross=[]
  for a,b in zip(xy,np.roll(xy,-1,axis=0)):
   edge=b-a;den=float(edge@edge)
   q=a+edge*(np.clip(-float(a@edge)/den,0,1) if den>1e-20 else 0)
   min_rad=min(min_rad,float(np.linalg.norm(q)));cross.append(float(a[0]*b[1]-a[1]*b[0]))
  if len(xy)>=3 and (all(v>=-1e-12 for v in cross) or all(v<=1e-12 for v in cross)):
   # For a degenerate collinear projection, edge distance is authoritative.
   if abs(sum(cross))>1e-12:min_rad=0.
  if min_rad<outer:return True
 return False

def screen(head,design,mesh=None):
 rows=solids(head['scene']);collisions=[];eps=.001
 sampled={r['id']:samples(r) for r in rows}
 for i,a in enumerate(rows):
  for b in rows[i+1:]:
   if np.any(inside(sampled[a['id']],b,eps)) or np.any(inside(sampled[b['id']],a,eps)):collisions.append(dict(objects=[a['id'],b['id']],type='hardware_envelope_intersection'))
 clearance=design.get('minimum_clearance_mm',2)
 for a in rows[:2]:
  b=rows[2]
  if not any(set(c['objects'])=={a['id'],b['id']} for c in collisions) and (np.any(inside(sampled[a['id']],b,-clearance)) or np.any(inside(sampled[b['id']],a,-clearance))):
   collisions.append(dict(objects=[a['id'],b['id']],type='insufficient_hardware_clearance',required_mm=clearance))
 if head['lighting_strategy']=='front_ring' and head['scene']['light_inner_mm']-head['scene']['lens_diameter_mm']<2*clearance:
  collisions.append(dict(objects=['lens','light'],type='insufficient_ring_mount_clearance',required_radial_mm=clearance))
 # Test every display-mesh vertex, not only FAI anchors. This is a sampled
 # intersection screen: triangles spanning a solid and containment can be missed.
 if mesh and not collisions:
  mesh=prepare_mesh(mesh)
  active=[]
  for r in rows:
   center=r['origin']+r['axis']*r['height']/2
   half=np.r_[r['half'],r['height']/2] if r['shape']=='box' else np.array([r['outer'],r['outer'],r['height']/2])
   extent=np.abs(r['basis']).T@half;r['bbox']=(center-extent,center+extent)
   if not mesh.get('bbox') or np.all(r['bbox'][0]<np.asarray(mesh['bbox'][3:])+0.01) and np.all(r['bbox'][1]>np.asarray(mesh['bbox'][:3])-0.01):active.append(r)
  low,high=mesh['_screen_bounds'];near=np.zeros(len(low),dtype=bool)
  for r in active:near|=np.all(high>=r['bbox'][0],axis=1)&np.all(low<=r['bbox'][1],axis=1)
  for fi in np.flatnonzero(near):
   face=mesh['faces'][fi]
   if not active:break
   points=np.asarray(face['positions'],dtype=float).reshape(-1,3)
   lo=low[fi];hi=high[fi]
   for r in active:
    if np.any(hi<r['bbox'][0]) or np.any(lo>r['bbox'][1]):continue
    if any(c.get('face_id')==face['id'] and r['id'] in c['objects'] for c in collisions):continue
    triangle_hit=(triangles_hit_box(points,face['indices'],r) if r['shape']=='box' else triangles_hit_cylinder(points,face['indices'],r)) if face.get('indices') else False
    if np.any(inside(points,r,eps)) or triangle_hit:
     collisions.append(dict(objects=[r['id'],'workpiece'],type='display_mesh_envelope_intersection',face_id=face['id']))
    if len(collisions)>=30:break
   if len(collisions)>=30:break
 for zone in design.get('forbidden_zones',[]):
  lo=np.asarray(zone['min_mm']);hi=np.asarray(zone['max_mm'])
  for r in rows:
   points=sampled[r['id']]
   if np.any(np.all((points>lo)&(points<hi),axis=1)):collisions.append(dict(objects=[r['id'],zone['name']],type='forbidden_zone_intersection'))
 return dict(status='rejected' if collisions else 'conditional',collisions=collisions,method='box_triangle_SAT_and_axially_clipped_annular_cylinder',mesh_checked=bool(mesh),minimum_mount_clearance_mm=clearance,
  installation_status='unverified',limitations=['Hardware pair checks remain sampled; solid containment and exact BREP are not checked','Display mesh and dimension envelopes are not supplier BREP','Fixture/support/screws/cable bend/service and loading clearance are incomplete','Inter-head collisions and travel paths are separate checks'],ready_for_installation=False)
