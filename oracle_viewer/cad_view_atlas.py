"""Render evidence-only orthographic CAD views for local visual FAI matching."""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import threading

import numpy as np
from PIL import Image, ImageDraw, ImageFont


VIEWS = {
    '+X': (np.array([1., 0, 0]), np.array([0., 1, 0]), np.array([0., 0, 1])),
    '-X': (np.array([-1., 0, 0]), np.array([0., -1, 0]), np.array([0., 0, 1])),
    '+Y': (np.array([0., 1, 0]), np.array([-1., 0, 0]), np.array([0., 0, 1])),
    '-Y': (np.array([0., -1, 0]), np.array([1., 0, 0]), np.array([0., 0, 1])),
    '+Z': (np.array([0., 0, 1]), np.array([1., 0, 0]), np.array([0., 1, 0])),
    '-Z': (np.array([0., 0, -1]), np.array([1., 0, 0]), np.array([0., -1, 0])),
}

_GEOMETRY_CACHE = OrderedDict()
_GEOMETRY_CACHE_LOCK = threading.RLock()
_PROJECTION_CACHE = OrderedDict()
_PROJECTION_CACHE_LOCK = threading.RLock()


def cached_surface_geometry(mesh):
    """Reuse decoded mesh triangles and their exact spatial index per Viewer mesh."""
    key=(id(mesh),mesh.get('source_sha256'),len(mesh.get('faces',[])))
    with _GEOMETRY_CACHE_LOCK:
        cached=_GEOMETRY_CACHE.get(key)
        if cached is not None and cached[0] is mesh:
            _GEOMETRY_CACHE.move_to_end(key)
            return cached[1],cached[2],cached[3]
    triangles,face_ids=_valid_triangles(mesh)
    from .surface_index import SurfaceIndex
    index=SurfaceIndex(triangles)
    with _GEOMETRY_CACHE_LOCK:
        _GEOMETRY_CACHE[key]=(mesh,triangles,face_ids,index)
        _GEOMETRY_CACHE.move_to_end(key)
        while len(_GEOMETRY_CACHE)>2:
            _GEOMETRY_CACHE.popitem(last=False)
    return triangles,face_ids,index


def _valid_triangles(mesh, *, max_triangles=None):
    triangles, face_ids = [], []
    for face in mesh.get('faces', []):
        if face.get('occt_face_valid') is False:
            continue
        vertices = np.asarray(face.get('positions', []), dtype=float).reshape(-1, 3)
        indices = np.asarray(face.get('indices', []), dtype=np.int64)
        if not len(vertices) or len(indices) < 3:
            continue
        indices = indices[:len(indices) // 3 * 3].reshape(-1, 3)
        tri = vertices[indices]
        valid = np.isfinite(tri).all(axis=(1, 2))
        if valid.any():
            triangles.append(tri[valid])
            face_ids.append(np.full(int(valid.sum()), face['id'], dtype=object))
    if not triangles:
        raise ValueError('CAD 网格中没有可用于视觉定位的有效三角面')
    triangles=np.concatenate(triangles,axis=0)
    face_ids=np.concatenate(face_ids,axis=0)
    if max_triangles and len(triangles)>max_triangles:
        # Keep a deterministic, evenly spaced surface sample plus at least one
        # triangle from every represented CAD face. The atlas is visual
        # evidence only; exact point-to-face projection always uses full CAD.
        sampled=np.linspace(0,len(triangles)-1,max_triangles,dtype=np.int64)
        _unique,first=np.unique(face_ids,return_index=True)
        indices=np.unique(np.concatenate((sampled,first)))
        triangles,face_ids=triangles[indices],face_ids[indices]
    return triangles,face_ids


def registration_atlas_mesh(mesh, *, max_triangles=60000):
    """Return a bounded display-only mesh for repeated registration overlays."""
    triangles,_face_ids=_valid_triangles(mesh,max_triangles=max_triangles)
    count=len(triangles)
    return {'faces':[{'id':'registration-atlas-sample','occt_face_valid':True,
        'positions':triangles.reshape(-1).tolist(),
        'indices':np.arange(count*3,dtype=np.int32).tolist()}]}


def _view_projection(triangles, view_id, panel_size, margin):
    normal, right, up = VIEWS[view_id]
    vertices = triangles.reshape(-1, 3)
    right_values, up_values = vertices @ right, vertices @ up
    center_right = (float(right_values.min()) + float(right_values.max())) / 2
    center_up = (float(up_values.min()) + float(up_values.max())) / 2
    span_right = max(float(right_values.max() - right_values.min()), 1e-6)
    span_up = max(float(up_values.max() - up_values.min()), 1e-6)
    usable = panel_size - 2 * margin
    scale = min(usable / span_right, usable / span_up)
    depth = float(np.ptp(vertices @ normal)) + 2.0
    return normal, right, up, center_right, center_up, scale, depth


def render_cad_view_atlas(mesh, output_path: str | Path, *, panel_size=480):
    """Write six shaded orthographic evidence views, without FAI decisions."""
    triangles, _face_ids = _valid_triangles(mesh,max_triangles=60000)
    margin = 22
    width, height = panel_size * 3, panel_size * 2
    image = Image.new('RGB', (width, height), '#f7f8fa')
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 17)
        small = ImageFont.truetype('DejaVuSans.ttf', 10)
    except OSError:
        font = ImageFont.load_default()
        small = ImageFont.load_default()
    for index, view_id in enumerate(('+Z', '-Z', '+X', '-X', '+Y', '-Y')):
        col, row = index % 3, index // 3
        ox, oy = col * panel_size, row * panel_size
        panel = (ox, oy, ox + panel_size - 1, oy + panel_size - 1)
        draw.rectangle(panel, fill='white', outline='#bdc4cc', width=1)
        normal, right, up, cr, cu, scale, _depth = _view_projection(
            triangles, view_id, panel_size, margin)
        projected_x = (triangles @ right - cr) * scale + (panel_size / 2)
        projected_y = (cu - triangles @ up) * scale + (panel_size / 2)
        centers = triangles.mean(axis=1)
        order = np.argsort(centers @ normal)  # far-to-near painter order
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        lengths = np.linalg.norm(normals, axis=1)
        shade = np.clip(np.abs(normals @ normal) / np.maximum(lengths, 1e-12), 0, 1)
        for tri_index in order:
            polygon = [(ox + float(projected_x[tri_index, j]),
                        oy + float(projected_y[tri_index, j])) for j in range(3)]
            intensity = int(120 + 90 * (1 - shade[tri_index]))
            color = (intensity, min(255, intensity + 2), min(255, intensity + 5))
            draw.polygon(polygon, fill=color)
        # Give the local vision model an explicit panel-local coordinate
        # reference. Without ticks, it often returns plausible-looking but
        # off-part UVs because the six-view atlas is a composite image. The
        # light dashed grid is evidence only; final rays are still checked
        # against the full-resolution CAD triangles.
        usable = panel_size - 2 * margin
        for tick in range(1, 10):
            position = margin + tick * usable / 10
            for start in range(margin, panel_size - margin, 12):
                end = min(start + 6, panel_size - margin)
                draw.line((ox + position, oy + start, ox + position, oy + end),
                          fill='#bfd0dc', width=1)
                draw.line((ox + start, oy + position, ox + end, oy + position),
                          fill='#bfd0dc', width=1)
            if tick % 2 == 0:
                draw.text((ox + position - 9, oy + 34), f'{tick/10:.1f}',
                          fill='#526576', font=small)
                draw.text((ox + 24, oy + position - 5), f'{tick/10:.1f}',
                          fill='#526576', font=small)
        draw.rectangle((ox + 5, oy + 5, ox + 75, oy + 30), fill='white', outline='#8d98a5')
        draw.text((ox + 12, oy + 8), view_id, fill='#182433', font=font)
        draw.text((ox + 8, oy + panel_size - 18),
                  'CAD orthographic view · normalized image coordinates', fill='#586474', font=small)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + '.tmp')
    image.save(temporary, format='PNG', optimize=True)
    temporary.replace(target)
    return target


def render_focused_view_retry(atlas_path: str | Path, output_path: str | Path,
                              markers, registration_points, *, panel_size=480, scale=2,
                              surface_candidates=()):
    """Upscale one failed model-selected view and overlay its prior UV picks."""
    view_order=('+Z','-Z','+X','-X','+Y','-Y')
    rows=[*markers,*registration_points]
    views={row.get('view_id') for row in rows if isinstance(row,dict)}
    if len(views)!=1 or next(iter(views),None) not in view_order:
        return None
    view_id=next(iter(views));index=view_order.index(view_id)
    col,row=index%3,index//3
    with Image.open(atlas_path) as atlas:
        panel=atlas.convert('RGB').crop((col*panel_size,row*panel_size,
                                        (col+1)*panel_size,(row+1)*panel_size))
    focused=panel.resize((panel_size*scale,panel_size*scale),Image.Resampling.LANCZOS)
    global_view_candidates=[candidate for candidate in surface_candidates
        if isinstance(candidate,dict) and candidate.get('view_id')==view_id
        and candidate.get('candidate_role')=='global_surface_search']
    if global_view_candidates:
        canvas=Image.new('RGB',(focused.width+300,focused.height),'#f7f8fa')
        canvas.paste(focused,(0,0));focused=canvas
    draw=ImageDraw.Draw(focused)
    try:
        font=ImageFont.truetype('DejaVuSans.ttf',14*scale)
        label_font=ImageFont.truetype('DejaVuSans.ttf',9*scale)
    except OSError:
        font=label_font=ImageFont.load_default()
    margin=22
    for rows,color,prefix in ((markers,'#d42363','M'),
                              (registration_points,'#087f8c','R')):
        for point_index,point in enumerate(rows,1):
            if not isinstance(point,dict) or point.get('view_id')!=view_id:
                continue
            uv=point.get('uv')
            if not isinstance(uv,list) or len(uv)!=2:
                continue
            x=(margin+float(uv[0])*(panel_size-2*margin))*scale
            y=(margin+float(uv[1])*(panel_size-2*margin))*scale
            radius=6*scale
            draw.ellipse((x-radius,y-radius,x+radius,y+radius),outline=color,width=3*scale)
            draw.line((x-radius,y,x+radius,y),fill=color,width=2*scale)
            draw.line((x,y-radius,x,y+radius),fill=color,width=2*scale)
            label=str(point.get('feature') or point.get('source_point_name') or point.get('name') or point_index)
            label=f'{prefix}{point_index} {label[:18]}'
            draw.rectangle((x+radius+2,y-radius,x+radius+2+len(label)*7*scale,y+radius),fill='white')
            draw.text((x+radius+3,y-radius),label,fill=color,font=label_font)
    # These green points are computed intersections of nearby view rays with
    # valid CAD triangles. They are search aids only: the local model must
    # still identify the feature and return its chosen UV in the retry JSON.
    for candidate in surface_candidates:
        if not isinstance(candidate,dict) or candidate.get('view_id')!=view_id:
            continue
        uv=candidate.get('uv')
        if not isinstance(uv,list) or len(uv)!=2:
            continue
        x=(margin+float(uv[0])*(panel_size-2*margin))*scale
        y=(margin+float(uv[1])*(panel_size-2*margin))*scale
        radius=5*scale
        draw.ellipse((x-radius,y-radius,x+radius,y+radius),fill='#16834a',outline='white',width=2*scale)
        label=(f"C{candidate.get('id','?')}" if candidate.get('candidate_role')=='global_surface_search'
               else f"C{candidate.get('id','?')} {str(candidate.get('feature') or '')[:10]}")
        draw.rectangle((x+radius+2,y-radius,x+radius+2+len(label)*8*scale,y+radius),fill='white')
        draw.text((x+radius+3,y-radius),label,fill='#12643a',font=label_font)
    if global_view_candidates:
        legend_x=panel_size*scale+18
        draw.text((legend_x,24),'VERIFIED CAD RAY HITS',fill='#172b44',font=font)
        draw.text((legend_x,58),'Choose a different C# for each distinct XYZ point.',
                  fill='#263649',font=label_font)
        for index,candidate in enumerate(global_view_candidates):
            uv=candidate.get('uv') or [0,0]
            y=94+index*30
            draw.ellipse((legend_x,y+2,legend_x+12,y+14),fill='#16834a',outline='white',width=1)
            draw.text((legend_x+20,y),
                f"C{candidate.get('id')}   UV {float(uv[0]):.3f}, {float(uv[1]):.3f}",
                fill='#12643a',font=label_font)
    draw.rectangle((0,0,focused.width-1,focused.height-1),outline='#d42363',width=4*scale)
    target=Path(output_path);target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_suffix(target.suffix+'.tmp')
    focused.save(temporary,format='PNG',optimize=True);temporary.replace(target)
    return target


def nearby_surface_view_candidates(mesh, view_id, uv, *, panel_size=480,
                                   radii=(0.003,0.006,0.02,0.05,0.10),
                                   max_candidates=3, triangle_data=None):
    """Find nearby normalized pixels whose orthographic rays hit valid CAD.

    Results are visual search evidence for the local model, not placements.
    The small deterministic neighborhood is useful when a model's UV lands
    just outside a silhouette; the final point still needs semantic model
    selection and the standard independent geometry gates.
    """
    if view_id not in VIEWS or not isinstance(uv,(list,tuple)) or len(uv)!=2:
        return []
    try:
        u,v=map(float,uv)
    except (TypeError,ValueError):
        return []
    if not np.isfinite([u,v]).all() or not (0<=u<=1 and 0<=v<=1):
        return []
    angles=np.arange(8,dtype=float)*(np.pi/4)
    offsets=tuple((float(np.cos(angle)),float(np.sin(angle))) for angle in angles)
    try:
        triangles,face_ids=triangle_data[:2] if triangle_data is not None else cached_surface_geometry(mesh)[:2]
    except (ValueError,TypeError):
        return []
    projection=_view_projection_cache(triangles,view_id,panel_size,22)
    candidates=[]
    for radius in radii:
        for du,dv in offsets:
            point=[u+du*radius,v+dv*radius]
            if not (0<=point[0]<=1 and 0<=point[1]<=1):
                continue
            hit=_ray_hit_view_triangles(triangles,face_ids,view_id,*point,
                                        panel_size=panel_size,projection=projection)
            if hit is None:
                continue
            if any(np.linalg.norm(np.asarray(hit['position'])-np.asarray(old['position']))<0.10
                   for old in candidates):
                continue
            candidates.append(dict(id=len(candidates)+1,view_id=view_id,uv=point,
                                   face_id=hit['face_id'],position=hit['position'],
                                   offset_uv=[point[0]-u,point[1]-v]))
            if len(candidates)>=max_candidates:
                return candidates
    return candidates


def distributed_surface_view_candidates(mesh, *, view_ids=None, panel_size=480,
                                        grid_size=32, max_candidates_per_view=10,
                                        triangle_data=None, allowed_face_ids=None):
    """Sample verified ray hits across whole orthographic panels for recovery.

    These are visual search choices, never placements. They let the local
    vision model recover when its prior UV is too far from the CAD silhouette.
    Farthest-point sampling keeps a compact, spatially distributed set.
    """
    view_ids=tuple(view_ids or ('+Z','-Z','+X','-X','+Y','-Y'))
    if type(grid_size) is not int or grid_size<8 or type(max_candidates_per_view) is not int or max_candidates_per_view<1:
        return []
    try:
        triangles,face_ids=(triangle_data[:2] if triangle_data is not None
                             else cached_surface_geometry(mesh)[:2])
    except (ValueError,TypeError):
        return []
    result=[]
    allowed_face_ids=set(allowed_face_ids) if allowed_face_ids is not None else None
    ticks=(np.arange(grid_size,dtype=float)+0.5)/grid_size
    for view_id in view_ids:
        if view_id not in VIEWS:
            continue
        projection=_view_projection_cache(triangles,view_id,panel_size,22)
        hits=[]
        for v in ticks:
            for u in ticks:
                hit=_ray_hit_view_triangles(triangles,face_ids,view_id,float(u),float(v),
                    panel_size=panel_size,projection=projection)
                if hit is not None and (allowed_face_ids is None or hit.get('face_id') in allowed_face_ids):
                    hits.append(dict(view_id=view_id,uv=[float(u),float(v)],
                        face_id=hit['face_id'],position=hit['position']))
        if not hits:
            continue
        coords=np.asarray([row['uv'] for row in hits],float)
        # Start near the view center, then greedily spread choices over the
        # visible part and its distinct lobes rather than returning a cluster.
        selected=[int(np.argmin(np.sum((coords-np.array([.5,.5]))**2,axis=1)))]
        while len(selected)<min(max_candidates_per_view,len(hits)):
            distance=np.min(np.sum((coords[:,None,:]-coords[np.asarray(selected)][None,:,:])**2,axis=2),axis=1)
            distance[np.asarray(selected)]=-1
            selected.append(int(np.argmax(distance)))
        for local_index,index in enumerate(selected,1):
            row=dict(hits[index]);row['id']=len(result)+1
            row['candidate_role']='global_surface_search'
            row['panel_candidate']=f'{view_id}:{local_index}'
            result.append(row)
    return result


def render_surface_candidate_atlas(atlas_path: str | Path, output_path: str | Path,
                                   surface_candidates, *, panel_size=480):
    """Overlay globally sampled exact CAD ray hits on the six-view atlas."""
    target=Path(output_path)
    with Image.open(atlas_path) as source:
        image=source.convert('RGB').copy()
    draw=ImageDraw.Draw(image)
    try:
        font=ImageFont.truetype('DejaVuSans.ttf',9)
    except OSError:
        font=ImageFont.load_default()
    margin=22;grouped={}
    for candidate in surface_candidates:
        view_id=candidate.get('view_id')
        if view_id not in VIEWS:
            continue
        view_index=('+Z','-Z','+X','-X','+Y','-Y').index(view_id)
        col,row=view_index%3,view_index//3
        uv=candidate.get('uv') or []
        if len(uv)!=2:
            continue
        x=col*panel_size+margin+float(uv[0])*(panel_size-2*margin)
        y=row*panel_size+margin+float(uv[1])*(panel_size-2*margin)
        grouped.setdefault((view_id,round(x),round(y)),[]).append((candidate,x,y))
    for (_view_id,_x,_y),rows in grouped.items():
        candidate,x,y=rows[0];radius=5
        draw.ellipse((x-radius,y-radius,x+radius,y+radius),fill='#16834a',outline='white',width=1)
        labels=[]
        for candidate,_,_ in rows:
            names=candidate.get('source_point_names') or []
            suffix=(str(names[0]).split('_')[-1] if len(names)==1 else '')
            labels.append(f"C{candidate.get('id','?')} {suffix}".strip())
        row_height=11;stack_height=row_height*len(labels)
        label_x=(x+radius+2 if x+radius+max((draw.textbbox((0,0),label,font=font)[2]
                    for label in labels),default=20)+4 < (_x//panel_size+1)*panel_size
                 else x-radius-3-max(draw.textbbox((0,0),label,font=font)[2] for label in labels))
        label_y=y-radius-2
        if label_y+stack_height>((_y//panel_size+1)*panel_size-4):
            label_y=(_y//panel_size+1)*panel_size-4-stack_height
        for index,(candidate,label) in enumerate(zip(rows,labels)):
            yy=label_y+index*row_height
            width=draw.textbbox((0,0),label,font=font)[2]+3
            draw.rectangle((label_x-1,yy,label_x+width,yy+row_height-1),fill='white')
            draw.text((label_x,yy),label,fill='#12643a',font=font)
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_suffix(target.suffix+'.tmp')
    image.save(temporary,format='PNG',optimize=True)
    temporary.replace(target)
    return target


def render_registration_hypothesis_atlas(mesh, source_xyz, rotation, translation_mm,
                                         output_path: str | Path, hypothesis_id: str,
                                         *, panel_size=480):
    """Overlay one complete source frame on the CAD six-view atlas for local-model comparison."""
    target=Path(output_path)
    base=target.with_name(target.stem+'.base.png')
    render_cad_view_atlas(mesh,base,panel_size=panel_size)
    with Image.open(base) as source_image:
        image=source_image.convert('RGB').copy()
    base.unlink(missing_ok=True)
    triangles,_face_ids=_valid_triangles(mesh,max_triangles=60000)
    points=np.asarray(source_xyz,dtype=float)
    rotation=np.asarray(rotation,dtype=float);translation=np.asarray(translation_mm,dtype=float)
    if (points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all()
            or rotation.shape!=(3,3) or not np.isfinite(rotation).all()
            or translation.shape!=(3,) or not np.isfinite(translation).all()):
        raise ValueError('Registration atlas requires finite XYZ and a rigid transform')
    mapped=points@rotation.T+translation
    header=32
    canvas=Image.new('RGB',(image.width,image.height+header),'#f7f8fa')
    canvas.paste(image,(0,header));draw=ImageDraw.Draw(canvas)
    try:font=ImageFont.truetype('DejaVuSans.ttf',17)
    except OSError:font=ImageFont.load_default()
    color='#1268c4'
    draw.rectangle((0,0,canvas.width-1,canvas.height-1),outline=color,width=5)
    draw.text((12,7),f'MSOP → CAD registration candidate {hypothesis_id}',fill='#172b44',font=font)
    margin=22
    for view_index,view_id in enumerate(('+Z','-Z','+X','-X','+Y','-Y')):
        col,row=view_index%3,view_index//3
        ox,oy=col*panel_size,row*panel_size+header
        _normal,right,up,center_right,center_up,scale,_depth=_view_projection(
            triangles,view_id,panel_size,margin)
        px=(mapped@right-center_right)*scale+panel_size/2+ox
        py=(center_up-mapped@up)*scale+panel_size/2+oy
        for index,(x,y) in enumerate(zip(px,py),start=1):
            if not (ox+margin<=x<=ox+panel_size-margin and oy+margin<=y<=oy+panel_size-margin):
                continue
            radius=5
            draw.ellipse((x-radius,y-radius,x+radius,y+radius),fill='#ff365f',outline='white',width=2)
            if len(mapped)<=32:
                draw.text((x+radius+1,y-radius-2),str(index),fill='#991d39',font=font)
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_suffix(target.suffix+'.tmp')
    canvas.save(temporary,format='PNG',optimize=True)
    temporary.replace(target)
    return target


def render_registration_group_atlas(mesh, groups, output_path: str | Path, *, panel_size=480,
        title='MSOP XYZ point groups · model-selected frame proposals · display check'):
    """Overlay complete FAI point groups after their locally selected frame transforms."""
    target=Path(output_path)
    base=target.with_name(target.stem+'.base.png')
    render_cad_view_atlas(mesh,base,panel_size=panel_size)
    with Image.open(base) as source_image:
        image=source_image.convert('RGB').copy()
    base.unlink(missing_ok=True)
    triangles,_face_ids=_valid_triangles(mesh,max_triangles=60000)
    palette=('#ff365f','#087f8c','#8a4fff','#e07a00','#14833b','#c22e94')
    mapped_groups=[]
    for index,group in enumerate(groups):
        points=np.asarray(group['source_xyz'],dtype=float)
        rotation=np.asarray(group['rotation'],dtype=float)
        translation=np.asarray(group['translation_mm'],dtype=float)
        if (points.ndim!=2 or points.shape[1]!=3 or not len(points) or not np.isfinite(points).all()
                or rotation.shape!=(3,3) or not np.isfinite(rotation).all()
                or translation.shape!=(3,) or not np.isfinite(translation).all()):
            raise ValueError('Registration group atlas requires finite XYZ and rigid transforms')
        mapped=points@rotation.T+translation
        mapped_groups.append((group,mapped,palette[index%len(palette)]))
    header=38+18*max(0,(len(mapped_groups)-1)//3)
    canvas=Image.new('RGB',(image.width,image.height+header),'#f7f8fa')
    canvas.paste(image,(0,header));draw=ImageDraw.Draw(canvas)
    try:font=ImageFont.truetype('DejaVuSans.ttf',16);small=ImageFont.truetype('DejaVuSans.ttf',11)
    except OSError:font=small=ImageFont.load_default()
    draw.rectangle((0,0,canvas.width-1,canvas.height-1),outline='#1268c4',width=5)
    draw.text((12,5),title,fill='#172b44',font=font)
    for index,(group,_mapped,color) in enumerate(mapped_groups):
        label=f"{group.get('program','?')} / {group.get('coordinate_system','?')}  {group.get('hypothesis_id','')}"
        x=12+(index%3)*panel_size
        y=22+(index//3)*18
        draw.ellipse((x,y+2,x+9,y+11),fill=color,outline='white',width=1)
        draw.text((x+14,y),label[:44],fill='#263649',font=small)
    margin=22
    for view_index,view_id in enumerate(('+Z','-Z','+X','-X','+Y','-Y')):
        col,row=view_index%3,view_index//3
        ox,oy=col*panel_size,row*panel_size+header
        _normal,right,up,center_right,center_up,scale,_depth=_view_projection(
            triangles,view_id,panel_size,margin)
        for group_index,(_group,mapped,color) in enumerate(mapped_groups):
            px=(mapped@right-center_right)*scale+panel_size/2+ox
            py=(center_up-mapped@up)*scale+panel_size/2+oy
            for point_index,(x,y) in enumerate(zip(px,py),start=1):
                if not (ox+margin<=x<=ox+panel_size-margin and oy+margin<=y<=oy+panel_size-margin):
                    continue
                radius=4 if len(mapped)>32 else 5
                draw.ellipse((x-radius,y-radius,x+radius,y+radius),fill=color,outline='white',width=2)
                # Marker identity must remain visible for full-group model review,
                # including dense FAI groups. White callouts separate IDs from mesh edges.
                label=str(point_index)
                tx=min(max(x+radius+2,ox+margin),ox+panel_size-margin-20)
                ty=min(max(y-radius-3,oy+margin),oy+panel_size-margin-14)
                box=draw.textbbox((tx,ty),label,font=small)
                draw.rectangle((box[0]-1,box[1]-1,box[2]+1,box[3]+1),fill='white')
                draw.text((tx,ty),label,fill='#172b44',font=small)
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_suffix(target.suffix+'.tmp')
    canvas.save(temporary,format='PNG',optimize=True)
    temporary.replace(target)
    return target


def ray_hit_view(mesh, view_id: str, u: float, v: float, *, panel_size=480):
    """Intersect a model-selected normalized orthographic pixel with the CAD."""
    if view_id not in VIEWS or not (0 <= float(u) <= 1 and 0 <= float(v) <= 1):
        raise ValueError('Invalid CAD atlas view or normalized image coordinate')
    triangles, face_ids, _index = cached_surface_geometry(mesh)
    projection=_view_projection_cache(triangles,view_id,panel_size,22)
    return _ray_hit_view_triangles(triangles,face_ids,view_id,u,v,
                                   panel_size=panel_size,projection=projection)


def project_point_to_view_uv(mesh,view_id,point,*,panel_size=480):
    """Project a model-space XYZ point into the exact normalized atlas view."""
    if view_id not in VIEWS:
        raise ValueError('Invalid CAD atlas view')
    xyz=np.asarray(point,float)
    if xyz.shape!=(3,) or not np.isfinite(xyz).all():
        raise ValueError('Expected a finite model-space XYZ point')
    triangles,face_ids,_index=cached_surface_geometry(mesh)
    projection=_view_projection_cache(triangles,view_id,panel_size,22)
    _normal,right,up,center_right,center_up,scale,_depth=projection[:7]
    pixel_x=(float(xyz@right)-center_right)*scale+panel_size/2
    pixel_y=(center_up-float(xyz@up))*scale+panel_size/2
    usable=panel_size-44
    return ([float((pixel_x-22)/usable),float((pixel_y-22)/usable)],float(scale))


def _view_projection_cache(triangles,view_id,panel_size,margin):
    key=(id(triangles),view_id,int(panel_size),int(margin))
    with _PROJECTION_CACHE_LOCK:
        cached=_PROJECTION_CACHE.get(key)
        if cached is not None and cached[0] is triangles:
            _PROJECTION_CACHE.move_to_end(key)
            return cached[1]
    normal,right,up,center_right,center_up,scale,depth=_view_projection(
        triangles,view_id,panel_size,margin)
    vertices=triangles.reshape(-1,3)
    model_center=(vertices.min(axis=0)+vertices.max(axis=0))/2
    projected_right=triangles@right
    projected_up=triangles@up
    x=(projected_right-center_right)*scale+panel_size/2
    y=(center_up-projected_up)*scale+panel_size/2
    min_x=x.min(axis=1);min_y=y.min(axis=1);max_x=x.max(axis=1);max_y=y.max(axis=1)
    grid=48;usable=panel_size-2*margin
    overlaps=((max_x>=margin)&(min_x<=panel_size-margin)&
              (max_y>=margin)&(min_y<=panel_size-margin))
    ids=np.flatnonzero(overlaps)
    x0=np.clip(np.floor((min_x[ids]-margin)*grid/usable).astype(np.int32),0,grid-1)
    x1=np.clip(np.floor((max_x[ids]-margin)*grid/usable).astype(np.int32),0,grid-1)
    y0=np.clip(np.floor((min_y[ids]-margin)*grid/usable).astype(np.int32),0,grid-1)
    y1=np.clip(np.floor((max_y[ids]-margin)*grid/usable).astype(np.int32),0,grid-1)
    widths=x1-x0+1;heights=y1-y0+1;counts=widths*heights
    # Huge projected triangles are rare; keep them in a tiny global list so
    # they do not inflate every spatial bucket.
    large=counts>256
    large_ids=ids[large]
    ids=ids[~large];x0=x0[~large];y0=y0[~large]
    widths=widths[~large];heights=heights[~large];counts=counts[~large]
    repeated=np.repeat(ids,counts)
    starts=np.cumsum(counts,dtype=np.int64)-counts
    owner=np.repeat(np.arange(len(counts),dtype=np.int64),counts)
    local=np.arange(int(counts.sum()),dtype=np.int64)-np.repeat(starts,counts)
    cell_x=x0[owner]+local%widths[owner]
    cell_y=y0[owner]+local//widths[owner]
    cell_codes=cell_x*grid+cell_y
    order=np.argsort(cell_codes,kind='stable')
    sorted_codes=cell_codes[order];sorted_ids=repeated[order]
    bucket_counts=np.bincount(sorted_codes,minlength=grid*grid)
    bucket_offsets=np.concatenate(([0],np.cumsum(bucket_counts,dtype=np.int64)))
    result=(normal,right,up,center_right,center_up,scale,depth,
            min_x,min_y,max_x,max_y,(grid,margin,usable,bucket_offsets,sorted_ids,large_ids),
            model_center)
    with _PROJECTION_CACHE_LOCK:
        _PROJECTION_CACHE[key]=(triangles,result)
        _PROJECTION_CACHE.move_to_end(key)
        while len(_PROJECTION_CACHE)>3:
            _PROJECTION_CACHE.popitem(last=False)
    return result


def _ray_hit_view_triangles(triangles, face_ids, view_id, u, v, *, panel_size=480,projection=None):
    """Intersect a normalized UV against already-decoded valid mesh triangles."""
    margin = 22
    if projection is None:
        projection=_view_projection_cache(triangles,view_id,panel_size,margin)
    (normal,right,up,center_right,center_up,scale,depth,
     min_x,min_y,max_x,max_y,*extra)=projection
    pixel_x = margin + float(u) * (panel_size - 2 * margin)
    pixel_y = margin + float(v) * (panel_size - 2 * margin)
    right_coord = center_right + (pixel_x - panel_size / 2) / scale
    up_coord = center_up - (pixel_y - panel_size / 2) / scale
    all_vertices=triangles.reshape(-1,3)
    center=(extra[1] if len(extra)>1 else
            (all_vertices.min(axis=0)+all_vertices.max(axis=0))/2)
    if extra:
        grid,bin_margin,usable,bucket_offsets,bucket_ids,large_ids=extra[0]
        ix=int(np.clip(np.floor((pixel_x-bin_margin)*grid/usable),0,grid-1))
        iy=int(np.clip(np.floor((pixel_y-bin_margin)*grid/usable),0,grid-1))
        bucket=ix*grid+iy
        candidate_indices=bucket_ids[bucket_offsets[bucket]:bucket_offsets[bucket+1]]
        if len(large_ids):candidate_indices=np.concatenate((candidate_indices,large_ids))
        if len(candidate_indices):
            candidate_indices=candidate_indices[(min_x[candidate_indices]<=pixel_x+1e-9)&
                (max_x[candidate_indices]>=pixel_x-1e-9)&(min_y[candidate_indices]<=pixel_y+1e-9)&
                (max_y[candidate_indices]>=pixel_y-1e-9)]
    else:
        candidate_indices=np.flatnonzero((min_x<=pixel_x+1e-9)&(max_x>=pixel_x-1e-9)&
                                         (min_y<=pixel_y+1e-9)&(max_y>=pixel_y-1e-9))
    if not len(candidate_indices):
        return None
    triangles=triangles[candidate_indices]
    face_ids=face_ids[candidate_indices]
    vertices = triangles.reshape(-1, 3)
    origin = center + right * (right_coord - center @ right) + up * (up_coord - center @ up) + normal * depth
    direction = -normal
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    edge1, edge2 = b - a, c - a
    pvec = np.cross(np.broadcast_to(direction, edge2.shape), edge2)
    determinant = np.einsum('ij,ij->i', edge1, pvec)
    safe = np.where(np.abs(determinant) > 1e-12, determinant, 1.0)
    inverse = 1.0 / safe
    tvec = origin - a
    bary_u = np.einsum('ij,ij->i', tvec, pvec) * inverse
    qvec = np.cross(tvec, edge1)
    bary_v = np.einsum('j,ij->i', direction, qvec) * inverse
    distance = np.einsum('ij,ij->i', edge2, qvec) * inverse
    valid = ((np.abs(determinant) > 1e-12) & (bary_u >= -1e-9) & (bary_v >= -1e-9)
             & (bary_u + bary_v <= 1 + 1e-9) & (distance >= 0))
    if not valid.any():
        return None
    hit_index = int(np.argmin(np.where(valid, distance, np.inf)))
    point = origin + distance[hit_index] * direction
    return dict(face_id=str(face_ids[hit_index]), position=point.tolist(),
                view_id=view_id, uv=[float(u), float(v)], ray_distance_mm=float(distance[hit_index]))
