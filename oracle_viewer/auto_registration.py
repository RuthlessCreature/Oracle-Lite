"""Drawing-independent multistart rigid ICP proposals; never metrology approval."""
from itertools import permutations, product
import hashlib
import numpy as np
from scipy.spatial import cKDTree
from .visibility import closest_surface
from .surface_index import SurfaceIndex

MIN_REGISTRATION_POINTS = 5


def registration_algorithm_sha256(root=None):
    """Fingerprint every source file that can change registration evidence."""
    from pathlib import Path
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    files = [root / 'oracle_viewer' / name for name in (
        'auto_registration.py', 'surface_index.py', 'candidate_mining.py',
        'point_context.py', 'visibility.py', 'msop_coordinates.py', 'msop_tables.py')]
    files.extend((root / 'scripts' / name for name in ('auto_register_msop.py',)))
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b'\0')
        digest.update(path.read_bytes())
    return digest.hexdigest()


def axis_rotations():
    for order in permutations(range(3)):
        for signs in product((-1, 1), repeat=3):
            r=np.eye(3)[list(order)]*np.asarray(signs)[:,None]
            if np.linalg.det(r)>0:yield r


def automatic_fit(source, surface_samples, *, threshold_mm=.1, iterations=60):
    a=np.asarray(source,dtype=float);b=np.asarray(surface_samples,dtype=float)
    # Five points leave four independent fit points and one held-out point
    # under the 1-in-5 split below. Candidate generation is not authorization:
    # every transform still requires local-model review of the complete group.
    if a.ndim!=2 or a.shape[1]!=3 or len(a)<MIN_REGISTRATION_POINTS or not np.isfinite(a).all():
        return dict(status='insufficient_unambiguous_points',automatic_application=False)
    if b.ndim!=2 or b.shape[1]!=3 or not len(b) or not np.isfinite(b).all():raise ValueError('Invalid CAD samples')
    # Rank-two source groups are common in MSOP: the measured locations lie on
    # one physical plane, but their XY spread still carries a useful shape
    # signature. Rigid Kabsch updates are well-defined for a non-collinear
    # planar set. They do not, by themselves, prove handedness or semantic
    # correspondence, so retain the alternative-fit and holdout gates below.
    if np.linalg.matrix_rank(a-a.mean(0),tol=1e-6)<2:
        return dict(status='insufficient_source_spread',automatic_application=False)
    tree=cKDTree(b);fit=a[np.arange(len(a))%5!=0];holdout=a[np.arange(len(a))%5==0]
    results=[]
    for initial in axis_rotations():
        r=initial.copy();rotated=fit@r.T
        t=(b.min(0)+b.max(0))/2-(rotated.min(0)+rotated.max(0))/2
        for _ in range(iterations):
            x=fit@r.T+t;dist,idx=tree.query(x);mask=dist<=np.quantile(dist,.85)
            aa=fit[mask];bb=b[idx[mask]]
            if len(aa)<4:break
            u,_,vt=np.linalg.svd((aa-aa.mean(0)).T@(bb-bb.mean(0)))
            nr=vt.T@u.T
            if np.linalg.det(nr)<0:vt[-1]*=-1;nr=vt.T@u.T
            nt=bb.mean(0)-nr@aa.mean(0)
            delta=np.linalg.norm(nt-t)+np.linalg.norm(nr-r);r,t=nr,nt
            if delta<1e-7:break
        dist,idx=tree.query(a@r.T+t);hd,_=tree.query(holdout@r.T+t)
        fd=dist[np.arange(len(a))%5!=0]
        results.append(dict(rotation=r.tolist(),translation_mm=t.tolist(),
            fit_p95_residual_mm=float(np.quantile(fd,.95)),fit_median_residual_mm=float(np.median(fd)),
            median_residual_mm=float(np.median(dist)),p95_residual_mm=float(np.quantile(dist,.95)),
            max_residual_mm=float(dist.max()),inlier_ratio=float(np.mean(dist<=threshold_mm)),
            held_out_max_mm=float(hd.max()),held_out_p95_mm=float(np.quantile(hd,.95)),
            residuals_mm=dist.tolist(),sample_indices=idx.tolist()))
    results.sort(key=lambda x:(x['fit_p95_residual_mm'],x['fit_median_residual_mm']))
    best=results[0]
    alternatives=[x for x in results[1:] if np.linalg.norm(np.asarray(x['rotation'])-best['rotation'])>.1 or
                  np.linalg.norm(np.asarray(x['translation_mm'])-best['translation_mm'])>threshold_mm]
    gap=alternatives[0]['fit_p95_residual_mm']-best['fit_p95_residual_mm'] if alternatives else None
    supported=best['max_residual_mm']<=threshold_mm and best['held_out_max_mm']<=threshold_mm and (gap is None or gap>threshold_mm)
    return dict(status='geometry_supported_unreviewed' if supported else 'registration_not_validated',
        best=best,alternatives=alternatives[:3],alternative_p95_gap_mm=gap,threshold_mm=threshold_mm,
        fit_point_count=len(fit),held_out_point_count=len(holdout),
        source_to_cad_registered=False,automatic_application=False,metrology_accepted=False,
        authority='unpaired_display_surface_icp_only',source_rank=int(np.linalg.matrix_rank(a-a.mean(0),tol=1e-6)),
        planar_source=bool(np.linalg.matrix_rank(a-a.mean(0),tol=1e-6)==2),
        sample_distance_is_surface_upper_bound=True,
        selection_uses_fit_points_only=True)


def surface_seed_candidates(source, triangles, *, threshold_mm=.1, index=None, limit=8):
    """Search CAD-derived height levels on fit points only, with no project offsets."""
    a=np.asarray(source,float);tri=np.asarray(triangles,float)
    fit=a[np.arange(len(a))%5!=0]
    index=index or SurfaceIndex(tri)
    extent=tri.reshape(-1,3).max(0)-tri.reshape(-1,3).min(0)
    rotations=list(axis_rotations())
    # Prioritize orientations with compatible aspect ratios. This is retrieval,
    # not a registration proof; the original ICP alternatives remain available.
    ranked=sorted(rotations,key=lambda r:np.linalg.norm(np.ptp(fit@r.T,axis=0)/np.maximum(extent,1e-6)-1))
    raw=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);area=np.linalg.norm(raw,axis=1)
    centers=tri.mean(1);horizontal=(area>1e-9)&(np.abs(raw[:,2])/np.maximum(area,1e-20)>.995)
    levels=[]
    if horizontal.any():
        h=centers[horizontal,2];w=area[horizontal];bin_width=threshold_mm/2
        bins=np.floor(h/bin_width).astype(int)
        for key in sorted(set(bins),key=lambda k:float(w[bins==k].sum()),reverse=True)[:12]:
            mask=bins==key;levels.append(float(np.average(h[mask],weights=w[mask])))
    results=[]
    for rotation in ranked[:4]:
        x=fit@rotation.T
        centered=(tri.reshape(-1,3).min(0)+tri.reshape(-1,3).max(0))/2-(x.min(0)+x.max(0))/2
        starts=[centered,np.zeros(3)]
        source_levels=np.unique(np.round(x[:,2]/(threshold_mm/2))*(threshold_mm/2))
        source_levels=sorted(source_levels,key=lambda z:sum(np.abs(x[:,2]-z)<=threshold_mm/4),reverse=True)[:4]
        for level in levels:
            for z in source_levels:
                starts.append(np.array([centered[0],centered[1],level-z]))
        for translation in starts:
            distance=np.array([index.nearest(p,exact=False)['distance_mm'] for p in x+translation])
            results.append(dict(rotation=rotation.tolist(),translation_mm=translation.tolist(),
                                seed_p95_mm=float(np.quantile(distance,.95)),
                                seed_median_mm=float(np.median(distance)),
                                initialization='CAD_area_weighted_surface_levels',fit_only=True))
    results.sort(key=lambda row:(row['seed_p95_mm'],row['seed_median_mm']))
    unique=[]
    for row in results:
        if any(np.linalg.norm(np.asarray(row['rotation'])-np.asarray(old['rotation']))<1e-6 and
               np.linalg.norm(np.asarray(row['translation_mm'])-np.asarray(old['translation_mm']))<threshold_mm/2
               for old in unique):continue
        unique.append(row)
        if len(unique)>=limit:break
    return unique


def refine_surface_fit(source,triangles,candidates,*,threshold_mm=.1,iterations=25,index=None):
    """Refine sample fits against triangles; certify final nearest-distance queries."""
    a=np.asarray(source,float);tri=np.asarray(triangles,float)
    index=index or SurfaceIndex(tri)
    fit=a[np.arange(len(a))%5!=0];results=[]
    def project(points,exact=False):
        hits=[index.nearest(p,exact=exact) for p in points]
        return np.asarray([h['point'] for h in hits]),np.asarray([h['distance_mm'] for h in hits])
    for candidate in candidates:
        r=np.asarray(candidate['rotation']);t=np.asarray(candidate['translation_mm'])
        for _ in range(iterations):
            b,dist=project(fit@r.T+t);mask=dist<=np.quantile(dist,.9)
            x=fit[mask];y=b[mask];u,_,vt=np.linalg.svd((x-x.mean(0)).T@(y-y.mean(0)))
            nr=vt.T@u.T
            if np.linalg.det(nr)<0:vt[-1]*=-1;nr=vt.T@u.T
            nt=y.mean(0)-nr@x.mean(0);delta=np.linalg.norm(nt-t)+np.linalg.norm(nr-r);r,t=nr,nt
            if delta<1e-7:break
        b,dist=project(a@r.T+t,exact=True);hd=dist[np.arange(len(a))%5==0]
        fd=dist[np.arange(len(a))%5!=0]
        results.append(dict(rotation=r.tolist(),translation_mm=t.tolist(),cad_points_mm=b.tolist(),
            fit_p95_residual_mm=float(np.quantile(fd,.95)),fit_max_residual_mm=float(fd.max()),
            initialization=candidate.get('initialization','sample_ICP'),
            residuals_mm=dist.tolist(),max_residual_mm=float(dist.max()),p95_residual_mm=float(np.quantile(dist,.95)),
            median_residual_mm=float(np.median(dist)),held_out_max_mm=float(hd.max()),
            inlier_ratio=float(np.mean(dist<=threshold_mm))))
    results.sort(key=lambda r:(r['fit_p95_residual_mm'],r['fit_max_residual_mm']))
    best=results[0]
    return dict(best=best,alternatives=results[1:],threshold_mm=threshold_mm,
        status='surface_fit_unreviewed' if best['max_residual_mm']<=threshold_mm else 'surface_residual_rejected',
        source_to_cad_registered=False,automatic_application=False,metrology_accepted=False,
        authority='unpaired_display_triangle_fit_only',independent_semantic_correspondence_verified=False,
        selection_uses_fit_points_only=True)
