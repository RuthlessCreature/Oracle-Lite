"""Exact triangle proximity without scanning every display triangle per point."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .visibility import closest_surface


class SurfaceIndex:
    """Enclosing-sphere buckets retain large faces missed by centroid-only search."""
    def __init__(self, triangles):
        tri = np.asarray(triangles, dtype=float).reshape(-1, 3, 3)
        valid = np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1) > 1e-9
        self.source_indices = np.flatnonzero(valid)
        self.triangles = tri[valid]
        if not len(self.triangles):
            raise ValueError('CAD has no nondegenerate display triangles')
        self.centers = self.triangles.mean(1)
        radii = np.linalg.norm(self.triangles-self.centers[:, None], axis=2).max(1)
        self.tree = cKDTree(self.centers)
        groups = np.floor(np.log2(np.maximum(radii, 1e-12))).astype(int)
        self.buckets = []
        for group in np.unique(groups):
            ids = np.flatnonzero(groups == group)
            self.buckets.append((ids, cKDTree(self.centers[ids]), float(radii[ids].max())))

    def nearest(self, point, *, exact=True):
        p = np.asarray(point, float)
        if p.shape != (3,) or not np.isfinite(p).all():
            raise ValueError('Expected finite XYZ')
        _, ids = self.tree.query(p, k=min(64, len(self.triangles)))
        ids = np.atleast_1d(ids)
        best = self._hit(p, ids)
        if exact:
            for bucket_ids, tree, radius in self.buckets:
                # A triangle cannot be closer than center distance minus its
                # enclosing radius. Test every sphere that can improve the hit.
                near = tree.query_ball_point(p, best['distance_mm']+radius+1e-7)
                if not near:
                    continue
                hit = self._hit(p, bucket_ids[near])
                if hit['distance_mm'] < best['distance_mm']:
                    best = hit
        return best

    def nearby_triangles(self, point, distance_mm):
        """All triangles that could intersect a sphere, retaining original IDs."""
        p=np.asarray(point,float)
        if p.shape!=(3,) or not np.isfinite(p).all() or distance_mm<0 or not np.isfinite(distance_mm):
            raise ValueError('Expected finite XYZ and a nonnegative radius')
        ids=[]
        for bucket_ids,tree,radius in self.buckets:
            near=tree.query_ball_point(p,float(distance_mm)+radius+1e-7)
            if near:ids.extend(self.source_indices[bucket_ids[near]].tolist())
        return np.asarray(ids,dtype=int)

    def _hit(self, p, ids):
        hit = closest_surface(p, self.triangles[ids])
        # Find the winning triangle without relying on equality of rounded
        # positions or triangle winding; only a few closed triangles hit here.
        q = np.asarray(hit['point'])
        t = self.triangles[ids]
        normals = np.cross(t[:, 1]-t[:, 0], t[:, 2]-t[:, 0])
        length = np.linalg.norm(normals, axis=1)
        plane = np.abs(np.einsum('ij,ij->i', q-t[:, 0], normals))/length
        candidates = np.flatnonzero(plane < 1e-7)
        from .annotations import triangle_distance
        winner = next((int(i) for i in candidates if triangle_distance(q, *t[i]) < 1e-12), None)
        if winner is None:
            winner = min(range(len(t)), key=lambda i: triangle_distance(q, *t[i]))
        hit['triangle_index'] = int(self.source_indices[int(ids[winner])])
        hit['normal'] = (normals[winner]/length[winner]).tolist()
        return hit
