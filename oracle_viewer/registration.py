"""Rigid frame fitting with explicit correspondences and held-out evidence.

Mesh/marker fits are proposals, never independent metrology calibration.
"""
from __future__ import annotations
import hashlib
import json
import numpy as np


def fit_rigid(source, target, *, max_residual_mm=0.1, allow_reflection=False):
    a, b = np.asarray(source, float), np.asarray(target, float)
    if a.ndim != 2 or a.shape[1] != 3 or a.shape != b.shape or len(a) < 4:
        raise ValueError('A frame fit requires at least four paired XYZ points')
    if not np.isfinite(a).all() or not np.isfinite(b).all() or not np.isfinite(max_residual_mm) or max_residual_mm <= 0:
        raise ValueError('Coordinates and residual limit must be finite')
    # Four non-coplanar points constrain handedness; a planar fit cannot resolve
    # which side/layer the frame belongs to, even with a zero training residual.
    if np.linalg.matrix_rank(a-a.mean(0), tol=1e-6) < 3 or np.linalg.matrix_rank(b-b.mean(0), tol=1e-6) < 3:
        return dict(status='underdetermined', reason='Non-coplanar correspondence evidence is missing', source_to_cad_registered=False)

    def solve(x, y):
        u, _, vt = np.linalg.svd((x-x.mean(0)).T @ (y-y.mean(0)))
        r = vt.T @ u.T
        if np.linalg.det(r) < 0 and not allow_reflection:
            vt[-1] *= -1; r = vt.T @ u.T
        return r, y.mean(0)-r@x.mean(0)

    r, t = solve(a, b)
    residuals = np.linalg.norm(a@r.T+t-b, axis=1)
    # No correspondence is silently dropped; outliers remain in the report.
    held_out = []
    for i in range(len(a)):
        mask = np.arange(len(a)) != i
        if np.linalg.matrix_rank(a[mask]-a[mask].mean(0), tol=1e-6) < 3:
            held_out.append(None); continue
        ri, ti = solve(a[mask], b[mask])
        held_out.append(float(np.linalg.norm(ri@a[i]+ti-b[i])))
    tested = [v for v in held_out if v is not None]
    supported = bool(tested) and max(tested) <= max_residual_mm and max(residuals) <= max_residual_mm
    return dict(status='geometrically_supported_proposal' if supported else 'residual_rejected',
                rotation=r.tolist(), translation_mm=t.tolist(), determinant=float(np.linalg.det(r)),
                residuals_mm=residuals.tolist(), max_residual_mm=float(max(residuals)),
                rms_residual_mm=float(np.sqrt(np.mean(residuals**2))),
                held_out_residuals_mm=held_out, held_out_max_mm=max(tested) if tested else None,
                threshold_mm=max_residual_mm, source_to_cad_registered=False,
                authority='paired_source_points_and_display_mesh_only',
                metrology_accepted=False)


def frame_proposals(tasks, max_residual_mm=0.1):
    buckets = {}
    for task in tasks:
        mapping = task.get('msop_coordinate_map') or {}
        for p in mapping.get('mapped_points', []):
            frame = p.get('coordinate_system')
            if not frame:
                continue
            row = dict(name=p['name'], source_xyz=p['source_xyz'], cad_xyz=p['cad_position_mm'],
                       face_id=p['face_id'], source_pages=p.get('source_pages', []), fai=task['number'])
            row['program'] = p.get('program')
            buckets.setdefault((frame, p.get('program')), []).append(row)
    proposals = []
    for (frame, program), rows in sorted(buckets.items(), key=lambda pair: (pair[0][0], pair[0][1] or '')):
        # Repeated point names are not independent support. Contradictory names
        # are reported instead of choosing the first occurrence.
        by_name = {}
        for row in rows: by_name.setdefault(row['name'], []).append(row)
        unique, conflicts = [], []
        for name, values in by_name.items():
            if any(not np.allclose(v['source_xyz'], values[0]['source_xyz'], atol=1e-6, rtol=0)
                   or not np.allclose(v['cad_xyz'], values[0]['cad_xyz'], atol=max_residual_mm, rtol=0)
                   for v in values[1:]):
                conflicts.append(dict(name=name, occurrences=values)); continue
            unique.append(values[0])
        if len(unique) >= 4:
            result = fit_rigid([v['source_xyz'] for v in unique], [v['cad_xyz'] for v in unique],
                               max_residual_mm=max_residual_mm)
        else: result = dict(status='insufficient_correspondences', source_to_cad_registered=False)
        evidence = dict(frame=frame, program=program, correspondences=unique, conflicts=conflicts)
        proposals.append(evidence | result | dict(evidence_sha256=hashlib.sha256(
            json.dumps(evidence, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
            automatic_application=False))
    return proposals
