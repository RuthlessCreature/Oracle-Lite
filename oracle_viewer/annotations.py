"""Validate visual pins against the active display mesh, never as measurements."""

from math import isfinite

MAX_ANNOTATION_POINTS = 2000


def vector(values, size):
    if (
        not isinstance(values, (list, tuple))
        or len(values) != size
        or any(
            isinstance(v, bool) or not isinstance(v, (float, int)) or not isfinite(v)
            for v in values
        )
    ):
        raise ValueError(f"Expected {size} finite coordinates")
    return [float(v) for v in values]


def sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def segment_distance(p, a, b):
    ab, ap = sub(b, a), sub(p, a)
    length = dot(ab, ab)
    t = min(1.0, max(0.0, dot(ap, ab) / length)) if length else 0.0
    d = tuple(ap[i] - t * ab[i] for i in range(3))
    return dot(d, d)


def triangle_distance(p, a, b, c):
    """Squared distance to the closed triangle, including degenerate triangles."""
    ab, ac, ap = sub(b, a), sub(c, a), sub(p, a)
    d00, d01, d11 = dot(ab, ab), dot(ab, ac), dot(ac, ac)
    denominator = d00 * d11 - d01 * d01
    if denominator > 1e-24:
        u = (d11 * dot(ap, ab) - d01 * dot(ap, ac)) / denominator
        v = (d00 * dot(ap, ac) - d01 * dot(ap, ab)) / denominator
        if u >= 0 and v >= 0 and u + v <= 1:
            d = tuple(ap[i] - u * ab[i] - v * ac[i] for i in range(3))
            return dot(d, d)
    return min(
        segment_distance(p, a, b), segment_distance(p, b, c), segment_distance(p, c, a)
    )


def validate_anchors(mesh, anchors):
    if not isinstance(anchors, list) or not 1 <= len(anchors) <= MAX_ANNOTATION_POINTS:
        raise ValueError(f"Select between 1 and {MAX_ANNOTATION_POINTS} annotation points")
    faces = {f["id"]: f for f in mesh["faces"]}
    result = []
    for anchor in anchors:
        if not isinstance(anchor, dict) or anchor.get("face_id") not in faces:
            raise ValueError("Annotation must reference a face in the active CAD")
        position = vector(anchor.get("position"), 3)
        offset = vector(anchor.get("label_offset", [24, -24]), 2)
        if max(abs(v) for v in offset) > 2000:
            raise ValueError("Label offset exceeds the viewport layout limit")
        face = faces[anchor["face_id"]]
        xyz, indices = face["positions"], face["indices"]
        # Ray picking uses Float32; allow 1 micron for transport/rounding only.
        found = False
        for i in range(0, len(indices), 3):
            points = [xyz[3 * j : 3 * j + 3] for j in indices[i : i + 3]]
            if triangle_distance(position, *points) <= 1e-6:
                found = True
                break
        if not found:
            raise ValueError("Annotation point is not on its selected display face")
        result.append(
            {
                "face_id": face["id"],
                "position": position,
                "label_offset": offset,
                "body": face.get("body"),
                "source_face_id": face.get("face"),
                "instance_path": face.get("instance_path", []),
            }
        )
    return result
