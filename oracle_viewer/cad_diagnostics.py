"""Source decoding evidence independent of whether a display mesh can be built."""
from collections import Counter
import hashlib
from pathlib import Path


def source_report(decoded,source):
    brep=decoded.brep
    return dict(source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                schema_key=brep.schema_key,source_decoding_complete=brep.complete,
                source_topology_valid=brep.topology.valid,source_face_count=len(brep.faces),
                source_edge_count=len(brep.edges),source_vertex_count=len(brep.vertices),
                surface_types=dict(Counter(type(s.definition).__name__ for s in brep.surfaces)),
                raw_node_count=len(decoded.document.nodes),metrology_ready=False,
                evidence_scope='source_entities_only_not_display_or_metrology_acceptance')
