"""Explicit SCH_1500000_15003 layout subset for legacy assembly exports.

The reviewed standard profile already declares the 15003 BODY/geometry layouts
and its LIST delta. This profile extends its observed membership with the
public XT assembly, attribute, spline and transform declarations. It does not
rename an external catalog or choose the nearest schema version.
"""
from pathlib import Path
from dataclasses import replace
import hashlib

BASE_SHA256='0dd291ea706fc306f16a78140e05e595e75c85ab63e4077e68941b127fcfdcc3'
KEY='SCH_1500000_15003'
# Membership observed in the retained legacy assembly; all fields are explicit
# declarations, not token-count guessing. Unknown future types still fail.
TYPES={10,11,12,13,14,15,16,17,18,19,29,30,31,32,38,40,41,45,50,51,52,53,54,56,59,60,67,70,74,79,80,81,82,83,84,100,124,125,126,127,128,133,134,135,136,137,141}


def legacy_provider(source, schema_dir):
    from parasolid_kit import load_schema_catalog,InMemorySchemaProvider,inspect_xt
    if Path(source).suffix.lower()!='.x_t':return None
    header=inspect_xt(source)
    if header.schema_key!=KEY: return None
    from parasolid_kit.schema import SchemaCatalog,FieldDefinition,FieldType
    path=Path(schema_dir)/'sch_13006.sch_txt'
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=BASE_SHA256:
        raise ValueError('Reviewed public base declarations are missing or changed')
    catalog=load_schema_catalog(path)
    declarations=[]
    for definition in catalog.definitions:
        if definition.node_type not in TYPES:continue
        if definition.node_type==70:
            f=definition.fields
            # Exact delta from icad_standard.rs::LIST for standard 15003.
            fields=(f[0],FieldDefinition('entry_kind',FieldType.UNSIGNED_BYTE,0,0,True),
                    FieldDefinition('transmission_flag',FieldType.LOGICAL,0,0,True),
                    *f[1:4],*f[5:7],FieldDefinition('cursor_index',FieldType.INTEGER,0,0,True),
                    FieldDefinition('cursor_block',FieldType.POINTER_INDEX,1012,0,True),f[8])
            definition=replace(definition,fields=fields,variable=False)
        declarations.append(definition)
    if {d.node_type for d in declarations}!=TYPES:
        raise ValueError('Legacy profile has missing declared types')
    profile=SchemaCatalog('15003',tuple(declarations),source_path='oracle-public-reference-15003-subset-v1')
    return InMemorySchemaProvider([profile])
