"""Decode and audit a Parasolid source without certifying its display conversion."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from oracle_viewer.cad_diagnostics import source_report
from oracle_viewer.parasolid_legacy import legacy_provider
from parasolid_kit import read_brep

p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('--schema-root',type=Path,default=Path('/home/cyg/Oracle/third_party/parasolid-schemas'));p.add_argument('--output',type=Path,required=True);args=p.parse_args()
try:decoded=read_brep(args.source)
except Exception:
    provider=legacy_provider(args.source,args.schema_root)
    decoded=(read_brep(args.source,schema_provider=provider) if provider is not None
             else read_brep(args.source,schema_dir=str(args.schema_root)))
args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(source_report(decoded,args.source),indent=2))
