#!/usr/bin/env python3
"""Validate one source CAD file with the Viewer FreeCAD reader."""
from __future__ import annotations
import json,sys,time
from pathlib import Path

def main():
 name=Path(sys.argv[1]).name;started=time.time()
 try:
  sys.path.append('/usr/lib/freecad-python3/lib')
  import FreeCAD,Part
  shape=Part.read(sys.argv[1])
  if shape.isNull() or not shape.Faces:
   row={'file':name,'status':'no_faces'}
  else:
   valid=bool(shape.isValid());bb=shape.BoundBox
   row={'file':name,'status':'renderable' if valid else 'invalid_brep','brep_valid':valid,'body_count':len(shape.Solids) or 1,'face_count':len(shape.Faces),'bbox_mm':[round(bb.XLength,4),round(bb.YLength,4),round(bb.ZLength,4)]}
 except Exception as exc: row={'file':name,'status':'error','error':repr(exc)[:1000]}
 row['seconds']=round(time.time()-started,2);print('ORACLE_RESULT '+json.dumps(row,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
