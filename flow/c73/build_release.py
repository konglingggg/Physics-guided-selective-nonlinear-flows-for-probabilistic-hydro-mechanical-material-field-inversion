#!/usr/bin/env python3
"""Create verified standalone C73 release. Zip includes physical source + all checkpoints."""
from pathlib import Path
import json,hashlib,zipfile
root=Path(__file__).resolve().parent
out=root.parent/'AI4THMC_C73_HM_DualGeometry_ConditionalFlow_20261111.zip'
manifest=[]
for p in sorted(root.rglob('*')):
    if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc':
        content=p.read_bytes()
        manifest.append({'path':str(p.relative_to(root)),'size':len(content),'sha256':hashlib.sha256(content).hexdigest()})
(root/'SHA256_manifest.json').write_text(json.dumps(manifest,indent=2))
with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
    for p in sorted(root.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc':
            z.write(p,arcname=f'C73_HM_Information_Routing_20261111/{p.relative_to(root)}')
with zipfile.ZipFile(out) as z:
    bad=z.testzip();assert bad is None,bad
    n=len(z.namelist())
print(json.dumps({'zip':str(out),'zip_bytes':out.stat().st_size,'n_files':n,'integrity':'passed'},indent=2))
