"""Add only the unchanged inherited config and sole-lease driver to this release."""
from pathlib import Path
import json
root=Path(__file__).parent;release=root/'release';release.mkdir(exist_ok=True)
inputs=[(root/'run_dfl_queue.py',release/'run_dfl_queue.py'),
        (root.parent/'2026-09-08_probe_同帧选择覆盖/witness_release/llvip_N_s42_FT3.yaml',release/'llvip_N_s42_FT3.yaml')]
rows=[]
for src,dest in inputs:
    b=src.read_bytes()
    with dest.open('xb') as f:f.write(b)
    rows.append(dict(source=str(src),destination=str(dest),bytes=len(b),byte_exact=dest.read_bytes()==b))
with (root/'RELEASE_INPUTS.json').open('x',encoding='utf-8') as f:json.dump(dict(files=rows,new_hash_computed=False),f,ensure_ascii=False,indent=2)
print('RELEASE_INPUTS_READY',len(rows))
