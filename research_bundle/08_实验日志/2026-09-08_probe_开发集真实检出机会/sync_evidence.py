"""Mirror accepted CPU evidence to a fresh directory on the 94 project data disk."""
from pathlib import Path
import json,base64,subprocess
root=Path(__file__).parent
if 'ACCEPTED' not in (root/'REVIEW.md').read_text(encoding='utf-8'):raise ValueError('Review required')
payload=[]
for p in sorted(root.rglob('*')):
    if not p.is_file():continue
    rel=p.relative_to(root)
    if '__pycache__' in rel.parts or p.suffix.lower() not in {'.py','.md','.json','.jsonl','.csv','.tsv','.yaml','.txt','.gz'}:continue
    b=p.read_bytes()
    if len(b)>8000000:raise ValueError('Unexpected large file')
    payload.append(dict(path=rel.as_posix(),bytes=len(b),data=base64.b64encode(b).decode()))
if sum(r['bytes'] for r in payload)>20000000:raise ValueError('Unexpected large evidence package')
dest='/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_dev_detection_opportunities_20260908/review_v1'
script="""from pathlib import Path
import json,base64
root=Path(%r);root.mkdir(parents=True,exist_ok=False)
rows=[]
for item in json.loads(%r):
 p=root/item['path'];b=base64.b64decode(item['data'])
 if root not in p.resolve().parents:raise ValueError('Invalid destination')
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(b)
 if p.read_bytes()!=b:raise ValueError('Copy differs')
 rows.append(dict(path=str(p),bytes=len(b),byte_exact=True))
r=dict(status='CPU_EVIDENCE_MIRRORED',local_origin='E:/SHARE/光sar/08_实验日志/2026-09-08_probe_开发集真实检出机会',destination=str(root),files=rows,new_hash_computed=False)
(root/'mirror_receipt.json').write_text(json.dumps(r,ensure_ascii=False,indent=2))
print(json.dumps(r,ensure_ascii=False))
"""%(dest,json.dumps(payload))
py='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
r=subprocess.run(['ssh','94',py,'-'],input=script.encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace'))
receipt=json.loads(r.stdout)
with (root/'REMOTE_MIRROR_RECEIPT.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,ensure_ascii=False,indent=2)
print(receipt['status'],len(receipt['files']),sum(x['bytes'] for x in receipt['files']))
