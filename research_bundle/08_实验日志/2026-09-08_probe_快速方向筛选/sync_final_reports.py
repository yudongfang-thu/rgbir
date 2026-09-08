"""Mirror final small analysis artifacts to a new directory on the project data disk."""
from pathlib import Path
import base64,json,subprocess
root=Path(__file__).parent
names=['FINAL_REPORT.md','README.md','ERRATA.md','RESOURCE_ADMISSION_EXCERPT.json',
       'DIRECTIONS_AND_PROXY_LIMITS.md','OLD_QUEUE_PROGRESS_1220.json',
       'feature_gm_analysis_1225/README.md','feature_gm_analysis_1225/summary.json',
       'confidence_analysis_1210/README.md','confidence_analysis_1210/summary.json',
       'cpu_selection_coverage_audit_v1/README.md','cpu_selection_coverage_audit_v1/summary.json']
payload=[]
for name in names:
    data=(root/name).read_bytes()
    if len(data)>1000000:raise ValueError('Unexpected report size')
    payload.append(dict(path=name,content=base64.b64encode(data).decode()))
dest='/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_direction_screen_20260908/reports_20260908_1230'
script="""from pathlib import Path
import json,base64
root=Path(%r);root.mkdir(exist_ok=False)
rows=[]
for row in json.loads(%r):
 p=root/row['path']
 if root not in p.resolve().parents:raise ValueError('Invalid destination')
 p.parent.mkdir(parents=True,exist_ok=True);b=base64.b64decode(row['content'])
 with p.open('xb') as f:f.write(b)
 if p.read_bytes()!=b:raise AssertionError('Mirror differs')
 rows.append(dict(path=str(p),bytes=len(b),byte_exact=True))
receipt=dict(status='FINAL_SMALL_REPORTS_MIRRORED',local_origin='E:/SHARE/光sar/08_实验日志/2026-09-08_probe_快速方向筛选',destination=str(root),files=rows,new_hash_computed=False)
(root/'mirror_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
print(json.dumps(receipt,ensure_ascii=False))
"""%(dest,json.dumps(payload))
PY='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
r=subprocess.run(['ssh','94',PY,'-'],input=script.encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace'))
receipt=json.loads(r.stdout)
with (root/'FINAL_REMOTE_MIRROR_RECEIPT.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,ensure_ascii=False,indent=2)
print(receipt['status'],len(receipt['files']),receipt['destination'])
