"""Mirror small final narrative/analysis to a fresh project-data directory on 94."""
from pathlib import Path
import base64,json,subprocess
root=Path(__file__).parent
names=['FINAL_REPORT.md','README.md','RESOURCE_EXCERPT.json','DEV_RECHECK_INPUT_AUDIT.md',
 'analysis_final/README.md','analysis_final/summary.json',
 'witness_analysis_final/README.md','witness_analysis_final/summary.json',
 'WITNESS_REAL_READOUT_REVIEW.md','WITNESS_ANALYZER_INDEPENDENT_REVIEW.md']
payload=[]
for name in names:
    b=(root/name).read_bytes()
    if len(b)>1000000:raise ValueError('Unexpected large report')
    payload.append(dict(path=name,data=base64.b64encode(b).decode()))
dest='/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_selection_coverage_20260908/reports_20260908_1330'
script="""from pathlib import Path
import base64,json
root=Path(%r);root.mkdir(exist_ok=False)
rows=[]
for item in json.loads(%r):
 p=root/item['path'];b=base64.b64decode(item['data'])
 if root not in p.resolve().parents:raise ValueError('Invalid path')
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(b)
 if p.read_bytes()!=b:raise ValueError('Copy differs')
 rows.append(dict(path=str(p),bytes=len(b),byte_exact=True))
r=dict(status='FINAL_REPORTS_MIRRORED',destination=str(root),local_origin='E:/SHARE/光sar/08_实验日志/2026-09-08_probe_同帧选择覆盖',files=rows,new_hash_computed=False)
(root/'mirror_receipt.json').write_text(json.dumps(r,ensure_ascii=False,indent=2))
print(json.dumps(r,ensure_ascii=False))
"""%(dest,json.dumps(payload))
py='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
r=subprocess.run(['ssh','94',py,'-'],input=script.encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace'))
receipt=json.loads(r.stdout)
with (root/'FINAL_REMOTE_MIRROR_RECEIPT.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,ensure_ascii=False,indent=2)
print(receipt['status'],len(receipt['files']))
