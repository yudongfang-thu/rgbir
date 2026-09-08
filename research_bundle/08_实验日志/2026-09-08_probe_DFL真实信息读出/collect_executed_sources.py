"""Bind the producer's executed copies and pinned test logs without hashes."""
from pathlib import Path
import base64,json,subprocess
root=Path(__file__).parent;B='/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_dfl_information_20260908'
files={'run_dfl_probe.py':B+'/attempt1/probe/sources/run_dfl_probe.py','dfl_export.py':B+'/attempt1/probe/sources/dfl_export.py',
 'test_dfl_cpu.py':B+'/attempt1/probe/sources/test_dfl_cpu.py','run_dfl_queue.py':B+'/attempt1/queue/executed_driver.py'}
code="""from pathlib import Path
import base64,json
files=%r
print(json.dumps({k:base64.b64encode(Path(v).read_bytes()).decode() for k,v in files.items()}))
"""%files
py='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
p=subprocess.run(['ssh','94',py,'-'],input=code.encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace'))
rows=json.loads(p.stdout);dest=root/'executed_source_copies';dest.mkdir(exist_ok=False);receipt=[]
for name,b64 in rows.items():
 b=base64.b64decode(b64)
 assert b==(root/'independent_review/reviewed_source'/name).read_bytes()==(root/'release'/name).read_bytes()
 with (dest/name).open('xb') as f:f.write(b)
 receipt.append(dict(file=name,remote=files[name],bytes=len(b),byte_exact_to_reviewed_source=True))
with (dest/'receipt.json').open('x',encoding='utf-8') as f:json.dump(dict(status='EXECUTED_COPIES_MATCH_REVIEWED_SOURCE',files=receipt,new_hash_computed=False),f,indent=2)
print('EXECUTED_COPIES_MATCH_REVIEWED_SOURCE',len(receipt))
