from pathlib import Path
import urllib.request, shutil, json, time
r=Path('/mnt/dataX/ydf/projects/RGBT_campaign_90')
a=r/'artifacts/cft_author_protocol_20260909_attempt1'
out=r/'external_reproductions/cft/author_initialization/yolov5l.pt'
out.parent.mkdir(parents=True,exist_ok=True)
url='https://drive.usercontent.google.com/download?id=12OFGLF73CqTgOCMJAycZ8lB4eW19D0nb&export=download&authuser=0&confirm=t'
t=time.perf_counter()
try:
 with urllib.request.urlopen(url,timeout=45) as f,out.open('xb') as dest:shutil.copyfileobj(f,dest)
 assert out.stat().st_size>1000000
 receipt=dict(status='DOWNLOADED',url=url,path=str(out),bytes=out.stat().st_size,elapsed_seconds=time.perf_counter()-t,no_new_digest=True)
 (a/'initialization_download.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(receipt),flush=True)
except Exception as exc:
 (a/'initialization_failure.json').write_text(json.dumps(dict(error=repr(exc)),indent=2)+'\n')
 raise
