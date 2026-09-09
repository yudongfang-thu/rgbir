import urllib.request,json,time,subprocess,os
from pathlib import Path
r=Path("/mnt/dataX/ydf/projects/RGBT_campaign_90/artifacts/reproduction_20260909_attempt1")
t=Path("/mnt/dataX/ydf/projects/RGBT_campaign_90/external_reproductions/cft")
url="https://drive.usercontent.google.com/download?id=18yLDUOxNXQ17oypQ-fAV9OS9DESOZQtV&export=download&authuser=0&confirm=t"
start=time.time()
try:
 p=subprocess.run(["git","clone","--depth","1","https://github.com/DocF/multispectral-object-detection.git",str(t/"author_source")],timeout=240,check=True)
 (r/"source_downloaded.json").write_text(json.dumps({"source_url":"https://github.com/DocF/multispectral-object-detection","status":"DOWNLOADED","revision":subprocess.check_output(["git","-C",str(t/"author_source"),"rev-parse","HEAD"],text=True).strip()}))
 with urllib.request.urlopen(url,timeout=90) as resp,(t/"author_llvip.pt").open("xb") as out:
  total=0
  while True:
   b=resp.read(1024*1024)
   if not b:break
   out.write(b);total+=len(b)
   if total%(32*1024*1024)==0:print("weight_bytes",total,flush=True)
 assert total==413453231,total
 result={"status":"DOWNLOADED","weight":str(t/"author_llvip.pt"),"bytes":total,"url":url,"elapsed_s":time.time()-start,"new_digest_calculated":False}
 (r/"weight_downloaded.json").write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
except Exception as e:
 (r/"fetch_failed.json").write_text(json.dumps({"error":repr(e),"elapsed_s":time.time()-start}));raise
