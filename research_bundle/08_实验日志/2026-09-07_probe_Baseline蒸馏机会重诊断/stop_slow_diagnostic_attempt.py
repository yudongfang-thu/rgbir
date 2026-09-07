"""Stop only this task's diagnostic tree; preserve all partial raw artifacts."""
import subprocess
CODE=r'''
from pathlib import Path
import psutil,json,time
root=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_baseline_information_20260907/attempt2')
target=psutil.Process(135827)
assert str(root/'run_export_campaign.py') in target.cmdline()
children=target.children(recursive=True)
record={'status':'STOPPED_TECHNICAL_THROUGHPUT','reason':'per-object GPU synchronization bottleneck; 25 images in 112.56 seconds; no outcome selection','pid':target.pid,'children':[p.pid for p in children],'time':time.time(),'next_attempt':'attempt3 image-level CPU postprocessing','original_artifacts_preserved':True}
with (root/'throughput_stop_receipt.json').open('x') as f:json.dump(record,f,indent=2)
target.terminate()
for child in reversed(children):
    try:child.terminate()
    except psutil.NoSuchProcess:pass
_,alive=psutil.wait_procs(children+[target],timeout=5)
for child in alive:child.kill()
print(json.dumps(record))
'''
result=subprocess.run(['ssh','94','/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python','-'],input=CODE.encode(),capture_output=True,check=True)
print(result.stdout.decode());print(result.stderr.decode())
