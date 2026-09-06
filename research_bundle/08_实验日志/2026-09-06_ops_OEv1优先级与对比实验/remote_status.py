import json, subprocess
from pathlib import Path
out={}
for pid in (3894177,3894186,878944,878988):
    p=Path('/proc')/str(pid)
    if p.exists():
        out[str(pid)]={'cmd':(p/'cmdline').read_bytes().decode().split('\0'), 'status':(p/'status').read_text().splitlines()[:8]}
out['screens']=subprocess.run(['screen','-ls'],capture_output=True,text=True).stdout
out['osssl_processes']=[line for line in subprocess.run(['ps','-u','yudongfang','-o','pid,ppid,pgid,stat,args'],capture_output=True,text=True).stdout.splitlines() if 'osssl' in line or '878988' in line or '878944' in line]
print(json.dumps(out,indent=2))
