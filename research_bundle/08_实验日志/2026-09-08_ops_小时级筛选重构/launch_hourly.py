"""Launch the authorized small FT matrix through screen and the existing lease."""
import base64
import argparse
import datetime
import json
from pathlib import Path
import subprocess
from prepare_hourly_release import PY,CAMPAIGN,HERE

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--release',default='release_v2');args=parser.parse_args()
    release=CAMPAIGN+'/'+args.release;output=CAMPAIGN+'/ft_screen_attempt1'
    sources={str(p.relative_to(HERE/'release')).replace('\\','/'):base64.b64encode(p.read_bytes()).decode()
        for p in (HERE/'release').rglob('*') if p.is_file() and p.suffix in ('.py','.yaml') and '__pycache__' not in p.parts}
    code="""import base64,json,subprocess,sys,time
from pathlib import Path
release=Path(RELEASE);output=Path(OUTPUT)
assert not output.exists()
for name,content in SOURCES.items():assert (release/name).read_bytes()==base64.b64decode(content),name
sys.path.insert(0,str(release))
import hourly_common,run_hourly_queue as queue
for p in (release/'configs').glob('*.yaml'):hourly_common.load_config(p)
assert queue.NATIVE_BINDING.is_file()
for p in release.glob('*.py'):compile(p.read_text(),str(p),'exec')
snapshot=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total,memory.used,memory.free','--format=csv,noheader']).decode()
log=release.parent/'ft_screen_attempt1_screen.log'
assert not log.exists()
subprocess.run(['screen','-dmS','rgbirhourly_screen_42','-L','-Logfile',str(log),sys.executable,
 str(release/'run_hourly_queue.py'),'--release-dir',str(release),'--output',str(output)],check=True)
print(json.dumps(dict(status='SCREEN_QUEUE_DISPATCHED',release=str(release),output=str(output),
 screen='rgbirhourly_screen_42',gpu_before=snapshot,time=time.time(),new_hash_computed=False)))
""".replace('RELEASE',repr(release)).replace('OUTPUT',repr(output)).replace('SOURCES',repr(sources))
    r=subprocess.run(['ssh','94',PY,'-'],input=code.encode(),capture_output=True,timeout=60)
    receipt=dict(recorded_at=datetime.datetime.now().astimezone().isoformat(),returncode=r.returncode,
        stdout=r.stdout.decode(),stderr=r.stderr.decode(),new_hash_computed=False)
    with (HERE/('launch_'+args.release+'.json')).open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
    print(receipt['stdout']);print(receipt['stderr']);r.check_returncode()

if __name__=='__main__':main()
