import subprocess,sys
from pathlib import Path
base=Path(__file__).resolve().parent
source=base/sys.argv[1]
target=base/sys.argv[2]
assert not target.exists(), 'Do not overwrite prior receipt'
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','94','/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python','-'],input=source.read_bytes(),capture_output=True)
target.write_bytes(r.stdout)
target.with_suffix('.stderr.txt').write_bytes(r.stderr)
print(r.stdout.decode('utf-8',errors='replace'))
if r.returncode: raise RuntimeError(r.stderr.decode('utf-8',errors='replace'))
