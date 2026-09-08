"""Run an explicitly supplied CPU/read-only Python script through SSH stdin."""
import subprocess,sys
from pathlib import Path
PY='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
r=subprocess.run(['ssh','94',PY,'-'],input=Path(sys.argv[1]).read_bytes(),stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
sys.stdout.buffer.write(r.stdout)
sys.exit(r.returncode)
