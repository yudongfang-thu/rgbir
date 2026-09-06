import json,subprocess
from pathlib import Path
H=Path(__file__).resolve().parent
p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','94',
                  '/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python','-'],
                 input=(H/'independent_read_launch.py').read_bytes(),capture_output=True)
(H/'independent_launch_raw_v2.json').write_bytes(p.stdout);(H/'independent_launch_stderr_v2.txt').write_bytes(p.stderr)
if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace'))
j=json.loads(p.stdout)
for seed in [0,123]:
    print('SEED',seed,j['files'].get(f'seed{seed}_status.json'),j['files'].get(f'parallel_canary_comparison_s{seed}.json'))
    p=j['files'].get(f'concurrent_canary_s{seed}.json',{})
    print('CONCURRENCY', {k:v for k,v in p.items() if k!='samples'})
print('GPUS',j['gpus']);print('USAGE',j['usage']);print('CUDA',j['cuda_processes'])
print('RUNS', {k:{'exists':v['exists'],'progress':v['progress']} for k,v in j['runs'].items()})
