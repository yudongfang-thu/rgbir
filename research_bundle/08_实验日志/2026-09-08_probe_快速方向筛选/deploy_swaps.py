from pathlib import Path
import subprocess
ROOT=Path(__file__).parent
REMOTE='/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_direction_screen_20260908/swap_release_v1'
subprocess.run(['ssh','94','mkdir','-p',REMOTE],check=True)
for name in ('evaluate_state_swap.py','dispatch_swaps.py','STATE_SWAP_CONTRAST.md'):
    subprocess.run(['scp',str(ROOT/name),'94:'+REMOTE+'/'+name],check=True)
cmd=['ssh','94','screen','-dmS','direction_bn_swap_s42','bash','-lc',"'exec /mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python "+REMOTE+"/dispatch_swaps.py > "+REMOTE+"/launch.log 2>&1'"]
r=subprocess.run(cmd,capture_output=True,text=True);print(r.stdout,r.stderr);r.check_returncode()
print('DISPATCHED_EXISTING_GLOBAL_LEASE')
