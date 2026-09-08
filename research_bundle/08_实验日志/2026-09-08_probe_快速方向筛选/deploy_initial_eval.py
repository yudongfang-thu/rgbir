from pathlib import Path
import subprocess
root=Path(__file__).parent
remote='/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_direction_screen_20260908/initial_eval_release_v1'
subprocess.run(['ssh','94','mkdir','-p',remote],check=True)
for name in ('initial_baseline_eval.py','dispatch_initial_eval.py'):
    subprocess.run(['scp',str(root/name),'94:'+remote+'/'+name],check=True)
subprocess.run(['ssh','94','screen','-dmS','direction_llvip_initial_s42','bash','-lc',"'exec /mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python "+remote+"/dispatch_initial_eval.py > "+remote+"/launch.log 2>&1'"],check=True)
print('INITIAL_EVAL_DISPATCHED_EXISTING_GLOBAL_LEASE')
