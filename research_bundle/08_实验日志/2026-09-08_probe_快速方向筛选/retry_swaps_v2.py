from pathlib import Path
import subprocess
root=Path(__file__).parent
remote='/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_direction_screen_20260908/swap_release_v2'
s=(root/'dispatch_swaps.py').read_text().replace('state_swap_attempt1','state_swap_attempt2').replace('swap_release_v1','swap_release_v2')
(root/'dispatch_swaps_v2.py').write_text(s)
subprocess.run(['ssh','94','mkdir','-p',remote],check=True)
subprocess.run(['scp',str(root/'evaluate_state_swap.py'),'94:'+remote+'/evaluate_state_swap.py'],check=True)
subprocess.run(['scp',str(root/'dispatch_swaps_v2.py'),'94:'+remote+'/dispatch_swaps.py'],check=True)
subprocess.run(['ssh','94','screen','-dmS','direction_bn_swap_s42_v2','bash','-lc',"'exec /mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python "+remote+"/dispatch_swaps.py > "+remote+"/launch.log 2>&1'"],check=True)
print('SWAP_ATTEMPT2_DISPATCHED')
