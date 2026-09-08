import json,subprocess,sys
from pathlib import Path
root=Path(__file__).parent/'state_swap_evidence';root.mkdir(exist_ok=True)
remote='/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_direction_screen_20260908/state_swap_attempt2'
for variant in ('parameter_only','buffer_only'):
    folder=root/variant;folder.mkdir(exist_ok=True)
    for name in ('composition.json','evaluation_contract.json','swap_evaluation_receipt.json'):
        p=folder/name
        if not p.exists():subprocess.run(['scp','94:'+remote+'/'+variant+'/'+name,str(p)],check=True)
    r=json.loads((folder/'swap_evaluation_receipt.json').read_text())
    print(variant,json.dumps({k:r[k] for k in ('mAP50_95','AP50','AP75','seconds','changed_buffers_all_bn')}))
