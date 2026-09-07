"""Write explicit spec, upload only new diagnostic sources, launch named screen."""
from pathlib import Path
import json, subprocess, shutil
ROOT=Path(__file__).resolve().parent
REMOTE='/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_baseline_information_20260907/attempt3'
BASE='/mnt/dataset/yudongfang/projects/RGBT_campaign'
OLD=BASE+'/artifacts/rgbir_task_conditional_v1_20260907'
PY='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
spec={}
for dataset,short in [('dronevehicle','drone'),('llvip','llvip')]:
    root=OLD+'/d2_'+short+'_train_attempt1'
    models={'N42':BASE+'/runs/rgbt_p3_causal_v1/formal_native/'+dataset+'/'+('rgb' if short=='drone' else 'visible')+'_seed42_native_b32a2/weights/last.pt',
            'T42':BASE+'/runs/rgbt_p3_causal_v1/formal_native/'+dataset+'/infrared_seed42_native_b32a2/weights/last.pt'}
    if short=='drone':
        models['N42']=BASE+'/runs/rgbir_object_evidence_v1_20260906/full_weight0_s42_attempt1/weights/last.pt'
        models['N0']=BASE+'/runs/rgbir_object_evidence_expand_20260906/full_weight0_s0_attempt1/weights/last.pt'
    spec[dataset]={'config':root+'/protocol_config.yaml','train_roster':root+'/frozen_roster.json',
                   'val_roster':OLD+'/d2_'+short+'_val_attempt1/frozen_roster.json','models':models}
(ROOT/'probe_spec.json').write_text(json.dumps(spec,indent=2)+'\n',encoding='utf-8')
subprocess.run(['ssh','94','mkdir',REMOTE],check=True)
for name in ['probe_spec.json','export_baseline_information.py','run_export_campaign.py','README.md']:
    subprocess.run(['scp',str(ROOT/name),'94:'+REMOTE+'/'+name],check=True)
subprocess.run(['ssh','94','screen','-dmS','baselineinfo_probe_s20260907','bash','-lc',
                "'"+PY+' '+REMOTE+'/run_export_campaign.py > '+REMOTE+"/campaign.log 2>&1'"],check=True)
(ROOT/'launch_receipt.json').write_text(json.dumps({'remote':REMOTE,'screen':'baselineinfo_probe_s20260907','status':'DISPATCHED'},indent=2)+'\n')

if __name__=='__main__':pass
