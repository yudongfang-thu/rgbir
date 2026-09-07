"""Read bound LLVIP metadata and prepare a new diagnostic deployment only."""
from pathlib import Path
import argparse,json,shlex,subprocess
ROOT=Path(__file__).resolve().parent
PY='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
REMOTE='/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_evidence_priority_20260907/llvip_full_eval_attempt2'
CODE=r'''
import json,yaml
from pathlib import Path
from ultralytics.cfg import DEFAULT_CFG_DICT
base=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign')
cfg=yaml.safe_load((base/'artifacts/rgbir_task_conditional_v1_20260907/d2_llvip_train_attempt1/protocol_config.yaml').read_text())
models={}
for key,modality,dk in [('N42','visible','student_data_yaml'),('T42','infrared','privileged_data_yaml')]:
    cp=base/'runs/rgbt_p3_causal_v1/formal_native/llvip'/(modality+'_seed42_native_b32a2')/'weights/last.pt'
    data=Path(cfg['paths'][dk]);args=cp.parents[1]/'args.yaml'
    assert cp.is_file() and args.is_file() and data.is_file()
    models[key]=dict(checkpoint=str(cp),data=str(data),original_args=yaml.safe_load(args.read_text()),data_value=yaml.safe_load(data.read_text()))
print(json.dumps(dict(models=models,defaults={k:DEFAULT_CFG_DICT.get(k,'ABSENT') for k in ['half','quantize','rect','batch','conf','iou','max_det']},remote=__REMOTE__)))
'''.replace('__REMOTE__',repr(REMOTE))

def main():
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');a=p.parse_args()
    if not a.launch:
        r=subprocess.run(['ssh','94',PY,'-'],input=CODE.encode(),capture_output=True,check=True)
        data=json.loads(r.stdout);(ROOT/'remote_metadata.json').write_text(json.dumps(data,indent=2)+'\n')
        spec=dict(models={key:{k:v[k] for k in ('checkpoint','data')} for key,v in data['models'].items()},
                  data_role='development_val',expected_images=2406,official_test_accessed=False)
        (ROOT/'spec.json').write_text(json.dumps(spec,indent=2)+'\n')
        print(json.dumps(dict(defaults=data['defaults'],data={k:v['data_value'] for k,v in data['models'].items()})))
        return
    review=ROOT/'INDEPENDENT_CODE_REVIEW_ATTEMPT2.md'
    if not review.is_file():raise ValueError('Actual independent review required before dispatch')
    subprocess.run(['ssh','94','mkdir','-p',str(Path(REMOTE).parent).replace('\\','/')],check=True)
    subprocess.run(['ssh','94','mkdir',REMOTE],check=True)
    for name in ('export_full_dev.py','run_campaign.py','PROTOCOL.md','spec.json','INDEPENDENT_CODE_REVIEW_ATTEMPT2.md','ATTEMPT1_INVALID.md'):
        subprocess.run(['scp',str(ROOT/name),'94:'+REMOTE+'/'+name],check=True)
    command=shlex.quote(PY)+' '+shlex.quote(REMOTE+'/run_campaign.py')+' > '+shlex.quote(REMOTE+'/campaign.log')+' 2>&1'
    remote_command='screen -dmS evidence_llvip_eval_s42 bash -lc '+shlex.quote(command)
    subprocess.run(['ssh','94',remote_command],check=True)
    with (ROOT/'launch_receipt_attempt2.json').open('x') as f:json.dump(dict(status='DISPATCHED',remote=REMOTE,screen='evidence_llvip_eval_s42'),f,indent=2)

if __name__=='__main__':main()
