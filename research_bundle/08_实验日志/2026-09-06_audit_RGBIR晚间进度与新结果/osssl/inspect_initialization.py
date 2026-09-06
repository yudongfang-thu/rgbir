"""Read checkpoint metadata and tensor equality on CPU; never launch inference."""
import json
import pathlib
import subprocess
OUT = pathlib.Path(__file__).resolve().parent
REMOTE = r'''
import datetime,json,pathlib,torch
torch.set_num_threads(2)
R=pathlib.Path('/mnt/dataset/yudongfang/projects/RGBT_campaign')
O=R/'artifacts/osssl_ir_20260906'
S=pathlib.Path('/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction')
report={'timestamp':datetime.datetime.now().astimezone().isoformat(),'device':'cpu','hashes_generated':False}
head_evidence=[]
for p in (R/'artifacts/cgkd_20260905/w1_native_s42.log',O/'ft_sar_only_s42.log',O/'ft_shuffled_s123.log'):
    with p.open('rb') as f: head=f.read(32768).decode('utf-8',errors='replace')
    lines=[l for l in head.splitlines() if any(w in l for w in ('Remapped','Transferred','Overriding model.yaml','Ultralytics','engine/trainer:'))]
    head_evidence.append({'source':str(p),'source_first_bytes':32768,'selected_lines':lines})
report['training_log_headers']=head_evidence
states={}
for arm in ('paired','sar_only','shuffled'):
    p=O/f'clean_{arm}.pt'
    d=torch.load(p,map_location='cpu',weights_only=False)
    m=d.get('ema') or d['model']
    states[arm]={k:v.detach().float() for k,v in m.state_dict().items()}
    report[arm]={'source':str(p),'checkpoint_keys':list(d.keys()),'nc':getattr(m,'yaml',{}).get('nc'),'names':getattr(m,'names',None),'model_type':type(m).__name__}
    del d,m
def is_backbone(k):
    a=k.split('.')
    return len(a)>2 and a[0]=='model' and a[1].isdigit() and int(a[1])<=10
reference=states['paired']
report['nonbackbone_comparisons']={}
for arm,state in states.items():
    keys=[k for k in reference if not is_backbone(k)]
    diffs=[k for k in keys if k not in state or not torch.equal(reference[k],state[k])]
    report['nonbackbone_comparisons'][arm]={'compared_tensors':len(keys),'different_tensors':diffs}
native_path=S/'artifacts/int8_cross_modal_stage2_v1/weights/yolo11n.pt'
d=torch.load(native_path,map_location='cpu',weights_only=False)
m=d.get('ema') or d['model']
report['native_pretrain_source']={'source':str(native_path),'nc':getattr(m,'yaml',{}).get('nc'),'names':getattr(m,'names',None),'classification_bias_shapes':{k:list(v.shape) for k,v in m.state_dict().items() if '.cv3.' in k and k.endswith('.2.bias')}}
report['clean_paired_bias_shapes']={k:list(v.shape) for k,v in reference.items() if '.cv3.' in k and k.endswith('.2.bias')}
print(json.dumps(report,ensure_ascii=False))
'''
python='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
p=subprocess.run(['ssh','-o','BatchMode=yes','94',f'CUDA_VISIBLE_DEVICES= {python} -'],input=REMOTE.encode(),capture_output=True,check=True)
d=json.loads(p.stdout)
(OUT/'initialization_inspection.json').write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(d,ensure_ascii=False,indent=2))
