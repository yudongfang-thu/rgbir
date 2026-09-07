"""Build factual historical-native/current-N/C comparison without touching raw evidence."""
import json, shutil, statistics, subprocess
from pathlib import Path
import yaml

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD=ROOT/'08_实验日志/2026-09-06_audit_RGBIR晚间进度与新结果/osssl/raw/runs/cgkd_w1'
NEW=ROOT/'08_实验日志/2026-09-07_audit_RGBIR实施起点/snapshots/2026-09-07T111931.041292_0800'
OUT=HERE/'native_raw_evidence'
OUT.mkdir(exist_ok=True)
rows=[]
records=json.loads((NEW/'derived/results_analysis.json').read_text(encoding='utf-8'))['records']
for seed in (0,42,123):
    old_dir=OLD/f'native_rgb_s{seed}_e200'
    metric=json.loads((old_dir/'metrics_record.json').read_text(encoding='utf-8'))
    n=next(r for r in records if r['arm']=='N' and r['seed']==seed)
    c=next(r for r in records if r['arm']=='C' and r['seed']==seed)
    old_map=100*metric['metrics']['mAP50_95']
    new_map=n['metrics_percent']['mAP50_95']
    c_map=c['metrics_percent']['mAP50_95']
    rows.append({'seed':seed,'historical_native':old_map,'current_weight0':new_map,'C':c_map,'weight0_minus_historical_pp':new_map-old_map,'C_minus_weight0_pp':c_map-new_map,'C_minus_historical_pp':c_map-old_map})
    target=OUT/f'historical_s{seed}'
    target.mkdir(exist_ok=True)
    for name in ('args.yaml','metrics_record.json','completion_receipt.json'):
        shutil.copy2(old_dir/name,target/name)
old_args=yaml.safe_load((OLD/'native_rgb_s42_e200/args.yaml').read_text(encoding='utf-8'))
new_args=yaml.safe_load((NEW/'raw/N42/args.yaml').read_text(encoding='utf-8'))
diff={k:{'historical':old_args.get(k),'current':new_args.get(k)} for k in sorted(old_args.keys()|new_args.keys()) if old_args.get(k)!=new_args.get(k)}
loader_text=(HERE/'native_cpu_replay_stdout.txt').read_text(encoding='utf-8-sig')
replay=json.loads(next(line.split('=',1)[1] for line in loader_text.splitlines() if line.startswith('AUDIT_RESULT_JSON=')))
(HERE/'native_cpu_replay_result.json').write_text(json.dumps(replay,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
stats={key:{'mean':statistics.mean(r[key] for r in rows),'sample_sd':statistics.stdev(r[key] for r in rows),'positive_seeds':sum(r[key]>0 for r in rows)} for key in ('historical_native','current_weight0','C','weight0_minus_historical_pp','C_minus_weight0_pp','C_minus_historical_pp')}
payload={'units':'percent and percentage points','seeds':[0,42,123],'rows':rows,'summary':stats,'all_args_differences_s42':diff,'source_old':str(OLD),'source_new':str(NEW),'cpu_scope':replay['scope']}
(HERE/'native_comparison.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
for name in ('args.yaml','evaluation_val.json','completion_receipt.json'):
    shutil.copy2(NEW/'raw/N42'/name,OUT/('current_N42_'+name))
source=ROOT/'08_实验日志/2026-09-06_train_RGBIR对象判别蒸馏首轮/framework_notes.md'
shutil.copy2(source,OUT/'framework_notes_prior.md')
shutil.copy2(ROOT/'08_实验日志/2026-09-05_train_CGKD-W1_native/train_native_rgbt.py',OUT/'historical_train_native_rgbt.py')
UL='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/lib/python3.10/site-packages/ultralytics'
for rel in ('engine/model.py','engine/trainer.py','models/yolo/detect/train.py','data/build.py','engine/validator.py'):
    p=subprocess.run(['ssh','94','cat '+UL+'/'+rel],capture_output=True,check=True)
    (OUT/('pinned_'+rel.replace('/','_'))).write_bytes(p.stdout)
print(json.dumps(payload,ensure_ascii=False,indent=2))
