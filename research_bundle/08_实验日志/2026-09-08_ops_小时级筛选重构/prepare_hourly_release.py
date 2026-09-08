"""Freeze three FT configurations and deploy small sources into a new release."""
import argparse
import base64
import json
from pathlib import Path
import subprocess
import yaml

HERE=Path(__file__).resolve().parent
BASE='/mnt/dataset/yudongfang/projects/RGBT_campaign'
CAMPAIGN=BASE+'/artifacts/rgbir_hourly_screen_20260908'
OLD=BASE+'/artifacts/rgbir_throughput_20260908/short_screen_E8_release_attempt1'
PY='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'

def main():
    p=argparse.ArgumentParser();p.add_argument('--deploy',action='store_true');p.add_argument('--release',default='release_v1');a=p.parse_args()
    release=HERE/'release';configs=release/'configs';configs.mkdir(exist_ok=True)
    original=HERE.parent/'2026-09-08_ops_训练吞吐诊断/short_screen_E8_release/configs/drone_N_s42_E8.yaml'
    for arm,weight in [('N',0.),('C0',.1),('C1',.09227393550836771)]:
        cfg=yaml.safe_load(original.read_text(encoding='utf-8'));cfg.pop('short_screen')
        cfg.update(method_id='RGBIR-HOURLY-FT-E3-S42-'+arm,method_identity='HOURLY_SCREEN_FT',
            description='Fixed 2048-image mature RGB warm-start fine-tuning screen',arm=arm,epochs=3,
            expected_train_images=2048,lr0=.001,lrf=.1,warmup_epochs=0.,
            classification_coefficient=weight,protocol_status='FROZEN_HOURLY_SCREEN',
            model=BASE+'/runs/rgbir_object_evidence_v1_20260906/full_weight0_s42_attempt1/weights/last.pt')
        cfg['auxiliary_data_identity']={k:cfg['paths'][k] for k in ('student_data_yaml','privileged_data_yaml')}
        cfg['paths']=dict(student_data_yaml=CAMPAIGN+'/subset_v1/data_rgb.yaml',
            privileged_data_yaml=CAMPAIGN+'/subset_v1/data_infrared.yaml',
            paired_train_mapping=CAMPAIGN+'/subset_v1/rgb_to_infrared_train.json')
        cfg['native_contract_config']=OLD+'/short_screen_E8_release/configs/drone_N_s42_E8.yaml'
        cfg['hourly_screen']=dict(scope='HOURLY_SCREEN_FT',endpoint='HOURLY_SCREEN_FT_E3_LAST_EMA',
            single_seed=True,subset_seed=20260908,training_images=2048,batches_per_epoch=64,
            initialization='completed matched N42 last/EMA',fresh_optimizer=True,fresh_ema=True,
            coefficient_recalibrated=False,not_from_scratch=True,comparison_arms=['N','C0','C1'],
            candidate_source=OLD+'/performance_candidate/selected_only_v1.py',
            approved_candidate_source=OLD+'/approved/selected_only_v1.py')
        target=configs/('drone_'+arm+'_s42_FT3.yaml');content=yaml.safe_dump(cfg,sort_keys=False).encode('utf-8')
        if target.exists():assert target.read_bytes()==content
        else:target.write_bytes(content)
    if not a.deploy:return
    sources={str(f.relative_to(release)).replace('\\','/'):base64.b64encode(f.read_bytes()).decode()
        for f in release.rglob('*') if f.is_file() and f.suffix in ('.py','.yaml','.json') and '__pycache__' not in f.parts}
    source="""import base64,json,sys
from pathlib import Path
out=Path(OUTPUT);out.mkdir(parents=True,exist_ok=False)
for name,content in SOURCES.items():
 p=out/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(content))
sys.path.insert(0,str(out))
from hourly_common import load_config
for p in (out/'configs').glob('*.yaml'):
 c=load_config(p)
 for k in ('model','teacher','reference'):assert Path(c[k]).is_file(),c[k]
 for k in c['paths']:assert Path(c['paths'][k]).is_file(),c['paths'][k]
print(json.dumps(dict(status='DEPLOYED',output=str(out),source_files=len(SOURCES))))
""".replace('OUTPUT',repr(CAMPAIGN+'/'+a.release)).replace('SOURCES',repr(sources))
    r=subprocess.run(['ssh','94',PY,'-'],input=source.encode(),capture_output=True,timeout=60)
    print(r.stdout.decode());print(r.stderr.decode());r.check_returncode()
    (HERE/('deployment_'+a.release+'.json')).write_text(json.dumps(dict(remote=CAMPAIGN+'/'+a.release,
        files=list(sources),source_byte_identity=True,new_hash_computed=False),indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':main()
