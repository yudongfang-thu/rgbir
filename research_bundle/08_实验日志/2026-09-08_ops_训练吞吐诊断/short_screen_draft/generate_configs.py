"""Materialize DRAFT Drone seed42 E20 N/C0/C1; never queue or admit runs."""
import argparse
import copy
import json
from pathlib import Path
import shutil

import yaml

COEFFICIENT=0.09227393550836771
ARMS=('N','C0','C1')
CHANGED_FIELDS={'method_id','method_identity','description','arm','epochs','classification_coefficient',
    'protocol_status','formal_training_authorized','readiness_receipt','canary_acceptance','short_screen'}


def configs(base, source):
    for key,value in dict(dataset='dronevehicle',arm='C1',seed=42,epochs=200,batch=32,nbs=64,workers=4,
                           imgsz=640,classification_coefficient=COEFFICIENT,expected_train_images=17990,
                           expected_val_images=1469,amp=True).items():
        if base.get(key)!=value:raise ValueError('Expected original frozen C1 source: '+key)
    result={}
    for arm in ARMS:
        c=copy.deepcopy(base)
        c.update(method_id='RGBIR-SHORT-SCREEN-E20-S42-'+arm,method_identity='SHORT_SCREEN_SINGLE_SEED',
            description='DRAFT independent E20 schedule, matched N/C0/C1; not an E200 endpoint',
            arm=arm,epochs=20,classification_coefficient={'N':0.,'C0':.1,'C1':COEFFICIENT}[arm],
            protocol_status='DRAFT',formal_training_authorized=False,readiness_receipt=None,canary_acceptance=None)
        c['short_screen']=dict(schema='rgbir-short-screen-draft-v1',scope='SHORT_SCREEN',status='DRAFT',
            original_config=str(source),single_seed=True,seed=42,epochs=20,
            scheduler_horizon_epochs=20,endpoint='SHORT_SCREEN_E20_LAST_EMA',
            comparison_arms=list(ARMS),full_dev_images=1469,full_dev_gt_objects=22462,
            official_test_accessed=False,production_switch_admitted=False,queue_authorized=False,
            candidate=dict(kind='UNRESOLVED' if arm=='C1' else 'original',source=None,export=None),
            needs_new_24update_review=True,needs_measured_resource_and_budget=True,
            old_calibration_provenance_only=True,classification_coefficient_recalibrated=False)
        changed={k for k in set(c)|set(base) if c.get(k)!=base.get(k)}
        if not changed.issubset(CHANGED_FIELDS):raise AssertionError('Unexpected recipe change')
        result[arm]=c
    common=[{k:v for k,v in c.items() if k not in CHANGED_FIELDS} for c in result.values()]
    if any(c!=common[0] for c in common):raise AssertionError('Common recipe differs')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    base=yaml.safe_load(a.source_config.read_text(encoding='utf-8'));values=configs(base,a.source_config.resolve())
    a.output.mkdir(parents=True,exist_ok=False)
    source=a.output/'original_formal_C1_s42.yaml';shutil.copyfile(a.source_config,source)
    if source.read_bytes()!=a.source_config.read_bytes():raise AssertionError('Source bytes differ')
    for arm,c in values.items():
        (a.output/('drone_'+arm+'_s42_E20_DRAFT.yaml')).write_text(yaml.safe_dump(c,sort_keys=False,allow_unicode=True),encoding='utf-8')
    stat=a.source_config.stat()
    (a.output/'generation_receipt.json').write_text(json.dumps(dict(status='DRAFT_CONFIGS_ONLY',arms=list(ARMS),
        source=dict(path=str(a.source_config.resolve()),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,copy=str(source)),
        changed_fields=sorted(CHANGED_FIELDS),new_hash_computed=False,queue_authorized=False,
        production_switch_admitted=False,short_schedule_not_e200_prefix=True),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
