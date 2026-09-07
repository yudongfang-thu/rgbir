"""CPU-only evidence/source/first-epoch LR review for reusing accepted N/C0 tests."""
import json
from pathlib import Path

import numpy as np
import yaml

HERE=Path(__file__).resolve().parent
LOGS=HERE.parent.parent
OLD=LOGS/'2026-09-07_train_IndependentKD实施'/'remote_admission_1532'
REF=OLD/'formal_C1_gpu5_attempt2/runs/C1_seed42/implementation_snapshot'
SOURCES=['train_independent.py','runtime.py','independent_criterion.py','protocol.py',
         'task_conditional_reference/tracked_pair_data.py',
         'task_conditional_reference/legacy_oev1/train_object_evidence.py',
         'task_conditional_reference/legacy_oev1/paired_rgbir_data.py',
         'task_conditional_reference/legacy_oev1/object_evidence_loss.py']
KEYS=['model','teacher','reference','dataset','expected_nc','imgsz','batch','nbs','workers','seed','optimizer',
      'lr0','lrf','momentum','weight_decay','warmup_epochs','warmup_momentum','warmup_bias_lr','cos_lr',
      'close_mosaic','patience','amp','deterministic','augmentation','paths','evidence','teacher_cache_images',
      'log_every_batches','classification_coefficient','localization_coefficient','arm','source']


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def main():
    result=dict(status='CPU_REUSE_REVIEW',new_hash_computed=False,arms={},lr_prefix=[],gpu_run=False)
    for arm,name in [('N','compat_N_s42_attempt2'),('C0','compat_C0_s42_attempt1')]:
        root=OLD/name;child=root/('seed42_'+arm)/'new'
        receipt=read(root/'compatibility_receipt.json');worker=read(child/'worker_receipt.json')
        source=child/'execution_binding_compatibility'/'sources'
        rows=[]
        for relative in SOURCES:
            a,b=source/relative,REF/relative
            rows.append(dict(relative=relative,accepted=str(a),current_reference=str(b),
                bytes_a=a.stat().st_size,bytes_b=b.stat().st_size,byte_exact=a.read_bytes()==b.read_bytes()))
        oldcfg=yaml.safe_load((root/('seed42_'+arm)/'worker_config.yaml').read_text(encoding='utf-8'))
        newcfg=yaml.safe_load((HERE/'configs'/('drone_'+arm+'_s42_E20_DRAFT.yaml')).read_text(encoding='utf-8'))
        differences={k:dict(old=oldcfg.get(k),short=newcfg.get(k)) for k in KEYS if oldcfg.get(k)!=newcfg.get(k)}
        result['arms'][arm]=dict(compatibility_receipt=str(root/'compatibility_receipt.json'),
            status=receipt['status'],successful_updates=receipt['successful_updates'],loader_batches=receipt['loader_batches'],
            trajectory_exact=receipt['trajectory_exact'],resources=receipt['resources'],
            new_worker_result=worker['result'],sources=rows,active_config_differences=differences,
            original_horizon=oldcfg['epochs'],short_horizon=newcfg['epochs'])
    for index in range(30):
        states=[]
        for horizon in (200,20):
            nw=round(min(3.,max(horizon-1,0))*563)
            lf=max(1-0/horizon,0)*(.99)+.01
            states.append(dict(nw=nw,lf_epoch0=lf,
                accumulate=max(1,int(np.interp(index,[0,nw],[1,64/32]).round())),
                lr_regular=float(np.interp(index,[0,nw],[0.,.01*lf])),
                lr_bias=float(np.interp(index,[0,nw],[.1,.01*lf])),
                momentum=float(np.interp(index,[0,nw],[.8,.937]))))
        result['lr_prefix'].append(dict(batch=index+1,equal=states[0]==states[1],state=states[0]))
    result['all_active_sources_byte_equal']=all(r['byte_exact'] for arm in result['arms'].values() for r in arm['sources'])
    result['all_active_config_keys_equal']=all(not arm['active_config_differences'] for arm in result['arms'].values())
    result['first_30_batch_lr_accumulate_momentum_exact']=all(r['equal'] for r in result['lr_prefix'])
    result['scope']='Static source/config and CPU first-epoch LR equality; reuses existing executed 30batch/24update compatibility, does not manufacture an E20 GPU equivalence run.'
    output=HERE/'NC0_REUSE_REVIEW.json'
    if output.exists():raise FileExistsError(output)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('all_active_sources_byte_equal','all_active_config_keys_equal','first_30_batch_lr_accumulate_momentum_exact')}))


if __name__=='__main__':main()
