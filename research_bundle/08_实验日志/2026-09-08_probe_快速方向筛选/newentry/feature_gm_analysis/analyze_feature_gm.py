"""Fixed cross-scope F-GM / main N / main C1 receipt readout; CPU only."""
import argparse
import json
import math
from pathlib import Path
import shutil
import yaml

SCOPE='FEATURE_RELATION_GM_FT3'
CONTROL_SCOPE='DIRECTION_FT3_BNFROZEN'
ARMS=('N','C1','F-rel-GM')
DOSES={'N':0.,'C1':0.09227393550836771,'F-rel-GM':14.438521129817886}
AP=('mAP50_95','AP50','AP75')
METRICS=AP+('precision','recall')
COMMON=('dataset','model','teacher','reference','expected_nc','imgsz','epochs','batch','nbs','workers','seed',
    'optimizer','lr0','lrf','momentum','weight_decay','warmup_epochs','warmup_momentum','warmup_bias_lr',
    'cos_lr','close_mosaic','patience','amp','deterministic','augmentation','paths','evidence',
    'expected_train_images','expected_val_images','auxiliary_data_identity','freeze_bn_running_statistics',
    'torch_version','ultralytics_version')


def require(ok,message):
    if not ok:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def number(v):return type(v) in (int,float) and math.isfinite(v) and 0<=v<=1


def validate(r,c,arm):
    scope=SCOPE if arm=='F-rel-GM' else CONTROL_SCOPE
    fixed=dict(status='DIRECTION_EVALUATION_COMPLETED',scope=scope,endpoint=scope+'_LAST_EMA',dataset='drone',arm=arm,
        seed=42,single_seed=True,epochs=3,independent_lr_horizon=3,full_dev_images=1469,full_dev_gt_objects=22462,
        observed_images=1469,gt_objects_captured=22462,metric_units='fraction_0_to_1',formal_e200_complete=False,
        formal_paper_gain_claim=False,accepted_endpoint_claim=False,official_test_accessed=False,new_hash_computed=False)
    for k,v in fixed.items():require(type(r.get(k)) is type(v) and r[k]==v,'Receipt identity: '+k)
    require(type(r.get('kd_coefficient')) in (int,float) and r['kd_coefficient']==DOSES[arm],'Fixed dose differs')
    require(all(number(r.get(k)) for k in METRICS),'Invalid metric fraction')
    rows=r.get('per_class',[])
    require(len(rows)==5 and all(type(x.get('class_id')) is int for x in rows)
        and sorted(x['class_id'] for x in rows)==list(range(5)),'Class IDs differ')
    require(all(isinstance(x.get('name'),str) and x['name'] and all(number(x.get(k)) for k in AP) for x in rows),'Class metrics invalid')
    for k in AP:require(math.isclose(sum(x[k] for x in rows)/5,r[k],rel_tol=0,abs_tol=1e-12),'Macro AP differs')
    fixedcfg=dict(scope=scope,arm=arm,dataset='drone',expected_nc=5,expected_train_images=2048,expected_val_images=1469,
        seed=42,epochs=3,batch=32,nbs=64,workers=4,imgsz=640,optimizer='SGD',lr0=.0001,lrf=1.,warmup_epochs=0.,
        cos_lr=False,freeze_bn_running_statistics=True,amp=True,kd_coefficient=DOSES[arm])
    for k,v in fixedcfg.items():require(c.get(k)==v,'Actual config differs: '+k)
    require(c.get('classification_coefficient')==DOSES[arm] and c.get('localization_coefficient')==0.,'Loss branch dose differs')
    ds=c.get('direction_screen',{})
    require(ds.get('fresh_optimizer') is True and ds.get('fresh_ema') is True and ds.get('batches_per_epoch')==64,'Fresh optimizer/EMA or schedule missing')
    for k in COMMON:require(k in c,'Missing common config: '+k)
    require(r.get('training_model')==c['model'],'Model/config binding differs')
    require(all(isinstance(r.get(k),str) and r[k] for k in ('training_configuration','training_completion')),'Run provenance missing')
    p=r.get('evaluation_identity_projection',{})
    require(p.get('dataset')=='drone' and p.get('subset_dev_roster_exact_to_full') is True
        and p.get('expected_dev_images')==1469 and p.get('expected_dev_gt_objects')==22462,'Full dev projection missing')
    require(p.get('training_model')==c['model'] and p.get('training_data_yaml')==c['paths']['student_data_yaml']
        and p.get('actual_evaluation_data_yaml')==c['auxiliary_data_identity']['student_data_yaml'],'Evaluation config projection differs')
    bn=r.get('bn_training_evidence',{})
    require(bn.get('bn_running_buffers_unchanged') is True and bn.get('bn_affine_trainable') is True
        and type(bn.get('bn_buffer_count')) is int and bn['bn_buffer_count']>0,'Actual BN evidence missing')
    if arm=='F-rel-GM':
        require(r.get('matched_control_projection_required') is True and r.get('matched_control_scope')==CONTROL_SCOPE,'Cross-scope projection not declared')
        require(r.get('expected_train_images')==2048 and r.get('training_subset_identity')==c['paths'],'F subset identity differs')


def summarize(receipts,configs,projection):
    require(set(receipts)==set(ARMS) and set(configs)==set(ARMS),'All three completed arms required')
    for a in ARMS:validate(receipts[a],configs[a],a)
    fixed=dict(status='FEATURE_GM_MATCHED_CONTROL_VERIFIED',dataset='drone',seed=42,candidate_arm='F-rel-GM',
        candidate_scope=SCOPE,control_scope=CONTROL_SCOPE,fixed_candidate_coefficient=DOSES['F-rel-GM'])
    for k,v in fixed.items():require(projection.get(k)==v,'Matched projection differs: '+k)
    require(projection.get('batch_records')==30 and projection.get('batch_size')==32
        and projection.get('new_hash_computed') is False,'First30 projection scope differs')
    candidate=projection.get('candidate_configuration',{})
    require(candidate.get('path')==receipts['F-rel-GM']['training_configuration']
        and candidate.get('bytes',0)>0,'Candidate configuration projection binding differs')
    require(projection.get('candidate_canary_receipt',{}).get('path'),'Candidate canary provenance missing')
    controls=projection.get('controls',{})
    require(set(controls)=={'N','C1'},'Both explicitly verified controls required')
    initial=projection.get('initial_checkpoint',{})
    require(initial.get('path')==configs['N']['model'] and type(initial.get('bytes')) is int and initial['bytes']>0
        and type(initial.get('mtime_ns')) is int,'Shared initial checkpoint stat missing')
    for arm in ('N','C1'):
        p=controls[arm]
        for k in ('common_config_fields_exact','first30_canary_exact','initial_checkpoint_stat_exact','control_full_train_first30_exact'):
            require(p.get(k) is True,'Missing '+arm+' proof: '+k)
        for k,rk in (('control_training_receipt','training_completion'),('control_config','training_configuration')):
            require(p.get(k,{}).get('path')==receipts[arm][rk],'Projection receipt binding differs: '+arm+'/'+k)
        require(p.get('control_evaluation_receipt',{}).get('path') and p['control_evaluation_receipt'].get('bytes',0)>0,'Evaluation provenance missing')
    for arm in ('C1','F-rel-GM'):
        require(all(configs[arm][k]==configs['N'][k] for k in COMMON),'Common recipe/initialization/subset differs')
        require({x['class_id']:x['name'] for x in receipts[arm]['per_class']}==
                {x['class_id']:x['name'] for x in receipts['N']['per_class']},'Class mapping differs')
    require(len({r['training_completion'] for r in receipts.values()})==3,'Duplicate actual training run')
    raw={a:{k:receipts[a][k] for k in METRICS} for a in ARMS}
    differences={control:{k:100*(raw['F-rel-GM'][k]-raw[control][k]) for k in METRICS} for control in ('N','C1')}
    return dict(status='COMPLETE_FEATURE_GM_MATCHED_READOUT',scope=SCOPE,control_scope=CONTROL_SCOPE,seed=42,n_seeds=1,
        epochs=3,training_images=2048,full_dev_images=1469,full_dev_gt_objects=22462,coefficients=DOSES,
        raw_fraction=raw,display_percent={a:{k:v*100 for k,v in row.items()} for a,row in raw.items()},
        F_minus_control_pp=differences,per_class_raw_fraction={a:receipts[a]['per_class'] for a in ARMS},
        common_initialization=initial,common_training_subset=configs['N']['paths'],
        protocol_revised_after_calibration=True,original_F_rel_still_blocked=True,original_F_rel_AP_available=False,
        standard_deviation=None,formal_e200_complete=False,formal_paper_gain_claim=False,causal_modality_claim=False,
        automatically_extend_matrix=False,new_hash_computed=False,
        limits='Single seed FT3 matched cross-scope receipt readout. Revision followed calibration and preceded F AP; no original-plan outcome-blind claim, long-run effect, significance or modality attribution.')


def collect(main_campaign,feature_campaign):
    folders={a:Path(main_campaign)/'evaluations'/'drone'/a for a in ('N','C1')}
    folders['F-rel-GM']=Path(feature_campaign)/'evaluations'/'F-rel-GM'
    receipts={};configs={};sources=[]
    projection=read(Path(feature_campaign)/'queue'/'matched_control_projection.json')
    queue=read(Path(feature_campaign)/'queue'/'completion.json')
    require(queue.get('status')=='FEATURE_GM_QUEUE_COMPLETED' and queue.get('scope')==SCOPE
        and queue.get('arm')=='F-rel-GM' and queue.get('new_hash_computed') is False,'Feature queue incomplete')
    require(not (Path(feature_campaign)/'queue'/'failure.json').exists(),'Conflicting feature queue failure')
    stream=read(Path(feature_campaign)/'queue'/'candidate_training_stream_check.json')
    require(stream.get('status')=='PASS' and stream.get('first30_exact') is True
        and stream.get('new_hash_computed') is False,'Actual F full training first30 proof missing')
    for a,d in folders.items():
        require(not (d/'direction_evaluation_failure.json').exists(),'Conflicting endpoint failure')
        rp=d/'direction_evaluation_receipt.json'; cp=d/('feature_gm_config.yaml' if a=='F-rel-GM' else 'direction_config.yaml')
        receipts[a]=read(rp);configs[a]=yaml.safe_load(cp.read_text(encoding='utf-8-sig'))
        if a!='F-rel-GM':
            require(rp.stat().st_size==projection['controls'][a]['control_evaluation_receipt']['bytes'],'Collected control receipt size differs')
            require(cp.stat().st_size==projection['controls'][a]['control_config']['bytes'],'Collected control config size differs')
        else:
            require(cp.stat().st_size==projection['candidate_configuration']['bytes'],'Collected candidate config size differs')
        for p in (rp,cp):sources.append(dict(path=str(p.absolute()),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns))
    result=summarize(receipts,configs,projection);result['input_sources']=sources
    return result


def markdown(r):
    lines=['# F-rel-GM 与匹配 N/C1 的单seed FT3 原值','',
        '这是校准后、首次F AP前固定的新协议；原F-rel数字上限BLOCKED保留，不补写原F AP。跨scope身份及前30批必须由显式projection通过。AP显示百分数，差显示pp。','',
        '|臂|λ|mAP50–95|AP50|AP75|precision|recall|','|---|---:|---:|---:|---:|---:|---:|']
    for a in ARMS:lines.append('|'+a+'|'+str(DOSES[a])+'|'+'|'.join(f"{r['display_percent'][a][k]:.6f}" for k in METRICS)+'|')
    lines+=['','|差值|ΔmAP50–95|ΔAP50|ΔAP75|Δprecision|Δrecall|','|---|---:|---:|---:|---:|---:|']
    for c in ('N','C1'):lines.append('|F-rel-GM − '+c+'|'+'|'.join(f"{r['F_minus_control_pp'][c][k]:+.6f}" for k in METRICS)+'|')
    lines+=['','只描述本次固定2048图/seed42/3轮last-EMA；不算显著性/SD，不升级E200或模态归因，不按结果扫λ。完整精度见[summary.json](summary.json)。']
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser();p.add_argument('--main-campaign',type=Path,required=True);p.add_argument('--feature-campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();require(not a.output.exists(),'New output required');r=collect(a.main_campaign,a.feature_campaign)
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'summary.json').write_text(json.dumps(r,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    (a.output/'README.md').write_text(markdown(r),encoding='utf-8');shutil.copyfile(__file__,a.output/'analyze_feature_gm_source.py')
    print(r['status'])


if __name__=='__main__':main()
