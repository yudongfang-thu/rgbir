"""Small receipt-only cross-check; no weights, inference, SSH or hashing."""
import json
import math
from pathlib import Path

HERE=Path(__file__).absolute().parent
LOGS=HERE.parent
OLD=LOGS/'2026-09-07_train_IndependentKD实施/remote_admission_1532/evaluator_profile_attempt2'
FT=LOGS/'2026-09-08_ops_小时级筛选重构/snapshot_1016/evaluations/N'
AP=('mAP50_95','AP50','AP75')


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def require(x,msg):
    if not x:raise ValueError(msg)


initial=read(OLD/'native_metrics.json');full=read(FT/'hourly_evaluation_receipt.json')
original_contract=read(OLD/'native_contract.json')
ft_roster=(FT/'development_roster.txt').read_text(encoding='utf-8-sig').splitlines()
records={'initial':initial,'full_ft':full};checks={};inputs=[OLD/'native_metrics.json',OLD/'native_contract.json',FT/'hourly_evaluation_receipt.json',FT/'development_roster.txt']
for variant,psource,bsource in [('parameter_only','finetuned','initial'),('buffer_only','initial','finetuned')]:
    folder=HERE/'state_swap_evidence'/variant
    c=read(folder/'composition.json');r=read(folder/'swap_evaluation_receipt.json');contract=read(folder/'evaluation_contract.json')
    inputs.extend(folder/f for f in ('composition.json','swap_evaluation_receipt.json','evaluation_contract.json'))
    require(r['status']=='BN_PARAMETER_SWAP_EVALUATION_COMPLETED' and r['variant']==variant,'Swap completion')
    require(c['variant']==variant and c['parameter_source']==psource and c['buffer_source']==bsource,'Composition sources')
    require(c['scope']==r['scope']=='BN_PARAMETER_SWAP_DIAGNOSTIC','Swap scope')
    for key in ('strict_key_dtype_shape','fp32_projection_exact','all_changed_buffers_are_bn_statistics'):
        require(c[key] is True,key)
    for key in ('checkpoint_written','optimizer_created','new_hash_computed'):
        require(c[key] is False and r[key] is False,key)
    require(r['model_identity']['validator_received_memory_model'] is True and r['model_identity']['forward_memory_identity'] is True and r['model_identity']['forward_calls']>0,'Actual memory model')
    details=c['tensors'];require(len({x['name'] for x in details})==len(details)==499,'Unique complete tensor manifest')
    for kind,registered,changed in [('parameter','registered_parameters','changed_parameter_keys'),('buffer','registered_buffers','changed_buffer_keys')]:
        rows=[x for x in details if x['kind']==kind]
        require(len(rows)==c[registered],'Registered count')
        require([x['name'] for x in rows if x['changed']]==c[changed],'Changed group count/identity')
    require(all(x['bn_statistic'] for x in details if x['kind']=='buffer' and x['changed']) and not c['non_bn_changed_buffer_keys'],'Only BN statistics changed')
    require(c['initial_checkpoint']==r['initial_checkpoint']==contract['initial_checkpoint'],'Initial stat')
    require(c['finetuned_checkpoint']==r['finetuned_checkpoint']==contract['finetuned_checkpoint']==full['checkpoint'],'FT stat')
    require(r['initial_checkpoint']['path']==initial['checkpoint'],'Initial accepted metric path')
    require(r['full_dev_images']==contract['observed_images']==1469 and r['full_dev_gt_objects']==contract['gt_objects_before_inference']==contract['gt_objects_captured']==22462,'Full population')
    require(contract['effective_kwargs']==original_contract['effective_kwargs'],'Same native effective kwargs')
    require(contract['roster']==original_contract['roster']==ft_roster,'Same complete dev roster')
    require(contract['actual_loader_roster']==original_contract['actual_loader_roster'],'Same actual loader order')
    records[variant]=r
    checks[variant]=dict(composition_checked=True,registered_parameters=c['registered_parameters'],registered_buffers=c['registered_buffers'],
        changed_source_parameter_keys=len(c['changed_parameter_keys']),changed_source_buffer_keys=len(c['changed_buffer_keys']),
        changed_buffers_all_bn=True,model_identity=r['model_identity'],population_and_roster_checked=True,
        full_checkpoint_stat_matches_ft=True,seconds=r['seconds'],resources=r['resources'])
for name,r in records.items():
    require(r['metric_units']=='fraction_0_to_1','Metric units')
    require(len(r['per_class'])==5 and sorted(x['class_id'] for x in r['per_class'])==list(range(5)),'Five classes')
    for k in AP:
        require(math.isfinite(r[k]) and 0<=r[k]<=1,'AP fraction')
        require(math.isclose(sum(x[k] for x in r['per_class'])/5,r[k],rel_tol=0,abs_tol=1e-12),'Class mean')
percent={name:{k:100*r[k] for k in AP} for name,r in records.items()}
diff={name:{k:100*(r[k]-initial[k]) for k in AP} for name,r in records.items()}
full_minus_param={k:100*(full[k]-records['parameter_only'][k]) for k in AP}
nonadditivity={k:100*(full[k]-records['parameter_only'][k]-records['buffer_only'][k]+initial[k]) for k in AP}
receipt=dict(status='PASS_RECEIPT_LEVEL_SWAP_DESCRIPTIVE_REVIEW',scope='SINGLE_SEED_REGISTERED_STATE_SWAP_ONLY',checks=checks,
    percent=percent,vs_initial_pp=diff,full_minus_parameter_only_pp=full_minus_param,descriptive_nonadditivity_pp=nonadditivity,
    new_hash_computed=False,weights_loaded=False,new_gpu_or_ssh=False,new_direction_AP_read=False,
    sources=[dict(path=str(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in inputs])
with (HERE/'STATE_SWAP_RESULT_receipt.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,ensure_ascii=False,indent=2)
print(json.dumps(dict(percent=percent,vs_initial_pp=diff,full_minus_parameter_only_pp=full_minus_param,nonadditivity_pp=nonadditivity),ensure_ascii=False))
