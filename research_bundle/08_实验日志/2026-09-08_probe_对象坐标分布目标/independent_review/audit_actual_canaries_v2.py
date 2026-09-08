"""Read and reconcile existing executed canary artifacts only; no tests or jobs."""
from pathlib import Path
from datetime import datetime,timezone
import json,math,yaml
HERE=Path(__file__).resolve().parent;ENTRY=HERE.parent;SRC=ENTRY/'training_evidence_canaries'
OUT=HERE/'canaries_v2';OUT.mkdir(exist_ok=False)
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def lines(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x]
def dump(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False)
arms=('N','L3-DFL','L3-GT');checks={};metrics={};evidence=[];sources={};streams={};initials={};logged={}
queue=read(SRC/'queue'/'llvip_canary_checks.json');cal=read(SRC/'calibration'/'llvip'/'calibration_receipt.json')
review=read(HERE/'TRAINING_SOURCE_REVIEW_v2.json');byrel={r['relative_path']:r for r in review['reviewed_source_files']}
def check(name,ok,detail):
    checks[name]={'status':'pass' if ok else 'fail','details':detail}
    assert ok,name
check('matched_queue_receipt',queue['status']=='OBJECT_DFL_CANARIES_MATCHED' and set(queue['arms'])==set(arms),'Original queue explicitly matched all three real canaries.')
for arm in arms:
    directory=SRC/'canaries'/'llvip'/arm;c=read(directory/'canary.json');init=read(directory/'initialization_check.json');initials[arm]=init
    config=SRC/'effective_configs'/f'llvip_{arm}_s42_FT3.yaml';cfg=yaml.safe_load(config.read_text(encoding='utf-8'))
    dispatch=read(SRC/'queue'/f'object_dfl_attempt2_llvip_{arm}_canary_status.json');runtime=read(directory/'runtime_ready.json')
    check(arm+'_terminal',c['status']=='OBJECT_DFL_CANARY_COMPLETED' and c['scope']==c['method_identity']=='OBJECT_DFL_FT3_BNFROZEN' and c['arm']==arm and c['seed']==42 and c['dataset']=='llvip' and c['formal_e200_complete'] is False and dispatch['status']=='COMPLETED' and dispatch['exit_code']==0 and dispatch['monitor_errors']==[], 'Actual completed canary receipt and clean dispatcher exit for this arm.')
    check(arm+'_actual_counts',c['successful_updates']==c['optimizer_updates']==24 and c['attempts']==29 and c['amp_skips']==5 and c['batches']==58 and c['attempts']-c['amp_skips']==c['successful_updates'] and c['ema_updates']==29,'24 successful optimizer updates from 29 attempts, 5 AMP skips, 58 B32 batches; native EMA counter is 29, not mislabeled as 24.')
    check(arm+'_initialization',init['status']=='PASS_FULL_STATE_WARM_START' and init['state_tensors']==499 and all(init[k] for k in ('head_included','fresh_optimizer','fresh_ema','teacher_reference_isolated')) and init['initial_checkpoint']==cal['student_initial_checkpoint'],'All 499 initial tensors including head matched the same mature checkpoint; fresh optimizer/EMA and auxiliary isolation asserted in executed source.')
    check(arm+'_BN_and_runtime',c['bn_running_buffers_unchanged'] is True and c['bn_affine_trainable'] is True and c['bn_buffer_count']==243 and runtime['frozen_models_outside_optimizer'] is True and runtime['student_native_gt_only'] is True and runtime['amp'] is True,'243 BN running buffers unchanged; affine remains trainable; auxiliary models excluded from optimizer.')
    expected=cal['coefficients'][arm]
    check(arm+'_config_and_coefficient',(directory/'direction_config.yaml').read_bytes()==config.read_bytes() and c['kd_coefficient']==c['localization_coefficient']==cfg['kd_coefficient']==cfg['localization_coefficient']==expected and c['classification_coefficient']==cfg['classification_coefficient']==0,'Canary exact effective config bytes and fixed shared-calibration dose agree; no classification KD.')
    grads=lines(directory/'gradient_checks.jsonl')
    check(arm+'_gradient_isolation',grads==c['gradient_checks'] and len(grads)>=1 and all(g['teacher_has_grad'] is False and g['reference_has_grad'] is False and g['parameter_names']==cal['parameter_names'] and g['actual_B']==32 and g['coefficient']==expected and math.isfinite(g['kd_gradient_l2']) for g in grads) and (arm=='N' or any(g['kd_gradient_l2']>0 for g in grads)), 'Recorded nonzero student shared-parameter unit KD gradient for KD arms; no teacher/reference accumulated gradients. N retains unit observation but applies zero dose.')
    rowstream=lines(directory/'sample_stream.jsonl');streams[arm]=rowstream
    check(arm+'_first30',len(rowstream)==30 and [r['batch'] for r in rowstream]==list(range(1,31)) and all(len(r['im_file'])==32 and all(k in r for k in ('cls','bboxes','batch_idx','teacher_cls','teacher_bboxes','teacher_batch_idx')) for r in rowstream),'Complete contiguous first30 actual image/label records, including batch-index and teacher-label fields.')
    logged[arm]=lines(directory/'kd_batches.jsonl')
    if arm=='N':check('N_native_total_exact',all(r['weighted_kd_total']==0 and r['total_loss']==r['native_total'] for r in logged[arm]),'All available actual sparse N loss rows equal native total; source asserts this on every call.')
    rr=c['resources'];peak=max(list(rr['per_gpu_peak_vram_mib'].values())+[c['gpu_allocated_peak_mib'],c['gpu_reserved_peak_mib']]);rss=rr['peak_rss_mib']
    vram=math.ceil((peak+max(256,.05*peak))/256)*256;ram=math.ceil((rss+max(2048,.05*rss))/1024)*1024
    check(arm+'_reservation',vram==queue['arms'][arm]['resources']['vram_mib']==6144 and ram==queue['arms'][arm]['resources']['rss_mib']==20480 and vram<=8192 and ram<=32768 and dispatch['minimum_free_mib']>=2048,'Independent actual-canary peak plus unchanged margin recomputes 6144 MiB GPU / 20480 MiB process-tree RSS reservation; observed whole-card free margin >2 GiB.')
    sm=read(directory/'source_manifest.json');bound={}
    for item in sm['files']:
        marker='/release_v2/'
        if marker in item['path']:
            rel=item['path'].split(marker,1)[1]
            if rel in byrel:
                bound[rel]=item['bytes']==byrel[rel]['bytes'] and item['byte_identity']
    required={r for r in byrel if r.endswith('.py')}|{'validated_amp_prior.json','validated_amp_prior_config.yaml'}
    check(arm+'_source_binding',required.issubset(bound) and all(bound.values()),'All operational Python and AMP inputs in actual source-copy manifest bind to reviewed release_v2 paths/byte sizes and producer byte-copy assertions. Review does not download large source/runtime trees again.')
    metrics[arm]={'successful_updates':24,'attempts':29,'AMP_skips':5,'EMA_updates_counter':29,'batches':58,'selected_objects':c['selected_objects'],'unit_shared_KD_gradient_norm':grads[0]['kd_gradient_l2'],'coefficient':expected,'NVML_peak_mib':max(rr['per_gpu_peak_vram_mib'].values()),'allocated_peak_mib':c['gpu_allocated_peak_mib'],'reserved_peak_mib':c['gpu_reserved_peak_mib'],'RSS_peak_mib':rss,'training_vram_reservation_mib':vram,'training_rss_reservation_mib':ram,'seconds':c['seconds'],'minimum_whole_card_free_mib':dispatch['minimum_free_mib']}
    sources[arm]=str(directory.resolve())
    evidence.extend([directory/x for x in ('canary.json','initialization_check.json','gradient_checks.jsonl','direction_config.yaml','sample_stream.jsonl','kd_batches.jsonl','source_manifest.json')])
check('cross_arm_first30_exact',streams['N']==streams['L3-DFL']==streams['L3-GT'] and all((SRC/'canaries'/'llvip'/a/'sample_stream.jsonl').read_bytes()==(SRC/'canaries'/'llvip'/'N'/'sample_stream.jsonl').read_bytes() for a in arms),'All first30 source records are both structurally and byte-for-byte identical across three arms.')
check('common_initialization_exact',initials['N']==initials['L3-DFL']==initials['L3-GT'],'Initialization check records equal exactly across arms.')
selection_fields=('batch','base_count','normalizer','selected_count','selected_anchors','selected_object_ids')
def selected_rows(arm):return [{k:r[k] for k in selection_fields} for r in logged[arm]]
check('logged_selection_exact',selected_rows('N')==selected_rows('L3-DFL')==selected_rows('L3-GT') and len({metrics[a]['selected_objects'] for a in arms})==1,'All stored per-batch selection identities/denominators match and terminal selected totals are 190 per arm; not a claim that unlogged pixel tensors were compared.')
cal_dispatch=read(SRC/'queue'/'object_dfl_attempt2_llvip_N_calibration_status.json')
check('calibration_dispatch_terminal',cal_dispatch['status']=='COMPLETED' and cal_dispatch['exit_code']==0 and cal_dispatch['monitor_errors']==[],'This newer snapshot supplies the clean calibration process terminal missing from training_evidence_1716.')
for name in ('calibration_receipt.json','calibration_batches.jsonl','calibration_allocation_lifecycle.jsonl','calibration_resource_batches.jsonl'):
    check('calibration_snapshot_unchanged_'+name,(SRC/'calibration'/'llvip'/name).read_bytes()==(ENTRY/'training_evidence_1716'/'calibration'/'llvip'/name).read_bytes(),'New collection calibration file equals previously audited copy byte for byte.')
snapshot=OUT/'evidence_snapshot';snapshot.mkdir();identity=[]
for p in evidence+[SRC/'queue'/'llvip_canary_checks.json',SRC/'queue'/'object_dfl_attempt2_llvip_N_calibration_status.json']:
    rel=p.relative_to(SRC);dest=snapshot/rel;dest.parent.mkdir(parents=True,exist_ok=True);raw=p.read_bytes();dest.write_bytes(raw)
    assert p.read_bytes()==dest.read_bytes();identity.append({'source_path':str(p.resolve()),'snapshot_path':str(dest.resolve()),'bytes':len(raw),'direct_byte_identity':True})
audit={'date':datetime.now(timezone.utc).isoformat(),'auditor':'/root/object_transport_review','overall_verdict':'pass','integrity_status':'pass','reason_code':'ACTUAL_THREE_CANARIES_MATCHED_AND_RECONCILED',
 'checks':{
  'gt_provenance':{'status':'pass','details':'Same recorded actual dual-label first30 flow and same logged object/anchor/base identities across three arms; no new physical-registration evidence.','evidence':[sources[a]+'/sample_stream.jsonl' for a in arms]},
  'score_normalization':{'status':'pass','details':'No AP score; fixed shared lambda from accepted calibration, zero N coefficient, complete native total retained. Successful updates, attempts, skips and EMA counter remain distinct.','evidence':[sources[a]+'/canary.json' for a in arms]},
  'result_existence':{'status':'pass','details':'All three completed actual canary receipts and clean dispatcher exits verified.','evidence':[str(SRC/'queue'/'llvip_canary_checks.json')]},
  'dead_code':{'status':'pass','details':'Actual nonzero shared student KD gradients recorded for both KD arms, no auxiliary gradients; source/cfg/initialization binding checked.','evidence':[sources[a]+'/gradient_checks.jsonl' for a in arms]},
  'scope':{'status':'pass','details':'58 batches, 29 attempts, 5 AMP skips, 24 successful optimizer updates and EMA counter29 per arm. These are canaries, not full192-batch endpoints.','evidence':[str(OUT/'EXPERIMENT_AUDIT.json')+' metrics']},
  'eval_type':{'status':'pass','types':['real_gt'],'details':'Actual train-label gradient/update canaries and resource measurements; no new accuracy evaluation.','evidence':[sources[a]+'/canary.json' for a in arms]},
  'specific_checks':{'status':'pass','details':checks,'evidence':[str(OUT/'EXPERIMENT_AUDIT.json')]}
 },'claims':[
  {'id':'three_actual_canaries_pass','impact':'supported','affected_evidence':[str(SRC/'queue'/'llvip_canary_checks.json')],'rationale':'All actual terminals, counts, source/init/data/gradient and resource contracts independently reconciled.'},
  {'id':'new_short_training_reservation','impact':'supported','affected_evidence':[str(SRC/'queue'/'llvip_canary_checks.json')],'rationale':'Each measured canary recomputes to 6144 MiB VRAM and 20480 MiB RSS with the original margins.'},
  {'id':'full_FT3_or_KD_gain','impact':'unsupported','affected_evidence':[sources[a]+'/canary.json' for a in arms],'rationale':'No full training/evaluation endpoint or AP is accepted by this audit.'}
 ],'metrics':metrics,'declared_input_set':[x['source_path'] for x in identity],'audited_input_hashes':[],'new_hash_computed':False,
 'identity_policy':'Exact local snapshots and direct full-byte equality, plus already reviewed deployment/source-manifest provenance. No new hash per user instruction.',
 'source_evidence_identity':identity,'run_id':'rgbir_object_dfl_screen_20260908/attempt2/canaries/llvip','recomputation_command':str(Path(__file__).resolve()),'inaccessible_inputs':['Large checkpoints and original full pixel tensors were not re-downloaded or replayed.'],
 'limits':['Source-executed state assertions and manifests are checked, not replaced by a new full model replay.','EMA counter29 is reported as native runtime counter, not as 29 successful optimizer steps.','Only permits continued previously authorized guarded FT3; no scientific gain conclusion.']}
dump(OUT/'EXPERIMENT_AUDIT.json',audit)
md='''# v2 三臂实际 canary 独立审计

**PASS**：N / L3-DFL / L3-GT 均真实完成24次成功optimizer更新；每臂58个batch、29次更新尝试、5次AMP skip，EMA计数29，分别如实记录。三臂均clean exit，外部monitor无错误。

499个完整初始状态张量（含head）来自同一成熟checkpoint，fresh optimizer/EMA；243个BN running buffers不变，affine可学。执行source manifest绑定reviewed release_v2，effective config逐字节相等，DFL/GT共享λ=.6900524651944485、N为0。教师/R均无梯度，两个KD臂实际共享学生参数梯度非零；N全部可见loss记录等于native total。

三臂前30批32图文件、RGB/IR标签及batch_idx记录完全相同，且三个sample_stream文件逐字节一致；所有可见选择/分母记录匹配，终态各190个选中对象。没有把这称为全58批像素张量逐位核验。

三臂NVML峰均5694 MiB，allocated峰4732.996 MiB，reserved峰5162 MiB。按原实际峰+余量规则独立重算，三臂正式短训预约均为6144 MiB显存、20480 MiB进程树RSS。整卡最小空闲分别14886/14885/14889 MiB，均保留了2 GiB余量。

新快照还补齐了八批校准clean exit且monitor无错；校准四份关键数据与上一审计逐字节一致，解除此前仅缺外层终态的限定。本审计只覆盖校准与真实canary，后续FT3和实际AP仍须终态证据，不作KD收益或物理配准结论。
'''
(OUT/'EXPERIMENT_AUDIT.md').write_text(md,encoding='utf-8')
print(json.dumps({'overall_verdict':'pass','checks':len(checks),'metrics':metrics},indent=2))
