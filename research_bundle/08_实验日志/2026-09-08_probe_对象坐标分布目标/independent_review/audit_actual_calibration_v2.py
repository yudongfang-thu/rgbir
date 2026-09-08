from pathlib import Path
from datetime import datetime,timezone
import json,math,statistics
HERE=Path(__file__).resolve().parent;ENTRY=HERE.parent
SOURCE=ENTRY/'training_evidence_1716';CAL=SOURCE/'calibration'/'llvip'
OUT=HERE/'calibration_v2';OUT.mkdir(exist_ok=False)
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def lines(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s]
def dump(p,o):
    with p.open('x',encoding='utf-8') as f:json.dump(o,f,ensure_ascii=False,indent=2,allow_nan=False)
receipt=read(CAL/'calibration_receipt.json');batches=lines(CAL/'calibration_batches.jsonl')
lifecycle=lines(CAL/'calibration_allocation_lifecycle.jsonl');resources=lines(CAL/'calibration_resource_batches.jsonl')
runtime=read(CAL/'runtime_ready.json');source_manifest=read(CAL/'source_manifest.json')
old_batches=lines(ENTRY/'training_evidence_1703'/'calibration'/'llvip'/'calibration_batches.jsonl')
review=read(HERE/'TRAINING_SOURCE_REVIEW_v2.json');checks={};defects=[]
def check(name,passed,details):
    checks[name]={'status':'pass' if passed else 'fail','details':details}
    if not passed:defects.append(name)
check('executed_fixed8',receipt['status']=='OBJECT_DFL_CALIBRATION_COMPLETED' and receipt['batches']==8 and all([r['batch'] for r in seq]==list(range(1,9)) for seq in (batches,lifecycle,resources)),'Three independent eight-row ledgers and completed calibration receipt; no failed retry rows substituted.')
check('runtime_identity',(runtime['train_images'],runtime['val_images'],runtime['train_batches'],runtime['batch'],runtime['workers'])==(2048,2406,64,32,4) and runtime['amp'] is True and receipt['method_identity']=='OBJECT_DFL_FT3_BNFROZEN' and receipt['seed']==42,'Pinned LLVIP FT3 subset and seed42 runtime identity.')
check('no_state_updates',receipt['optimizer_updates']==receipt['ema_updates']==0 and all(receipt[k] is True for k in ('reset_all_parameters_buffers_each_batch','bn_running_buffers_unchanged','full_initial_checkpoint_state_verified')),'Executed receipt reflects source assertions for full initial state, reset each batch, zero optimizer/EMA, frozen BN; no independent model replay in this audit.')
check('firstbatch_attempt1_exact',batches[0]==old_batches[0],'All first-batch files, gradient norms, cosines, complete selection statistics and object records equal the preserved attempt1 first batch exactly.')
check('fixed_B32_stream',all(len(r['files'])==32 for r in batches),'Eight sequential B32 file lists preserved. Eight-batch pixel tensors were not recopied or independently replayed.')
config_equal=(CAL/'direction_config.yaml').read_bytes()==(HERE/'training_reviewed_source_v2'/'configs'/'llvip_N_s42_FT3.yaml').read_bytes()
check('config_bytes_equal',config_equal,'Executed calibration config equals independently reviewed v2 N template byte for byte.')
ratios={'L3-DFL':[],'L3-GT':[]};summary_rows=[];masks_equal=True;base_total=selected_total=0
for row in batches:
    d,g=row['stats']['L3-DFL'],row['stats']['L3-GT']
    masks_equal=masks_equal and all(d[k]==g[k] for k in ('base_records','base_count','normalizer','selected_count','selected_anchors','selected_object_ids'))
    base_total+=d['base_count'];selected_total+=d['selected_count']
    assert len(d['base_records'])==d['base_count'] and d['normalizer']==max(1,d['base_count'])
    assert d['selected_count']==len(d['selected_anchors'])==sum(x['selected'] for x in d['base_records'])
    for x in d['base_records']:
        if x['selected']:
            assert x['teacher_anchor']==x['reference_anchor'] and x['full_probability_support'] and x['target_float32_supported'] and x['teacher_same_anchor_unique']
    nn=row['native_norm'];out={'batch':row['batch'],'base':d['base_count'],'selected':d['selected_count'],'native_norm':nn,'unit_B_kd_norms':row['unit_B_kd_norms'],'native_cosines':row['native_cosines'],'actual_scaled_gradient_ratios':{}}
    for arm in ratios:
        kn=row['unit_B_kd_norms'][arm];cos=row['native_cosines'][arm]
        assert math.isfinite(nn) and nn>0 and math.isfinite(kn) and kn>=0
        if kn>0:
            ratios[arm].append(.1*nn/kn)
            assert cos is not None and math.isfinite(cos) and -1<=cos<=1
        else:assert cos is None and d['selected_count']==0
        out['actual_scaled_gradient_ratios'][arm]=receipt['coefficients'][arm]*kn/nn
    summary_rows.append(out)
check('same_mask_base',masks_equal,f'Both variants share all {base_total} logged base records and {selected_total} selected objects; normalizer stays max(1,base), including zero-selected batch3.')
for arm in ratios:
    detail=receipt['details'][arm]
    check('ratio_recompute_'+arm,ratios[arm]==detail['ratios'] and len(ratios[arm])==detail['finite_nonzero_batches'] and statistics.median(ratios[arm])==detail['raw_median'],'Independent .1*native_norm/(B*unit_KD_norm) recomputation from every finite nonzero batch.')
dose=min(1,statistics.median(ratios['L3-DFL']))
check('shared_fixed_coefficient',len(ratios['L3-DFL'])>=4 and receipt['coefficients']=={'N':0.,'L3-DFL':dose,'L3-GT':dose} and receipt['blocked']=={},'Exactly one DFL-derived coefficient; GT keeps the same lambda despite its different raw gradient calibration median.')
resource_rows=[]
for row in resources:
    rr=row['resources'];peak=max(list(rr['per_gpu_peak_vram_mib'].values())+[row['gpu_allocated_peak_mib'],row['gpu_reserved_peak_mib']]);rss=rr['peak_rss_mib']
    vram=math.ceil((peak+max(256,.05*peak))/256)*256
    ram=math.ceil((rss+max(2048,.05*rss))/1024)*1024
    assert vram==row['measured_with_margin_vram_mib'] and ram==row['measured_with_margin_rss_mib']
    assert row['limits']=={'vram_mib':8192,'rss_mib':32768}
    resource_rows.append({'batch':row['batch'],'recomputed_vram_reservation_mib':vram,'recomputed_rss_reservation_mib':ram,'pass':vram<=8192 and ram<=32768 and row['status']=='PASS'})
check('all_resource_gates',all(x['pass'] for x in resource_rows) and receipt['all_eight_batch_resource_checks_passed'],'All eight recorded peaks plus unchanged margin independently satisfy 8192 MiB VRAM and 32768 MiB RSS ceilings.')
starts=[r['allocated_start_mib'] for r in lifecycle];after=[r['allocated_after_cleanup_mib'] for r in lifecycle];reserved=[r['reserved_after_cleanup_mib'] for r in lifecycle]
check('lifetime_plateau',len(set(starts[1:]))==1 and max(after)-min(after)<=.001 and len(set(reserved))==1 and all(not r['empty_cache_called'] and r['graph_loop_references_deleted'] for r in lifecycle),'Batches2-8 start allocated is exactly constant; cleanup allocation range <.001 MiB, reserved constant, no empty_cache. This establishes stability in the observed eight-batch window only.')
check('peak_ledger_closure',receipt['gpu_allocated_peak_mib']==max(r['gpu_allocated_peak_mib'] for r in resources) and receipt['gpu_reserved_peak_mib']==max(r['gpu_reserved_peak_mib'] for r in resources) and receipt['resources']['per_gpu_peak_vram_mib']==resources[-1]['resources']['per_gpu_peak_vram_mib'],'Terminal framework/NVML peaks agree with per-batch cumulative ledger. RSS terminal may rise during serialization; terminal peak remains below ceiling.')
check('terminal_RSS_within_budget',math.ceil((receipt['resources']['peak_rss_mib']+max(2048,.05*receipt['resources']['peak_rss_mib']))/1024)*1024<=32768,'Terminal process-tree RSS independently remains within the fixed ceiling after margin.')
names=receipt['parameter_names']
check('P3_P4_parameter_scope',len(names)==len(set(names))==24 and all(n.startswith(('model.16.','model.19.')) for n in names) and any(n.startswith('model.16.') for n in names) and any(n.startswith('model.19.') for n in names),'24 unique parameter tensors in the frozen P3/P4 source modules; names are explicit in executed receipt.')
manifest_index={Path(x['path']).name:x for x in source_manifest['files']}
source_bound=True
for entry in review['reviewed_source_files']:
    if entry['relative_path'] in ('configs/llvip_L3-DFL_s42_FT3.yaml','configs/llvip_L3-GT_s42_FT3.yaml','configs/llvip_native_evaluation.yaml'):continue
    file=manifest_index[Path(entry['relative_path']).name]
    source_bound=source_bound and file['bytes']==entry['bytes'] and file['byte_identity'] and '/release_v2/' in file['path']
    assert Path(entry['source_path']).read_bytes()==Path(entry['snapshot_path']).read_bytes()
check('source_execution_binding',source_bound,'All 15 calibration source-copy records bind reviewed release_v2 paths/bytes and producer byte-identity assertions; local reviewed/current bytes rechecked. Remote source-copy bytes were not downloaded again.')
assert not defects,defects
snapshot=OUT/'evidence_snapshot';snapshot.mkdir();identity=[]
for name in ('calibration_receipt.json','calibration_batches.jsonl','calibration_allocation_lifecycle.jsonl','calibration_resource_batches.jsonl','direction_config.yaml','runtime_ready.json','source_manifest.json'):
    source=CAL/name;dest=snapshot/name;raw=source.read_bytes();dest.write_bytes(raw);assert raw==source.read_bytes()==dest.read_bytes()
    identity.append({'source_path':str(source.resolve()),'snapshot_path':str(dest.resolve()),'bytes':len(raw),'direct_byte_identity':True})
measurements={'lambda_shared':dose,'finite_nonzero_batches':len(ratios['L3-DFL']),'base_objects_total':base_total,'selected_objects_total':selected_total,
 'allocated_peak_mib':receipt['gpu_allocated_peak_mib'],'NVML_peak_mib':max(receipt['resources']['per_gpu_peak_vram_mib'].values()),'reserved_peak_mib':receipt['gpu_reserved_peak_mib'],'RSS_peak_mib':receipt['resources']['peak_rss_mib'],
 'start_allocated_batches2_8_mib':starts[1],'after_cleanup_allocated_range_mib':[min(after),max(after)],'reserved_after_cleanup_mib':reserved[0],
 'median_actual_scaled_gradient_ratio_nonzero':{arm:statistics.median(row['actual_scaled_gradient_ratios'][arm] for row in summary_rows if row['unit_B_kd_norms'][arm]>0) for arm in ratios},'seconds':receipt['seconds']}
dump(OUT/'RECOMPUTATION.json',{'checks':checks,'measurements':measurements,'batch_rows':summary_rows,'resource_rows':resource_rows,'source_evidence_identity':identity,'new_hash_computed':False})
audit={'date':datetime.now(timezone.utc).isoformat(),'auditor':'/root/object_transport_review','overall_verdict':'pass','integrity_status':'pass','reason_code':'ACTUAL_FIXED8_CALIBRATION_AND_RESOURCE_COMPUTATION_ACCEPTED',
 'checks':{
  'gt_provenance':{'status':'pass','details':'Actual original LLVIP train annotations used through frozen dual-label loader; audit checks logged same-mask base/anchor identities, not independent physical correspondence.','evidence':[str(CAL/'calibration_batches.jsonl')+' records1-8']},
  'score_normalization':{'status':'pass','details':'Shared lambda independently recomputed from .1 native/B-unit-DFL norms, seven valid rows; GT shares lambda and actual GT gradient dose is separately reported. No AP involved.','evidence':[str(OUT/'RECOMPUTATION.json')]},
  'result_existence':{'status':'pass','details':'Completed calibration receipt and three materialized eight-row numerical/resource/lifecycle ledgers exist. This snapshot outer dispatcher status was still RUNNING, so this audit accepts computation terminal, not outer queue process completion.','evidence':[str(CAL/'calibration_receipt.json'),str(SOURCE/'queue'/'object_dfl_attempt2_llvip_N_calibration_status.json')]},
  'dead_code':{'status':'pass','details':'V2 cleanup branch exercised all eight times; direct lifecycle diagnostics show no cross-batch accumulation over this window. Source-copy manifest binds to reviewed v2.','evidence':[str(CAL/'calibration_allocation_lifecycle.jsonl'),str(CAL/'source_manifest.json')]},
  'scope':{'status':'pass','details':'Exactly eight B32 batches, zero optimizer/EMA updates; first complete batch content equals preserved attempt1. No new canary, FT3 endpoint, AP or KD effect accepted here.','evidence':[str(OUT/'RECOMPUTATION.json'),str(CAL/'calibration_receipt.json')]},
  'eval_type':{'status':'pass','types':['real_gt'],'details':'Real model forward/gradient calibration on train labels and GPU resource observation; not a detection accuracy evaluation.','evidence':[str(CAL/'calibration_batches.jsonl')]},
  'resource_and_lifetime':{'status':'pass','details':'All eight peak-plus-margin gates pass; cleanup/start allocation stable without empty_cache. Formal subsequent training must still use actual canary peaks.','evidence':[str(OUT/'RECOMPUTATION.json')]},
  'all_specific_checks':{'status':'pass','details':checks,'evidence':[str(OUT/'RECOMPUTATION.json')]}
 },'claims':[
  {'id':'v2_calibration_completed','impact':'supported','affected_evidence':[str(CAL/'calibration_receipt.json')],'rationale':'Eight real rows and independently recomputed shared coefficient agree.'},
  {'id':'v2_no_graph_accumulation_in_fixed8','impact':'supported','affected_evidence':[str(CAL/'calibration_allocation_lifecycle.jsonl')],'rationale':'Batch2-8 starts and cleanup memory show a plateau; original batch2 resource failure does not recur within fixed8.'},
  {'id':'GT_DFL_equal_actual_gradient_dose','impact':'unsupported','affected_evidence':[str(OUT/'RECOMPUTATION.json')],'rationale':'Lambda is shared, but actual gradient magnitudes/cosines differ; GT median actual ratio is separately reported.'},
  {'id':'canary_FT3_KD_utility_or_physical_registration','impact':'unsupported','affected_evidence':[str(CAL/'calibration_receipt.json')],'rationale':'This scoped review accepts calibration only, not subsequent endpoints or scientific efficacy.'}
 ],'run_id':'rgbir_object_dfl_screen_20260908/attempt2/calibration/llvip','code_revision':'release_v2, reviewed exact-byte source snapshot',
 'declared_input_set':[x['source_path'] for x in identity],'audited_input_hashes':[],'new_hash_computed':False,'identity_policy':'Direct full-byte snapshots and resolved paths; user no-new-hash scope overrides skill hash default.',
 'measurements':measurements,'recomputation_command':str((HERE/'audit_actual_calibration_v2.py').resolve()),'inaccessible_inputs':['Original per-parameter gradient arrays and complete GPU tensors were not stored/recomputed by the reviewer.','Outer dispatcher clean-exit terminal was not yet present in this snapshot.'],
 'limits':['Accepts actual calibration computation and resource/lifetime evidence only.','Does not block or bypass the already encoded subsequent canary gates.','No AP, efficacy, physical registration or full-training memory guarantee.']}
dump(OUT/'EXPERIMENT_AUDIT.json',audit)
md=f'''# 实际 v2 八批校准独立审计

**PASS（限校准计算、剂量与资源记录）**。从8条实际梯度记录独立重算，7/8批有限非零，固定共享 λ={dose:.16g}；DFL与GT全部{base_total}个基础对象和{selected_total}个选中对象同掩码、同anchor、同分母。第3批无选中，保留为0梯度，没有替换批次。

八批资源门全部通过。NVML峰5454 MiB、framework allocated峰4390.573 MiB、reserved峰4954 MiB、进程树RSS峰17280 MiB；原规则计算的显存预约5888 MiB，低于不变的8192 MiB。batch2–8开始allocated均为82.584 MiB，清理后为382.5889–382.5898 MiB，reserved全程4954 MiB，没有empty_cache；固定八批内未再出现原跨批累积现象。

v2首批完整JSON记录与失败attempt1首批逐值一致，包括32文件、梯度norm/cosine和完整对象选择统计。执行config与v2独立快照逐字节一致；source manifest指向reviewed release_v2。全模型初始状态/逐批恢复/BN冻结/零optimizer与EMA更新由执行路径断言和终态记录支持，本审计没有另跑模型重建梯度。

共享λ不等于相同梯度剂量：非零批实际 `λ·norm(B·KD)/norm(native)` 的中位数，DFL={measurements['median_actual_scaled_gradient_ratio_nonzero']['L3-DFL']:.9f}，GT={measurements['median_actual_scaled_gradient_ratio_nonzero']['L3-GT']:.9f}；每批余弦和剂量都保留于RECOMPUTATION.json。GT没有单独重校准。

校准计算耗时{receipt['seconds']:.3f}秒。收集快照的外层dispatcher状态仍为RUNNING，因此本审计不宣称外层进程或后续canary/FT3已完成；它不阻断也不绕过原encoded queue的后续实际门槛。没有读取新AP，不作KD效用或物理配准结论。原失败attempt1保留，v2仅在本固定八批范围证实资源恢复稳定。
'''
(OUT/'EXPERIMENT_AUDIT.md').write_text(md,encoding='utf-8')
print(json.dumps({'overall_verdict':'pass','measurements':measurements,'checks':len(checks)},indent=2))
