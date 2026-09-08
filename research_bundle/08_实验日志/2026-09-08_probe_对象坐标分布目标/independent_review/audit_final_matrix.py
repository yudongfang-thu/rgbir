"""Read existing final artifacts only. No inference, training, tests or hashes."""
import csv, gzip, io, json, math, pathlib, datetime
import yaml

ENTRY = pathlib.Path(__file__).resolve().parent.parent
ROOT = ENTRY / 'training_evidence_final'
OUT = ENTRY / 'independent_review/final_matrix'
OUT.mkdir(exist_ok=False)
seen = {}
checks = []
def read(p):
    p = pathlib.Path(p).resolve()
    b = p.read_bytes()
    if p in seen and seen[p] != b:
        raise RuntimeError('Input changed during inspection: '+str(p))
    seen[p] = b
    return b
def js(p): return json.loads(read(p).decode('utf-8-sig'))
def yl(p): return yaml.safe_load(read(p).decode('utf-8-sig'))
def lines(p): return [json.loads(x) for x in read(p).decode('utf-8-sig').splitlines() if x]
def check(name, value, evidence, detail=None):
    checks.append(dict(name=name, passed=bool(value), evidence=evidence, details=detail))
def finite(v):
    if isinstance(v, float): return math.isfinite(v)
    if isinstance(v, dict): return all(finite(x) for x in v.values())
    if isinstance(v, list): return all(finite(x) for x in v)
    return True
def equal(p,q): return read(p)==read(q)
def pathlabel(p): return str(pathlib.Path(p).resolve())

arms = ['N','L3-DFL','L3-GT']
scope = 'OBJECT_DFL_FT3_BNFROZEN'
endpoint = 'OBJECT_DFL_FT3_BNFROZEN_LAST_EMA'
metrics = ['mAP50_95','AP50','AP75','precision','recall']
collection = js(ROOT/'collection_receipt.json')
check('collection_163_files_present_exact_sizes',len(collection['files'])==163 and all(len(read(ROOT/r['path']))==r['bytes'] for r in collection['files']),pathlabel(ROOT/'collection_receipt.json'))
review = js(ENTRY/'independent_review/TRAINING_SOURCE_REVIEW_v2.json')
deployed = js(ROOT/'pinned_source/deployment.json')
deploy = {r['path']:r for r in deployed['files']}
for r in review['reviewed_source_files']:
    check('reviewed_source_'+r['relative_path'],equal(r['source_path'],r['snapshot_path']) and len(read(r['source_path']))==r['bytes'] and deploy[r['relative_path']]['bytes']==r['bytes'] and deploy[r['relative_path']]['byte_exact'],r)
check('executed_analyzer_source_exact',equal(ENTRY/'analysis/output_attempt1/analyze_object_dfl_source.py',ENTRY/'independent_review/analyzer_reviewed_source/analyze_object_dfl.py') and equal(ENTRY/'analysis/analyze_object_dfl.py',ENTRY/'analysis/output_attempt1/analyze_object_dfl_source.py'),pathlabel(ENTRY/'analysis/output_attempt1/analyze_object_dfl_source.py'))

completed = js(ROOT/'queue/llvip_completion.json')
matrix = js(ROOT/'queue/completion.json')
expected = [('N','calibration')]+[(a,'canary') for a in arms]+[(a,s) for a in arms for s in ['train','eval']]
check('ten_stage_complete_order',completed['status']=='COMPLETED' and not completed['blocked'] and [(r['arm'],r['stage']) for r in completed['completed']]==expected and matrix['status']=='OBJECT_DFL_MATRIX_COMPLETED',pathlabel(ROOT/'queue/llvip_completion.json'))
stage_stats = []
for r in completed['completed']:
    status=js(ROOT/'queue'/f"{r['id']}_status.json")
    profile=js(ROOT/'queue'/f"{r['id']}_resource_profile.json")
    admission=js(ROOT/'queue'/f"{r['id']}_admission.json")
    job=js(ROOT/'queue'/f"{r['id']}_job.json")
    rp=ROOT/r['receipt'].split('/attempt2/')[1]
    receipt=js(rp)
    check(r['id']+'_terminal',status['status']=='COMPLETED' and status['exit_code']==0 and not status['monitor_errors'] and profile['exit_code']==0 and not profile['monitor_errors'] and r['receipt_validated'] and job['expected_receipt']==r['receipt'],pathlabel(ROOT/'queue'/f"{r['id']}_status.json"))
    samples=profile['samples']; minfree=min(s['memory_free_mib'] for s in samples); maxrss=max(s['project_rss_mib'] for s in samples)
    check(r['id']+'_resources',minfree>=2048 and maxrss<=300*1024 and len(admission['active_gpus_after'])<=4 and (len(admission['active_gpus_after'])<=3 or (admission['four_gpu_exception'] and len(admission['empty_gpus_after'])>=2)),pathlabel(ROOT/'queue'/f"{r['id']}_admission.json"))
    stage_stats.append(dict(id=r['id'],seconds=receipt['seconds'],minimum_free_mib=minfree,peak_project_rss_mib=maxrss,admission=admission,launch=status['launch']))

# Already independently audited calibration/canary records must be the same bytes.
prior_equal=[]
for subtree,previous in [('calibration','training_evidence_1716'),('canaries','training_evidence_canaries')]:
    for p in (ROOT/subtree).rglob('*'):
        if p.is_file():
            old=ENTRY/previous/p.relative_to(ROOT)
            if old.exists(): prior_equal.append(equal(p,old))
check('prior_actual_calibration_canaries_unchanged',all(prior_equal),[pathlabel(ENTRY/'independent_review/calibration_v2/EXPERIMENT_AUDIT.json'),pathlabel(ENTRY/'independent_review/canaries_v2/EXPERIMENT_AUDIT.json')],{'compared_files':len(prior_equal)})
cal=js(ROOT/'calibration/llvip/calibration_receipt.json')
initial=js(ENTRY/'initial_endpoint_reference/initial_evaluation_receipt.json')
initial_contract=js(ENTRY/'initial_endpoint_reference/initial_evaluation_contract.json')
initial_identity=js(ENTRY/'initial_endpoint_reference/REFERENCE_IDENTITY.json')
check('initial_existing_receipt_copy_identity',equal(ENTRY/'initial_endpoint_reference/initial_evaluation_receipt.json',ENTRY/'initial_endpoint_reference/remote_source/initial_evaluation_receipt.json'),pathlabel(ENTRY/'initial_endpoint_reference/REMOTE_COLLECTION.json'))
roster=read(ENTRY/'initial_endpoint_reference/development_roster.txt').decode('utf-8-sig').splitlines()
check('initial_full_roster_identity',roster==initial_contract['roster'] and len(set(roster))==2406 and set(roster)==set(initial_contract['actual_loader_roster']),pathlabel(ENTRY/'initial_endpoint_reference/initial_evaluation_contract.json'))

results={}; training={}; kd={}; eval_gt={}; capture_stats={}; config_diffs={}; csv_lengths={}
for a in arms:
    tr=ROOT/'runs/llvip'/a; ev=ROOT/'evaluations/llvip'/a
    t=js(tr/'completion_receipt.json'); training[a]=t
    e=js(ev/'direction_evaluation_receipt.json'); results[a]=e
    ini=js(tr/'initialization_check.json')
    check(a+'_training_completion',t['status']=='OBJECT_DFL_TRAINING_COMPLETED' and t['scope']==scope and t['method_identity']==scope and t['endpoint']==endpoint and (t['epochs_configured'],t['last_epoch'],t['batches'],t['successful_updates'],t['attempts'],t['amp_skips'],t['ema_updates'],t['selected_objects'])==(3,3,192,91,96,5,96,673),pathlabel(tr/'completion_receipt.json'))
    check(a+'_full_warm_start_isolation',ini['status']=='PASS_FULL_STATE_WARM_START' and ini['initial_checkpoint']==initial['checkpoint'] and ini['state_tensors']==499 and all(ini[k] for k in ['head_included','fresh_optimizer','fresh_ema','teacher_reference_isolated']) and t['bn_buffer_count']==243 and t['bn_running_buffers_unchanged'] and t['bn_affine_trainable'],pathlabel(tr/'initialization_check.json'))
    check(a+'_independent_last_checkpoint_binding',e['checkpoint']==t['checkpoint'] and equal(ev/'training_completion_copy.json',tr/'completion_receipt.json') and e['training_model']==initial['checkpoint']['path'],pathlabel(ev/'direction_evaluation_receipt.json'))
    effective=ROOT/f'effective_configs/llvip_{a}_s42_FT3.yaml'
    check(a+'_all_config_copies_exact',all(equal(effective,q/'direction_config.yaml') for q in [tr,ev,ROOT/'canaries/llvip'/a]),pathlabel(effective))
    cfg=yl(effective); frozen=yl(ENTRY/f'trainer_release/configs/llvip_{a}_s42_FT3.yaml')
    diff={k:[frozen.get(k),cfg.get(k)] for k in set(frozen)|set(cfg) if frozen.get(k)!=cfg.get(k)}; config_diffs[a]=diff
    coefficient=0. if a=='N' else .6900524651944485
    check(a+'_only_frozen_calibration_config_updates',set(diff)==({'calibration_receipt'} if a=='N' else {'calibration_receipt','localization_coefficient','kd_coefficient'}) and cfg['kd_coefficient']==coefficient and t['kd_coefficient']==coefficient and e['kd_coefficient']==coefficient,pathlabel(effective),diff)
    check(a+'_first30_cross_arm_and_canary_bytes',equal(tr/'sample_stream.jsonl',ROOT/'canaries/llvip'/a/'sample_stream.jsonl') and equal(tr/'sample_stream.jsonl',ROOT/'runs/llvip/N/sample_stream.jsonl') and len(lines(tr/'sample_stream.jsonl'))==30,pathlabel(tr/'sample_stream.jsonl'))
    ready=js(tr/'runtime_ready.json')
    check(a+'_runtime_population',all(ready[k]==v for k,v in [('train_images',2048),('val_images',2406),('train_batches',64),('batch',32),('workers',4)]),pathlabel(tr/'runtime_ready.json'))
    for sub in [tr,ev]:
        manifest=js(sub/'source_manifest.json')
        records=[r for r in manifest['files'] if '/release_v2/' in r['path']]
        check(a+'_'+sub.parts[-3]+'_runtime_source_binding',all(r['byte_identity'] and r['bytes']==deploy[r['path'].split('/release_v2/')[1]]['bytes'] for r in records) and all(r['byte_identity'] for r in manifest['files']),pathlabel(sub/'source_manifest.json'),{'release_v2_records':len(records),'all_manifest_records':len(manifest['files']),'remote_source_copies_not_redownloaded':True})
    kd[a]=lines(tr/'kd_batches.jsonl')
    check(a+'_saved_diagnostics_finite',finite(kd[a]) and finite(lines(tr/'gradient_checks.jsonl')) and finite(t),pathlabel(tr/'kd_batches.jsonl'))
    check(a+'_positive_aux_gradient_isolated',all(g['kd_gradient_l2']>0 and not g['teacher_has_grad'] and not g['reference_has_grad'] for g in t['gradient_checks']),pathlabel(tr/'gradient_checks.jsonl'))
    check(a+'_native_loss_retained_and_once_scaled',all(not x['native_loss_replaced'] and not x['classification_kd_evaluated'] and x['temperature']==2 and x['temperature_squared_applied'] and x['four_edges_mean'] and x['coefficient']==coefficient and abs(x['weighted_kd_total']-x['actual_B']*coefficient*x['loss_unweighted'])<1e-12 and abs(x['total_loss']-(x['native_total']+x['weighted_kd_total']))<=2e-5 for x in kd[a]),pathlabel(tr/'kd_batches.jsonl'))
    check(a+'_n_exact_or_nonzero_aux',all(x['native_total']==x['total_loss'] for x in kd[a]) if a=='N' else any(x['weighted_kd_total']>0 for x in kd[a]),pathlabel(tr/'kd_batches.jsonl'))
    ct=js(ev/'direction_evaluation_contract.json'); pr=js(ev/'actual_native_evaluation_profile.json')
    check(a+'_full_dev_actual_profile',e['status']=='OBJECT_DFL_EVALUATION_COMPLETED' and e['scope']==scope and e['endpoint']==endpoint and e['method_identity']==scope and all(e[k]==v for k,v in [('full_dev_images',2406),('observed_images',2406),('full_dev_gt_objects',7879),('gt_objects_captured',7879)]) and ct['roster']==initial_contract['roster'] and ct['actual_loader_roster']==initial_contract['actual_loader_roster'] and ct['effective_kwargs']==pr['effective_kwargs']==initial['actual_effective_kwargs'] and ct['evaluator_identity']==initial_contract['evaluator_identity'] and pr['actual_class_names']=={'0':'person'} and pr['gt_objects_before_inference']==7879 and e['evaluation_identity_projection']['actual_evaluation_data_yaml']==initial['actual_evaluation_data_yaml'],pathlabel(ev/'direction_evaluation_contract.json'))
    check(a+'_native_yaml_exact_to_initial',equal(ev/'native_config.yaml',ENTRY/'initial_endpoint_reference/remote_source/requested_native_config.yaml'),pathlabel(ev/'native_config.yaml'))
    capture=[json.loads(x) for x in gzip.decompress(read(ev/'native_capture/objects.jsonl.gz')).decode('utf-8').splitlines()]
    # Grouped-dev image paths are aliases of canonical raw/train paths. Only their
    # names can be compared locally; runtime contract records resolved paths.
    names=[pathlib.PurePosixPath(x['image']).name for x in capture]
    check(a+'_actual_capture_population_and_finite',len(capture)==len(set(names))==2406 and names==[pathlib.PurePosixPath(x).name for x in ct['actual_loader_roster']] and sum(len(x['gt_boxes']) for x in capture)==7879 and finite(capture) and all(len(x['gt_boxes'])==len(x['gt_classes']) and len(x['pred_boxes'])==len(x['pred_classes'])==len(x['pred_confidence']) and all(0<=z<=1 for z in x['pred_confidence']) for x in capture),pathlabel(ev/'native_capture/objects.jsonl.gz'))
    eval_gt[a]=[{k:x[k] for k in ['image','canvas_shape','original_shape','gt_boxes','gt_classes']} for x in capture]
    capture_stats[a]=dict(images=len(capture),gt_objects=sum(len(x['gt_boxes']) for x in capture),predictions=sum(len(x['pred_boxes']) for x in capture),empty_prediction_images=sum(not x['pred_boxes'] for x in capture),all_stored_numbers_finite=finite(capture))
    check(a+'_native_metric_fraction_class_mean',e['metric_units']=='fraction_0_to_1' and all(math.isfinite(e[m]) and 0<=e[m]<=1 for m in metrics) and len(e['per_class'])==1 and e['per_class'][0]['class_id']==0 and all(e[m]==e['per_class'][0][m] for m in metrics[:3]),pathlabel(ev/'direction_evaluation_receipt.json'))
    rows=list(csv.reader(read(tr/'results.csv').decode().splitlines())); csv_lengths[a]=[len(x) for x in rows]

check('actual_gt_capture_exact_all_three_arms',eval_gt['N']==eval_gt['L3-DFL']==eval_gt['L3-GT'],[pathlabel(ROOT/'evaluations/llvip'/a/'native_capture/objects.jsonl.gz') for a in arms])
identity_keys=[k for k in kd['N'][0] if k.endswith('_count')]+['fullsupport_checked','fullsupport_rejected','target_float32_rejected','normalizer','selected_anchors','selected_object_ids','base_records','selection_identity','config','batch','epoch']
check('all_saved_mask_base_anchor_identity',len(kd['N'])==len(kd['L3-DFL'])==len(kd['L3-GT'])==15 and all(all(x[k]==y[k]==z[k] for k in identity_keys) for x,y,z in zip(kd['N'],kd['L3-DFL'],kd['L3-GT'])),[pathlabel(ROOT/'runs/llvip'/a/'kd_batches.jsonl') for a in arms])
base=[r for x in kd['N'] for r in x['base_records']]; selected=[r for r in base if r['selected']]
check('saved_selected_masks_satisfy_frozen_bounds',all(r['reference_anchor']==r['teacher_anchor'] and r['reference_conf']>=.25 and r['reference_iou']<.7 and r['teacher_conf']>=.25 and r['teacher_own_iou']>=.5 and r['mapped_teacher_rgb_iou']>=.6 and r['mapped_teacher_rgb_iou']>r['reference_iou']+.05 and r['teacher_same_anchor_unique'] and r['full_probability_support'] and r['target_float32_supported'] for r in selected) and all(len(x['base_records'])==x['base_count']==x['normalizer'] and sum(r['selected'] for r in x['base_records'])==x['selected_count'] for x in kd['N']),pathlabel(ROOT/'runs/llvip/N/kd_batches.jsonl'))
mask_summary={k:sum(x[k] for x in kd['N']) for k in kd['N'][0] if k.endswith('_count') or k in ['fullsupport_checked','fullsupport_rejected','target_float32_rejected']}
mask_summary.update(saved_batches=[x['batch'] for x in kd['N']],saved_zero_selected_batches=[x['batch'] for x in kd['N'] if x['selected_count']==0],total_selected_occurrences_per_arm=673,all_saved_base_rgb_ir_gt_equal=all(r['rgb_gt']==r['ir_gt'] for r in base),all_saved_selected_mapped_box_identity=all(r['teacher_box']==r['mapped_teacher_box'] for r in selected))

summary=js(ENTRY/'analysis/output_attempt1/summary.json'); ds=summary['datasets']['llvip']
check('analyzer_complete_scope_units',summary['status']=='COMPLETE_OBJECT_DFL_READOUT' and summary['scope']==scope and summary['endpoint']==endpoint and summary['seed']==42 and summary['n_seeds']==1 and summary['standard_deviation'] is None and not summary['formal_paper_gain_claim'] and not summary['automatically_admit_e200'],pathlabel(ENTRY/'analysis/output_attempt1/summary.json'))
raw={a:{m:results[a][m] for m in metrics} for a in arms}
for r in ds['arms']:
    check('analyzer_'+r['arm']+'_all_values',r['raw_fraction']==raw[r['arm']] and r['display_percent']=={m:100*raw[r['arm']][m] for m in metrics},pathlabel(ENTRY/'analysis/output_attempt1/summary.json')+'#datasets.llvip.arms')
differences={}
for r in ds['comparisons']:
    delta={m:100*(raw[r['arm']][m]-raw[r['control']][m]) for m in metrics};differences[r['contrast']]=delta
    check('analyzer_'+r['contrast']+'_all_differences',delta==r['delta_pp'],pathlabel(ENTRY/'analysis/output_attempt1/summary.json')+'#datasets.llvip.comparisons')
initial_delta={a:{m:100*(raw[a][m]-initial[m]) for m in metrics} for a in arms}
report=read(ENTRY/'FINAL_REPORT.md').decode('utf-8-sig')
report_numbers=[f'{100*initial[m]:.6f}' for m in metrics]+[f'{100*raw[a][m]:.6f}' for a in arms for m in metrics]+[f'{abs(initial_delta[a]["mAP50_95"]):.6f}' for a in arms]+[f'{abs(d[m]):.6f}' for d in differences.values() for m in metrics[:3]]+[f'{x["seconds"]:.3f}' for x in stage_stats]+[f'{matrix["seconds"]:.3f}']
check('final_report_display_arithmetic_and_seconds',all(x in report for x in report_numbers),pathlabel(ENTRY/'FINAL_REPORT.md'),{'numeric_tokens_verified':len(report_numbers),'not_claiming_unrelated_historical_C0_L2_sources_reviewed':True})
check('final_report_four_gpu_example',js(ROOT/'queue/object_dfl_attempt2_llvip_N_train_admission.json')==dict(active_gpus_after=[0,2,4,5],empty_gpus_after=[1,3],four_gpu_exception=True,minimum_empty_required=2),pathlabel(ENTRY/'FINAL_REPORT.md')+'#实测代价与有效性')

recomputed=dict(status='PASS' if all(x['passed'] for x in checks) else 'FAIL',artifact_checks=checks,raw_metrics_fraction=raw,display_percent={a:{m:100*v for m,v in vals.items()} for a,vals in raw.items()},differences_pp=differences,initial_reference_fraction={m:initial[m] for m in metrics},difference_from_initial_pp=initial_delta,stage_resources_and_seconds=stage_stats,queue_seconds=matrix['seconds'],mask_coverage=mask_summary,capture_coverage=capture_stats,config_changes_from_frozen=config_diffs,results_csv_column_counts=csv_lengths,new_hash_computed=False,new_GPU=False,new_tests=False,new_inference=False)
(OUT/'RECOMPUTATION.json').write_text(json.dumps(recomputed,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
manifest=[]
for p,b in seen.items():
    stable=p.read_bytes()==b
    manifest.append(dict(path=str(p),bytes=len(b),mtime_ns=p.stat().st_mtime_ns,direct_bytes_unchanged_at_end=stable))
check('all_declared_inputs_unchanged_at_end',all(x['direct_bytes_unchanged_at_end'] for x in manifest),'INPUT_MANIFEST.json')
(OUT/'INPUT_MANIFEST.json').write_text(json.dumps(dict(identity_method='Direct byte comparisons against bytes actually inspected and reviewed snapshots; no new hash per explicit task scope',new_hash_computed=False,files=manifest),ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(status=recomputed['status'],checks=len(checks),failed=[x for x in checks if not x['passed']],input_files=len(manifest),mask_summary=mask_summary,capture=capture_stats,initial_delta_pp=initial_delta),ensure_ascii=False,indent=2))
