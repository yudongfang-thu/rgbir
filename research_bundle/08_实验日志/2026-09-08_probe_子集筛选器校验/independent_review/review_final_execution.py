"""Bounded terminal artifact audit. No testing, inference, new metrics or hashes."""
import pathlib,json,gzip,math,datetime
P=pathlib.Path(__file__).resolve().parent.parent;E=P/'evidence_final';O=P/'independent_review'
seen={};checks=[]
def b(p):
 p=pathlib.Path(p).resolve();v=p.read_bytes()
 if p in seen and seen[p]!=v:raise RuntimeError('Input changed: '+str(p))
 seen[p]=v;return v
def j(p):return json.loads(b(p).decode('utf-8-sig'))
def jl(p):return [json.loads(x) for x in b(p).decode('utf-8-sig').splitlines() if x]
def check(name,v,e):checks.append(dict(name=name,status='pass' if v else 'fail',evidence=str(e)))
scope='DRONE_SUBSET2048_PRETRAIN_E8_CHECK';endpoint='SUBSET2048_PRETRAIN_E8_LAST_EMA'
src=j(O/'SOURCE_REVIEW.json');files={r['relative_path']:r for r in src['reviewed_source_files']}
for r in files.values():check('source_'+r['relative_path'],b(r['source_path'])==b(r['snapshot_path']),r['source_path'])
check('actual_analyzer_source_accepted',b(P/'analysis/output_attempt1/analyze_subset_e8_source.py')==b(O/'analyzer_reviewed_source_v2/analyze_subset_e8.py'),P/'analysis/output_attempt1/analyze_subset_e8_source.py')
q=j(E/'queue/completion.json');collect=j(E/'collection_receipt.json');summary=j(P/'analysis/output_attempt1/summary.json')
check('six_actual_stages_fixed_order',q['status']=='SUBSET_SCREEN_MATRIX_COMPLETED' and q['scope']==scope and q['error'] is None and [(x['arm'],x['stage']) for x in q['completed']]==[('N','canary'),('C0','canary'),('N','train'),('N','eval'),('C0','train'),('C0','eval')] and all(x['validated'] for x in q['completed']),E/'queue/completion.json')
check('collection102_without_private_tensors',len(collect['files'])==102 and not collect['private_tensor_files_collected'],E/'collection_receipt.json')
check('no_failed_final_attempt_artifacts',not any(x.is_file() for x in E.rglob('*failure*')) and not any(x.is_file() for x in E.rglob('*timeout*')),E)
resource_summary={}
for row in q['stage_timings']:
 name=row['job_id'];status=j(E/'queue'/(name+'_status.json'));profile=j(E/'queue'/(name+'_resource_profile.json'));admission=j(E/'queue'/(name+'_admission.json'))
 check(name+'_clean_exit',status['status']=='COMPLETED' and status['exit_code']==0 and not status['monitor_errors'] and profile['exit_code']==0 and not profile['monitor_errors'] and row['actual_launch_observed'] and not row['budget_timed_out'],E/'queue'/(name+'_status.json'))
 check(name+'_shared_resource_limits',status['minimum_free_mib']>=2048 and max(s['project_rss_mib'] for s in profile['samples'])<=300*1024 and len(admission['active_gpus_after'])<=4 and (len(admission['active_gpus_after'])<=3 or admission['four_gpu_exception'] and len(admission['empty_gpus_after'])>=2),E/'queue'/(name+'_resource_profile.json'))
 resource_summary[name]=dict(reserved_vram_mib=status['launch']['expected_vram_mib'],reserved_rss_mib=status['launch']['expected_rss_mib'],minimum_free_mib=status['minimum_free_mib'],observed_execution_seconds=row['observed_execution_seconds'],admission_wait_seconds=row['admission_wait_seconds'])
check('execution_wait_wall_close',q['execution_seconds']<2700 and abs(q['execution_seconds']+q['admission_wait_seconds']-q['total_wall_seconds'])<1e-4 and abs(sum(x['admission_wait_seconds'] for x in q['stage_timings'])-q['admission_wait_seconds'])<1e-3 and 0<=q['execution_seconds']-sum(x['observed_execution_seconds'] for x in q['stage_timings'])<1,E/'queue/completion.json')
arms={};evals={};contracts={};flows={};initial={};GT={};metrics=['mAP50_95','AP50','AP75','precision','recall']
baseflow=j(E/'canaries/N/flow_check.json');basestate=j(E/'canaries/N/initialization_check.json')
for a in ('N','C0'):
 tr=E/'runs'/a;ev=E/'evaluations'/a;t=j(tr/'short_training_receipt.json');e=j(ev/'short_evaluation_receipt.json');ct=j(ev/'short_evaluation_contract.json');evals[a]=e;contracts[a]=ct
 init=j(tr/'initialization_check.json');flow=j(tr/'flow_check.json');initial[a]=init;flows[a]=jl(tr/'sample_stream.jsonl')
 check(a+'_canary_unchanged',b(E/'canaries'/a/'canary.json')==b(P/'evidence_canaries/canaries'/a/'canary.json') and b(E/'canaries'/a/'flow_check.json')==b(P/'evidence_canaries/canaries'/a/'flow_check.json'),E/'canaries'/a/'canary.json')
 check(a+'_E8_512_completed',t['status']=='SUBSET_SCREEN_TRAINING_COMPLETED' and t['scope']==scope and t['endpoint']==endpoint and t['arm']==a and t['seed']==42 and (t['epochs_configured'],t['last_epoch'],t['batches'],t['expected_train_images'])==(8,8,512,2048) and t['loader_workers_cleaned'],tr/'short_training_receipt.json')
 check(a+'_BN_optimizer_EMA_accounting',t['bn_running_statistics']=='normal_training' and t['bn_running_buffers_changed']==243 and t['fresh_optimizer'] and t['fresh_ema'] and t['attempts']==t['optimizer_updates']+t['amp_skips']==304 and t['successful_updates']==t['optimizer_updates'] and t['ema_updates']==304,tr/'short_training_receipt.json')
 check(a+'_common_fiveclass_state',init['status']=='PASS' and init['cross_arm_initial_state_exact'] and init['tensors']==499 and init['head_included'] and init['reference_state']==basestate['reference_state'] and init['initialization']==t['initialization'] and init['fresh_optimizer'] and init['fresh_ema'] and init['ema_updates']==0,tr/'initialization_check.json')
 check(a+'_actual_server30_pixel_label_equality',flow['status']=='PASS' and flow['exact_pixels_and_labels'] and flow['prefix_batches']==len(flow['files'])==len(flows[a])==30 and flow['reference_dir']==baseflow['reference_dir'] and [x['reference'] for x in flow['files']]==[x['reference'] for x in baseflow['files']] and all(x['exact'] for x in flow['files']),tr/'flow_check.json')
 check(a+'_all512_timing_records',len(jl(tr/'batch_timing.jsonl'))==len(t['batch_duration_seconds'])==len(t['normal_cadence_batch_seconds'])==512,tr/'batch_timing.jsonl')
 check(a+'_fixed_config_all_copies',b(tr/'short_screen_config.yaml')==b(ev/'short_screen_config.yaml')==b(P/f'release/configs/drone_{a}_s42_E8.yaml')==b(E/'canaries'/a/'short_screen_config.yaml'),tr/'short_screen_config.yaml')
 for d in (tr,ev):
  manifest=j(d/'source_manifest.json');bound=[x for x in manifest['files'] if '/rgbir_subset_e8_check_20260908/release_v1/' in x['path']]
  check(a+'_'+d.parts[-2]+'_source_manifest',all(x['byte_identity'] for x in manifest['files']) and all(x['bytes']==files[x['path'].split('/release_v1/')[1]]['bytes'] for x in bound if x['path'].split('/release_v1/')[1] in files),d/'source_manifest.json')
 check(a+'_lastEMA_receipt_stat_binding',e['checkpoint']==t['checkpoint'] and b(tr/'short_training_receipt.json')==b(ev/'short_training_receipt_copy.json') and len(b(tr/'short_training_receipt.json'))==e['training_completion']['bytes'] and e['training_completion']['path'].endswith('/runs/'+a+'/short_training_receipt.json') and e['training_configuration']['bytes']==len(b(tr/'short_screen_config.yaml')) and e['initialization']==t['initialization'],ev/'short_evaluation_receipt.json')
 check(a+'_full_dev_actual_completion',e['status']=='SUBSET_SCREEN_EVALUATION_COMPLETED' and e['scope']==scope and e['endpoint']==endpoint and e['epochs']==8 and e['independent_lr_horizon']==8 and e['full_dev_images']==ct['observed_images']==1469 and e['full_dev_gt_objects']==ct['gt_objects_captured']==ct['gt_objects_before_inference']==22462 and not e['official_test_accessed'] and e['classification_coefficient']==(0 if a=='N' else .1),ev/'short_evaluation_contract.json')
 roster=b(ev/'development_roster.txt').decode().splitlines();check(a+'_full_unique_roster',roster==ct['roster'] and len(set(roster))==1469 and set(ct['actual_loader_roster'])==set(roster),ev/'development_roster.txt')
 captures=[json.loads(x) for x in gzip.decompress(b(ev/'native_capture/objects.jsonl.gz')).decode().splitlines() if x]
 GT[a]=[{k:x[k] for k in ('image','canvas_shape','original_shape','gt_boxes','gt_classes')} for x in captures]
 check(a+'_actual_GT_capture_population',len(captures)==1469 and sum(len(x['gt_boxes']) for x in captures)==22462 and len({x['image'] for x in captures})==1469,ev/'native_capture/objects.jsonl.gz')
 kd=jl(tr/'kd_batches.jsonl')
 check(a+'_saved_actual_aux_path',all(x['weighted_kd_total']==0 and x['total_loss']==x['native_total'] for x in kd) if a=='N' else all(x['kd_weight']==.1 and x['weighted_kd_total']>0 and math.isfinite(x['total_loss']) for x in kd),tr/'kd_batches.jsonl')
 row=next(x for x in summary['arms'] if x['arm']==a)
 check(a+'_native_fraction_and_analyzer_values',e['metric_units']=='fraction_0_to_1' and all(math.isfinite(e[k]) and 0<=e[k]<=1 and row['raw_fraction'][k]==e[k] and row['display_percent'][k]==100*e[k] for k in metrics) and len(e['per_class'])==5 and {x['class_id'] for x in e['per_class']}==set(range(5)) and all(abs(sum(x[k] for x in e['per_class'])/5-e[k])<1e-12 for k in metrics[:3]),P/'analysis/output_attempt1/summary.json')
 arms[a]=dict(training_seconds=t['seconds'],evaluation_seconds=e['seconds'],epochs=8,batches=512,optimizer_attempts=t['attempts'],successful_updates=t['successful_updates'],amp_skips=t['amp_skips'],ema_updates=t['ema_updates'],selected_occurrences=t['selected_objects'],bn_buffers_changed=t['bn_running_buffers_changed'],raw_fraction={k:e[k] for k in metrics},nvml_peak_mib=list(t['resources']['per_gpu_peak_vram_mib'].values())[0],rss_peak_mib=t['resources']['peak_rss_mib'],full_train_gradient_probe_enabled=False,saved_nonzero_C0_loss_records=len(kd) if a=='C0' else None)
check('both_actual_eval_contracts_and_GT_match',contracts['N']==contracts['C0'] and GT['N']==GT['C0'] and evals['N']['evaluation_identity_projection']==evals['C0']['evaluation_identity_projection'],E/'evaluations')
keys=['batch','reference','tensors','im_file','pair_info','labels','scope']
check('both_train30_saved_metadata_exact',all(all(n[k]==c[k] for k in keys) for n,c in zip(flows['N'],flows['C0'])),E/'runs')
delta={k:100*(evals['C0'][k]-evals['N'][k]) for k in metrics}
check('existing_metric_differences_exact',delta==summary['comparison']['delta_pp'] and summary['primary_direction']=='negative' and summary['status']=='COMPLETE_SUBSET_E8_READOUT' and summary['standard_deviation'] is None and not summary['general_method_ranking_validated'],P/'analysis/output_attempt1/summary.json')
manifest=[dict(path=str(p),bytes=len(raw),unchanged_at_end=p.read_bytes()==raw) for p,raw in seen.items()]
check('inspected_inputs_stable',all(x['unchanged_at_end'] for x in manifest),'FINAL_EXECUTION_INPUTS.json')
verdict='pass' if all(x['status']=='pass' for x in checks) else 'fail'
result=dict(date=datetime.datetime.now(datetime.timezone.utc).isoformat(),auditor=dict(agent='/root/object_transport_review',model_identity='Not exposed; inherited parent settings',cross_model_claim=False),status='PASS_EXECUTED_SUBSET_SCREEN' if verdict=='pass' else 'FAIL_EXECUTION_AUDIT',overall_verdict=verdict,integrity_status=verdict,scope=scope,endpoint=endpoint,eval_type='real_gt',checks=checks,arms=arms,delta_pp=delta,execution_seconds=q['execution_seconds'],admission_wait_seconds=q['admission_wait_seconds'],total_wall_seconds=q['total_wall_seconds'],driver_handoff_seconds=q['execution_seconds']-sum(x['observed_execution_seconds'] for x in q['stage_timings']),stage_resources=resource_summary,new_tests=0,new_GPU=0,new_inference=0,new_metrics=0,new_hash_computed=False,audited_input_hashes=[],claims=[dict(id='actual_E8_endpoints_complete',impact='supported' if verdict=='pass' else 'unsupported',rationale='Both actual512batch/E8/full-dev1469/22462/fixedlastEMA chains and clean6stage terminal records checked.'),dict(id='known_C0_direction_not_retained',impact='supported' if verdict=='pass' else 'unsupported',rationale='C0-N mAP50-95=-1.128104587625961pp in this fixedsubset/seed42/independentE8. This screener version did not preserve the known direction.'),dict(id='C0_ineffective_or_general_rank_validity',impact='unsupported',rationale='Single known-control calibration and lower subset exposure do not establish C0 inefficacy, general ranking, significance or E200 predictivity.')],limits=['Pixel and initial-state tensor equality is supported by actual server torch.equal receipts bound to reviewed code; tensors were not downloaded or compared again. Only first30 pixel prefix is covered.','N/C0 have298/297 successful updates becauseAMP skips differ6/7, although both512batches/304attempts/EMA304; no matched successful-update claim.','Full training disables gradient probes: gradient_checks=[] and nonzero_kd_gradient=false mean unobserved, not zero KD. Actual canary gradient was nonzero and all8saved C0loss rows are positive.','62774 selected occurrences are cumulative, not unique objects or full per-batchmask observation.','No original native AP re-inference/recomputation from predictions, full weight reload, new metric, test or hash. Existing APfractions and declared differences only recomputed.'],input_manifest='FINAL_EXECUTION_INPUTS.json')
(O/'FINAL_EXECUTION_INPUTS.json').write_text(json.dumps(dict(new_hash_computed=False,files=manifest),ensure_ascii=False,indent=2),encoding='utf-8')
(O/'FINAL_EXECUTION_REVIEW.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
md=f'''**{result['status']}。** 两臂实际完成通用预训练初始化、正常BN、固定2048子集的E8/512batch及各一次完整dev1469图/22462GT固定last/EMA评价。当前筛选器未保留已知C0正向对照；不能据此判断C0无效。

| 臂 | mAP50–95（百分数） | 成功更新 / 尝试 | AMP skips / EMA调用 | 训练秒数 | 评价秒数 |
|---|---:|---:|---:|---:|---:|
| N | 27.588046 | 298 / 304 | 6 / 304 | 224.976281 | 18.347238 |
| C0 | 26.459942 | 297 / 304 | 7 / 304 | 214.619190 | 19.024567 |

C0−N = **−1.1281045876 pp**。已核原fraction、五类宏均值及已有分析器单位/差值；无新增指标或AP重推理。两臂并非成功更新数完全相同。

共同499tensor五类初始化、fresh optimizer/EMA、前30批服务器完整双模态uint8像素/双标签等值回执闭合；正常BN有243buffer改变。该像素证据只限前缀，原张量未下载。全训关闭梯度探针，因此gradient_checks为空及nonzero_kd_gradient=false不表示KD为零：真实canary已证明非零梯度，保存的8条C0训练记录均有正KD损失。两臂62774为累计选择次数。

6个实际阶段均clean exit0、monitor_errors为空，无failure/timeout冲突。训练NVML峰6704MiB、RSS17727/17964MiB，处于7168/20480MiB预约内；完整评价NVML1370MiB，RSS4113/4159MiB。原lease显存余量/项目RSS/第四卡条件通过。源码仍与10输入快照逐字节相同，实际source manifests、配置副本、初始化来源及训练/评价checkpoint stat相互闭合；两份actual评价合同、完整roster和capture GT相同。

累计执行 **571.548243秒**，排队 **10.687456秒**，总墙钟 **582.235702秒**，均由实际6阶段回执闭合；执行低于2700秒硬限时。排队不包装成计算时间。

这是固定子集/seed42/独立E8的已知对照校验，未通过方向保留检验；不证明一般方法排序、E200预测性或C0整体无效。未重复CPU测试/流测试，未启动GPU、读取权重、修改源码或计算新hash。详细检查与来源见FINAL_EXECUTION_REVIEW.json和FINAL_EXECUTION_INPUTS.json。
'''
(O/'FINAL_EXECUTION_REVIEW.md').write_text(md,encoding='utf-8')
print(json.dumps(dict(status=result['status'],checks=len(checks),failed=[x for x in checks if x['status']!='pass'],delta_mAP_pp=delta['mAP50_95'],inputs=len(manifest)),ensure_ascii=False,indent=2))
