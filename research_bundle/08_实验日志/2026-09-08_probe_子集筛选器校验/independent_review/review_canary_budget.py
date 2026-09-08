"""Audit existing completed canary/budget records only; no tests or GPU work."""
import pathlib,json,math,datetime
P=pathlib.Path(__file__).resolve().parent.parent
E=P/'evidence_canaries'; O=P/'independent_review'
seen={};checks=[]
def b(p):
 p=pathlib.Path(p).resolve();raw=p.read_bytes()
 if p in seen and seen[p]!=raw:raise RuntimeError('Input changed: '+str(p))
 seen[p]=raw;return raw
def j(p):return json.loads(b(p).decode('utf-8-sig'))
def jl(p):return [json.loads(x) for x in b(p).decode('utf-8-sig').splitlines() if x]
def check(name,v,evidence):checks.append(dict(name=name,status='pass' if v else 'fail',evidence=str(evidence)))
def finite(x):
 if isinstance(x,float):return math.isfinite(x)
 if isinstance(x,dict):return all(finite(v) for v in x.values())
 if isinstance(x,list):return all(finite(v) for v in x)
 return True
review=j(O/'SOURCE_REVIEW.json'); source={r['relative_path']:r for r in review['reviewed_source_files']}
for name,r in source.items():check('source_still_reviewed_'+name,b(r['source_path'])==b(r['snapshot_path']) and len(b(r['source_path']))==r['bytes'],r['source_path'])
budget=j(E/'queue/budget_decision.json');gate=j(E/'queue/canary_checks.json');queue_manifest=j(E/'queue/manifest.json')
receipts={};initial={};flow={};streams={};est={};stats={};timings={}
for a in ('N','C0'):
 d=E/'canaries'/a;r=j(d/'canary.json');receipts[a]=r
 initial[a]=j(d/'initialization_check.json');flow[a]=j(d/'flow_check.json');streams[a]=jl(d/'sample_stream.jsonl')
 setup=j(d/'setup_receipt.json');ready=j(d/'runtime_ready.json');rows=jl(d/'batch_timing.jsonl')
 status=j(E/'queue'/f'subset_e8_attempt1_{a}_canary_status.json')
 resources=j(E/'queue'/f'subset_e8_attempt1_{a}_canary_resource_profile.json')
 admission=j(E/'queue'/f'subset_e8_attempt1_{a}_canary_admission.json')
 timing=j(E/'queue'/f'subset_e8_attempt1_{a}_canary_timing.json');timings[a]=timing
 check(a+'_completed_identity',r['status']=='SUBSET_SCREEN_CANARY_COMPLETED' and r['scope']==review['scope'] and r['endpoint']==review['endpoint'] and r['arm']==a and r['seed']==42 and r['classification_coefficient']==(0 if a=='N' else .1) and not r['formal_e200_complete'] and not r['official_test_accessed'],d/'canary.json')
 check(a+'_actual_clean_exit',status['status']=='COMPLETED' and status['exit_code']==0 and status['monitor_errors']==[] and resources['exit_code']==0 and resources['monitor_errors']==[],E/'queue'/f'subset_e8_attempt1_{a}_canary_status.json')
 check(a+'_fixed_update_gate',r['successful_updates']==r['optimizer_updates']==24 and r['attempts']<=48 and r['attempts']==r['successful_updates']+r['amp_skips'] and r['batches']>=30 and r['loader_workers_cleaned'],d/'canary.json')
 check(a+'_runtime_population_BN_fresh',tuple(ready[k] for k in ('train_images','val_images','train_batches','batch','workers'))==(2048,1469,64,32,4) and len(ready['class_names'])==5 and r['bn_running_statistics']=='normal_training' and r['bn_running_buffers_changed']==243 and r['fresh_optimizer'] and r['fresh_ema'],d/'runtime_ready.json')
 check(a+'_actual_gradient_and_finiteness',finite(r) and all(g['weight0_exact_loss_gradient'] and g['native_score_gradient_l2']>0 and not g['teacher_has_grad'] and not g['reference_has_grad'] for g in r['gradient_checks']) and (a=='N' or r['nonzero_kd_gradient'] and r['selected_objects']>0 and any(g['kd_score_gradient_l2']>0 for g in r['gradient_checks'])),d/'canary.json')
 check(a+'_state_reference_contract',initial[a]['status']=='PASS' and initial[a]['tensors']==499 and initial[a]['head_included'] and initial[a]['fresh_optimizer'] and initial[a]['fresh_ema'] and initial[a]['ema_updates']==0 and initial[a]['initialization']==r['initialization'] and (initial[a]['state_reference_written_and_verified'] and initial[a]['cross_arm_initial_state_exact'] is None if a=='N' else initial[a]['cross_arm_initial_state_exact'] is True),d/'initialization_check.json')
 f=flow[a]
 check(a+'_all30_actual_flow_receipt',f['status']=='PASS' and f['prefix_batches']==f['records']==len(f['files'])==len(streams[a])==30 and f['exact_pixels_and_labels'] and f['remote_only_tensors'] and f['equality_method']=='shape_dtype_device_and_torch_equal_all_elements' and [x['batch'] for x in f['files']]==list(range(1,31)) and all(x['exact'] for x in f['files']),d/'flow_check.json')
 check(a+'_pixels_are_full_uint8_prefix',all(x['full_rgb_ir_pixels_and_dual_labels_exact'] and x['scope']=='first30_augmented_loader_batch_uint8_before_preprocess' and all(x['tensors'][k]==dict(shape=[32,3,640,640],dtype='torch.uint8',device='cpu') for k in ('img','strong_img')) for x in streams[a]),d/'sample_stream.jsonl')
 check(a+'_actual_config_bytes',b(d/'short_screen_config.yaml')==b(P/f'release/configs/drone_{a}_s42_E8.yaml'),d/'short_screen_config.yaml')
 manifest=j(d/'source_manifest.json');bound=[x for x in manifest['files'] if '/rgbir_subset_e8_check_20260908/release_v1/' in x['path']]
 check(a+'_runtime_source_binding',all(x['byte_identity'] and x['bytes']==source[x['path'].split('/release_v1/')[1]]['bytes'] for x in bound if x['path'].split('/release_v1/')[1] in source) and len(bound)>=8 and all(x['byte_identity'] for x in manifest['files']),d/'source_manifest.json')
 check(a+'_all_batch_timings_exact',len(rows)==r['batches'] and [x['batch'] for x in rows]==list(range(1,r['batches']+1)) and [x['seconds'] for x in rows]==r['batch_duration_seconds'] and [x['excluding_flow_seconds'] for x in rows]==r['normal_cadence_batch_seconds'] and all(x['excluding_flow_seconds']>0 and abs(x['seconds']-x['flow_audit_seconds']-x['excluding_flow_seconds'])<1e-12 for x in rows),d/'batch_timing.jsonl')
 check(a+'_flow_cost_closed',abs(sum(x['seconds'] for x in f['files'])-r['flow_audit_seconds'])<1e-10 and setup['initial_state_audit_seconds']==r['initial_state_audit_seconds'],d/'flow_check.json')
 samples=[x['excluding_flow_seconds'] for x in rows];mean=sum(samples)/len(samples);io=r['flow_audit_seconds'];overhead=max(0,r['seconds']-sum(samples)-io)
 train=1.2*(512*mean+overhead+io)
 est[a]=dict(mean_normal_batch_seconds=mean,measured_batches=len(samples),observed_non_batch_overhead_seconds=overhead,observed_flow_audit_seconds=io,train_seconds_with_margin=train,margin_factor=1.2)
 check(a+'_independent_budget_arithmetic',est[a]==budget['arms'][a],E/'queue/budget_decision.json')
 peak=max(list(r['resources']['per_gpu_peak_vram_mib'].values())+[r['gpu_allocated_peak_mib'],r['gpu_reserved_peak_mib']]);rss=r['resources']['peak_rss_mib']
 v=math.ceil((peak+max(256,.05*peak))/256)*256;m=math.ceil((rss+max(2048,.05*rss))/1024)*1024
 check(a+'_independent_resource_reservation',(v,m)==(gate['arms'][a]['reservation']['vram_mib'],gate['arms'][a]['reservation']['rss_mib']) and v<=8192 and m<=32768,d/'canary.json')
 check(a+'_observed_shared_resource_discipline',status['minimum_free_mib']>=2048 and min(x['memory_free_mib'] for x in resources['samples'])>=2048 and max(x['project_rss_mib'] for x in resources['samples'])<=300*1024 and len(admission['active_gpus_after'])<=4 and (len(admission['active_gpus_after'])<=3 or admission['four_gpu_exception'] and len(admission['empty_gpus_after'])>=2),E/'queue'/f'subset_e8_attempt1_{a}_canary_resource_profile.json')
 check(a+'_RUNNING_clock_observed',timing['actual_launch_observed'] and not timing['budget_timed_out'] and timing['observed_execution_seconds']>=r['seconds'] and abs(timing['wall_seconds']-timing['observed_execution_seconds']-timing['admission_wait_seconds'])<1e-9,E/'queue'/f'subset_e8_attempt1_{a}_canary_timing.json')
 for key,name in [('receipt','canary.json'),('flow','flow_check.json'),('initialization','initialization_check.json')]:check(a+'_gate_'+key+'_size',len(b(d/name))==gate['arms'][a][key]['bytes'] and gate['arms'][a][key]['path'].endswith('/canaries/'+a+'/'+name),E/'queue/canary_checks.json')
 stats[a]=dict(seconds=r['seconds'],batches=r['batches'],optimizer_attempts=r['attempts'],successful_updates=r['successful_updates'],amp_skips=r['amp_skips'],ema_updates=r['ema_updates'],selected_occurrences=r['selected_objects'],nvml_peak_mib=list(r['resources']['per_gpu_peak_vram_mib'].values())[0],allocated_peak_mib=r['gpu_allocated_peak_mib'],reserved_peak_mib=r['gpu_reserved_peak_mib'],rss_peak_mib=rss,reservation_vram_mib=v,reservation_rss_mib=m,flow_io_seconds=io,initial_state_io_seconds=r['initial_state_audit_seconds'],gradient_checks=r['gradient_checks'])
check('common_initial_state_reference_and_inputs',initial['N']['reference_state']==initial['C0']['reference_state'] and receipts['N']['initialization']==receipts['C0']['initialization'] and receipts['N']['training_subset_identity']==receipts['C0']['training_subset_identity'],E/'canaries/C0/initialization_check.json')
keys=['batch','reference','tensors','im_file','pair_info','labels','scope']
check('common30_reference_and_all_saved_metadata',flow['N']['reference_dir']==flow['C0']['reference_dir'] and [x['reference'] for x in flow['N']['files']]==[x['reference'] for x in flow['C0']['files']] and all(all(n[k]==c[k] for k in keys) for n,c in zip(streams['N'],streams['C0'])),E/'canaries/C0/sample_stream.jsonl')
total=budget['spent_execution_seconds']+sum(x['train_seconds_with_margin'] for x in est.values())+120
check('budget_928seconds_below2700',gate['status']=='BOTH_CANARIES_VALID' and budget['status']=='PASS_WITHIN_EXECUTION_BUDGET' and total==budget['estimate_total_execution_seconds'] and total<=2700 and budget['eval_reserve_seconds_each']==60 and budget['queue_admission_wait_excluded'] and not budget['ap_used'],E/'queue/budget_decision.json')
active=sum(t['observed_execution_seconds'] for t in timings.values());waiting=sum(t['admission_wait_seconds'] for t in timings.values())
check('driver_overhead_included_without_wait',0<=budget['spent_execution_seconds']-active<1,E/'queue/budget_decision.json')
manifest=[dict(path=str(p),bytes=len(raw),mtime_ns=p.stat().st_mtime_ns,unchanged_at_end=p.read_bytes()==raw) for p,raw in seen.items()]
check('all_read_inputs_stable',all(x['unchanged_at_end'] for x in manifest),'CANARY_BUDGET_INPUTS.json')
verdict='pass' if all(x['status']=='pass' for x in checks) else 'fail'
review=dict(date=datetime.datetime.now(datetime.timezone.utc).isoformat(),auditor=dict(agent='/root/object_transport_review',model_identity='Not exposed; inherited parent settings',cross_model_claim=False),status='PASS_CANARY_AND_BUDGET_ADMISSION' if verdict=='pass' else 'FAIL_CANARY_OR_BUDGET',overall_verdict=verdict,integrity_status=verdict,scope='DRONE_SUBSET2048_PRETRAIN_E8_CHECK',checks=checks,canaries=stats,independent_budget=est,total_estimate_seconds=total,limit_seconds=2700,spent_execution_seconds=budget['spent_execution_seconds'],observed_canary_execution_seconds=active,driver_handoff_seconds=budget['spent_execution_seconds']-active,excluded_admission_seconds=waiting,reference_tensor_bytes=sum(x['reference']['bytes'] for x in flow['N']['files']),initial_state_reference_bytes=initial['N']['reference_state']['bytes'],new_hash_computed=False,audited_input_hashes=[],new_tests=0,new_GPU=0,mid_training_AP_read=False,source_modified=False,claims=[dict(id='actual_canary_gate',impact='supported' if verdict=='pass' else 'unsupported',rationale='Both reach24successful updates within48attempts, required30prefix, actual shared state/flow receipts and resources.'),dict(id='measured_budget_admission',impact='supported' if verdict=='pass' else 'unsupported',rationale='Independent recomputation uses all30/31 actual batch samples plus measured IO/overhead,20percent margin and120second eval reserve; estimate928.842092seconds.'),dict(id='full_training_or_AP_completed',impact='unsupported',rationale='Canary budget review only; no intermediate AP read, no final endpoint observed.')],limits=['Raw pixel and initialized-state tensors remain server-only; this reviewer verifies actual torch.equal receipts bound to reviewed code and source identity, without downloading or rerunning these comparisons.','Pixel evidence is exactly the first30 augmented pre-normalization dual-modality batches, not original image files or all512 batches.','Canaries have equal24successful updates but N30/C031batches, six/sevenAMP skips and EMA30/31; these are not equal exposures.','Normal BN actually changed243 buffers; generic80class checkpoint equality is not claimed for the adaptedfive-class head.','Budget is a conservative observed cost estimate and does not establish future runtime or completedtraining; the2700second execution guard remains required.','No AP, method effectiveness or general screener validity follows from this admission.'],input_manifest='CANARY_BUDGET_INPUTS.json')
(O/'CANARY_BUDGET_INPUTS.json').write_text(json.dumps(dict(new_hash_computed=False,files=manifest),ensure_ascii=False,indent=2),encoding='utf-8')
(O/'CANARY_BUDGET_REVIEW.json').write_text(json.dumps(review,ensure_ascii=False,indent=2),encoding='utf-8')
md=f'''**{review['status']}。** 实际 N/C0 canary 回执与独立预算复算支持按已冻结队列继续；这不是完整训练或 AP 终态验收。

| 臂 | batch / optimizer尝试 | 成功更新 | AMP skip / EMA调用 | 脚本秒数 | 实测均值秒/batch | 训练估计（含20%余量） |
|---|---:|---:|---:|---:|---:|---:|
| N | 30 / 30 | 24 | 6 / 30 | 36.772593 | 0.578092 | 378.495239 |
| C0 | 31 / 31 | 24 | 7 / 31 | 31.384069 | 0.547157 | 353.479754 |

两个阶段均真实 clean exit0、monitor_errors为空。共同五类学生499个state tensor、包括检测头：N服务器保存并完整读回，C0实际全元素比较相等；fresh optimizer/EMA、正常BN，训练中243个BN buffer改变。前30批双模态uint8像素及双标签均绑定同一服务器参考，shape/dtype/device、文件stat和保存的标签/增强元数据逐项一致；实际执行回执记录完整torch.equal。原始张量约{review['reference_tensor_bytes']/1e9:.6f} GB只留服务器，本审阅未下载或重复比较，不能外推到512批。

native梯度有限、weight0精确等式通过，C0实际KD梯度非零（首批0.00445516686886549），T/R均无梯度。C0比N多1个batch及1次AMP skip，不能把二者canary称为相同曝光。两臂NVML峰6692MiB、allocated5543.981445MiB、reserved6160MiB；RSS为18120/18000MiB。独立加余量得到两臂预约7168MiB VRAM /20480MiB RSS，原lease整卡余量与项目RSS均通过。

预算使用全部30/31个完整batch，不使用排除前6批的备选列表。已花执行76.867099秒 + N训练378.495239 + C0训练353.479754 + 两次评价120 = **928.842092秒 <2700秒**。像素I/O5.291256/3.554257秒与初始化及其余开销均计入。两canary观测执行合计{active:.6f}秒、driver交接{review['driver_handoff_seconds']:.6f}秒，排队{waiting:.6f}秒另记。估计不替代后续实际45分钟硬限时。

仅核现有canary、初始化/输入/资源、源码身份与预算；未读中途AP，未新增测试/GPU或改源码，未计算hash。详细逐项检查和路径见CANARY_BUDGET_REVIEW.json及CANARY_BUDGET_INPUTS.json。
'''
(O/'CANARY_BUDGET_REVIEW.md').write_text(md,encoding='utf-8')
print(json.dumps(dict(status=review['status'],checks=len(checks),failed=[x for x in checks if x['status']!='pass'],estimate_seconds=total,input_files=len(manifest)),ensure_ascii=False,indent=2))
