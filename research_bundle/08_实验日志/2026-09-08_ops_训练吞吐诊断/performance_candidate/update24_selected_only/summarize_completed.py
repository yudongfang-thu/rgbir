"""Read-only completed selected-only 24-update JSON/JSONL readout. No torch/GPU."""
import argparse
import json
import math
from pathlib import Path
import statistics
import struct

ATOL, RTOL, COEFFICIENT = 1e-6, 1e-5, 0.09227393550836771
STATUS = 'COMPLETED_COMPARISON_NOT_PRODUCTION_ADMISSION'
LOSS_FIELDS = ('native_total','loss_unweighted','weighted_kd_total','total_loss',
               'target_loss_unweighted','off_target_loss_unit','off_target_loss_unweighted')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def require(condition, reason):
    if not condition:
        raise AssertionError(reason)


def distribution(values):
    require(bool(values) and all(math.isfinite(v) for v in values),'Finite nonempty timing sample required')
    return dict(n=len(values),sum=sum(values),mean=statistics.mean(values),median=statistics.median(values),
                minimum=min(values),maximum=max(values),values=values)


def json_difference(old,new):
    result=dict(exact=True,numeric_agreement=True,all_finite=True,max_abs=0.,
                finite_outside_tolerance=0,leaves=0,mismatches=[])
    def fail(path,reason,numeric=True):
        result['exact']=False
        if numeric:result['numeric_agreement']=False
        if len(result['mismatches'])<20:result['mismatches'].append(dict(path=path,reason=reason))
    def visit(a,b,path):
        if type(a) is not type(b):fail(path,'type differs');return
        if isinstance(a,dict):
            if set(a)!=set(b):fail(path,'keys differ')
            for key in a.keys() & b.keys():visit(a[key],b[key],path+'.'+key)
        elif isinstance(a,list):
            if len(a)!=len(b):fail(path,'length differs')
            for i,(aa,bb) in enumerate(zip(a,b)):visit(aa,bb,path+'[%d]'%i)
        elif isinstance(a,float):
            result['leaves']+=1
            if not (math.isfinite(a) and math.isfinite(b)):
                result['all_finite']=False;fail(path,'nonfinite JSON number');return
            delta=abs(b-a);result['max_abs']=max(result['max_abs'],delta)
            if delta>ATOL+RTOL*abs(a):
                result['finite_outside_tolerance']+=1;fail(path,'outside original fixed tolerance')
            if struct.pack('d',a)!=struct.pack('d',b):fail(path,'JSON float bytes differ',False)
        else:
            result['leaves']+=1
            if a!=b:fail(path,'value differs')
    visit(old,new,'root')
    return result


def worker_readout(path,expected_worker):
    receipt=read_json(path/'worker_receipt.json')
    require(receipt['status']=='COMPLETED_24_UPDATE_DIAGNOSTIC','Worker incomplete')
    require(receipt['worker']==expected_worker,'Wrong worker identity')
    for field in ('new_hash_computed','long_training_switch_admitted','official_test_accessed'):
        require(receipt[field] is False,'Invalid worker scope: '+field)
    cfg,r,coverage,timing=(receipt[k] for k in ('configuration','result','candidate_coverage','timing'))
    require(cfg['arm']=='C1' and cfg['seed']==42 and cfg['epochs']==200 and cfg['batch']==32
            and cfg['amp'] is True and cfg['classification_coefficient']==COEFFICIENT,'Frozen C1 recipe mismatch')
    require(r['updates']==r['optimizer_calls']==24 and 24<=r['attempts']<=96
            and r['amp_skips']==r['attempts']-24 and r['selected_calls']==r['batches']
            and r['frozen_auxiliaries_unchanged'] is True,'Contradictory update/flow counters')
    if expected_worker=='selected_only':
        require(coverage['thin_learning_batches']==coverage['full_diagnostics_batches']==r['batches']
                and coverage['fallback_batches']==0,'Actual thin learning coverage missing')
    batches=timing['batches']
    require(len(batches)==r['batches'] and [b['batch'] for b in batches]==list(range(1,r['batches']+1)),
            'Timing rows do not close against actual batches')
    require(timing['warmup_batches']==6,'Frozen timing warmup changed')
    for row in batches:
        require(row['warmup'] is (row['batch']<=6),'Warmup identity differs')
        require(math.isclose(row['wall_seconds']-row['audit_seconds'],row['wall_minus_audit_seconds'],rel_tol=1e-12,abs_tol=1e-9),
                'Audit timing arithmetic mismatch')
        require(math.isclose(row['wall_minus_audit_seconds']-row['extra_full_diagnostics_seconds'],
            row['wall_minus_audit_and_extra_diagnostics_seconds'],rel_tol=1e-12,abs_tol=1e-9),'Diagnostic timing arithmetic mismatch')
    warm=[row for row in batches if not row['warmup']]
    require(len(warm)==timing['post_warmup_batches'],'Post-warmup count mismatch')
    require([r['wall_minus_audit_seconds'] for r in warm]==timing['post_warmup_batch_seconds'],
            'Stored audit-subtracted samples differ')
    require([r['wall_minus_audit_and_extra_diagnostics_seconds'] for r in warm]==timing['post_warmup_minus_audit_and_extra_diagnostics_seconds'],
            'Stored diagnostic-subtracted samples differ')
    require(math.isclose(sum(row['extra_full_diagnostics_seconds'] for row in batches),
                        coverage['extra_full_diagnostics_seconds'],rel_tol=1e-12,abs_tol=1e-8),'Full diagnostic time does not close')
    rows=read_jsonl(path/'kd_batches.jsonl')
    require([row['batch'] for row in rows]==list(range(1,r['batches']+1)),'Sanity KD log is incomplete')
    for row in rows:
        require(row['actual_B']==32 and row['coefficient']==COEFFICIENT,'Actual logged dose mismatch')
    summary=dict(worker=expected_worker,result=r,candidate_coverage=coverage,
        actual_classification_coefficient=COEFFICIENT,resources=receipt['resources'],
        gpu_allocated_peak_mib=receipt['gpu_allocated_peak_mib'],gpu_reserved_peak_mib=receipt['gpu_reserved_peak_mib'],
        timing=dict(training_span_seconds=timing['training_span_seconds'],
            training_audit_copy_save_check_seconds=timing['training_audit_copy_save_check_seconds'],
            training_span_minus_audit_seconds=timing['training_span_minus_audit_seconds'],
            all_audit_including_final_check_seconds=timing['all_audit_including_final_check_seconds'],
            extra_full_diagnostics_seconds=coverage['extra_full_diagnostics_seconds'],
            training_span_minus_audit_and_extra_diagnostics_seconds=timing['training_span_minus_audit_seconds']-coverage['extra_full_diagnostics_seconds'],
            warmup_batches=6,post_warmup_batches=len(warm),
            post_warmup={field:distribution([row[field] for row in warm]) for field in
                ('wall_seconds','audit_seconds','extra_full_diagnostics_seconds','wall_minus_audit_seconds',
                 'wall_minus_audit_and_extra_diagnostics_seconds')},
            all_batch_rows=batches,production_epoch_throughput=False))
    return receipt,summary,rows


def summarize(run_dir):
    run_dir=Path(run_dir).resolve()
    top=read_json(run_dir/'receipt.json')
    require(top['status']==STATUS and top['fresh_sequential_workers'] is True
            and top['coordinator_cuda_initialized'] is False,'Comparison coordinator has not completed')
    require(top['new_hash_computed'] is False and top['long_training_switch_admitted'] is False,'Invalid coordinator scope')
    comparison=read_json(run_dir/'comparison'/'comparison.json')
    require(comparison['status']==STATUS and comparison['atol']==ATOL and comparison['rtol']==RTOL,'Missing complete fixed-tolerance comparison')
    require(comparison['new_hash_computed'] is False and comparison['long_training_switch_admitted'] is False,'Invalid comparison scope')
    require(comparison['compared_delta_scope']=='selected_S_T_only','Wrong delta comparison scope')
    workers=[worker_readout(run_dir/name,name) for name in ('old','selected_only')]
    require(workers[0][0]['configuration']==workers[1][0]['configuration'],'Worker recipes differ')
    require(workers[1][1]['candidate_coverage']==comparison['candidate_coverage'],'Candidate coverage differs across receipts')
    for key in ('trajectory_bitwise_exact','trajectory_numeric_agreement'):
        require(top[key] is comparison[key],'Coordinator comparison flag differs: '+key)
    # Independently close the published category summary against per-file
    # comparisons. This is a JSON audit, not a re-read of GPU .pt state arrays.
    aggregate={}
    for row in comparison['records']:
        cat=aggregate.setdefault(row['category'],dict(records=0,bitwise_exact=True,numeric_agreement=True,
            all_finite=True,max_abs=0.,finite_outside_tolerance=0,nonfinite_elements=0))
        cat['records']+=1
        for key in ('bitwise_exact','numeric_agreement','all_finite'):cat[key]&=row[key]
        cat['max_abs']=max(cat['max_abs'],row['max_abs'])
        for key in ('finite_outside_tolerance','nonfinite_elements'):cat[key]+=row[key]
    require(aggregate==comparison['categories'],'Published categories do not match per-file records')
    required=('initial','first_batch_pixels','flow_worker_rng_gt','selection_identity','selection_floating',
        'per_batch_losses','losses','gradients_scaled','applied_gradients','student','optimizer','ema','control','final_control')
    require(all(key in aggregate and aggregate[key]['records']>0 for key in required),'Required trajectory category missing')
    losses={field:json_difference([r[field] for r in workers[0][2]],[r[field] for r in workers[1][2]]) for field in LOSS_FIELDS}
    shared=json_difference(read_jsonl(run_dir/'old'/'shared_gradient_observations.jsonl'),
                           read_jsonl(run_dir/'selected_only'/'shared_gradient_observations.jsonl'))
    sanity=json_difference(read_jsonl(run_dir/'old'/'gradient_checks.jsonl'),
                           read_jsonl(run_dir/'selected_only'/'gradient_checks.jsonl'))
    ratios={field:workers[0][1]['timing']['post_warmup'][field]['median']/workers[1][1]['timing']['post_warmup'][field]['median']
        for field in ('wall_seconds','wall_minus_audit_seconds','wall_minus_audit_and_extra_diagnostics_seconds')}
    return dict(status='COMPLETED_JSON_READOUT_NOT_PRODUCTION_ADMISSION',input_run=str(run_dir),
        inputs=[str(run_dir/'receipt.json'),str(run_dir/'comparison'/'comparison.json')]
               +[str(run_dir/name/file) for name in ('old','selected_only') for file in
                 ('worker_receipt.json','kd_batches.jsonl','shared_gradient_observations.jsonl','gradient_checks.jsonl')],
        trajectory_bitwise_exact=comparison['trajectory_bitwise_exact'],
        trajectory_numeric_agreement=comparison['trajectory_numeric_agreement'],
        initial_flow_selection_control_exact=comparison['initial_flow_selection_and_control_exact'],
        selected_delta_quality_c0=comparison['categories']['selection_floating'],
        categories=aggregate,workers=[worker[1] for worker in workers],per_batch_loss_fields=losses,
        shared_gradient_observation_json=shared,sanity_gradient_checks_json=sanity,
        observed_post_warmup_old_over_new_median=ratios,
        uncollected_learning_delta_fields=comparison['uncollected_learning_delta_fields'],
        raw_state_arrays_reopened=False,category_json_reaggregation_exact=True,
        new_hash_computed=False,long_training_switch_admitted=False,production_epoch_throughput=False,
        limits=['This reads completed JSON/JSONL; full tensor byte/tolerance comparisons are from the executed analyzer.',
                'Student state includes parameters and buffers; the student max_abs is not a parameter-only maximum.',
                'Only the first batch has independently compared saved pixels; later flow exact means source/GT/metadata/worker RNG.',
                'Shared-gradient JSON summarizes the fixed observer, not all parameter gradients; full gradient tensors use trajectory categories.',
                'Matching AMP nonfinite bits are not finite usable gradients.',
                'Full wall/audit/full diagnostics are reported separately; subtraction changes cache/overlap and is not production epoch throughput.',
                'Same formula or raw-bundle equality does not imply update trajectory equality; 24 updates does not imply E200 identity.'])


def write_report(summary,output):
    old,new=summary['workers']
    exact=summary['trajectory_bitwise_exact'];numeric=summary['trajectory_numeric_agreement']
    lines=['# Selected-only：24 次更新结果读出','',
        '**本次训练轨迹：字节精确=%s；原容差通过=%s。** 这是已完成 JSON/JSONL 的只读汇总，不是正式长训准入。'%(exact,numeric),'',
        '| 路径 | 实际批次 | 成功更新 | 尝试 | AMP 跳步 | thin / full diagnostics / fallback |',
        '|---|---:|---:|---:|---:|---|']
    for worker in (old,new):
        r,c=worker['result'],worker['candidate_coverage']
        lines.append('| %s | %d | %d | %d | %d | %d / %d / %d |'%(worker['worker'],r['batches'],r['updates'],r['attempts'],r['amp_skips'],c['thin_learning_batches'],c['full_diagnostics_batches'],c['fallback_batches']))
    lines += ['', '实际 λ=`%.17g`；原初始化、E200 配方、B32/workers4/AMP。原生损失与 KD 组件按同一容差分别核对。'%COEFFICIENT,'',
        '| 比较项 | bitwise exact | 原容差 | max abs |', '|---|---|---|---:|']
    for key in ('initial','first_batch_pixels','flow_worker_rng_gt','selection_identity','selection_floating',
                'per_batch_losses','gradients_scaled','applied_gradients','student','optimizer','ema','control'):
        row=summary['categories'][key]
        lines.append('| %s | %s | %s | %.9g |'%(key,row['bitwise_exact'],row['numeric_agreement'],row['max_abs']))
    lines += ['', 'student 包括参数和 buffers；scaled gradient 的 AMP 非有限模式另列，不能称为有限可用梯度。学习 delta 仅比较 selected S/T；未选与 R delta 未采集到学习快照。','',
        '| 逐批 loss / 组件 | JSON exact | 原容差 | max abs |','|---|---|---|---:|']
    for key,row in summary['per_batch_loss_fields'].items():
        lines.append('| %s | %s | %s | %.9g |'%(key,row['exact'],row['numeric_agreement'],row['max_abs']))
    lines += ['', '| 时间（秒） | old | selected-only |','|---|---:|---:|']
    for key in ('training_span_seconds','training_audit_copy_save_check_seconds','extra_full_diagnostics_seconds',
                'training_span_minus_audit_seconds','training_span_minus_audit_and_extra_diagnostics_seconds'):
        lines.append('| %s | %.6f | %.6f |'%(key,old['timing'][key],new['timing'][key]))
    for field in ('wall_seconds','wall_minus_audit_seconds','wall_minus_audit_and_extra_diagnostics_seconds'):
        lines.append('| 去前 6 批后 median：%s | %.6f | %.6f |'%(field,old['timing']['post_warmup'][field]['median'],new['timing']['post_warmup'][field]['median']))
    lines += ['', '完整 wall、审计、额外完整诊断均保留；每个热身后样本在 summary.json。回调时间不含 loader fetch，扣除诊断会受到缓存与重叠影响，**不能当作正式 epoch 吞吐**。','',
        '首批像素与后续 source/GT/增强元数据/worker RNG 的证据范围分开；没有重新打开大规模 state .pt。共享梯度观察 JSON exact=%s、容差=%s；sanity JSON exact=%s。'%(summary['shared_gradient_observation_json']['exact'],summary['shared_gradient_observation_json']['numeric_agreement'],summary['sanity_gradient_checks_json']['exact']),'',
        '同公式或同 raw 通过不推导训练轨迹通过；本次结果不覆盖旧 block16 的失败，也不外推 E200 或多 seed 增益。','',
        '输入：`%s`。输入路径逐项保留于 summary.json；原始结果未改，无新增 hash。'%summary['input_run']]
    (output/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    require(not args.output.exists(),'Output must be a new directory')
    summary=summarize(args.run_directory)  # Refuse partial inputs before writing.
    args.output.mkdir(parents=True)
    (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    write_report(summary,args.output)
    print(json.dumps({k:summary[k] for k in ('status','trajectory_bitwise_exact','trajectory_numeric_agreement')},ensure_ascii=False))


if __name__=='__main__':main()
