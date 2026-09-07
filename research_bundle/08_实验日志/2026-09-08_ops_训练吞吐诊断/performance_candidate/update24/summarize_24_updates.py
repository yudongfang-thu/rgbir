"""Read-only CPU summary of collected update24 JSON; no state/GPU rerun.

The accepted compare_24_updates.py remains frozen. This entry checks JSON
closure and reports instrumentation timing without claiming production speed.
"""
import argparse
import json
import math
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def finite(value, positive=False):
    if type(value) not in (int,float) or not math.isfinite(value) or value < 0 or positive and value == 0:
        raise ValueError('Invalid finite timing/count value: '+repr(value))
    return value


def close(a,b):
    return math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-8)


def timing_summary(receipt):
    t=receipt['timing'];rows=t['batches'];n=receipt['result']['batches']
    if [r['batch'] for r in rows] != list(range(1,n+1)):
        raise ValueError('Per-batch timing inventory incomplete')
    warm=[];cold=[]
    for r in rows:
        span=finite(r['wall_seconds'],True);audit=finite(r['audit_seconds'])
        corrected=finite(r['wall_minus_audit_seconds'],True)
        if not close(span-audit,corrected):raise ValueError('Per-batch audit subtraction differs')
        if r['warmup'] is not (r['batch'] <= t['warmup_batches']):raise ValueError('Warmup rule changed')
        (cold if r['warmup'] else warm).append(corrected)
    if not warm or t['post_warmup_batches'] != len(warm) or t['post_warmup_batch_seconds'] != warm:
        raise ValueError('Post-warmup series does not close')
    total=finite(t['training_span_seconds'],True)
    audit=finite(t['training_audit_copy_save_check_seconds'])
    corrected=finite(t['training_span_minus_audit_seconds'],True)
    if not close(total-audit,corrected):raise ValueError('Trainer span audit subtraction differs')
    return dict(batches=n,successful_updates=receipt['result']['updates'],
        training_span_seconds=total,training_audit_seconds=audit,
        training_span_minus_audit_seconds=corrected,training_audit_fraction=audit/total,
        post_warmup_batches=len(warm),post_warmup_batch_median_seconds=statistics.median(warm),
        post_warmup_batch_mean_seconds=statistics.mean(warm),
        post_warmup_batch_min_seconds=min(warm),post_warmup_batch_max_seconds=max(warm),
        post_warmup_batch_sum_seconds=sum(warm),
        observed_batch_audit_seconds=sum(r['audit_seconds'] for r in rows),
        warmup_batch_sum_seconds=sum(cold),gpu_allocated_peak_mib=finite(receipt['gpu_allocated_peak_mib'],True),
        gpu_reserved_peak_mib=finite(receipt['gpu_reserved_peak_mib'],True))


def optional_kd_compare(old,new):
    paths=[p/'kd_batches.jsonl' for p in (old,new)]
    if not all(p.is_file() for p in paths):return dict(status='NOT_COLLECTED',scope='Optional native C1 scalar log comparison')
    rows=[[json.loads(line) for line in p.read_text(encoding='utf-8').splitlines() if line.strip()] for p in paths]
    result=dict(status='COMPARED_FIXED_SCALAR_LOGS',counts=list(map(len,rows)),batch_identity_exact=True,fields={})
    if len(rows[0]) != len(rows[1]):result['batch_identity_exact']=False
    for a,b in zip(*rows):
        result['batch_identity_exact'] &= all(a.get(k)==b.get(k) for k in ('batch','arm','actual_B','coefficient','student_files'))
        for k in ('native_total','loss_unweighted','weighted_kd_total','total_loss'):
            x,y=a[k],b[k]
            if not all(type(z) in (int,float) and math.isfinite(z) for z in (x,y)):
                raise ValueError('Nonfinite actual C1 scalar log')
            r=result['fields'].setdefault(k,dict(records=0,numeric_agreement=True,python_scalar_equal=True,max_abs=0.,outside=0))
            delta=abs(y-x);ok=delta<=1e-6+1e-5*abs(x)
            r['records']+=1;r['numeric_agreement'] &= ok;r['python_scalar_equal'] &= x==y
            r['outside']+=int(not ok);r['max_abs']=max(r['max_abs'],delta)
    return result


def summarize(root):
    old,new=root/'old',root/'block16'
    receipts=[read(p/'worker_receipt.json') for p in (old,new)]
    c=read(root/'comparison'/'comparison.json');parent=read(root/'receipt.json')
    for name,r in zip(('old','block16'),receipts):
        if r['status']!='COMPLETED_24_UPDATE_DIAGNOSTIC' or r['worker']!=name or r['result']['updates']!=24:
            raise ValueError('Missing complete 24-update worker')
        if r['new_hash_computed'] is not False or r['long_training_switch_admitted'] is not False:
            raise ValueError('Worker scope contract changed')
    if receipts[0]['configuration']!=receipts[1]['configuration']:raise ValueError('Worker config mismatch')
    if c['status']!='COMPLETED_COMPARISON_NOT_PRODUCTION_ADMISSION' or c['atol']!=1e-6 or c['rtol']!=1e-5:
        raise ValueError('Comparison status/tolerance changed')
    if c['new_hash_computed'] is not False or c['long_training_switch_admitted'] is not False:
        raise ValueError('Comparison scope contract changed')
    recomputed={}
    for row in c['records']:
        cat=recomputed.setdefault(row['category'],dict(records=0,bitwise_exact=True,numeric_agreement=True,
            all_finite=True,max_abs=0.,finite_outside_tolerance=0,nonfinite_elements=0))
        cat['records']+=1
        for key in ('bitwise_exact','numeric_agreement','all_finite'):cat[key] &= row[key]
        cat['max_abs']=max(cat['max_abs'],row['max_abs'])
        for key in ('finite_outside_tolerance','nonfinite_elements'):cat[key]+=row[key]
    if recomputed!=c['categories']:raise ValueError('Category summary differs from individual state comparisons')
    strict=('initial','first_batch_pixels','flow_worker_rng_gt','selection_identity','control','final_control')
    numeric=('losses','gradients_scaled','student','optimizer','ema','applied_gradients')
    contract=all(c[k] for k in ('file_inventory_exact','source_snapshots_byte_exact','input_model_stats_exact'))
    contract &= all(recomputed.get(k,{}).get('bitwise_exact',False) for k in strict)
    for key,expected in [('initial_flow_selection_and_control_exact',contract),
                         ('trajectory_bitwise_exact',contract and all(recomputed[k]['bitwise_exact'] for k in numeric)),
                         ('trajectory_numeric_agreement',contract and all(recomputed[k]['numeric_agreement'] for k in numeric))]:
        if c[key] is not expected:raise ValueError('Aggregate decision differs: '+key)
        if key in parent and parent[key] is not expected:raise ValueError('Parent decision differs: '+key)
    timings={name:timing_summary(r) for name,r in zip(('old','block16'),receipts)}
    ratio=timings['old']['post_warmup_batch_median_seconds']/timings['block16']['post_warmup_batch_median_seconds']
    return dict(status='COMPLETED_JSON_READOUT_NOT_PRODUCTION_ADMISSION',
        trajectory_bitwise_exact=c['trajectory_bitwise_exact'],trajectory_numeric_agreement=c['trajectory_numeric_agreement'],
        initial_flow_selection_control_exact=contract,
        intermediate_selection_numeric_agreement=recomputed['selection_floating']['numeric_agreement'],
        categories=recomputed,timings=timings,observed_post_warmup_median_old_over_block16=ratio,
        optional_original_c1_scalar_logs=optional_kd_compare(old,new),
        comparison_cpu_seconds=c['comparison_cpu_seconds'],new_hash_computed=False,long_training_switch_admitted=False,
        scope='CPU recomputation of collected JSON closure. Does not reload multi-GB state .pt or independently reproduce GPU trajectory.',
        timing_scope='Synchronized, audit-subtracted short diagnostic batch spans excluding loader fetch; not production epoch speed.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    result=summarize(args.campaign)
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    (args.output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    table=['|路径|成功更新|训练跨度秒|审计秒|扣审计后秒|后热身每批中位秒|','|---|---:|---:|---:|---:|---:|']
    for name,t in result['timings'].items():
        table.append('|{}|{}|{:.3f}|{:.3f}|{:.3f}|{:.4f}|'.format(name,t['successful_updates'],t['training_span_seconds'],t['training_audit_seconds'],t['training_span_minus_audit_seconds'],t['post_warmup_batch_median_seconds']))
    lines=['# 24次更新诊断 JSON 复核',
        '**轨迹逐位一致={}；固定容差一致={}；中间selection固定容差一致={}。**'.format(result['trajectory_bitwise_exact'],result['trajectory_numeric_agreement'],result['intermediate_selection_numeric_agreement']),
        '这三个判断分开报告；比较完成不等于生产切换准入。','\n'.join(table),
        '后热身观测batch中位数 old/block16 = {:.3f}。计时带同步且扣除审计复制/写盘，不含loader fetch，不能称整轮训练加速。'.format(result['observed_post_warmup_median_old_over_block16']),
        '本次CPU仅重算已保存JSON逐项到汇总的闭合；未重读完整state .pt，未运行GPU/SSH。完整梯度/参数精度见summary.json分类字段。',
        '原生kd_batches.jsonl存在时另比实际C1 native/KD/加权KD/total标量；缺失时明确NOT_COLLECTED，不以C0兼容scalar冒充C1损失。']
    (args.output/'README.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
