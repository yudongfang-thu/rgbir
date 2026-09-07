# coding: utf-8
"""CPU-only timing arithmetic from completed receipts; never read CSV AP or weights."""
from pathlib import Path
from decimal import Decimal
from datetime import datetime, timezone, timedelta
import argparse
import csv
import io
import json
import math
import statistics

ENDPOINTS = {'N': 'endpoint_N_20260908_043812', 'C0': 'endpoint_C0_20260908_053913', 'C1': 'endpoint_C1_20260908_063705'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.root.resolve()
    provenance = []

    def read(path, as_json=True):
        data = path.read_bytes()
        st = path.stat()
        provenance.append({'path': str(path.relative_to(root)), 'bytes': len(data), 'mtime_ns': st.st_mtime_ns})
        return json.loads(data.decode('utf-8-sig')) if as_json else data.decode('utf-8-sig')

    snapshot = read(root / 'heartbeat_20260908_063654/snapshot.json')
    assert snapshot['queue_failure'] is None
    completion = read(root / 'three_arm_summary_20260908/queue_completion.json')
    assert completion['status'] == 'SHORT_SCREEN_MATRIX_COMPLETED'
    expected_order = [(arm, stage) for arm in ENDPOINTS for stage in ('train', 'eval')]
    assert [(x['arm'], x['stage']) for x in completion['completed']] == expected_order
    freeze = read(root / 'launch_evidence_attempt1/freeze_receipt.json')
    launch = read(root / 'launch_evidence_attempt1/short_N_s42_E8_train_status.json')
    assert launch['status'] == 'RUNNING'
    rows = {}
    phases = []
    for arm, folder in ENDPOINTS.items():
        base = root / folder
        train = read(base / 'runs' / arm / 'short_training_receipt.json')
        evaluation = read(base / 'evaluations' / arm / 'short_evaluation_receipt.json')
        assert train['status'] == 'SHORT_SCREEN_TRAINING_COMPLETED'
        assert evaluation['status'] == 'SHORT_SCREEN_EVALUATION_COMPLETED'
        assert train['arm'] == evaluation['arm'] == arm and train['seed'] == evaluation['seed'] == 42
        assert train['last_epoch'] == train['epochs_configured'] == evaluation['epochs'] == 8
        assert train['batches'] == 4504 and train['optimizer_updates'] + train['amp_skips'] == train['attempts']
        assert train['seconds'] == snapshot['arms'][arm]['training']['seconds']
        assert evaluation['seconds'] == snapshot['arms'][arm]['evaluation']['seconds']
        assert snapshot['arms'][arm]['training_failure'] is None and snapshot['arms'][arm]['evaluation_failure'] is None
        content = list(csv.reader(io.StringIO(read(base / 'runs' / arm / 'results.csv', False))))
        assert content[0][:2] == ['epoch', 'time'] and [int(x[0]) for x in content[1:]] == list(range(1, 9))
        cumulative = [Decimal(x[1]) for x in content[1:]]
        duration = [t - previous for t, previous in zip(cumulative, [Decimal(0)] + cumulative[:-1])]
        assert all(x > 0 for x in duration) and sum(duration) == cumulative[-1]
        values = [float(x) for x in duration]
        assert train['seconds'] > float(cumulative[-1])
        rows[arm] = {'training_seconds': train['seconds'], 'training_minutes': train['seconds'] / 60,
                     'evaluation_seconds': evaluation['seconds'], 'epochs': 8, 'batches': train['batches'],
                     'optimizer_updates': train['optimizer_updates'], 'amp_skips': train['amp_skips'], 'ema_updates': train['ema_updates'],
                     'csv_header_columns': len(content[0]), 'csv_row_columns': [len(x) for x in content[1:]],
                     'csv_cumulative_seconds': [float(x) for x in cumulative], 'csv_epoch_seconds': values,
                     'csv_epoch_min_seconds': min(values), 'csv_epoch_max_seconds': max(values),
                     'csv_epoch_median_seconds': statistics.median(values), 'csv_total_seconds': float(cumulative[-1]),
                     'training_receipt_minus_csv_seconds': train['seconds'] - float(cumulative[-1]),
                     'whole_training_entry_seconds_per_batch': train['seconds'] / train['batches']}
        for stage in ('train', 'eval'):
            profile = read(base / 'queue' / f'short_{arm}_s42_E8_{stage}_resource_profile.json')
            assert profile['status'] == 'COMPLETED' and profile['exit_code'] == 0 and not profile['monitor_errors']
            sample_times = [x['time'] for x in profile['samples']]
            assert all(a <= b for a, b in zip(sample_times, sample_times[1:]))
            phases.append({'arm': arm, 'stage': stage, 'first_monitor_sample_unix': sample_times[0],
                           'last_monitor_sample_unix': sample_times[-1], 'guard_completed_unix': profile['time'],
                           'monitor_samples': len(sample_times)})
    assert all(a['guard_completed_unix'] < b['first_monitor_sample_unix'] for a, b in zip(phases, phases[1:]))
    training_total = sum(x['training_seconds'] for x in rows.values())
    evaluation_total = sum(x['evaluation_seconds'] for x in rows.values())
    task_total = training_total + evaluation_total
    queue_seconds = completion['seconds']
    observed_span = phases[-1]['guard_completed_unix'] - launch['time']
    assert queue_seconds > observed_span > task_total > training_total
    assert math.isclose(freeze['estimated_hours_with_margin'], freeze['pure_training_hours'] * 1.2 + 1, abs_tol=1e-12)
    total = {'training_seconds': training_total, 'training_hours': training_total / 3600,
             'evaluation_seconds': evaluation_total, 'worker_task_seconds': task_total, 'worker_task_hours': task_total / 3600,
             'queue_seconds': queue_seconds, 'queue_hours': queue_seconds / 3600,
             'queue_minus_worker_timers_seconds': queue_seconds - task_total,
             'csv_training_seconds': sum(x['csv_total_seconds'] for x in rows.values()),
             'observed_first_launch_to_last_guard_seconds': observed_span,
             'first_launch_running_unix': launch['time'], 'last_guard_completed_unix': phases[-1]['guard_completed_unix']}
    budget = {'frozen_pure_training_hours': freeze['pure_training_hours'],
              'frozen_total_hours_with_margin': freeze['estimated_hours_with_margin'],
              'actual_training_fraction_of_training_budget': total['training_hours'] / freeze['pure_training_hours'],
              'actual_queue_fraction_of_total_budget': total['queue_hours'] / freeze['estimated_hours_with_margin'],
              'total_budget_minus_actual_queue_hours': freeze['estimated_hours_with_margin'] - total['queue_hours']}
    result = {'status': 'COMPLETED_CPU_TIMING_SUMMARY', 'arms': rows, 'totals': total, 'budget': budget,
              'guard_phases': phases, 'snapshot_captured_at': snapshot['captured_at'], 'provenance': provenance,
              'queue_timer_definition': 'run_short_screen_queue.py lines 103-111: started=time.time() before six jobs; seconds written after all six stages and terminal checks.',
              'training_timer_definition': 'train_short_screen.py lines 77-95: perf_counter before trainer construction through train() and terminal checks; earlier source copying and interpreter startup are outside.',
              'epoch_timer_definition': 'Difference of CSV time column, using only epoch and time; rounded source precision, not an isolated training-kernel timer.',
              'strict_code_speedup_claim': False, 'csv_ap_used': False, 'gpu_used': False, 'network_used': False,
              'weights_loaded': False, 'new_hash_computed': False, 'parent_readme_modified': False}
    lines = ['# E8 三臂实际吞吐与队列耗时', '',
             f'**三臂训练合计 {total["training_hours"]:.6f} 小时，独立评价合计 {evaluation_total:.3f} 秒；完整六阶段队列实际耗时 {total["queue_hours"]:.6f} 小时（{queue_seconds / 60:.3f} 分钟）。** 这是本次顺序运行的实际墙钟，不是严格代码加速倍数。', '',
             '|臂|训练入口分钟|CSV 八轮耗时范围（秒）|CSV 八轮中位数（秒）|独立评价秒|训练入口平均秒/批|',
             '|---|---:|---:|---:|---:|---:|']
    for arm, row in rows.items():
        lines.append(f'|{arm}|{row["training_minutes"]:.3f}|{row["csv_epoch_min_seconds"]:.3f}–{row["csv_epoch_max_seconds"]:.3f}|{row["csv_epoch_median_seconds"]:.3f}|{row["evaluation_seconds"]:.3f}|{row["whole_training_entry_seconds_per_batch"]:.6f}|')
    lines += ['', '每臂均完成 8 轮、4504 批。训练入口计时从构建 trainer 前开始，覆盖 train() 与结束检查；不含更早的源码复制和解释器启动。平均秒/批包含该入口内的设置与结束开销。独立评价秒数沿用各评估回执的原计时，不把进程完整生命周期混入其中。', '',
              '|已完成轮次|N 秒|C0 秒|C1 秒|', '|---|---:|---:|---:|']
    for i in range(8):
        lines.append('|' + str(i + 1) + '|' + '|'.join(f'{rows[a]["csv_epoch_seconds"][i]:.3f}' for a in ENDPOINTS) + '|')
    lines += ['', '上表只使用 CSV 的 epoch/time 前两列，对累计 time 做差分。三臂表头和前七轮均为 15 列、末轮均仅 8 列；末尾 LR 及禁用训练内评价的占位不能作为 AP。差分继承原 CSV 的舍入精度，包含原 trainer 计时边界内的开销，不是独立 GPU kernel 计时。', '',
              f'三臂 CSV 累计合计 {total["csv_training_seconds"]:.3f} 秒；训练入口合计 {training_total:.3f} 秒，两者相差 {training_total-total["csv_training_seconds"]:.3f} 秒，计时边界不同，保持分列。训练入口与评估回执合计 {task_total:.3f} 秒（{total["worker_task_hours"]:.6f} 小时）；队列回执为 {queue_seconds:.3f} 秒，差额 {queue_seconds-task_total:.3f} 秒包含各入口外的启动、准备、调度与完成检查，现有计时不能逐项分摊。', '',
              f'完整队列来源为 [queue_completion.json](queue_completion.json)，状态 SHORT_SCREEN_MATRIX_COMPLETED，顺序 N train/eval → C0 train/eval → C1 train/eval。冻结队列源码第 103–111 行在六阶段开始前记录 started，全部成功并完成终态检查后写 seconds。这个回执补齐了完整队列时间，无需用心跳采集时间代替完成时刻。另以首次 N RUNNING 到末次 C1 eval guard 完成算出的观察窗口为 {observed_span:.3f} 秒；它只是交叉核对，不替代队列自身计时。', '',
              f'冻结预算为纯训练 {freeze["pure_training_hours"]:.6f} 小时，另加 20% 训练波动余量与 1 小时准备/评估/调度，总预算 {freeze["estimated_hours_with_margin"]:.6f} 小时。实际训练用量为纯训练预算的 {budget["actual_training_fraction_of_training_budget"]*100:.2f}%；实际完整队列为总预算的 {budget["actual_queue_fraction_of_total_budget"]*100:.2f}%，低于预算 {budget["total_budget_minus_actual_queue_hours"]:.6f} 小时。这是对原预算的实际结算，不能解释为优化带来的同负载加速。', '',
              'N 的前四轮约 440–531 秒，后四轮约 271–287 秒；C0 的第 2–8 轮约 237–240 秒，C1 约 541–560 秒/轮。运行时段不同且共享 GPU 负载发生变化，预热/缓存也可能影响计时；不能将臂间墙钟差或相对短窗口预算的下降归因于代码或蒸馏方法；没有同负载配对计时，也不以这些时间推断 E200 收敛或方法收益。', '',
              '输入是三个 endpoint 的训练/评价回执、CSV、六阶段资源时间戳、launch 冻结预算与首次状态，以及 06:36 心跳中的本项目完成字段。心跳训练/评价 seconds 与端点回执逐值一致。只引用完整主机 snapshot 的路径并读取所需字段，不复制其主机进程明细。', '',
              '复算：[summarize_throughput.py](summarize_throughput.py)；数值与来源 stat：[throughput_summary.json](throughput_summary.json)。使用 Python 标准库即可；脚本拒绝覆盖已有输出。未联网、使用 GPU、加载权重、读取 CSV AP、计算新文件 hash 或修改父 README。', '']
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {'throughput_summary.json': (json.dumps(result, ensure_ascii=False, indent=2) + '\n').encode('utf-8'),
               'THROUGHPUT.md': '\n'.join(lines).encode('utf-8')}
    for name in outputs:
        if (args.output_dir / name).exists():
            raise FileExistsError(args.output_dir / name)
    for name, payload in outputs.items():
        (args.output_dir / name).write_bytes(payload)
    print(json.dumps({'status': result['status'], 'totals': total, 'budget': budget}, ensure_ascii=True))


if __name__ == '__main__':
    main()
