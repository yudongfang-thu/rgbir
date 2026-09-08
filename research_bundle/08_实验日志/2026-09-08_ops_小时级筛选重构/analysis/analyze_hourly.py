"""Pure CPU descriptive FT3 summary; no training, inference or scientific promotion."""
import argparse
import json
import math
from pathlib import Path
import shutil

ARMS = ('N', 'C0', 'C1')
METRICS = ('mAP50_95', 'AP50', 'AP75', 'precision', 'recall')
AP = METRICS[:3]
PAIRS = (('C0', 'N'), ('C1', 'N'), ('C1', 'C0'))
ENDPOINT = 'HOURLY_SCREEN_FT_E3_LAST_EMA'
CLASSES = ('car', 'freight car', 'truck', 'bus', 'van')
COEFFICIENTS = {'N': 0., 'C0': .1, 'C1': 0.09227393550836771}
SEQUENCE = [(arm, 'canary') for arm in ARMS] + [(arm, stage) for arm in ARMS for stage in ('train', 'eval')]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, label, minimum=0, maximum=None):
    require(type(value) in (int, float) and math.isfinite(value), 'nonfinite/non-numeric ' + label)
    require(value >= minimum and (maximum is None or value <= maximum), 'out-of-range ' + label)
    return value


def positive_seconds(value, label):
    number(value, label)
    require(value > 0, 'nonpositive ' + label)
    return value


def integer(value, label):
    require(type(value) is int and value >= 0, 'invalid integer ' + label)
    return value


def false_flags(receipt):
    for name in ('formal_e200_complete', 'formal_paper_gain_claim', 'official_test_accessed', 'new_hash_computed'):
        require(receipt.get(name) is False, 'forbidden/missing flag: ' + name)


def analyze(evaluations, trainings, queue):
    """Validate small executed records, then compute only prespecified quantities."""
    require(set(evaluations) == set(ARMS) and set(trainings) == set(ARMS), 'missing or extra arm')
    base, base_train = evaluations['N'], trainings['N']
    common_eval = ('seed', 'dataset', 'endpoint', 'epochs', 'independent_lr_horizon',
                   'full_dev_images', 'full_dev_gt_objects', 'native_profile_binding', 'training_model')
    common_train = ('model', 'teacher', 'reference', 'seed', 'dataset', 'epochs_configured', 'last_epoch', 'batches')
    values, elapsed, counters = {}, {}, {}
    for arm in ARMS:
        r, t = evaluations[arm], trainings[arm]
        require(r['status'] == 'HOURLY_SCREEN_EVALUATION_COMPLETED', 'incomplete evaluation: ' + arm)
        require(t['status'] == 'HOURLY_SCREEN_TRAINING_COMPLETED', 'incomplete training: ' + arm)
        for item in (r, t):
            require(item['arm'] == arm and item['seed'] == 42 and item['dataset'] == 'dronevehicle', 'arm/seed/dataset')
            require(item['scope'] == 'HOURLY_SCREEN_FT' and item['single_seed'] is True, 'FT single-seed scope')
            require(item['endpoint'] == ENDPOINT, 'fixed FT3 endpoint')
            false_flags(item)
        require(r['epochs'] == 3 and r['independent_lr_horizon'] == 3, 'evaluation horizon')
        require(t['epochs_configured'] == 3 and t['last_epoch'] == 3 and t['batches'] == 192, 'training duration')
        require(type(t['classification_coefficient']) in (int, float)
                and t['classification_coefficient'] == COEFFICIENTS[arm]
                and type(t['localization_coefficient']) in (int, float)
                and t['localization_coefficient'] == 0., 'fixed arm coefficient identity')
        require(r['full_dev_images'] == 1469 and r['full_dev_gt_objects'] == 22462, 'full dev population')
        require(all(r[k] == base[k] for k in common_eval), 'unmatched evaluator/initialization protocol')
        require(all(t[k] == base_train[k] for k in common_train), 'unmatched train/initialization protocol')
        require(t['model'] == r['training_model'], 'student warm-start identity')
        require(t['checkpoint'] == r['checkpoint'], 'train/eval checkpoint stat mismatch')
        require(t['canary_receipt'] == r['canary_receipt'], 'train/eval canary identity')
        require(r['metric_units'] == 'fraction_0_to_1', 'metric units')
        counts = {k: integer(t[k], k) for k in ('batches', 'optimizer_updates', 'attempts', 'amp_skips', 'ema_updates')}
        require(counts['optimizer_updates'] > 0, 'no successful update')
        require(counts['attempts'] <= counts['batches'], 'more update attempts than batches')
        require(counts['optimizer_updates'] + counts['amp_skips'] == counts['attempts'], 'AMP/update counter closure')
        require(counts['ema_updates'] == counts['attempts'], 'EMA/attempt counter closure')
        if 'successful_updates' in t:
            require(t['successful_updates'] == counts['optimizer_updates'], 'successful update alias differs')
        counters[arm] = counts
        for metric in METRICS:
            number(r[metric], arm + ' ' + metric, maximum=1)
        rows = sorted(r['per_class'], key=lambda p: p['class_id'])
        require(len(rows) == 5 and [p['class_id'] for p in rows] == list(range(5)), 'five unique classes')
        require(all(type(p['class_id']) is int for p in rows), 'class ID type')
        require([p['name'] for p in rows] == list(CLASSES), 'Drone class mapping')
        for metric in AP:
            for p in rows:
                number(p[metric], 'class ' + metric, maximum=1)
            require(math.isclose(sum(p[metric] for p in rows) / 5, r[metric], rel_tol=0, abs_tol=1e-12), 'class mean mismatch')
        values[arm] = dict(percent={k: 100 * r[k] for k in METRICS},
            per_class_percent=[dict(class_id=p['class_id'], name=p['name'], **{k: 100*p[k] for k in AP}) for p in rows])
        elapsed[arm] = dict(training_seconds=positive_seconds(t['seconds'], 'train seconds'),
                            evaluation_seconds=positive_seconds(r['seconds'], 'eval seconds'))
    require(queue['status'] == 'HOURLY_SCREEN_MATRIX_COMPLETED', 'queue is not complete')
    require(queue['single_seed'] is True and queue['seed'] == 42 and queue['epochs'] == 3, 'queue seed/horizon')
    require(queue['train_images'] == 2048 and queue['batches_per_arm'] == 192, 'queue train population')
    for flag in ('new_hash_computed', 'formal_e200_complete', 'formal_paper_gain_claim', 'ap_adaptive'):
        require(queue.get(flag) is False, 'queue forbidden/missing flag: ' + flag)
    stages = queue['completed']
    require([(row['arm'], row['stage']) for row in stages] == SEQUENCE, 'incomplete or reordered queue stages')
    require(len({row['id'] for row in stages}) == len(SEQUENCE), 'duplicate queue stage ID')
    for row in stages:
        if row['stage'] in ('train', 'eval'):
            expected = ('/runs/' + row['arm'] + '/hourly_training_receipt.json' if row['stage'] == 'train'
                        else '/evaluations/' + row['arm'] + '/hourly_evaluation_receipt.json')
            require(row['receipt'].replace('\\', '/').endswith(expected), 'queue receipt target')
    differences = {}
    for left, right in PAIRS:
        left_pc, right_pc = values[left]['per_class_percent'], values[right]['per_class_percent']
        differences[left + '-' + right] = dict(
            pp={k: 100 * (evaluations[left][k] - evaluations[right][k]) for k in METRICS},
            per_class_pp=[dict(class_id=p['class_id'], name=p['name'], **{k: p[k]-q[k] for k in AP})
                          for p, q in zip(left_pc, right_pc)])
    training_seconds = sum(row['training_seconds'] for row in elapsed.values())
    evaluation_seconds = sum(row['evaluation_seconds'] for row in elapsed.values())
    queue_seconds = positive_seconds(queue['seconds'], 'queue seconds')
    remainder = queue_seconds - training_seconds - evaluation_seconds
    require(remainder >= -1e-6, 'serial queue duration shorter than contained train/eval intervals')
    timing = dict(per_arm=elapsed, training_seconds=training_seconds, evaluation_seconds=evaluation_seconds,
                  queue_seconds=queue_seconds, queue_hours=queue_seconds/3600,
                  other_including_canaries_seconds=remainder,
                  scope='Queue includes three canaries, waiting/startup and entry-external work; not a code-speedup estimate.')
    return dict(scope='DESCRIPTIVE_SINGLE_SEED_FT3_ONLY', seed=42, n_seeds=1, epochs=3, train_images=2048,
                endpoint=ENDPOINT, sample_sd=None, significance_test=None, formal_gain_claim=False,
                automatic_expansion=False, E200_decision=False, C0_review_released=False,
                initial_model=base_train['model'], values=values, differences=differences,
                training_counters=counters, timing=timing, new_hash_computed=False)


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def source_stat(path):
    s = path.stat()
    return dict(path=str(path.absolute()), bytes=s.st_size, mtime_ns=s.st_mtime_ns)


def load_campaign(campaign):
    evaluations, trainings, sources = {}, {}, []
    common_roster = None
    for arm in ARMS:
        train = campaign / 'runs' / arm / 'hourly_training_receipt.json'
        evaluation = campaign / 'evaluations' / arm / 'hourly_evaluation_receipt.json'
        roster = evaluation.parent / 'development_roster.txt'
        trainings[arm], evaluations[arm] = read(train), read(evaluation)
        raw = roster.read_bytes()
        entries = raw.decode('utf-8-sig').splitlines()
        require(len(entries) == 1469 and len(set(entries)) == 1469 and all(entries), 'invalid full dev roster')
        if common_roster is None:
            common_roster = raw
        require(raw == common_roster, 'three-arm roster bytes differ')
        sources.extend(source_stat(p) for p in (train, evaluation, roster))
    queue = campaign / 'queue' / 'completion.json'
    result = analyze(evaluations, trainings, read(queue))
    sources.append(source_stat(queue))
    return result, sources


def comparison_markdown(result):
    lines = ['# FT3 三臂原值、差值与耗时', '',
             '仅单 seed、成熟 RGB 初始化、固定 2048 图/3 轮的描述性结果。百分制原值，差值 pp；没有跨 seed SD、显著性或自动升级判断。', '',
             '|臂|mAP50–95|AP50|AP75|Precision|Recall|', '|---|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        lines.append('|' + arm + '|' + '|'.join('{:.6f}'.format(result['values'][arm]['percent'][k]) for k in METRICS) + '|')
    lines += ['', '|比较|ΔmAP50–95|ΔAP50|ΔAP75|ΔPrecision|ΔRecall|', '|---|---:|---:|---:|---:|---:|']
    for key, diff in result['differences'].items():
        lines.append('|' + key + '|' + '|'.join('{:+.6f}'.format(diff['pp'][k]) for k in METRICS) + '|')
    for metric in AP:
        lines += ['', '## 逐类 ' + metric, '', '|类别|N|C0|C1|C0−N|C1−N|C1−C0|', '|---|---:|---:|---:|---:|---:|---:|']
        for i, name in enumerate(CLASSES):
            raw = [result['values'][arm]['per_class_percent'][i][metric] for arm in ARMS]
            delta = [result['differences'][a+'-'+b]['per_class_pp'][i][metric] for a, b in PAIRS]
            lines.append('|' + name + '|' + '|'.join('{:.6f}'.format(x) for x in raw) + '|'
                         + '|'.join('{:+.6f}'.format(x) for x in delta) + '|')
    lines += ['', '## 独立计时边界', '', '|臂|训练入口秒|评价入口秒|成功更新|AMP skips|attempt/EMA|', '|---|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        t, c = result['timing']['per_arm'][arm], result['training_counters'][arm]
        lines.append('|{}|{:.6f}|{:.6f}|{}|{}|{} / {}|'.format(arm, t['training_seconds'], t['evaluation_seconds'],
                     c['optimizer_updates'], c['amp_skips'], c['attempts'], c['ema_updates']))
    timing = result['timing']
    lines += ['', '队列总耗时 {:.6f} 秒（{:.6f} 小时）；训练入口合计 {:.6f} 秒，评价入口合计 {:.6f} 秒。'.format(
                 timing['queue_seconds'], timing['queue_hours'], timing['training_seconds'], timing['evaluation_seconds']),
              '差额 {:.6f} 秒包含三臂 canary、队列等待、启动与入口外处理，不能全称训练开销，不能据此推断严格代码加速。'.format(timing['other_including_canaries_seconds']), '',
              'P/R 是原生 evaluator 汇总值，不是新增固定阈值对象损伤统计。初始化已见过完整 train；本结果不代表仅用 2048 图从头训练，也不判断 E200、非目标类因果或解除 C0 REVIEW_REQUIRED。']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Output exists; preserve prior attempts')
    result, sources = load_campaign(args.campaign)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, value in (('summary.json', result), ('inputs.json', dict(files=sources, new_hash_computed=False))):
        with (args.output/name).open('x', encoding='utf-8') as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n')
    (args.output/'COMPARISON_TABLES.md').write_text(comparison_markdown(result), encoding='utf-8')
    (args.output/'README.md').write_text(
        '**仅完成单 seed FT3 描述性汇总，不生成正式增益或自动扩展结论。**\n\n'
        '固定共同成熟 RGB 初始化、2048 图/3轮/192批、完整 dev1469 图/22462 GT。'
        '原值、百分点差及分开计时见 [表格](COMPARISON_TABLES.md)，原始计算见 [summary.json](summary.json)，'
        '来源 path/stat 见 [inputs.json](inputs.json)。未读取权重、运行 GPU 或计算新 hash。\n', encoding='utf-8')
    shutil.copyfile(Path(__file__), args.output/'executed_analyze_hourly.py')
    require((args.output/'executed_analyze_hourly.py').read_bytes() == Path(__file__).read_bytes(), 'Executed source copy differs')
    print(json.dumps(dict(output=str(args.output), scope=result['scope']), ensure_ascii=False))


if __name__ == '__main__':
    main()
