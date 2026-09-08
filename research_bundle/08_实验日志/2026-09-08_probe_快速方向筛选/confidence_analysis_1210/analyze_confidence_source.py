# coding: utf-8
"""Prospectively frozen LLVIP confidence N/C0 receipt-only readout; no model load."""
import argparse
import json
import math
from pathlib import Path
import shutil

SCOPE = 'LLVIP_CONFIDENCE_FT3'
ENDPOINT = 'LLVIP_CONFIDENCE_FT3_LAST_EMA'
ARMS = ('N', 'C0')
AP = ('mAP50_95', 'AP50', 'AP75')
METRICS = AP + ('precision', 'recall')
DATA_KEYS = ('student_data_yaml', 'privileged_data_yaml', 'paired_train_mapping')


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def fraction(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def validate(receipt, arm):
    fixed = dict(status='DIRECTION_EVALUATION_COMPLETED', scope=SCOPE, endpoint=ENDPOINT,
                 dataset='llvip', arm=arm, seed=42, single_seed=True, epochs=3, independent_lr_horizon=3,
                 expected_train_images=2048, full_dev_images=2406, full_dev_gt_objects=7879,
                 observed_images=2406, gt_objects_captured=7879, metric_units='fraction_0_to_1',
                 formal_e200_complete=False, formal_paper_gain_claim=False, accepted_endpoint_claim=False,
                 official_test_accessed=False, new_hash_computed=False)
    for key, value in fixed.items():
        actual = receipt.get(key)
        require(type(actual) is type(value) and actual == value, 'Receipt identity differs: ' + key)
    expected_coefficient = 0. if arm == 'N' else .1
    require(type(receipt.get('kd_coefficient')) in (int, float)
            and receipt['kd_coefficient'] == expected_coefficient, 'Fixed N/C0 coefficient differs')
    require(all(fraction(receipt.get(key)) for key in METRICS), 'Invalid metric fraction')
    classes = receipt.get('per_class')
    require(isinstance(classes, list) and len(classes) == 1, 'Exactly one person class required')
    person = classes[0]
    require(type(person.get('class_id')) is int and person['class_id'] == 0 and person.get('name') == 'person',
            'Single-class mapping differs')
    for key in AP:
        require(fraction(person.get(key)) and math.isclose(person[key], receipt[key], rel_tol=0, abs_tol=1e-12),
                'Person AP and overall AP differ: ' + key)
    for key in ('training_model', 'training_configuration', 'training_completion'):
        require(isinstance(receipt.get(key), str) and bool(receipt[key]), 'Missing training identity: ' + key)
    subset = receipt.get('training_subset_identity')
    require(isinstance(subset, dict), 'Missing this-confidence-run subset identity')
    for key in DATA_KEYS:
        require(isinstance(subset.get(key), str) and bool(subset[key]), 'Missing subset identity: ' + key)
    projection = receipt.get('evaluation_identity_projection', {})
    require(projection.get('dataset') == 'llvip' and projection.get('subset_dev_roster_exact_to_full') is True,
            'Missing original-full-dev projection')
    require(type(projection.get('expected_dev_images')) is int and projection['expected_dev_images'] == 2406
            and type(projection.get('expected_dev_gt_objects')) is int and projection['expected_dev_gt_objects'] == 7879,
            'Full-dev projection counts differ')
    require(projection.get('training_model') == receipt['training_model']
            and projection.get('training_data_yaml') == subset['student_data_yaml'], 'Training/full-dev projection binding differs')
    require(isinstance(projection.get('actual_evaluation_data_yaml'), str) and bool(projection['actual_evaluation_data_yaml']),
            'Missing complete dev YAML')
    bn = receipt.get('bn_training_evidence', {})
    require(bn.get('bn_running_buffers_unchanged') is True and bn.get('bn_affine_trainable') is True,
            'Actual BN frozen-buffer/trainable-affine evidence missing')
    require(type(bn.get('bn_buffer_count')) is int and bn['bn_buffer_count'] > 0, 'Missing actual BN buffer count')
    seconds = receipt.get('seconds')
    require(type(seconds) in (int, float) and math.isfinite(seconds) and seconds >= 0, 'Invalid evaluation duration')
    return receipt


def summarize(receipts):
    require(set(receipts) == set(ARMS), 'Both new-protocol N and C0 completed receipts are required; no partial/old-N substitution')
    n, c = [validate(receipts[a], a) for a in ARMS]
    require(n['training_model'] == c['training_model'], 'N/C0 initialization path differs')
    require(all(n['training_subset_identity'][k] == c['training_subset_identity'][k] for k in DATA_KEYS),
            'N/C0 fixed training data or mapping differs')
    require(n['evaluation_identity_projection']['actual_evaluation_data_yaml']
            == c['evaluation_identity_projection']['actual_evaluation_data_yaml'], 'N/C0 full-dev YAML differs')
    require(n['training_completion'] != c['training_completion'] and n['training_configuration'] != c['training_configuration'],
            'N and C0 cannot reuse the same actual training run/configuration')
    raw = {a: {key: receipts[a][key] for key in METRICS} for a in ARMS}
    delta = {key: (c[key] - n[key]) * 100. for key in METRICS}
    return dict(status='COMPLETE_CONFIDENCE_PAIR_READOUT', scope=SCOPE, endpoint=ENDPOINT,
                dataset='llvip', seed=42, n_seeds=1, standard_deviation=None, epochs=3,
                expected_train_images=2048, full_dev_images=2406, full_dev_gt_objects=7879,
                raw_metric_units='fraction_0_to_1', display_metric_units='percent', difference_units='percentage_points',
                raw_fraction=raw, display_percent={a: {k: v*100 for k, v in r.items()} for a, r in raw.items()},
                coefficients={'N': 0., 'C0': .1}, comparison=dict(contrast='C0 - N', delta_pp=delta,
                    direction={k: 'positive' if v > 0 else 'negative' if v < 0 else 'zero' for k, v in delta.items()}),
                evaluation_seconds={a: receipts[a]['seconds'] for a in ARMS},
                common_initialization=n['training_model'], common_training_subset=n['training_subset_identity'],
                full_dev_yaml=n['evaluation_identity_projection']['actual_evaluation_data_yaml'],
                formal_e200_complete=False, formal_paper_gain_claim=False, causal_modality_claim=False,
                automatically_extend_matrix=False, old_L2_N_reused=False, new_hash_computed=False,
                limitation='Receipt-only single-seed fixed confidence FT3 comparison. No significance/SD, checkpoint reread, GT-box re-audit, old-L2-N comparison or content-causality claim.')


def collect(campaign):
    receipts, inputs = {}, []
    for arm in ARMS:
        folder = Path(campaign) / 'evaluations' / arm
        completed = folder / 'direction_evaluation_receipt.json'
        failures = [p for p in (folder / 'direction_evaluation_failure.json', folder / 'confidence_evaluation_failure.json') if p.exists()]
        require(completed.is_file() and not failures, 'Missing or conflicting completed confidence endpoint: ' + arm)
        receipts[arm] = read(completed)
        st = completed.stat()
        inputs.append(dict(path=str(completed.absolute()), bytes=st.st_size, mtime_ns=st.st_mtime_ns))
    result = summarize(receipts)
    result['input_sources'] = inputs
    return result


def markdown(result):
    lines = ['# LLVIP 置信度 N/C0 单seed FT3 原值', '',
        '仅比较本轮匹配 N/C0；2048训练图、完整dev2406图/7879GT、固定3轮last/EMA。AP以百分数显示，差值为pp。正负号仅描述观测方向，不计算SD/显著性，不自动扩展或升级E200增益。', '',
        '|臂|λ|mAP50–95|AP50|AP75|precision|recall|', '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        lines.append('|'+arm+'|'+str(result['coefficients'][arm])+'|'+'|'.join(f"{result['display_percent'][arm][k]:.6f}" for k in METRICS)+'|')
    lines += ['', '|固定比较|ΔmAP50–95|ΔAP50|ΔAP75|Δprecision|Δrecall|', '|---|---:|---:|---:|---:|---:|',
        '|C0−N|'+'|'.join(f"{result['comparison']['delta_pp'][k]:+.6f}" for k in METRICS)+'|', '',
        '完整精度原始fraction、百分点差与输入stat见[summary.json](summary.json)。仅检查完成回执及其中的初始化/子集/dev身份，不重新证明权重字节或GT框版本；没有复用原L2的N端点，也不将不同λ/损失的数字混为同剂量比较。']
    return '\n'.join(lines)+'\n'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    require(args.campaign.is_dir() and not args.output.exists(), 'Existing campaign and new output required')
    result = collect(args.campaign)
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / 'summary.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    (args.output / 'README.md').write_text(markdown(result), encoding='utf-8')
    shutil.copyfile(__file__, args.output / 'analyze_confidence_source.py')
    require(Path(__file__).read_bytes() == (args.output / 'analyze_confidence_source.py').read_bytes(), 'Analyzer source copy differs')
    print(result['status'])


if __name__ == '__main__':
    main()
