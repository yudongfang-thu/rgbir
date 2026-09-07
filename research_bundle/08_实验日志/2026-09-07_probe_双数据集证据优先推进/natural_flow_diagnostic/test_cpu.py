"""CPU source/stream/selection contracts, not a detector or GPU canary."""
import importlib.util
import json
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
WS = HERE.parents[2]
RELEASE = WS / '03_现行工程/SpaceNet6_OTD_official_reproduction/experiments/rgbir_independent_kd_v2'
sys.path.insert(0, str(RELEASE))
sys.path.insert(0, str(RELEASE / 'task_conditional_reference'))
import test_classification_logit as cfixture
import test_localization_loss as lfixture
from selection_adapter import build_classification_selection, original, EvidenceConfig
from localization_loss import LocalizationConfig, build_localization_selection
import diagnose_natural_flow as diag


def main():
    torch.set_num_threads(1)
    tests = {}
    batch = lfixture.labels(indices=[0])
    batch.update(teacher_batch=lfixture.labels(indices=[0]), im_file=['image0'])
    cfg = LocalizationConfig(input_size=64)
    teacher = lfixture.raw(distance=1.5)
    reference = lfixture.raw(distance=1.)
    primary = build_localization_selection(teacher, reference, batch, config=cfg, return_records=True)
    replay = diag.replay_by_image(teacher, reference, batch, cfg, (8,16,32), build_localization_selection, primary)
    assert primary.stats['selected_count'] == 1 and primary.stats['selected_anchors'] == [(0,27,0)]
    assert not primary.stats['geometry_verified'] and not primary.stats['geometry_mask_supplied']
    tests['one_object_primary_and_replay_exact_unverified'] = True

    batch = lfixture.labels(boxes=((16,16,40,40), (40,16,64,40), (16,16,40,40)), indices=[0,0,2])
    batch.update(teacher_batch=lfixture.labels(boxes=((16,16,40,40), (40,16,64,40), (16,16,40,40)), indices=[0,0,2]), im_file=['image0','empty1','image2'])
    teacher, reference = lfixture.raw(3,1.5), lfixture.raw(3,1.)
    primary = build_localization_selection(teacher, reference, batch, config=cfg, return_records=True)
    replay = diag.replay_by_image(teacher, reference, batch, cfg, (8,16,32), build_localization_selection, primary)
    assert primary.stats['selected_object_ids'] == [(0,0,0),(2,2,2)]
    assert primary.stats['normalizer'] == 2 and replay[1]['gate_counts']['base_count'] == 0
    assert sum(r['gate_counts']['base_count'] for r in replay) == 2
    tests['multi_image_empty_image_global_gt_ids_and_batch_normalizer'] = True

    batch = lfixture.labels(); batch.update(teacher_batch=lfixture.labels(), im_file=['image0'])
    teacher, reference = lfixture.raw(distance=1.5), lfixture.raw()
    reference['scores'][0,0,28] = 5.
    primary = build_localization_selection(teacher, reference, batch, config=cfg, return_records=True)
    assert primary.anchor_indices.tolist() == [28]
    tests['real_selector_is_reference_confidence_first_not_gt_best_iou'] = True

    batch = cfixture.labels(); batch.update(teacher_batch=cfixture.labels(), im_file=['image0'])
    r, t = cfixture.evidence_raw(1, requires_grad=False), cfixture.evidence_raw(4, requires_grad=False)
    with torch.no_grad():
        selection = build_classification_selection(r,t,r,batch,config=EvidenceConfig(input_size=64))
        _, old = original.object_evidence_loss(r,t,r,batch,config=EvidenceConfig(input_size=64))
        changed = cfixture.evidence_raw(-3, requires_grad=False)
        changed_selection = build_classification_selection(changed,t,r,batch,config=EvidenceConfig(input_size=64))
    assert selection.c0_stats == old
    assert selection.c0_stats['selected_object_ids'] == changed_selection.c0_stats['selected_object_ids']
    record = diag.classification_record(selection,batch,[{'governed_group':'fixture'}])
    assert record['class_counts']['0']['selected'] == 1 and record['selection_only_no_student_loss']
    tests['whole_batch_C0_adapter_exact_and_student_values_do_not_select'] = True

    # Actual old source trace validates serialization semantics, then tampering rejects.
    base = WS / '08_实验日志/2026-09-07_train_IndependentKD实施/remote_cpu1'
    previous_trace_counts = {}
    for dataset in ('llvip','drone'):
        root = base / ('coverage_' + dataset + '_attempt1')
        records = [json.loads(line) for line in (root / 'natural_batches.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len(records) == 64 and [r['batch'] for r in records] == list(range(64))
        assert all(len(r['images']) == 32 for r in records)
        for i, row in enumerate(records):
            diag.compare_trace(row['images'], row['images'], i)
            for image in row['images']:
                assert {'rgb_matrix','ir_matrix','coverage','geometry_trace'} <= set(image['pair_info'])
        previous_trace_counts[dataset] = len(records) * 32
        changed = json.loads(json.dumps(records[0]['images']))
        changed[0]['pair_info']['rgb_matrix'][0][2] += 1e-8
        try: diag.compare_trace(changed, records[0]['images'], 0)
        except ValueError: pass
        else: raise AssertionError('Changed augmentation accepted')
        receipt = json.loads((root / 'coverage_receipt.json').read_text(encoding='utf-8'))
        live = json.loads(json.dumps(receipt))
        for item in live['train_manifests']:
            item['names'] = {int(k):v for k,v in item['names'].items()}
        diag.metadata_matches(live, receipt)
        live['workers'] = 0
        try: diag.metadata_matches(live, receipt)
        except ValueError: pass
        else: raise AssertionError('Changed loader workers accepted')
    tests['both_prior_real_64x32_trace_contracts_and_augmentation_tamper_rejection'] = previous_trace_counts
    tests['serialized_int_class_keys_match_and_recipe_drift_rejects'] = True
    assert not torch.cuda.is_initialized()
    result = {'status':'PASSED_CPU_ONLY_NOT_REMOTE_ADMISSION','tests':tests,
              'torch_version':str(torch.__version__),'pinned_training_runtime':False,'cuda_initialized':False,
              'new_training':False,'new_hash':False,'new_forward_on_real_data':False}
    with (HERE / 'cpu_checks.json').open('x',encoding='utf-8') as f:
        json.dump(result,f,indent=2,ensure_ascii=False)
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__ == '__main__': main()
