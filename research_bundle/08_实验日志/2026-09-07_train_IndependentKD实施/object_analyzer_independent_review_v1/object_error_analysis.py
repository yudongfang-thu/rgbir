"""CPU-only fixed-point object repair/damage analysis of bound evaluator output.

This file belongs to the experiment log, not the immutable training release.
Default outputs are NOT_ACCEPTED drafts. An independently signed source-copy
review is needed to publish the contract consumed by analyze_independent.py.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import gzip
import json
import math
from pathlib import Path, PurePosixPath
import shutil

ERROR_CONTRACT = {'confidence': .25, 'match_iou': .50, 'coarse_iou': .10,
    'background_definition': 'all_gt_iou_below_0.10',
    'background_unit': 'false_positives_per_image'}
SCHEMA = 'rgbir-object-error-analysis-v1'
UNKNOWN = 'UNKNOWN'
SCALE_NAMES = ('canvas640_small_lt32sq', 'canvas640_medium_32sq_to_lt96sq', 'canvas640_large_ge96sq')
REVIEW_FILES = ('object_error_analysis.py', 'test_object_error_analysis.py', 'OBJECT_ERROR_ANALYSIS_RULES.md')
POPULATIONS = {'dronevehicle':1469,'llvip':2406}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def class_id(value):
    if not finite(value) or value < 0 or int(value) != value:
        raise ValueError('Classes must be nonnegative integer-valued numbers')
    return int(value)


def box_area(box):
    return max(0., box[2] - box[0]) * max(0., box[3] - box[1])


def iou(a, b):
    intersection = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    union = box_area(a) + box_area(b) - intersection
    return intersection / union if union > 0 else 0.


def scale_bin(box):
    area = box_area(box)
    return SCALE_NAMES[0] if area < 32**2 else SCALE_NAMES[1] if area < 96**2 else SCALE_NAMES[2]


def validate_row(row, nc=None):
    if not isinstance(row.get('image'), str) or not row['image']:
        raise ValueError('Each image requires a nonempty exact image key')
    shape = row.get('canvas_shape')
    if (not isinstance(shape, list) or len(shape) != 2 or
            any(not finite(v) or int(v) != v or v <= 0 for v in shape) or max(shape) != 640):
        raise ValueError('Require actual network canvas with maximum dimension 640')
    for kind in ('gt', 'pred'):
        boxes, classes = row.get(kind+'_boxes'), row.get(kind+'_classes')
        if not isinstance(boxes, list) or not isinstance(classes, list) or len(boxes) != len(classes):
            raise ValueError('Mismatched box/class arrays: ' + kind)
        for box in boxes:
            if (not isinstance(box, list) or len(box) != 4 or not all(finite(x) for x in box)
                    or box[2] < box[0] or box[3] < box[1] or kind == 'gt' and box_area(box) <= 0):
                raise ValueError('Invalid xyxy box: ' + kind)
        for value in classes:
            ci = class_id(value)
            if nc is not None and ci >= nc:
                raise ValueError('Class outside bound dataset inventory')
    confidence = row.get('pred_confidence')
    if (not isinstance(confidence, list) or len(confidence) != len(row['pred_boxes'])
            or any(not finite(x) or not 0 <= x <= 1 for x in confidence)):
        raise ValueError('Invalid prediction confidence array')


def indexed_rows(rows, nc=None):
    result = {}
    for row in rows:
        validate_row(row, nc)
        if row['image'] in result:
            raise ValueError('Duplicate image row: ' + row['image'])
        result[row['image']] = row
    return result


def match_image(row):
    """Confidence-ordered greedy matching, independent of the other endpoint."""
    predictions = sorted((i for i,c in enumerate(row['pred_confidence']) if c >= .25),
                         key=lambda i: (-row['pred_confidence'][i], i))
    overlaps = {p: [iou(row['pred_boxes'][p], box) for box in row['gt_boxes']] for p in predictions}
    matched, pred_match, errors = {}, {}, {}
    for p in predictions:
        eligible = [g for g,c in enumerate(row['gt_classes'])
                    if g not in matched and class_id(c) == class_id(row['pred_classes'][p]) and overlaps[p][g] >= .5]
        if eligible:
            g = min(eligible, key=lambda j: (-overlaps[p][j], j))
            matched[g], pred_match[p], errors[p] = p, g, 'correct'
            continue
        if max(overlaps[p], default=0.) < .1:
            errors[p] = 'background'
        elif any(v >= .5 and class_id(row['gt_classes'][g]) == class_id(row['pred_classes'][p]) for g,v in enumerate(overlaps[p])):
            errors[p] = 'duplicate_or_assignment_competition'
        elif any(v >= .5 and class_id(row['gt_classes'][g]) != class_id(row['pred_classes'][p]) for g,v in enumerate(overlaps[p])):
            errors[p] = 'wrong_class'
        else:
            errors[p] = 'localization_or_mixed'
    objects = []
    for g, ci in enumerate(row['gt_classes']):
        same = [overlaps[p][g] for p in predictions if class_id(row['pred_classes'][p]) == class_id(ci)]
        other = [overlaps[p][g] for p in predictions if class_id(row['pred_classes'][p]) != class_id(ci)]
        best_same, best_other = max(same, default=0.), max(other, default=0.)
        if g in matched:
            state = 'correct'
        elif best_same >= .5:
            state = 'assignment_competition'
        elif best_other >= .5:
            state = 'wrong_class'
        elif best_same >= .1:
            state = 'localization_insufficient'
        elif best_other >= .1:
            state = 'coarse_wrong_class'
        else:
            state = 'no_coarse_candidate_at_conf025'
        p = matched.get(g)
        objects.append(dict(correct=g in matched, state=state, matched_prediction_index=p,
            matched_iou=None if p is None else overlaps[p][g],
            matched_confidence=None if p is None else row['pred_confidence'][p],
            max_same_class_iou=best_same, max_other_class_iou=best_other))
    counts = {name: sum(value == name for value in errors.values()) for name in
              ('correct','background','duplicate_or_assignment_competition','wrong_class','localization_or_mixed')}
    return dict(objects=objects, prediction_errors=errors, retained_predictions=predictions,
                prediction_counts=counts, background_fp=counts['background'])


def rates(counts):
    counts = dict(counts)
    incorrect = counts['gt_objects'] - counts['baseline_correct']
    counts.update(baseline_incorrect=incorrect,
        repair_rate=None if incorrect == 0 else counts['repaired']/incorrect,
        damage_rate=None if counts['baseline_correct'] == 0 else counts['damaged']/counts['baseline_correct'],
        repair_denominator='baseline_incorrect_objects', damage_denominator='baseline_correct_objects',
        rate_units='fraction_0_to_1')
    return counts


def zero_counts():
    return dict(gt_objects=0, baseline_correct=0, candidate_correct=0, repaired=0, damaged=0,
                stable_correct=0, still_incorrect=0)


def add_object(counts, baseline_correct, candidate_correct):
    counts['gt_objects'] += 1
    counts['baseline_correct'] += int(baseline_correct)
    counts['candidate_correct'] += int(candidate_correct)
    key = ('stable_correct' if candidate_correct else 'damaged') if baseline_correct else ('repaired' if candidate_correct else 'still_incorrect')
    counts[key] += 1


def image_metadata(image, mapping):
    row = mapping.get(image, {})
    return {key: str(row.get(key) or UNKNOWN) for key in ('source_group','brightness_bin')}


def analyze_pair(baseline_rows, candidate_rows, metadata=None, nc=None):
    """Pure kernel on known rows; it does not verify filesystem provenance."""
    baseline, candidate = indexed_rows(baseline_rows, nc), indexed_rows(candidate_rows, nc)
    if not baseline or set(baseline) != set(candidate):
        raise ValueError('Endpoints require the same nonempty image population')
    # Compare every GT array before matching anything; order is the object ID.
    for image in baseline:
        for field in ('canvas_shape','original_shape','gt_boxes','gt_classes'):
            if baseline[image].get(field) != candidate[image].get(field):
                raise ValueError('GT/canvas identity differs: ' + image + ': ' + field)
    metadata = metadata or {}
    total, groups = zero_counts(), {key: defaultdict(zero_counts) for key in ('class_id','scale','source_group','brightness_bin')}
    error_groups = {key: defaultdict(lambda: {'baseline':0,'candidate':0,'images':0})
                    for key in ('class_id','scale','source_group','brightness_bin')}
    rows, images = [], []
    all_class_ids = {str(c) for r in baseline.values() for c in map(class_id, r['gt_classes'])}
    all_class_ids |= {str(c) for r in list(baseline.values())+list(candidate.values()) for c in map(class_id,r['pred_classes'])}
    if nc is not None:
        all_class_ids |= {str(i) for i in range(nc)}
    # Prediction-class/scale FP rates use every image as their denominator.
    for c in sorted(all_class_ids):
        error_groups['class_id'][c]['images'] = len(baseline)
        groups['class_id'][c]
    for scale in SCALE_NAMES:
        error_groups['scale'][scale]['images'] = len(baseline)
        groups['scale'][scale]
    for image in sorted(baseline):
        a, b = baseline[image], candidate[image]
        ma, mb = match_image(a), match_image(b)
        meta = image_metadata(image, metadata)
        for key in ('source_group','brightness_bin'):
            error_groups[key][meta[key]]['images'] += 1
        for role, data, result in (('baseline',a,ma),('candidate',b,mb)):
            for p, error in result['prediction_errors'].items():
                if error != 'background':
                    continue
                keys = dict(class_id=str(class_id(data['pred_classes'][p])), scale=scale_bin(data['pred_boxes'][p]), **meta)
                for key,value in keys.items():
                    error_groups[key][value][role] += 1
        for g, ci in enumerate(a['gt_classes']):
            ac, bc = ma['objects'][g]['correct'], mb['objects'][g]['correct']
            add_object(total, ac, bc)
            keys = dict(class_id=str(class_id(ci)), scale=scale_bin(a['gt_boxes'][g]), **meta)
            for key,value in keys.items():
                add_object(groups[key][value], ac, bc)
            rows.append(dict(sample_key=json.dumps([image,g],ensure_ascii=False,separators=(',',':')),
                image=image,gt_index=g,gt_box=a['gt_boxes'][g],**keys,
                baseline=ma['objects'][g],candidate=mb['objects'][g],
                transition='stable_correct' if ac and bc else 'damaged' if ac else 'repaired' if bc else 'still_incorrect'))
        images.append(dict(image=image, **meta, gt_objects=len(a['gt_boxes']),
            baseline_prediction_counts=ma['prediction_counts'],candidate_prediction_counts=mb['prediction_counts']))
    result = dict(schema=SCHEMA, roster=sorted(baseline), summary=rates(total), objects=rows,images=images,
        object_groups={k:{v:rates(c) for v,c in sorted(table.items())} for k,table in groups.items()},
        background_groups={k:{v:dict(c, baseline_per_image=c['baseline']/c['images'],
            candidate_per_image=c['candidate']/c['images'],delta_per_image=(c['candidate']-c['baseline'])/c['images'])
            for v,c in sorted(table.items())} for k,table in error_groups.items()})
    for role in ('baseline','candidate'):
        background = sum(r[role+'_prediction_counts']['background'] for r in images)
        result[role] = dict(background_fp=background, images=len(images),background_fp_per_image=background/len(images),
            prediction_counts={key:sum(r[role+'_prediction_counts'][key] for r in images) for key in images[0][role+'_prediction_counts']})
    return result


def bound_equal(target, copies):
    content = Path(target).read_bytes()
    if not any(Path(p).is_file() and Path(p).read_bytes() == content for p in copies):
        raise ValueError('Input bytes not present in evaluation receipt snapshots: ' + str(target))


def load_evaluation(metric_path):
    """Accept only the new evaluator's receipt-bound dev metric/object pair."""
    import yaml
    metric_path = Path(metric_path).resolve()
    raw = read_json(metric_path)
    folder = metric_path.parent/'eval_evidence'
    receipt_path = folder/'run_receipt.json'
    receipt = read_json(receipt_path)
    if (raw.get('status') != 'completed' or raw.get('endpoint') != 'fixed_budget_last_ema'
            or raw.get('split') != 'val' or raw.get('official_test_accessed') is not False
            or receipt.get('terminal_status') != 'COMPLETED' or receipt.get('run_kind') != 'eval'
            or receipt.get('data_role') != 'development_val'):
        raise ValueError('A completed receipt-bound development endpoint is required')
    for key in ('dataset','seed'):
        if receipt.get(key) != raw.get(key):
            raise ValueError('Evaluation receipt identity mismatch: ' + key)
    for key in ('method_id','arm','source','checkpoint','endpoint'):
        if receipt.get('inputs',{}).get(key) != raw.get(key):
            raise ValueError('Evaluation input identity mismatch: ' + key)
    objects_path, contract_path = Path(raw['objects']), Path(raw['evaluation_contract'])
    metric_copies = [folder/p for p in receipt.get('metric_snapshots',[])]
    config_copies = [folder/p for p in receipt.get('source_snapshots',{}).get('config',[])]
    bound_equal(metric_path,metric_copies);bound_equal(objects_path,metric_copies);bound_equal(contract_path,config_copies)
    cfgs = []
    for path in config_copies:
        value = yaml.safe_load(path.read_text(encoding='utf-8'))
        if isinstance(value,dict) and 'model' in value and 'paths' in value:
            cfgs.append(value)
    if not cfgs or any(c != cfgs[0] for c in cfgs):
        raise ValueError('Missing or inconsistent bound effective training config')
    cfg = cfgs[0]
    if cfg.get('epochs') != 200 or cfg.get('imgsz') != 640:
        raise ValueError('Require the fixed E200/640 protocol')
    for key in ('dataset','seed','arm','source','method_id'):
        if cfg.get(key) != raw.get(key):
            raise ValueError('Bound config differs: ' + key)
    nc = cfg.get('expected_nc')
    if type(nc) is not int or nc <= 0:
        raise ValueError('Missing bound class inventory')
    contract = read_json(contract_path)
    roster = contract.get('roster')
    if (contract.get('schema') != 'rgbir-evaluation-contract-v1' or contract.get('official_test_accessed') is not False
            or contract.get('endpoint') != raw['endpoint'] or not isinstance(roster,list)
            or len(set(roster)) != len(roster) or len(roster) != POPULATIONS.get(raw.get('dataset'))
            or contract.get('observed_images') != len(roster)
            or contract.get('expected_val_images') != len(roster)):
        raise ValueError('Incomplete actual evaluator population contract')
    with gzip.open(objects_path,'rt',encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    indexed = indexed_rows(rows,nc)
    if set(indexed) != set(roster) or len(indexed) != len(roster):
        raise ValueError('Object rows differ from the bound full roster')
    return dict(raw=raw,contract=contract,config=cfg,rows=rows,nc=nc,
        files=[metric_path,objects_path,contract_path,receipt_path,*config_copies])


def load_metadata(path, roster, source_groups_tsv=None):
    """Read frozen labels only; never derive brightness/source from predictions."""
    metadata, classes, files, definition = {}, {}, [], None
    if path is not None:
        path = Path(path)
        raw = read_json(path)
        if raw.get('frozen') is not True or raw.get('key_type','image') not in ('image','stem'):
            raise ValueError('Metadata requires a frozen explicit image/stem mapping')
        mapping = raw.get('images',{})
        if not isinstance(mapping,dict):
            raise ValueError('Metadata images must be an explicit mapping')
        keys = [image if raw.get('key_type','image') == 'image' else PurePosixPath(image.replace('\\','/')).stem for image in roster]
        if len(set(keys)) != len(keys):
            raise ValueError('Ambiguous metadata stem keys')
        definition = raw.get('brightness_definition')
        for image,key in zip(roster,keys):
            row = mapping.get(key,{})
            if not isinstance(row,dict):
                raise ValueError('Metadata row must be an object')
            if row.get('brightness_bin') and not definition:
                raise ValueError('Brightness bins need their frozen proxy definition')
            metadata[image] = {k:row.get(k) or UNKNOWN for k in ('source_group','brightness_bin')}
        classes = {str(class_id(int(k))):str(v) for k,v in raw.get('class_names',{}).items()}
        files.append(path)
    if source_groups_tsv is not None:
        path = Path(source_groups_tsv)
        mapping = {}
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            if not line.strip():continue
            parts=line.split('\t')
            if len(parts)<2:raise ValueError('Source group TSV requires stem and group')
            if parts[0]=='stem' and parts[1] in ('source_group','sequence_prefix','group'):continue
            if parts[0] in mapping:raise ValueError('Duplicate source metadata stem')
            mapping[parts[0]]=parts[1] or UNKNOWN
        stems=[PurePosixPath(p.replace('\\','/')).stem for p in roster]
        if len(set(stems))!=len(stems):raise ValueError('Ambiguous source metadata stem keys')
        for image,stem in zip(roster,stems):
            metadata.setdefault(image,dict(source_group=UNKNOWN,brightness_bin=UNKNOWN))
            value=mapping.get(stem,UNKNOWN)
            old=metadata[image].get('source_group',UNKNOWN)
            if old!=UNKNOWN and value!=UNKNOWN and old!=value:raise ValueError('Frozen source mappings disagree')
            if value!=UNKNOWN:metadata[image]['source_group']=value
        files.append(path)
    return metadata,classes,definition,files


def verify_review(path):
    if path is None:return False
    receipt=read_json(path)
    if receipt.get('status')!='ACCEPTED' or receipt.get('schema')!='rgbir-object-error-review-v1':
        raise ValueError('Independent object analyzer review is not accepted')
    records=receipt.get('source_files',[])
    if {r.get('relative') for r in records} != set(REVIEW_FILES) or len(records)!=len(REVIEW_FILES):
        raise ValueError('Review must bind script, truth fixtures and frozen rules')
    for row in records:
        if (Path(__file__).parent/row['relative']).read_bytes()!=Path(row['accepted_copy']).read_bytes():
            raise ValueError('Object analyzer changed since independent review')
    return True


def run(args):
    accepted=verify_review(args.review_receipt)
    a,b=load_evaluation(args.baseline_evaluation),load_evaluation(args.candidate_evaluation)
    if a['raw']['arm'] not in ('N','weight0') or a['raw']['source']!='paired':
        raise ValueError('Baseline must be same-seed paired N/weight0')
    for key in ('seed','dataset','endpoint'):
        if a['raw'][key]!=b['raw'][key]:raise ValueError('Pair identity differs: '+key)
    if a['nc']!=b['nc']:raise ValueError('Class inventories differ')
    for key in ('roster','effective_kwargs','evaluator_identity'):
        if a['contract'].get(key)!=b['contract'].get(key):raise ValueError('Evaluation protocol differs: '+key)
    metadata,classes,brightness,meta_files=load_metadata(args.metadata,a['contract']['roster'],args.source_groups_tsv)
    result=analyze_pair(a['rows'],b['rows'],metadata,a['nc'])
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    inputs=[]
    for role,endpoint in (('baseline',a),('candidate',b)):
        for index,path in enumerate(endpoint['files']):
            dest=output/'input_evidence'/role/f'{index:03d}_{path.name}'
            dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes())
            inputs.append(dict(role=role,original=str(path.resolve()),copy=str(dest)))
    for index,path in enumerate(meta_files):
        dest=output/'input_evidence'/f'metadata_{index}_{path.name}';dest.write_bytes(path.read_bytes())
        inputs.append(dict(role='metadata',original=str(path.resolve()),copy=str(dest)))
    snapshot=output/'analysis_source';snapshot.mkdir()
    for name in REVIEW_FILES:shutil.copyfile(Path(__file__).parent/name,snapshot/name)
    contract_field='contract' if accepted else 'draft_contract'
    for role,endpoint in (('baseline',a),('candidate',b)):
        value=dict(schema=SCHEMA,analyzer_status='ACCEPTED' if accepted else 'NOT_ACCEPTED',
            checkpoint=endpoint['raw']['checkpoint'],seed=endpoint['raw']['seed'],roster=endpoint['contract']['roster'],
            dataset=endpoint['raw']['dataset'],arm=endpoint['raw']['arm'],source=endpoint['raw']['source'],
            **result[role],**{contract_field:ERROR_CONTRACT},
            localization_diagnostics_consistent=None,
            input_receipt=str(output/'analysis_receipt.json'),
            metric_scope='fixed threshold diagnosis, not AP or native best-F1 recall')
        write_json_new(output/(role+'_error_analysis.json'),value)
    for name in ('objects','images'):
        with gzip.open(output/(name+'.jsonl.gz'),'xt',encoding='utf-8') as stream:
            for row in result.pop(name):stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    result.update(analyzer_status='ACCEPTED' if accepted else 'NOT_ACCEPTED',**{contract_field:ERROR_CONTRACT},
        class_names={str(i):classes.get(str(i),UNKNOWN) for i in range(a['nc'])},
        brightness_definition=brightness,brightness_is_proxy_not_day_night=True,
        comparison_recipe_acceptance='deferred_to_accepted_independent_analyzer',stop_training=False)
    write_json_new(output/'pair_summary.json',result)
    write_json_new(output/'analysis_receipt.json',dict(schema='rgbir-object-error-receipt-v1',
        status='COMPLETED',analyzer_acceptance='ACCEPTED' if accepted else 'NOT_ACCEPTED',
        review_receipt=None if args.review_receipt is None else str(Path(args.review_receipt).resolve()),
        inputs=inputs,source_copies=[str(snapshot/name) for name in REVIEW_FILES],
        output_files=['pair_summary.json','baseline_error_analysis.json','candidate_error_analysis.json','objects.jsonl.gz','images.jsonl.gz'],
        inference_run=False,official_test_accessed=False,raw_inputs_overwritten=False))
    return dict(status='COMPLETED',analyzer_status=result['analyzer_status'],output=str(output))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-evaluation',type=Path,required=True)
    parser.add_argument('--candidate-evaluation',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--metadata',type=Path)
    parser.add_argument('--source-groups-tsv',type=Path)
    parser.add_argument('--review-receipt',type=Path)
    args=parser.parse_args();print(json.dumps(run(args),ensure_ascii=False))


if __name__=='__main__':main()
