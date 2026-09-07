"""Finalize independent real-flow audit using original governance TSV and persisted CPU results."""
import collections,csv,json,math,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;TASK=HERE.parent;WORKSPACE=HERE.parents[3]
RAW=TASK/'remote_completed_attempt2';GOV=TASK/'governance_group_sources'
OLD=WORKSPACE/'08_实验日志/2026-09-07_train_IndependentKD实施/remote_cpu1'
META=('seed','generator_seed','worker_init','shuffle','replacement','drop_last','prefetch_factor','workers','batch_size',
      'dataset_size','dataset','yaml_sources','paired_mapping','train_manifests','augmentation','test_accessed')
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s]
def eq(a,b,message):
    if a!=b:raise AssertionError(message)
def save(p,obj):
    with p.open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
def concentration(values):
    hist=collections.Counter(values);ranked=sorted(hist.items(),key=lambda pair:(-pair[1],pair[0]));n=sum(hist.values())
    return dict(objects=n,unique_units=len(hist),top1_objects=sum(v for _,v in ranked[:1]),top5_objects=sum(v for _,v in ranked[:5]),
        top1_fraction=sum(v for _,v in ranked[:1])/n if n else None,top5_fraction=sum(v for _,v in ranked[:5])/n if n else None)
def main():
    cpu=read(HERE/'real_results_v1/result.json');eq(cpu['status'],'PASS','Full CPU recomputation failed')
    collection=read(TASK/'collection_receipt.json');eq(collection['status'],'COLLECTED_BYTE_EXACT','collection exact')
    for item in collection['records']:
        local=RAW/item['relative_path']
        access=Path('\\\\?\\'+str(local)) if sys.platform=='win32' else local
        eq(access.stat().st_size,item['bytes'],'collection local file size')
    govreceipt=read(GOV/'receipt.json');eq(govreceipt['new_hashes'],False,'governance nohash')
    reports=[];overviews=[]
    for key in ('llvip','drone'):
        path=GOV/(key+'_0.tsv');origin=govreceipt['datasets'][key]['files'][0]
        eq(path.stat().st_size,origin['bytes'],'governance source size')
        if key=='llvip':
            with path.open(encoding='utf-8',newline='') as f:pairs=[(r['stem'],r['sequence_prefix']) for r in csv.DictReader(f,delimiter='\t')]
        else:pairs=[tuple(line.split('\t')[:2]) for line in path.read_text(encoding='utf-8').splitlines()]
        mapping=dict(pairs);eq(len(pairs),len(mapping),'governance duplicate stems')
        full=RAW/(key+'_full_attempt1');canary=RAW/(key+'_canary_attempt1')
        selections=rows(full/'selection_batches.jsonl');trace=rows(full/'natural_batches.jsonl');summary=read(full/'summary.json')
        eq(rows(canary/'selection_batches.jsonl'),selections[:2],'canary/full first-two selection exact')
        original=read(OLD/('coverage_'+key+'_attempt1')/'coverage_receipt.json')
        for field in META:eq(summary['loader_metadata'][field],original[field],'original loader metadata '+field)
        natural=[];groups=[];hist=collections.defaultdict(list)
        for entry,t in zip(selections,trace):
            for im,tr in zip(entry['images'],t['images']):
                stem=Path(im['rgb_source']).stem
                if stem not in mapping:raise AssertionError('Missing original governed stem '+stem)
                eq(im['governed_group'],mapping[stem],'original TSV group')
                natural.append(im['rgb_source']);groups.append(mapping[stem])
            for family,field in [('C','classification'),('L','localization')]:
                for r in entry[field]['base_records']:
                    if r['selected']:
                        im=entry['images'][r['batch_index']]
                        hist[(family,'image')].append(im['rgb_source']);hist[(family,'group')].append(im['governed_group'])
        eq(len(natural),2048,'natural count');eq(len(set(natural)),2048,'without replacement unique count')
        reports.append(dict(dataset=key,governance_source=origin,governance_rows=len(mapping),exact_mapped_natural_images=2048,
            unique_natural_groups=len(set(groups)),canary_full_first_two_selection_exact=True,all_loader_metadata_exact=True,
            concentrations={family+'_'+unit:concentration(v) for (family,unit),v in hist.items()}))
        cr=summary['classification'];lr=summary['localization'];run=next(r for r in cpu['runs'] if r['dataset']==key and r['batches']==64)
        overviews.append(dict(dataset=key,RGB_GT=lr['counts']['rgb_gt_count'],C_base=cr['counts']['base_count'],C_eligible=cr['counts']['eligible_count'],
            C_selected=cr['counts']['selected_count'],C_selected_batches=cr['selected_batches'],C_selected_images=cr['selected_unique_images'],
            C_selected_groups=len(cr['selected_source_groups']),L_base=lr['counts']['base_count'],L_selected=lr['counts']['selected_count'],
            L_selected_batches=lr['selected_batches'],L_selected_images=lr['each_gate']['selected_count']['unique_images'],
            L_selected_groups=len(lr['each_gate']['selected_count']['source_groups']),C_classes=run['C_class_counts'],L_classes=run['L_class_counts']))
    save(HERE/'real_results_v1/governance_and_metadata_review.json',dict(status='PASS',datasets=reports,overviews=overviews,
        collection_files=collection['files'],gpu_or_ssh_used=False,new_hash_computed=False))
    receipt=dict(date='2026-09-07',auditor='/root/ap_error',executor='/root/baseline_feature_analysis',runtime_executor='/root',
        status='PASS_FOR_OBSERVED_FIXED_FLOW_DIAGNOSTIC',overall_verdict='pass',integrity_status='pass',
        reason_code='REAL_2_AND_64_BATCHES_BOTH_DATASETS_CPU_RECOMPUTATION_AND_GOVERNANCE_PASS',
        tests=['real_results_v1/result.json','real_results_v1/governance_and_metadata_review.json'],
        audited_input_hashes=[],new_hash_computed=False,hash_omission_reason='User no-hash instruction; source bytes/stat and root collection byte-identity receipt used.',
        training_admitted=False,calibration_admitted=False,geometry_verified=False,all_anchor_predictions_recomputed=False,
        datasets=overviews,
        checks={
          'gt_provenance':dict(status='pass',details='Both real 64x32 streams equal old recorded trace exactly; original governance TSV maps all4096 unique full-run image entries; metadata and globalGT/class identities verified.',evidence='real_results_v1/governance_and_metadata_review.json'),
          'score_normalization':dict(status='pass',details='C every-batch stable-q/ceil quota and preteacher-base denominator recomputed. L original whole-batch denominator, perimage/batch/summary counters and postbase gates recomputed.',evidence='real_results_v1/result.json'),
          'result_existence':dict(status='pass',details='Both datasets have real2+64 successful runs, source bytes match, canary first2 selections exactly equal full first2. Root129-file byte-identical collection receipt checked.',evidence='../collection_receipt.json'),
          'dead_code':dict(status='pass',details='All132 real batch records visited. GT-only common/pair/geometry/inside/support/owner gates recomputed from trace in CPU source function; actual source/summaries show original calls with replay assertions.',evidence='review_real_natural.py'),
          'scope':dict(status='pass',details='Unverified geometry only; no optimizer/backward/calibration/test. Four lease profiles complete without monitor errors; measured freeGPU>=2048MiB/projectRSS<=300GiB. FourGPU exception admission leaves>=2empty.',evidence='real_results_v1/result.json:resource_review'),
          'eval_type':dict(status='pass',details='real_gt train-label selection diagnostic. Independent CPU gate reconstruction uses real labels with synthetic no-candidate scores only to isolate GT gates; no fresh model prediction or AP.',evidence='REAL_RESULTS_REVIEW_SCOPE.md')},
        claims=[
          dict(id='fixed_natural_flow_selection_observed',impact='supported',evidence='real_results_v1',rationale='Both actual64batch populations, all labels/source groups and record-derived selection stats verified.'),
          dict(id='full_model_prediction_independent_replay',impact='unsupported',evidence='No all-anchor raw logits cache',rationale='Reference candidate existence/confidence rank and Cq source values remain dependent on original forward artifacts; no independent reinference.'),
          dict(id='geometry_or_learning_or_KD_gain',impact='unsupported',evidence='UNVERIFIED_GEOMETRY_DIAGNOSTIC',rationale='Selected nonzero object/batch counts are not gradients, calibration, new-student learning, or attainable KD gain.')])
    save(HERE/'REAL_RESULTS_EXPERIMENT_AUDIT.json',receipt)
    print(json.dumps(dict(status=receipt['status'],datasets=overviews),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
