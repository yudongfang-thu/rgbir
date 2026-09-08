"""Read-only collected capture/population/metric/resource review; no Torch or model."""
from collections import Counter
import gzip
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import yaml

ENTRY=Path(__file__).resolve().parents[1]
E=ENTRY/'evidence_1401'
N=ENTRY.parent/'2026-09-07_train_IndependentKD实施/legacy_diagnostics_snapshot_20260907_161459/N_s42_attempt1'
NAMES=['car','freight car','truck','bus','van']
KW=dict(imgsz=640,batch=32,workers=4,quantize=None,conf=.001,iou=.7,max_det=300,agnostic_nms=False,single_cls=False,rect=True,augment=False,half=False)
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def lines(p):
    with gzip.open(p,'rt',encoding='utf-8') as f:return [json.loads(x) for x in f if x.strip()]
def stat(p):s=p.stat();return dict(path=str(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns)
def check_metrics(m,ids):
    assert [r['class_id'] for r in m['per_class']]==ids
    for k in ('AP50','AP75','mAP50_95','precision','recall'):assert math.isfinite(m[k]) and 0<=m[k]<=1
    for r in m['per_class']:
        assert r['name']==NAMES[r['class_id']]
        for k in ('AP50','AP75','mAP50_95'):assert math.isfinite(r[k]) and 0<=r[k]<=1
    return {k:abs(m[k]-statistics.mean(r[k] for r in m['per_class'])) for k in ('AP50','AP75','mAP50_95')}

def main():
    spec=read(ENTRY/'release/drone_teacher_spec.json')['models']['T42'];by_alias=dict(zip(spec['aliases'],spec['canonical']))
    assert len(spec['aliases'])==len(set(spec['aliases']))==len(spec['canonical'])==len(set(spec['canonical']))==1469
    assert len(spec['label_counts'])==1469 and sum(spec['label_counts'])==24490
    counts_by_alias=dict(zip(spec['aliases'],spec['label_counts']));reports={};captures={};contracts={}
    for name,canary,images,gt,classes in [('T42_canary_attempt1',True,32,514,[447,24,30,13,0]),('T42_full_attempt1',False,1469,24490,[20588,918,1470,789,725])]:
        p=E/name;s=read(p/'summary.json');pop=read(p/'population.json');con=read(p/'capture_contract.json');m=read(p/'capture_metrics.json');identity=read(p/'model_identity.json')
        assert (p/'input_spec.json').read_bytes()==(ENTRY/'release/drone_teacher_spec.json').read_bytes()
        assert s['status']=='completed' and s['scope']=='DRONE_IR42_FULL_DEV_CAPTURE' and s['dataset']=='dronevehicle' and s['model']=='T42'
        assert s['canary'] is canary and s['native_capture_exact'] is canary
        assert s['images']==images and s['gt_count']==gt and s['full_population']==1469 and s['full_gt']==24490
        assert s['metric_units']=='fraction_0_to_1' and s['metrics']==m and s['class_names']==NAMES and s['class_counts']==classes
        assert s['checkpoint']==spec['checkpoint'] and s['input_data']==spec['data']
        assert not s['official_test_accessed'] and not s['training_modified'] and not s['new_hash_computed'] and s['optimizer_updates']==0
        assert con['effective_kwargs']==KW and con['scope']==s['scope'] and con['loader_per_image_labels_exact']
        assert con['expected_val_images']==con['observed_images']==images and con['loader_gt']==con['expected_gt']==con['captured_gt']==gt
        assert con['evaluator_identity']==dict(native='ultralytics.DetectionValidator',torch='2.10.0+cu128',ultralytics='8.4.115',extension='read_only_post_metric_object_capture_v1')
        actual_aliases=spec['aliases'][:32] if canary else spec['aliases']
        canonical=[by_alias[a] for a in actual_aliases]
        assert pop['evaluated_aliases']==actual_aliases and pop['evaluated_roster']==canonical and con['roster']==canonical
        assert set(con['actual_loader_roster'])==set(canonical) and len(con['actual_loader_roster'])==images
        assert set(con['alias_to_canonical'])==set(actual_aliases) and con['alias_to_canonical']=={a:by_alias[a] for a in actual_aliases}
        assert (p/'evaluation_roster.txt').read_text().splitlines()==actual_aliases
        assert pop['full_gt']==24490 and pop['full_class_counts']==spec['class_counts'] and pop['empty_gt_images']==2
        assert pop['class_counts']==classes and pop['expected_evaluated_gt']==gt and pop['original_label_counts']==spec['label_counts']
        expected_ids=[i for i,n in enumerate(classes) if n];assert pop['expected_metric_class_ids']==expected_ids and con['expected_metric_class_ids']==expected_ids
        errors=check_metrics(m,expected_ids);assert max(errors.values())<1e-14
        args=yaml.safe_load((p/'original_args.yaml').read_text());assert args['epochs']==200 and args['seed']==42
        assert args['data']==spec['data']
        for value in identity.values():
            assert value['path']==spec['checkpoint'] and value['stat']==spec['checkpoint_stat'] and value['checkpoint_epoch'] in (-1,199)
            assert {int(k):v for k,v in value['names'].items()}==dict(enumerate(NAMES))
            assert value['training_args']==args
        if canary:
            nc=read(p/'native_contract.json');assert read(p/'native_metrics.json')==m
            assert nc['effective_kwargs']==con['effective_kwargs'] and nc['actual_loader_roster']==con['actual_loader_roster']
            assert nc['loader_per_image_labels_exact'] and nc['loader_gt']==514 and nc['observed_images']==32
            assert identity['native']==identity['capture']
        rows=lines(p/'capture/objects.jsonl.gz');assert len(rows)==images and len({r['image'] for r in rows})==images
        assert set(r['image'] for r in rows)==set(actual_aliases)
        classes_found=Counter();prediction_n=0;outside=0
        for r in rows:
            assert len(r['gt_boxes'])==len(r['gt_classes'])==counts_by_alias[r['image']]
            assert len(r['pred_boxes'])==len(r['pred_classes'])==len(r['pred_confidence'])<=300
            assert r['original_shape']==[512,640] and r['canvas_shape']==[544,672]
            classes_found.update(int(c) for c in r['gt_classes'])
            for prefix in ('gt','pred'):
                for box,cl in zip(r[prefix+'_boxes'],r[prefix+'_classes']):
                    assert len(box)==4 and all(math.isfinite(x) for x in box) and cl==int(cl) and 0<=cl<5
                    assert box[2]>box[0] and box[3]>box[1]
                    if prefix=='pred' and (box[0]<0 or box[1]<0 or box[2]>672 or box[3]>544):outside+=1
            assert all(math.isfinite(v) and .001<v<=1 for v in r['pred_confidence'])
            prediction_n+=len(r['pred_boxes'])
        assert [classes_found[i] for i in range(5)]==classes and sum(len(r['gt_boxes']) for r in rows)==gt
        captures[name]={r['image']:r for r in rows};contracts[name]=con
        reports[name]=dict(images=images,GT=gt,class_counts=classes,empty_GT_images=sum(not r['gt_boxes'] for r in rows),prediction_rows=prediction_n,outside_canvas_predictions_kept=outside,AP_macro_mean_max_error=max(errors.values()),native_capture_dual_path_exact=canary,seconds=s['seconds'])
    full=captures['T42_full_attempt1']
    for a,r in captures['T42_canary_attempt1'].items():
        assert all(r[k]==full[a][k] for k in ('gt_boxes','gt_classes','canvas_shape','original_shape'))
    nc=read(N/'evaluation_contract.json');nm=read(N/'evaluation_val.json');nr=lines(N/'predictions/objects.jsonl.gz')
    assert nc['effective_kwargs']==KW and nc['evaluator_identity']==contracts['T42_full_attempt1']['evaluator_identity']
    assert nm['dataset']=='dronevehicle' and nm['seed']==42 and nm['metric_units']=='fraction_0_to_1'
    assert [p['name'] for p in nm['per_class']]==NAMES and max(check_metrics(nm,list(range(5))).values())<1e-14
    assert len(nr)==1469 and sum(len(r['gt_boxes']) for r in nr)==22462
    assert len({PurePosixPath(r['image']).stem for r in nr})==1469
    n_by_stem={PurePosixPath(r['image']).stem:r for r in nr};t_by_stem={PurePosixPath(r['image']).stem:r for r in full.values()}
    assert set(n_by_stem)==set(t_by_stem) and len(t_by_stem)==1469
    assert all(n_by_stem[k]['canvas_shape']==t_by_stem[k]['canvas_shape'] and n_by_stem[k]['original_shape']==t_by_stem[k]['original_shape'] for k in n_by_stem)
    q=E/'queue';completion=read(q/'completion.json');admission=read(q/'canary_acceptance.json')
    assert completion['status']=='DRONE_IR42_FULL_CAPTURE_COMPLETED' and completion['images']==1469 and completion['gt_count']==24490
    assert admission['status']=='CANARY_ACCEPTED' and admission['vram_measured_mib']==1370 and admission['reservation_vram_mib']==1882
    resources={}
    for phase,folder in [('canary','T42_canary_attempt1'),('full','T42_full_attempt1')]:
        profile=read(q/('drone_teacher_attempt1_'+phase+'_resource_profile.json'));a=read(q/('drone_teacher_attempt1_'+phase+'_admission.json'));job=read(q/(phase+'_job.json'));s=read(E/folder/'summary.json')
        assert profile['status']=='COMPLETED' and profile['exit_code']==0 and profile['monitor_errors']==[]
        assert profile['launch']['gpu_ids']==[4] and profile['launch']['cuda_processes_per_gpu']==1
        assert a['active_gpus_after']==[2,4,5] and not a['four_gpu_exception']
        assert min(x['memory_free_mib'] for x in profile['samples'])==profile['minimum_free_mib']>=2048
        assert all(x['memory_used_mib']/x['memory_total_mib']<.70 and x['project_rss_mib']*2**20<300_000_000_000 for x in profile['samples'])
        assert max(s['resources']['per_gpu_peak_vram_mib'].values())<=job['vram_mib'] and s['resources']['peak_rss_mib']<=job['rss_mib']
        assert job['rss_mib']==profile['launch']['expected_rss_mib']==12288
        assert job['vram_mib']==profile['launch']['expected_vram_mib']
        resources[phase]=dict(minimum_free_mib=profile['minimum_free_mib'],max_sampled_project_rss_mib=max(x['project_rss_mib'] for x in profile['samples']),task_vram_peak_mib=max(s['resources']['per_gpu_peak_vram_mib'].values()),task_rss_peak_mib=s['resources']['peak_rss_mib'],reserved_vram_mib=job['vram_mib'],reserved_rss_mib=job['rss_mib'])
    result=dict(status='ACCEPTED_ACTUAL_DRONE_IR42_CAPTURE',populations=reports,canary_full_shared_GT_exact=True,N_T_native_kwargs_classes_canvas_roster_stems_exact=True,N_RGB_GT=22462,T_IR_GT=24490,GT_arrays_not_assumed_equal=True,N_checkpoint=nm['checkpoint'],T_checkpoint=spec['checkpoint'],same_training_protocol_claim=False,crossmodal_GT_pairing_not_yet_accepted=True,full_AP50_fraction=read(E/'T42_full_attempt1/summary.json')['metrics']['AP50'],full_AP75_fraction=read(E/'T42_full_attempt1/summary.json')['metrics']['AP75'],full_mAP_fraction=read(E/'T42_full_attempt1/summary.json')['metrics']['mAP50_95'],queue_seconds=completion['seconds'],resources=resources,resource_evidence_scope='existing lease guard completed without monitor errors plus retained samples; not exhaustive independent continuous sampling',new_GPU=False,new_model_forward=False,new_NMS=False,new_hash=False,checkpoint_loaded=False,inputs=[stat(E/'collection_receipt.json'),stat(E/'T42_full_attempt1/capture/objects.jsonl.gz'),stat(E/'T42_canary_attempt1/capture/objects.jsonl.gz')])
    with (ENTRY/'independent_review/ACTUAL_CAPTURE_RECEIPT.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ('inputs','N_checkpoint','T_checkpoint')},ensure_ascii=False))
if __name__=='__main__':main()
