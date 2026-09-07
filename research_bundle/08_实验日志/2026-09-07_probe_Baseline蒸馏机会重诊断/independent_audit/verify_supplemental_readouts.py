"""Independent CPU verification of existing readout artifacts; never refits."""
from pathlib import Path
import importlib.util
import json
import sys
import numpy as np

ROOT=Path(__file__).parent.parent
LOC=ROOT/'feature_review'/'localization_readout_v1'
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
loc=module('loc_readout',LOC/'localization_readout.py')
ana=module('baseline_analyzer',ROOT/'new_probe_analysis'/'analyze_baseline_probe.py')
result={'localization_known_truth_checks':loc.known_truth_tests(),'gpu_used':False,'hashes_computed':False,'datasets':{}}
for ds in ['dronevehicle','llvip']:
    source=ROOT/'remote_exports'/(ds+'_full_attempt1')
    rows=[json.loads(line) for line in (source/'objects.jsonl').open(encoding='utf8') if line.strip()]
    out=LOC/'outputs'/ds
    summ=json.loads((out/'summary.json').read_text(encoding='utf8'))
    with np.load(out/'predictions.npz',allow_pickle=False) as pred:
        selected=pred['cohort_source_row'];cohort=[rows[i] for i in selected]
        independent=[]
        for i,row in enumerate(rows):
            if row['is_background'] or row['paired_gt_iou'] is None or row['paired_gt_iou']<.5 or not row['anchor_has_reference_candidate']:continue
            box=np.asarray(row['gt_box_input']);center=np.asarray(row['anchor_center']);stride=row['anchor_stride']
            assert np.isfinite(box).all() and np.isfinite(center).all() and np.isfinite(stride) and stride>0
            d=np.r_[center-box[:2],box[2:]-center]/stride
            if np.all((d>=0)&(d<=14.99)):independent.append(i)
        assert np.array_equal(independent,selected)
        assert np.array_equal([row['object_id'] for row in cohort],pred['object_id'])
        assert np.array_equal([row['gt_box_input'] for row in cohort],pred['gt_box_input'])
        splits=pred['split'];donor=pred['shuffled_teacher_donor_cohort_index']
        assert np.array_equal(splits[donor],splits) and np.all(donor!=np.arange(len(donor))) and len(np.unique(donor))==len(donor)
        max_metric_error=0.
        for metric in summ['metrics']:
            use=splits==metric['split'];name=metric['arm']
            distance=pred[name+'_distance'][use];target=pred['target_ltrb_stride'][use]
            gt=pred['gt_box_input'][use];centers=pred['anchor_center'][use];strides=pred['anchor_stride'][use]
            xy=np.c_[centers-distance[:,:2]*strides[:,None],centers+distance[:,2:]*strides[:,None]]
            invalid=(distance<0).any(1)
            intersection=np.maximum(0,np.minimum(xy[:,2:],gt[:,2:])-np.maximum(xy[:,:2],gt[:,:2])).prod(1)
            area1=np.maximum(0,xy[:,2:]-xy[:,:2]).prod(1);area2=(gt[:,2:]-gt[:,:2]).prod(1)
            iou=intersection/(area1+area2-intersection);iou[invalid]=0
            assert np.allclose(iou,pred[name+'_iou'][use],atol=1e-12)
            error=max(abs(iou.mean()-metric['mean_iou_rgb_gt']),abs(((distance-target)**2).mean()-metric['four_edge_mse']))
            max_metric_error=max(max_metric_error,float(error));assert int(invalid.sum())==metric['invalid_negative_distance_n']
        native_box=loc.decode_distances(pred['native_N_dfl_expectation_distance'],pred['anchor_center'],pred['anchor_stride'])
        raw_box=np.asarray([row['N42']['same_anchor']['box'] for row in cohort])
        max_boxdiff=float(np.max(np.abs(native_box-raw_box)));assert max_boxdiff<.001
        with np.load(source/'logits.npz',allow_pickle=False) as logits:
            native=loc.native_expectation(logits['N42_dfl'][selected])
            assert np.array_equal(native,pred['native_N_dfl_expectation_distance'])
        loc_check={'cohort_n':len(cohort),'train_n':int((splits=='train').sum()),'val_n':int((splits=='val').sum()),
            'cohort_source_row_and_id_exact':True,'gt_boxes_exact':True,'donor_bijection_no_self_same_split':True,
            'all_saved_iou_mse_metrics_recomputed_max_difference':max_metric_error,'native_decoded_box_vs_export_fp32_max_difference_px':max_boxdiff,
            'native_dfl_expectation_exact':True}
    direct=json.loads((ROOT/'new_probe_analysis'/'direct_head_baseline_v1'/(ds+'_summary.json')).read_text(encoding='utf8'))
    train=np.asarray([r['split']=='train' for r in rows]);bg=np.asarray([r['is_background'] for r in rows]);y=np.asarray([r['class'] for r in rows]);groups,_=ana.group_columns(rows,train)
    with np.load(ROOT/'new_probe_analysis'/(ds+'_analysis_v1')/'probe_predictions.npz',allow_pickle=False) as cache:roi=cache['roi_common_valid'].copy();assert np.array_equal(cache['y'],y)
    direct_check={}
    with np.load(source/'logits.npz',allow_pickle=False) as logits:
        nc=logits['N42_cls'].shape[1]
        for model in direct['models']:
            z=logits[model+'_cls'].astype(float);p=z.argmax(1);p[z.max(1)<np.log(.25/.75)]=nc
            for cohort,mask in [('main_fixed_anchor',~train),('auxiliary_gt_roi_common_valid',~train&roi)]:
                recomputed=ana.plain(ana.evaluation_views(y,p,mask,nc+1,bg,groups))
                assert recomputed==direct['models'][model][cohort],(model,cohort)
            direct_check[model]={'rule_logit_threshold_log_one_third_equivalent':True,'all_metrics_exact':True,
                'foreground_macro_main':direct['models'][model]['main_fixed_anchor']['foreground']['balanced_accuracy']}
    result['datasets'][ds]={'localization':loc_check,'direct_head':direct_check}
(Path(__file__).parent/'supplemental_readout_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(result,ensure_ascii=False))
