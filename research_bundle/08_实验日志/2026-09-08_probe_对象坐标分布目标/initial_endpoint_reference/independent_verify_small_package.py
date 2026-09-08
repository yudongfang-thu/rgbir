"""Independent bounded provenance replay, no evaluator import/weights/remote access."""
import ast,json,math
from pathlib import Path
import yaml

HERE=Path(__file__).absolute().parent
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def effective(path):
    tree=ast.parse(Path(path).read_text(encoding='utf-8-sig'))
    nodes=[n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='EFFECTIVE' for t in n.targets)]
    assert len(nodes)==1
    value=nodes[0];assert isinstance(value,ast.Call) and isinstance(value.func,ast.Name) and value.func.id=='dict' and not value.args
    return {k.arg:ast.literal_eval(k.value) for k in value.keywords}

def run():
    r=read(HERE/'initial_evaluation_receipt.json');c=read(HERE/'initial_evaluation_contract.json')
    old=HERE.parents[1]/'2026-09-08_probe_快速方向筛选/initial_llvip_evidence'
    for name in ('initial_evaluation_receipt.json','initial_evaluation_contract.json','development_roster.txt'):
        assert (HERE/name).read_bytes()==(old/name).read_bytes()
    assert (HERE/'initial_evaluation_receipt.json').read_bytes()==(HERE/'remote_source/initial_evaluation_receipt.json').read_bytes()
    assert r['status']=='INITIAL_BASELINE_EVALUATION_COMPLETED' and r['scope']=='INITIAL_BASELINE_NATIVE_RECHECK'
    assert r['endpoint']=='INITIAL_FIXED_LAST_EMA_RECHECK' and r['dataset']=='llvip' and r['seed']==42
    assert r['accepted_endpoint_claim'] is False and r['metric_units']=='fraction_0_to_1'
    assert r['checkpoint']==c['checkpoint']
    assert r['full_dev_images']==r['observed_images']==2406 and r['full_dev_gt_objects']==r['gt_objects_captured']==7879
    assert c['gt_objects_before_inference']==c['gt_objects_captured']==7879
    roster=(HERE/'development_roster.txt').read_text(encoding='utf-8').splitlines()
    assert roster==c['roster'] and len(roster)==len(set(roster))==2406
    assert len(c['actual_loader_roster'])==2406 and set(roster)==set(c['actual_loader_roster'])
    assert c['actual_class_names']=={'0':'person'}
    assert len(r['per_class'])==1 and r['per_class'][0]['class_id']==0 and r['per_class'][0]['name']=='person'
    metrics={k:r[k] for k in ('mAP50_95','AP50','AP75','precision','recall')}
    assert all(type(v) is float and math.isfinite(v) and 0<=v<=1 for v in metrics.values())
    for k in ('mAP50_95','AP50','AP75'):assert r['per_class'][0][k]==r[k]
    release=HERE.parent/'trainer_release'
    prior=read(release/'validated_amp_prior.json')
    assert prior['initialization']['model']==prior['initialization']['reference']==r['checkpoint']
    for arm in ('N','L3-DFL','L3-GT'):
        cfg=yaml.safe_load((release/'configs'/('llvip_'+arm+'_s42_FT3.yaml')).read_text(encoding='utf-8'))
        assert cfg['model']==cfg['reference']==r['checkpoint']['path']
        assert cfg['auxiliary_data_identity']['student_data_yaml']==r['actual_evaluation_data_yaml']
        for name in ('torch','ultralytics'):assert cfg[name+'_version']==c['evaluator_identity'][name]
    assert (release/'configs/llvip_native_evaluation.yaml').read_bytes()==(HERE/'remote_source/requested_native_config.yaml').read_bytes()
    assert effective(release/'evaluate_object_dfl.py')==effective(HERE/'remote_source/0000_initial_baseline_eval.py')==r['actual_effective_kwargs']==c['effective_kwargs']
    result=dict(status='PASS_AS_PROVENANCE_BOUND_INITIAL_REFERENCE',auditor='/root/ap_error',
        scope='Independent small-package provenance only; no AP recomputation',metrics_fraction=metrics,
        checkpoint=r['checkpoint'],full_dev_images=2406,full_dev_gt_objects=7879,
        checks=dict(original_local_copies_byte_exact=True,remote_receipt_copy_byte_exact=True,
            receipt_contract_and_prior_DFL_init_stat_exact=True,canonical_roster_exact=True,actual_roster_set_exact=True,
            all_three_config_init_paths_and_versions_exact=True,native_projection_config_byte_exact=True,
            old_actual_profile_new_evaluator_constant_exact=True),
        original_scope=r['scope'],original_accepted_endpoint_claim=r['accepted_endpoint_claim'],
        current_three_arm_execution_initialization_verified=False,current_three_arm_actual_evaluation_verified=False,
        can_replace_matched_N=False,new_GPU=False,new_forward=False,new_weight_read=False,new_AP_recomputed=False,new_hash_computed=False)
    with (HERE/'INDEPENDENT_REFERENCE_REVIEW.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,ensure_ascii=False)
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':run()
