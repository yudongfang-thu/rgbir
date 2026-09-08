"""Verify existing initial endpoint provenance, without recomputing AP or reading weights."""
import argparse,importlib.util,json,pathlib,sys
import yaml

HERE=pathlib.Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))

def build(root=HERE):
    r=read(root/'initial_evaluation_receipt.json');c=read(root/'initial_evaluation_contract.json')
    remote=root/'remote_source';prior_cfg=yaml.safe_load((remote/'requested_native_config.yaml').read_text(encoding='utf-8'))
    roster=(root/'development_roster.txt').read_text(encoding='utf-8').splitlines()
    assert r['status']=='INITIAL_BASELINE_EVALUATION_COMPLETED' and r['scope']=='INITIAL_BASELINE_NATIVE_RECHECK'
    assert r['metric_units']=='fraction_0_to_1' and r['dataset']=='llvip' and r['seed']==42
    assert r['checkpoint']==c['checkpoint']
    assert roster==c['roster'] and len(roster)==len(set(roster))==2406
    assert len(c['actual_loader_roster'])==2406 and set(c['actual_loader_roster'])==set(roster)
    for k in ('full_dev_images','observed_images'):assert r[k]==2406
    for k in ('full_dev_gt_objects','gt_objects_captured'):assert r[k]==7879
    assert c['gt_objects_before_inference']==c['gt_objects_captured']==7879
    assert c['actual_class_names']=={'0':'person'}
    assert (remote/'initial_evaluation_receipt.json').read_bytes()==(root/'initial_evaluation_receipt.json').read_bytes()
    actual_sources=read(remote/'source_manifest.json')['files']
    script=[p for p in actual_sources if pathlib.PurePosixPath(p['path']).name=='initial_baseline_eval.py']
    assert len(script)==1
    copied_script=remote/pathlib.PurePosixPath(script[0]['copy']).name
    assert copied_script.stat().st_size==script[0]['bytes'] and script[0]['byte_identity'] is True
    spec=importlib.util.spec_from_file_location('_initial_reference_saved_source',copied_script)
    source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)
    assert source.EFFECTIVE==r['actual_effective_kwargs']==c['effective_kwargs']
    cfgdir=root.parent/'trainer_release/configs'
    native=yaml.safe_load((cfgdir/'llvip_native_evaluation.yaml').read_text(encoding='utf-8'))
    assert (cfgdir/'llvip_native_evaluation.yaml').read_bytes()==(remote/'requested_native_config.yaml').read_bytes()
    raw_prior=read(root.parent/'trainer_release/validated_amp_prior.json')
    assert r['checkpoint']==raw_prior['initialization']['model']==raw_prior['initialization']['reference']
    for arm in ('N','L3-DFL','L3-GT'):
        cfg=yaml.safe_load((cfgdir/('llvip_'+arm+'_s42_FT3.yaml')).read_text(encoding='utf-8'))
        assert cfg['model']==cfg['reference']==r['checkpoint']['path']
        assert cfg['auxiliary_data_identity']['student_data_yaml']==r['actual_evaluation_data_yaml']
        for name in ('torch','ultralytics'):assert cfg[name+'_version']==c['evaluator_identity'][name]
    p=root.parent/'trainer_release/evaluate_object_dfl.py'
    spec=importlib.util.spec_from_file_location('_prospective_l3_evaluation',p)
    target=importlib.util.module_from_spec(spec);spec.loader.exec_module(target)
    assert target.EFFECTIVE==source.EFFECTIVE
    for key in ('AP50','AP75','mAP50_95','precision','recall'):
        assert type(r[key]) is float and 0<=r[key]<=1
    for key in ('AP50','AP75','mAP50_95'):assert r['per_class'][0][key]==r[key]
    return dict(status='PASS_EXISTING_INITIAL_REFERENCE_PROVENANCE',checkpoint=r['checkpoint'],
        metrics_fraction={k:r[k] for k in ('AP50','AP75','mAP50_95','precision','recall')},
        original_evaluation_status=r['status'],scope=r['scope'],endpoint=r['endpoint'],
        metric_units=r['metric_units'],full_dev_images=2406,full_dev_gt_objects=7879,
        evaluator_identity=c['evaluator_identity'],actual_effective_kwargs=r['actual_effective_kwargs'],
        data_yaml=r['actual_evaluation_data_yaml'],actual_source_file=script[0],
        checks=dict(remote_and_local_receipt_byte_exact=True,source_manifest_snapshot_size_exact=True,
            canonical_roster_exact=True,actual_loader_roster_set_exact=True,
            receipt_contract_checkpoint_stat_exact=True,prior_DFL_forward_initialization_stat_exact=True,
            new_three_configs_initial_checkpoint_paths_exact=True,new_three_configs_versions_exact=True,
            new_evaluator_effective_kwargs_exact=True,new_native_projection_config_byte_exact=True),
        new_three_arm_results_read=False,new_AP_recomputed=False,new_weight_read=False,new_GPU=False,new_hash_computed=False,
        limits=['Initial readout has its own scope and accepted_endpoint_claim=false; this provenance check is not a new metric acceptance.',
            'New arm completion/init/checkpoint binding and actual evaluation profile/roster must still match before subtracting.',
            'This run did not re-open weights or recompute native AP from dense predictions.'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=pathlib.Path,required=True);args=p.parse_args()
    result=build()
    with args.output.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(result['status'],result['metrics_fraction'])
