"""CPU-only six-job candidate preparation. This script never starts a process."""
import argparse
import math
from pathlib import Path, PurePosixPath
import yaml
from legacy_checkpoint_evaluate import ARMS,derived_config,frozen_module,load_legacy_metadata,read,write_new,copy_new

ORDER=(42,0,123)


def measured_resources(profile):
    if (profile.get('schema')!='rgbir-independent-resource-profile-v1' or profile.get('status')!='COMPLETED'
        or profile.get('measurement_valid') is not True or profile.get('stage')!='evaluation_profile'):
        raise ValueError('Successful actual evaluation profile required; bootstrap cap is not measurement')
    resources=profile['resources']
    vram=max(resources.get('per_gpu_peak_vram_mib',{}).values(),default=0)
    rss=resources.get('peak_rss_mib',0)
    if any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in (vram,rss)):
        raise ValueError('Actual NVML and complete process-tree RSS measurements required')
    if vram>profile['reservation']['vram_mib'] or rss>profile['reservation']['rss_mib']:
        raise ValueError('Actual measured profile exceeded its reservation')
    return dict(vram_mib=math.ceil(vram)+256,rss_mib=max(8192,math.ceil(rss)+4096))


def prepare(args):
    if args.output.exists():raise FileExistsError('Preserve prior queue candidates')
    analyzer=frozen_module(args.analyzer_root,'analyze_independent')
    profile=read(args.resource_profile);resources=measured_resources(profile)
    manifest=read(args.old_manifest);selected=[]
    for spec in manifest['runs']:
        if spec['arm'] not in ('N','C','C0'):continue
        arm='N' if spec['arm']=='N' else 'C0'
        run=Path(spec['path']);train_receipt=read(run/'run_evidence/run_receipt.json')
        cfg,_=analyzer._bound_configs(run,train_receipt,'run_evidence')
        derived=derived_config(cfg,arm,spec['seed'])
        metadata=load_legacy_metadata(run,arm,spec['seed'],derived,analyzer)
        identity=dict({key:derived[key] for key in ('dataset','model','expected_nc','imgsz','batch','workers','torch_version','ultralytics_version')},
            student_data_yaml=derived['paths']['student_data_yaml'],split='val',expected_val_images=1469)
        if identity!=profile['profile_identity']['configuration']:raise ValueError('Old recipe differs from measured evaluator identity')
        selected.append((arm,spec['seed'],derived,metadata,run))
    if len(selected)!=6 or {(r[0],r[1]) for r in selected}!={(a,s) for a in ARMS for s in ORDER}:
        raise ValueError('Require exactly six legacy N/C0 seed42/0/123 endpoints')
    args.output.mkdir(parents=True,exist_ok=False);(args.output/'configs').mkdir()
    jobs=[];entries=[]
    for arm,seed,cfg,metadata,local_run in sorted(selected,key=lambda row:(ORDER.index(row[1]),list(ARMS).index(row[0]))):
        label=arm+'_s'+str(seed)
        cfg_name=label+'.yaml'
        with (args.output/'configs'/cfg_name).open('x',encoding='utf-8') as stream:yaml.safe_dump(cfg,stream,allow_unicode=True,sort_keys=False)
        checkpoint=metadata['completion']['checkpoint']
        old_run=str(PurePosixPath(checkpoint).parent.parent)
        remote_cfg=str(PurePosixPath(args.remote_code_dir)/'candidate_bundle_v1/configs'/cfg_name)
        output=str(PurePosixPath(args.remote_artifacts)/(label+'_attempt1'))
        command=[args.python,str(PurePosixPath(args.remote_code_dir)/'legacy_checkpoint_evaluate.py'),
            '--release',args.release,'--legacy-run',old_run,'--config',remote_cfg,'--output',output,
            '--evaluation-profile',args.remote_resource_profile,
            '--review-receipt',str(PurePosixPath(args.remote_code_dir)/'review_receipt.json'),
            '--arm',arm,'--seed',str(seed)]
        job=dict(id='ikdv2_legacy_diag_'+label+'_a1',kind='eval',stage='evaluation',formal=False,
            queue_owner='independent_v2',config=remote_cfg,profile_key=profile['profile_key'],
            requires_profile=args.remote_resource_profile,result_receipt=output+'/reevaluation_receipt.json',
            command=command,priority=len(jobs),**resources)
        jobs.append(job)
        entries.append(dict(normalized_method_arm=arm,original_arm=metadata['metric']['arm'],seed=seed,
            original_run=old_run,original_checkpoint=checkpoint,original_completion=old_run+'/completion_receipt.json',
            original_training_receipt=old_run+'/run_evidence/run_receipt.json',
            local_original_snapshot=str(local_run.resolve()),new_evaluation_attempt=output,
            new_metric=output+'/evaluation_val.json',new_objects=output+'/predictions/objects.jsonl.gz'))
    value=dict(schema='rgbir-legacy-six-endpoint-queue-candidate-v1',status='PREPARED_NOT_RUN_NOT_REVIEWED',
        jobs=jobs,entries=entries,requires_independent_wrapper_review=True,source_release=args.release,
        resource_profile=args.remote_resource_profile,queue_behavior='serial evaluations through frozen shared dispatcher; dynamic GPU',
        old_results_modified=False,new_training_receipts_created=False,official_test_accessed=False)
    write_new(args.output/'queue_candidate.json',value)
    copy_new(args.resource_profile,args.output/'actual_resource_profile_input.json')
    copy_new(args.old_manifest,args.output/'original_endpoint_manifest_input.json')
    write_new(args.output/'preparation_receipt.json',dict(status='CPU_PREPARED_NOT_RUN_NOT_REVIEWED',jobs=len(jobs),
        reservations=resources,metadata_validation='six actual legacy snapshots passed',new_gpu_jobs=0,
        all_outputs_new=True,official_test_accessed=False))
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('old-manifest','analyzer-root','resource-profile','output'):parser.add_argument('--'+name,type=Path,required=True)
    for name in ('remote-code-dir','remote-artifacts','release','remote-resource-profile','python'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args();value=prepare(args);print('CPU_PREPARED_NOT_RUN_NOT_REVIEWED: '+str(len(value['jobs']))+' jobs')


if __name__=='__main__':main()
