"""One sequential N->eval->C0->eval->C1->eval queue on existing run_job.

No new scheduler/lease pool, resume, AP adaptation, automatic admission or hash.
Preparation only until root supplies actual admissions and invokes this entry.
"""
import argparse
import math
from pathlib import Path
import shutil
import sys
import time
import traceback

from screen_common import ENDPOINT,data_output,load_config,read,require_admission,stat,write_new

ARMS=('N','C0','C1')
HERE=Path(__file__).resolve().parent


def jobs_for(args):
    jobs=[]
    for arm in ARMS:
        config=args.config_dir/('drone_'+arm+'_s42_E20_DRAFT.yaml')
        admission=args.admissions/(arm+'.json')
        run=args.campaign_root/'runs'/arm
        evaluation=args.campaign_root/'evaluations'/arm
        jobs.append(dict(id='short_'+arm+'_s42_E20_train',arm=arm,stage='train',kind='train',formal=False,
            vram_mib=args.train_vram_mib,rss_mib=args.train_rss_mib,
            command=[str(args.python),str(HERE/'train_short_screen.py'),'--reference-dir',str(args.reference_dir),
                '--config',str(config),'--admission',str(admission),'--output',str(run)],
            expected_receipt=str(run/'short_training_receipt.json')))
        jobs.append(dict(id='short_'+arm+'_s42_E20_eval',arm=arm,stage='eval',kind='eval',
            vram_mib=args.eval_vram_mib,rss_mib=args.eval_rss_mib,
            command=[str(args.python),str(HERE/'evaluate_short_screen.py'),'--reference-dir',str(args.reference_dir),
                '--run',str(run),'--admission',str(admission),'--native-profile-binding',str(args.native_profile_binding),
                '--output',str(evaluation)],expected_receipt=str(evaluation/'short_evaluation_receipt.json')))
    return jobs


def check_terminal(job,receipt):
    expected='SHORT_SCREEN_TRAINING_COMPLETED' if job['stage']=='train' else 'SHORT_SCREEN_EVALUATION_COMPLETED'
    if (receipt.get('status')!=expected or receipt.get('scope')!='SHORT_SCREEN' or receipt.get('arm')!=job['arm']
        or receipt.get('seed')!=42 or receipt.get('endpoint')!=ENDPOINT or receipt.get('single_seed') is not True
        or receipt.get('new_hash_computed') is not False or receipt.get('official_test_accessed') is not False
        or receipt.get('formal_e200_complete') is not False):
        raise ValueError('Stage lacks its own complete SHORT_SCREEN receipt: '+job['id'])
    if job['stage']=='train':
        if receipt.get('epochs_configured')!=20 or receipt.get('last_epoch')!=20:raise ValueError('Not complete E20 training')
    elif receipt.get('epochs')!=20 or receipt.get('full_dev_images')!=1469 or receipt.get('full_dev_gt_objects')!=22462:
        raise ValueError('Short evaluation population differs')


def verify_eval_reservation(path,vram,rss):
    r=read(path)
    if r.get('status')!='completed' or r.get('observed_images')!=1469 or r.get('official_test_accessed') is not False:
        raise ValueError('Existing complete full-dev evidence evaluator resource receipt required')
    peaks=list(r['resources']['per_gpu_peak_vram_mib'].values());host=r['resources']['peak_rss_mib']
    if not peaks or any(type(x) not in (int,float) or not math.isfinite(x) or x<=0 for x in peaks+[host]):
        raise ValueError('Missing measured evaluator peak')
    if max(peaks)>vram or host>rss:raise ValueError('Evaluator reservation below accepted measured peak')
    return dict(source=stat(path),measured_nvml_peak_mib=max(peaks),measured_rss_peak_mib=host,
                reserved_vram_mib=vram,reserved_rss_mib=rss)


def run(args):
    args.campaign_root=data_output(args.campaign_root)
    if args.campaign_root.exists():raise FileExistsError('Use a new campaign root; existing attempts are immutable')
    for key in ('reference_dir','dispatch_dir','config_dir','admissions','native_profile_binding'):
        setattr(args,key,getattr(args,key).resolve())
    # Preserve the venv invocation path even when bin/python is a symlink.
    args.python=args.python.absolute()
    if not args.python.is_file():raise FileNotFoundError(args.python)
    evaluation_evidence=args.native_profile_binding.parent.parent/'evidence_metrics.json'
    eval_resources=verify_eval_reservation(evaluation_evidence,args.eval_vram_mib,args.eval_rss_mib)
    # Preflight all three actual decisions before creating/running the queue.
    # This consumes existing reviewed evidence; it does not create any PASS.
    config_records=[]
    for arm in ARMS:
        config=args.config_dir/('drone_'+arm+'_s42_E20_DRAFT.yaml');cfg=load_config(config)
        if cfg['arm']!=arm:raise ValueError('Config filename/arm mismatch')
        admission=args.admissions/(arm+'.json')
        decision=require_admission(admission,cfg,config,HERE/'train_short_screen.py','training')
        require_admission(admission,cfg,config,HERE/'evaluate_short_screen.py','evaluation')
        budget=read(decision['resource_budget_review'])
        if (budget['measured_training_nvml_peak_mib']>args.train_vram_mib or
            budget['measured_training_rss_peak_mib']>args.train_rss_mib):
            raise ValueError('Training reservation below reviewed measured peak: '+arm)
        config_records.append(dict(arm=arm,config=stat(config),admission=stat(admission)))
    sys.path.insert(0,str(args.dispatch_dir))
    import resource_dispatch
    if Path(resource_dispatch.__file__).resolve()!=args.dispatch_dir/'resource_dispatch.py':
        raise RuntimeError('Wrong existing globallease dispatcher')
    args.campaign_root.mkdir(parents=True,exist_ok=False)
    queue=args.campaign_root/'queue';queue.mkdir()
    jobs=jobs_for(args)
    shutil.copyfile(Path(__file__),queue/'executed_driver.py')
    if (queue/'executed_driver.py').read_bytes()!=Path(__file__).read_bytes():raise AssertionError('Driver source copy differs')
    write_new(queue/'manifest.json',dict(scope='SHORT_SCREEN_SINGLE_SEED_E20',jobs=jobs,configs=config_records,
        evaluation_resource_basis=eval_resources,training_reservation_basis='Root-reviewed actual-cadence profiles and reusable historical measured peaks',
        independent_lr_horizon=20,seed=42,sequence=['N train','N eval','C0 train','C0 eval','C1 train','C1 eval'],
        native_profile_binding=stat(args.native_profile_binding),dispatcher=stat(args.dispatch_dir/'resource_dispatch.py'),
        new_hash_computed=False,new_scheduler_created=False,ap_adaptive=False,formal_e200_authorized=False))
    completed=[];started=time.time()
    try:
        for job in jobs:
            resource_dispatch.run_job(job,queue)
            receipt=read(job['expected_receipt']);check_terminal(job,receipt)
            completed.append(dict(id=job['id'],arm=job['arm'],stage=job['stage'],receipt=job['expected_receipt']))
            write_new(queue/(job['id']+'_completed.json'),dict(status='SHORT_SCREEN_STAGE_COMPLETED',step=completed[-1],new_hash_computed=False))
        write_new(queue/'completion.json',dict(status='SHORT_SCREEN_MATRIX_COMPLETED',completed=completed,
            seconds=time.time()-started,single_seed=True,epochs=20,new_hash_computed=False,
            formal_e200_complete=False,formal_paper_gain_claim=False,ap_adaptive=False))
    except BaseException as error:
        write_new(queue/'failure.json',dict(status='SHORT_SCREEN_MATRIX_STOPPED_ON_FAILURE',error=repr(error),
            traceback=traceback.format_exc(),completed=completed,seconds=time.time()-started,
            new_hash_computed=False,prior_attempts_preserved=True,automatic_retry=False))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('campaign-root','reference-dir','dispatch-dir','config-dir','admissions','native-profile-binding','python'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--train-vram-mib',type=int,default=8192);p.add_argument('--train-rss-mib',type=int,default=32768)
    p.add_argument('--eval-vram-mib',type=int,default=2048);p.add_argument('--eval-rss-mib',type=int,default=8192)
    args=p.parse_args()
    if min(args.train_vram_mib,args.train_rss_mib,args.eval_vram_mib,args.eval_rss_mib)<=0:p.error('Positive resource reservations required')
    run(args)


if __name__=='__main__':main()
