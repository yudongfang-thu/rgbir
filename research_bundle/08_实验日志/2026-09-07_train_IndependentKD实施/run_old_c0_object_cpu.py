"""Run the unchanged accepted object analyzer on six actual legacy endpoints.

Sequential CPU only. This runner records resource use and invocation, not a new
scientific acceptance. It neither loads models nor edits any endpoint identity.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import psutil
import yaml

SEEDS=(0,42,123)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


def rss_bytes():
    root=psutil.Process(os.getpid())
    total=0
    for proc in [root,*root.children(recursive=True)]:
        try:total+=proc.memory_info().rss
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    return total


def run(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Set empty CUDA_VISIBLE_DEVICES for this CPU analysis')
    completion=read(args.inputs/'dispatch_attempt1/completion.json')
    if completion.get('status')!='COMPLETED':raise ValueError('Six real reevaluations are not complete')
    records=[];recipes=[]
    for seed in SEEDS:
        for arm,actual in (('N','weight0'),('C0','paired')):
            folder=args.inputs/f'{arm}_s{seed}_attempt1';receipt=read(folder/'reevaluation_receipt.json')
            if (receipt.get('status')!='completed' or receipt.get('seed')!=seed or receipt.get('arm')!=actual
                or receipt.get('normalized_method_arm')!=arm or receipt.get('full_dev_images')!=1469
                or receipt.get('all_class_metrics')!=5 or receipt.get('historical_five_metrics_exact') is not True
                or receipt.get('old_results_modified') is not False or receipt.get('new_training_receipt_created') is not False
                or receipt.get('official_test_accessed') is not False):
                raise ValueError('Incomplete or mismatched actual legacy reevaluation: '+str(folder))
            cfg=yaml.safe_load((folder/'evaluation_config.yaml').read_text())
            recipes.append({k:v for k,v in cfg.items() if k not in ('seed','arm')})
            records.append(dict(normalized_method_arm=arm,seed=seed,receipt=str(folder/'reevaluation_receipt.json'),
                                checkpoint=receipt['checkpoint']))
    if any(row!=recipes[0] for row in recipes):raise ValueError('Actual old six effective recipes disagree beyond arm and seed')
    args.output.mkdir(parents=True,exist_ok=False)
    write_new(args.output/'invocation.json',dict(status='STARTED',time=time.time(),inputs=records,
        analyzer=str(args.analyzer),review_receipt=str(args.review_receipt),metadata=str(args.metadata),
        interpreter=sys.executable,CUDA_VISIBLE_DEVICES='',model_inference=False,ap_bridge_accepted=False,
        scope='Old C0 versus N fixed-threshold supplementary diagnosis; not C1'))
    resources=[]
    try:
        for seed in SEEDS:
            command=[sys.executable,str(args.analyzer),
                '--baseline-evaluation',str(args.inputs/f'N_s{seed}_attempt1/evaluation_val.json'),
                '--candidate-evaluation',str(args.inputs/f'C0_s{seed}_attempt1/evaluation_val.json'),
                '--output',str(args.output/f'seed{seed}'),'--metadata',str(args.metadata),
                '--review-receipt',str(args.review_receipt)]
            started=time.time();peak=rss_bytes()
            with (args.output/f'seed{seed}.log').open('x',encoding='utf-8') as log:
                proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
                try:
                    while proc.poll() is None:
                        peak=max(peak,rss_bytes());time.sleep(.25)
                    peak=max(peak,rss_bytes())
                except BaseException:
                    if proc.poll() is None:proc.terminate();proc.wait()
                    raise
            measurement=dict(seed=seed,command=command,exit_code=proc.returncode,seconds=time.time()-started,
                peak_process_tree_rss_bytes=peak,sampling_seconds=.25,includes_runner_and_analyzer_children=True,
                CUDA_VISIBLE_DEVICES='',model_inference=False)
            write_new(args.output/f'seed{seed}_cpu_receipt.json',measurement);resources.append(measurement)
            if proc.returncode:raise RuntimeError(f'Accepted analyzer failed for seed{seed}; preserve log/output')
            analysis=read(args.output/f'seed{seed}/analysis_receipt.json')
            if analysis.get('status')!='COMPLETED' or analysis.get('analyzer_acceptance')!='ACCEPTED':
                raise ValueError('Accepted diagnostic contract not produced')
        write_new(args.output/'completion.json',dict(status='COMPLETED',seeds=list(SEEDS),resources=resources,
            time=time.time(),raw_inputs_overwritten=False,model_inference=False,ap_bridge_accepted=False))
        print(json.dumps({'status':'COMPLETED','output':str(args.output)}))
    except BaseException as exc:
        write_new(args.output/'failure.json',dict(status='FAILED',error=repr(exc),traceback=traceback.format_exc(),time=time.time()))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('inputs','analyzer','review-receipt','metadata','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())
