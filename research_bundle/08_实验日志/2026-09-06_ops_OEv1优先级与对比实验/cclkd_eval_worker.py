"""One guarded, single-GPU canary followed by three fixed-checkpoint evals."""
import json,math,subprocess,sys,time
from pathlib import Path
ROOT=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign')
REPO=Path('/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction')
PY=REPO/'environments/sn6-int8-kd/bin/python'
HERE=ROOT/'artifacts/oev1_priority_comparators_20260906'
OUT=ROOT/'runs/oev1_comparators_20260906'
OUT.mkdir(parents=True,exist_ok=True)
snapshot=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free','--format=csv,noheader,nounits'],text=True)
rows=[list(map(int,x.split(','))) for x in snapshot.strip().splitlines()]
free=[x[0] for x in rows if x[1]<100]
assert len(free)>=3,'Fourth-card exception requires at least two empty cards after admission'
gpu=2 if 2 in free else free[0]
(HERE/'cclkd_eval_allocation.json').write_text(json.dumps({'gpu':gpu,'snapshot':snapshot,'empty_before':free,
    'policy':'User AGENTS 2.1 four-card exception; guard remains mandatory; at least two empty cards remain'},indent=2))

def run(seed,mode,expected_vram):
    out=OUT/(f'cclkd_partial_s{seed}_{mode}_attempt1')
    assert not out.exists(),out
    cmd=[str(PY),str(REPO/'tools/project_resource_guard.py'),'run','--job-id',out.name,'--kind','eval',
        '--candidate-gpu',str(gpu),'--expected-vram-mib',str(expected_vram),'--expected-rss-mib','32768',
        '--free-safety-mib','2048','--',str(PY),str(HERE/'evaluate_cclkd_partial.py'),
        '--seed',str(seed),'--output',str(out),'--mode',mode]
    print(json.dumps({'starting':out.name,'command':cmd}),flush=True)
    with (HERE/(out.name+'.log')).open('x') as f:
        status=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT).returncode
    if status:raise RuntimeError(f'{out.name}: exit {status}; inspect log; no automatic failed-run retry')
    result=json.loads((out/('canary_result.json' if mode=='canary' else 'evaluation_val.json')).read_text())
    assert (out/'eval_evidence/run_receipt.json').is_file()
    print(json.dumps({'finished':out.name,'metrics':result}),flush=True)
    return result

try:
    canary=run(42,'canary',14000) # capacity reservation on an observed empty card, not an empirical peak claim
    peak=max(canary['reserved_peak_mib'],max(canary['resources']['per_gpu_peak_vram_mib'].values()))
    reservation=math.ceil((peak+1024)/256)*256
    assert reservation<16000
    results=[run(seed,'full',reservation) for seed in (0,42,123)]
    summary={'status':'completed','canary_reserved_peak_mib':canary['reserved_peak_mib'],
             'formal_vram_reservation_mib':reservation,'gpu':gpu,'results':results}
except BaseException as error:
    (HERE/'cclkd_eval_worker_failure.json').write_text(json.dumps({'error':repr(error),'time':time.time()}))
    raise
with (HERE/'cclkd_eval_summary.json').open('x') as f:json.dump(summary,f,indent=2)
