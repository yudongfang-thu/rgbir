"""CPU audit of real paired_random vs same-seed original paired canaries."""
import argparse
import json
from pathlib import Path

import torch


def record(path):return json.loads(path.read_text())


def validate(paired,random,gpu):
    p=record(paired/'completion_receipt.json'); r=record(random/'completion_receipt.json')
    assert p['arm']=='paired' and r['arm']=='paired_random'
    assert p['seed']==r['seed'] and r['seed'] in (0,42,123)
    assert p['status']==r['status']=='canary_completed'
    assert p['optimizer_updates']>=24 and r['optimizer_updates']>=24
    assert r['selected_objects']>0
    assert any(x['kd_score_gradient_l2']>0 for x in r['gradient_checks'])
    assert all(x['weight0_exact_loss_gradient'] and not x['teacher_has_grad'] and not x['reference_has_grad'] for x in r['gradient_checks'])
    res=r['resources']; key=str(gpu)
    assert res['gpu_ids']==[gpu]
    assert res['cuda_pid_counts'][key]<=1
    assert res['per_gpu_peak_vram_mib'][key]<=10000
    assert res['peak_rss_mib']<=49152
    tensor_checks={}
    for name in ('initial_student.pt','first_batch.pt'):
        a=torch.load(paired/name,map_location='cpu',weights_only=True)
        b=torch.load(random/name,map_location='cpu',weights_only=True)
        equal=set(a)==set(b) and all(torch.equal(a[k],b[k]) for k in a)
        assert equal,name
        tensor_checks[name]=equal
    pm=record(paired/'launch_manifest.json'); rm=record(random/'launch_manifest.json')
    for k in ('model','teacher','reference'):
        assert pm['inputs'][k]==rm['inputs'][k],k
    assert (paired/'protocol_config.yaml').read_bytes()==(random/'protocol_config.yaml').read_bytes()
    unchanged={}
    for name in ('object_evidence_loss.py','paired_rgbir_data.py'):
        a=(paired/'implementation_snapshot'/name).read_bytes()
        b=(random/'implementation_snapshot'/name).read_bytes()
        assert a==b,name
        unchanged[name]={'direct_byte_equal':True,'bytes':len(b)}
    baseline=(paired/'implementation_snapshot/train_object_evidence.py').read_text()
    observed=(random/'implementation_snapshot/train_object_evidence.py').read_text()
    expected=baseline.replace("config=self.evidence_cfg, arm='paired', seed=self.cfg['seed']+self.calls)","config=self.evidence_cfg, arm=('paired' if self.arm == 'weight0' else self.arm), seed=self.cfg['seed']+self.calls)").replace("parser.add_argument('--arm',choices=['paired','weight0'],required=True)","parser.add_argument('--arm',choices=['paired','weight0','paired_random'],required=True)")
    assert observed==expected,'unexpected trainer source delta'
    def batches(path):return {x['batch']:x for x in (json.loads(line) for line in path.read_text().splitlines() if line.strip())}
    a=batches(paired/'kd_batches.jsonl'); b=batches(random/'kd_batches.jsonl')
    common=sorted(set(a)&set(b))
    assert len(common)>=24,'not enough common logged batches'
    different=0
    for i in common:
        for k in ('student_files','base_count','eligible_count','selected_count','normalizer','nominal_dose'):
            assert a[i][k]==b[i][k],(i,k)
        assert a[i]['arm']=='paired' and b[i]['arm']=='paired_random'
        assert b[i]['kd_weight']==.1
        different+=a[i]['selected_object_ids']!=b[i]['selected_object_ids']
    assert different>0,'canary did not demonstrate different selections'
    return {'status':'passed','seed':r['seed'],'physical_gpu':gpu,
            'paired_run':str(paired),'random_run':str(random),
            'tensor_equality':tensor_checks,'unchanged_implementation_direct_byte_comparison':unchanged,
            'trainer_delta_exactly_expected':True,'same_input_weight_paths_and_sizes':True,
            'common_batches_checked':len(common),'batches_with_different_selection':different,
            'dose_semantics':'same K and base normalizer; not equal actual loss/gradient magnitude',
            'optimizer_updates':[p['optimizer_updates'],r['optimizer_updates']],
            'amp_skipped_updates':[p['amp_skipped_updates'],r['amp_skipped_updates']],
            'resources':res,'gpu_allocated_peak_mib':r['gpu_allocated_peak_mib'],
            'gpu_reserved_peak_mib':r['gpu_reserved_peak_mib'],'official_test_accessed':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paired-run',type=Path,required=True)
    parser.add_argument('--random-run',type=Path,required=True)
    parser.add_argument('--gpu',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    try:r=validate(a.paired_run,a.random_run,a.gpu)
    except Exception as e:
        r={'status':'failed','error':repr(e),'paired_run':str(a.paired_run),'random_run':str(a.random_run)}
    with a.output.open('x') as f:json.dump(r,f,indent=2,allow_nan=False)
    print(json.dumps(r,allow_nan=False))
    raise SystemExit(0 if r['status']=='passed' else 1)


if __name__=='__main__':main()
