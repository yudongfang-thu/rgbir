"""Read-only independent capture of the bounded concurrent-training profile."""
import json, subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEST = HERE / 'independent_concurrency_audit'
DEST.mkdir(exist_ok=True)
BASE = '/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/oev1_concurrency_20260906'
FILES = ['concurrency_profile.json', 'concurrency_summary.json', 'profile_canary_comparison_s0.json', 'profile.log', 'worker.log']
for name in FILES:
    subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', f'94:{BASE}/{name}', str(DEST / name)], check=True)
remote = '''import json,time,subprocess
from pathlib import Path
root=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign')
profile=root/'runs/oev1_concurrency_20260906/profile_paired_random_s0_attempt1'
existing=root/'runs/rgbir_oev1_random_20260906/full_paired_random_s42_attempt1'
out={'captured_time':time.time(),'profile_run':str(profile),'existing_run':str(existing)}
for label,run in [('profile',profile),('existing',existing)]:
    for name in ['progress.json','completion_receipt.json','canary_summary.json']:
        p=run/name
        if p.is_file():out[label+'_'+name]=json.loads(p.read_text())
out['gpu2']=subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
out['gpu2_processes']=subprocess.check_output(['nvidia-smi','-i','2','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
print(json.dumps({k:v for k,v in out.items() if k not in ['completion','post']},indent=2))
'''
r = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','94','/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python','-'], input=remote.encode(), capture_output=True)
(DEST/'post_profile_snapshot.json').write_bytes(r.stdout)
(DEST/'fetch.stderr.txt').write_bytes(r.stderr)
if r.returncode: raise RuntimeError(r.stderr.decode(errors='replace'))
profile=json.loads((DEST/'concurrency_profile.json').read_text())
samples=profile['samples']
double=[s for s in samples if len(s['cuda_pids'])==2]
check=json.loads((DEST/'profile_canary_comparison_s0.json').read_text())
out={k:profile[k] for k in ['returncode','cuda_peak_process_count','observed_peak_total_used_mib','minimum_observed_free_mib','samples_with_two_cuda_pids']}
out.update(first_sample=samples[0],first_double_sample=double[0],last_double_sample=double[-1],last_sample=samples[-1],completion=profile.get('completion'),canary_check=check,post=json.loads(r.stdout))
(DEST/'audit_extracted.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
