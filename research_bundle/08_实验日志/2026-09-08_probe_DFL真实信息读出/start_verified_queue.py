"""Start the sole batch only after independent source and actual pinned CPU acceptance."""
from pathlib import Path
import base64,json,subprocess,sys
root=Path(__file__).parent;version,attempt=sys.argv[1:3]
if not version.isdigit() or not attempt.isdigit():raise ValueError('Numeric version/attempt')
review=json.loads((root/'independent_review/SOURCE_REVIEW.json').read_text(encoding='utf-8'))
if review.get('status')!='READY_FOR_SINGLE_BATCH':raise ValueError('Independent source review not ready')
bound={}
for name in review['reviewed_source_files']:
    p=root/'release'/name
    if (root/'release').resolve() not in p.resolve().parents:raise ValueError('Invalid review path')
    b=p.read_bytes()
    if b!=(root/'independent_review/reviewed_source'/name).read_bytes():raise ValueError('Source changed after review: '+name)
    bound[name]=base64.b64encode(b).decode()
if not {'run_dfl_probe.py','dfl_export.py','run_dfl_queue.py','test_dfl_cpu.py','test_installed_native_cpu.py','llvip_N_s42_FT3.yaml'}.issubset(bound):raise ValueError('Incomplete reviewed source binding')
cache=json.loads((root/'cache_audit/output_attempt1/receipt.json').read_text(encoding='utf-8'))
if cache.get('status')!='CACHE_MISSING_FOR_MATCHED_FIRST32_RAW_DFL':raise ValueError('New batch necessity not established')
B='/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_dfl_information_20260908'
release=B+'/release_v'+version;out=B+'/attempt'+attempt
py='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
screen='dfl_information_probe_s42_a'+attempt
code="""from pathlib import Path
import base64,json,subprocess
release=Path(%r);out=Path(%r)
if out.exists():raise FileExistsError('Preserve previous attempt')
for name,b64 in %r.items():
 if (release/name).read_bytes()!=base64.b64decode(b64):raise ValueError('Remote source differs from reviewed bytes: '+name)
r=json.loads((release/'pinned_cpu.json').read_text())
if r.get('status')!='PASS' or r.get('tests',0)<1 or r.get('failures',0)!=0 or r.get('errors',0)!=0:raise ValueError('Pinned CPU not accepted')
subprocess.run(['screen','-dmS',%r,'bash','-lc',%r],check=True)
print(json.dumps(dict(status='QUEUED_SINGLE_MEASURED_BATCH_EXISTING_LEASE',release=str(release),output=str(out),screen=%r,new_hash_computed=False)))
"""%(release,out,bound,screen,'exec '+py+' '+release+'/run_dfl_queue.py --release-dir '+release+' --output '+out+' > '+release+'/queue_launch_a'+attempt+'.log 2>&1',screen)
p=subprocess.run(['ssh','94',py,'-'],input=code.encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace'))
r=json.loads(p.stdout)
with (root/('queue_launch_attempt'+attempt+'.json')).open('x',encoding='utf-8') as f:json.dump(r,f,ensure_ascii=False,indent=2)
print(r['status'],r['output'])
