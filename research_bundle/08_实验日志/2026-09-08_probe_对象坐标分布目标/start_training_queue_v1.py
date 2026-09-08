"""Start only the reviewed new single-seed staged queue through the original lease."""
from pathlib import Path
import base64,json,subprocess,sys
root=Path(__file__).parent;version,attempt=sys.argv[1:3]
if not version.isdigit() or not attempt.isdigit():raise ValueError('Numeric versions')
r=json.loads((root/'independent_review/TRAINING_SOURCE_REVIEW.json').read_text(encoding='utf-8'))
if r['status']!='READY_FOR_GPU_PREFLIGHT':raise ValueError('Independent source/pinned CPU acceptance required')
cpu=json.loads((root/('PINNED_TRAINING_CPU_ACCEPTED_v'+version+'.json')).read_text(encoding='utf-8'))['receipt']
if cpu['status']!='PASS' or cpu['tests']!=36 or cpu['source_changed']:raise ValueError('Actual pinned CPU not accepted')
interface=json.loads((root/'independent_review/EXPERIMENT_AUDIT.json').read_text(encoding='utf-8'))
if interface['integrity_status']!='pass':raise ValueError('Executed CPU target interface not accepted')
bound={}
for reviewed in r['reviewed_source_files']:
 name=reviewed['relative_path']
 p=root/'trainer_release'/name
 if (root/'trainer_release').resolve() not in p.resolve().parents:raise ValueError('Invalid reviewed path')
 b=p.read_bytes()
 if b!=(root/'independent_review/training_reviewed_source'/name).read_bytes():raise ValueError('Reviewed source changed: '+name)
 bound[name]=base64.b64encode(b).decode()
required={'l3_distribution_loss.py','probability_transport.py','object_dfl_common.py','object_dfl_criterion.py','train_object_dfl.py','calibrate_object_dfl.py','evaluate_object_dfl.py','run_object_dfl_queue.py','validated_amp_prior.json','validated_amp_prior_config.yaml'}
required.update('configs/llvip_'+a+'_s42_FT3.yaml' for a in ('N','L3-DFL','L3-GT'))
required.add('configs/llvip_native_evaluation.yaml')
if not required.issubset(bound):raise ValueError('Incomplete source/config review binding')
B='/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_object_dfl_screen_20260908'
release=B+'/release_v'+version;out=B+'/attempt'+attempt
py='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
screen='object_dfl_matrix_s42_a'+attempt
code="""from pathlib import Path
import base64,json,subprocess
release=Path(%r);out=Path(%r)
if out.exists():raise FileExistsError('Preserve attempt')
for name,b64 in %r.items():
 if (release/name).read_bytes()!=base64.b64decode(b64):raise ValueError('Remote reviewed source differs: '+name)
r=json.loads((release/'pinned_cpu_accepted.json').read_text())
if r['status']!='PASS' or r['tests']!=36 or r['source_changed']:raise ValueError('Actual pinned acceptance missing')
subprocess.run(['screen','-dmS',%r,'bash','-lc',%r],check=True)
print(json.dumps(dict(status='QUEUED_REVIEWED_L3_STAGES_ORIGINAL_LEASE',release=str(release),output=str(out),screen=%r,
 stages='calibration8 -> all three canaries24 -> each arm FT3/full-dev; any calibration block prevents all training',new_hash_computed=False)))
"""%(release,out,bound,screen,'exec '+py+' '+release+'/run_object_dfl_queue.py --release-dir '+release+' --output '+out+' > '+release+'/queue_launch_a'+attempt+'.log 2>&1',screen)
p=subprocess.run(['ssh','94',py,'-'],input=code.encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace'))
result=json.loads(p.stdout)
with (root/('QUEUE_LAUNCH_attempt'+attempt+'.json')).open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
print(result['status'],result['output'])
