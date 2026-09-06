"""Read actual release_v2 over SSH; create an isolated two-line trainer delta."""
from pathlib import Path
import subprocess,json,hashlib,difflib,datetime
here=Path(__file__).resolve().parent
remote='/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_object_evidence_v1_20260906/release_v2'
names=['train_object_evidence.py','object_evidence_loss.py','paired_rgbir_data.py','evaluate_object_evidence.py','config_drone.yaml','test_object_evidence_loss.py','test_paired_rgbir_data.py']
script='from pathlib import Path\nimport json,base64\np=Path('+repr(remote)+')\nprint(json.dumps({n:base64.b64encode((p/n).read_bytes()).decode() for n in '+repr(names)+'}))\n'
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','94','/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python','-'],input=script.encode(),capture_output=True)
if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace'))
import base64
original={n:base64.b64decode(v) for n,v in json.loads(r.stdout).items()}
(here/'source_baseline').mkdir(exist_ok=True)
(here/'code').mkdir(exist_ok=True)
manifest={'captured_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'remote_source':remote,'files':[]}
for name,data in original.items():
    for directory in ('source_baseline','code'):
        target=here/directory/name
        if target.exists():raise FileExistsError(target)
        target.write_bytes(data)
    local=here.parent/'2026-09-06_train_RGBIR对象判别蒸馏首轮'/'code'/name
    manifest['files'].append({'file':name,'remote_sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'matches_first_campaign_local':local.exists() and local.read_bytes()==data})
target=here/'code/train_object_evidence.py'
before=target.read_text(encoding='utf-8')
replacements=[("config=self.evidence_cfg, arm='paired', seed=self.cfg['seed']+self.calls)","config=self.evidence_cfg, arm=('paired' if self.arm == 'weight0' else self.arm), seed=self.cfg['seed']+self.calls)"),("parser.add_argument('--arm',choices=['paired','weight0'],required=True)","parser.add_argument('--arm',choices=['paired','weight0','paired_random'],required=True)")]
after=before
for a,b in replacements:
    if after.count(a)!=1:raise ValueError('Expected exactly one replacement: '+a)
    after=after.replace(a,b)
target.write_bytes(after.encode('utf-8'))
manifest['isolated_trainer_sha256']=hashlib.sha256(target.read_bytes()).hexdigest()
(here/'source_manifest.json').write_bytes((json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
(here/'source_delta.patch').write_bytes(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='release_v2/train_object_evidence.py',tofile='random_release_v1/train_object_evidence.py')).encode('utf-8'))
print(json.dumps(manifest,ensure_ascii=False))
