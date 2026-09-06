import base64,json,subprocess
from pathlib import Path
here=Path(__file__).resolve().parent
random=here.parent/'2026-09-06_train_OEv1随机选择对照'
docs={'priority':(here/'README.md').read_text(encoding='utf-8'),'random':(random/'README.md').read_text(encoding='utf-8')}
code="""import base64,json
from pathlib import Path
root=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign')
docs=json.loads(DOCS)
targets={'priority':root/'artifacts/oev1_priority_comparators_20260906/PRIORITY_UPDATE_2325.md',
         'random':root/'artifacts/rgbir_oev1_random_20260906/LAUNCH_VERIFIED_2325.md'}
for key,p in targets.items():
    with p.open('x') as f:f.write(docs[key])
index=root/'PROJECTS_INDEX_OEV1_PRIORITY_20260906.md'
with index.open('x') as f:f.write('# OEv1 priority and comparators, 2026-09-06\\n\\n'+
    'New artifacts: artifacts/oev1_priority_comparators_20260906 and artifacts/rgbir_oev1_random_20260906.\\n'+
    'New results: runs/oev1_comparators_20260906 and runs/rgbir_oev1_random_20260906.\\n'+
    'OS-SSL stopped for user research priority; original checkpoints and completed outputs preserved.\\n'+
    'Local mirrors: 08 experiment entries ops_OEv1 and train_OEv1 random, 2026-09-06.\\n')
run=root/'runs/rgbir_oev1_random_20260906/canary_paired_random_s42_attempt1'
paths=[run/n for n in ['completion_receipt.json','launch_manifest.json','runtime_ready.json','protocol_config.yaml','kd_batches.jsonl','gradient_checks.jsonl']]
paths+=list((run/'run_evidence').rglob('*'))
files=[]
for p in paths:
    if not p.is_file() or p.suffix not in ('.json','.jsonl','.yaml','.py','.txt','.md'):continue
    assert p.stat().st_size<15000000,p
    files.append({'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'data':base64.b64encode(p.read_bytes()).decode()})
print(json.dumps({'files':files,'notes':[str(p) for p in targets.values()],'index':str(index)}))
""".replace('DOCS',repr(json.dumps(docs)))
r=subprocess.run(['ssh','-o','BatchMode=yes','94','/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python','-'],input=code.encode(),capture_output=True)
if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace'))
data=json.loads(r.stdout)
for item in data['files']:
    target=random/'canary_raw'/item['path'];target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('xb') as f:f.write(base64.b64decode(item.pop('data')))
(here/'sync_and_canary_manifest.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
print(json.dumps({'files':len(data['files']),'bytes':sum(x['bytes'] for x in data['files']),'notes':data['notes']}))
