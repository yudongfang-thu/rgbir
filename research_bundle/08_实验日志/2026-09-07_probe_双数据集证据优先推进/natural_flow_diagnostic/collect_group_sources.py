"""Read governance group TSVs after inference; no source modifications."""
from pathlib import Path
import base64,json,subprocess
ROOT=Path(__file__).resolve().parent
PY='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
CODE=r'''
import base64,json,sys,yaml
from pathlib import Path
r=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_independent_kd_v2_20260907')
sys.path.insert(0,str(r/'release_gpu5'))
import runtime
from diagnose_opportunities import dataset_config
result={}
for d in ['llvip','drone']:
 cfg=yaml.safe_load((r/'configs_draft_v1'/(d+'_C1.yaml')).read_text());root=dataset_config(cfg['paths']['student_data_yaml'])['root']
 candidates=[root.parent/'rgb_train_source_groups.tsv',root/'rgb_train_source_groups.tsv']
 if d=='llvip':candidates += [p/'splits/grouped_v1/fit.tsv' for p in [root]+list(root.parents)[:5]]
 files=[]
 for p in candidates:
  if p.is_file():
   s=p.stat();files.append(dict(path=str(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns,data=base64.b64encode(p.read_bytes()).decode()))
 result[d]=dict(root=str(root),searched_paths=[str(p) for p in candidates],files=files)
print(json.dumps(result))
'''
r=subprocess.run(['ssh','94',PY,'-'],input=CODE.encode(),capture_output=True,check=True)
data=json.loads(r.stdout);dest=ROOT/'governance_group_sources';dest.mkdir(exist_ok=False)
for d,info in data.items():
    for i,record in enumerate(info['files']):
        p=dest/(d+'_'+str(i)+'.tsv');raw=base64.b64decode(record.pop('data'));assert len(raw)==record['bytes'];p.write_bytes(raw);record['local']=str(p)
(dest/'receipt.json').write_text(json.dumps(dict(status='READ_ONLY_COLLECTED',new_hashes=False,datasets=data),indent=2)+'\n',encoding='utf-8')
print(json.dumps({d:[dict(path=x['path'],bytes=x['bytes']) for x in v['files']] for d,v in data.items()}))
