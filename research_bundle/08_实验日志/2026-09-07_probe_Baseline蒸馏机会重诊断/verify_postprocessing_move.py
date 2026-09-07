from pathlib import Path
import json,subprocess
ROOT=Path(__file__).resolve().parent
PY='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
CODE=r'''
from pathlib import Path
import json,numpy as np
r=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_baseline_information_20260907')
a=r/'attempt2/dronevehicle_canary_attempt1';b=r/'attempt3/dronevehicle_canary_attempt1'
aa=[json.loads(x) for x in (a/'objects.jsonl').read_text().splitlines()];bb=[json.loads(x) for x in (b/'objects.jsonl').read_text().splitlines()]
assert [x['object_id'] for x in aa]==[x['object_id'] for x in bb]
assert [x['anchor_index'] for x in aa]==[x['anchor_index'] for x in bb]
diff={}
for name in ('features.npz','logits.npz'):
 x=np.load(a/name);y=np.load(b/name);assert set(x.files)==set(y.files)
 diff[name]={k:float(np.max(np.abs(x[k]-y[k]))) for k in x.files}
v={'status':'PASS','objects':len(aa),'object_and_anchor_ids_equal':True,'absolute_max_differences':diff,'scope':'2 image canary CPU vs GPU float32 postprocessing; no claim of full-run bitwise equality'}
assert max(diff['features.npz'].values())<1e-4 and max(diff['logits.npz'].values())<1e-4
with (r/'attempt3/postprocessing_move_check.json').open('x') as f:json.dump(v,f,indent=2)
print(json.dumps(v))
'''
r=subprocess.run(['ssh','94',PY,'-'],input=CODE.encode(),capture_output=True,check=True)
v=json.loads(r.stdout);(ROOT/'postprocessing_move_check.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
