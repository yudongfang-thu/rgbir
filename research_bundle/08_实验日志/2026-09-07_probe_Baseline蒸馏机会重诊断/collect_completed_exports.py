from pathlib import Path
import argparse,json,subprocess
ROOT=Path(__file__).resolve().parent
BASE='/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_baseline_information_20260907/attempt3'
PY='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
p=argparse.ArgumentParser();p.add_argument('dataset',choices=['dronevehicle','llvip']);a=p.parse_args()
remote=BASE+'/'+a.dataset+'_full_attempt1'
code="from pathlib import Path\nimport json\np=Path("+repr(remote)+")\ns=json.loads((p/'summary.json').read_text())\nassert s['status']=='completed' and (p/'run_evidence/run_receipt.json').exists()\nprint(json.dumps(s))\n"
r=subprocess.run(['ssh','94',PY,'-'],input=code.encode(),capture_output=True,check=True)
summary=json.loads(r.stdout)
out=ROOT/'remote_exports'/(a.dataset+'_full_attempt1');out.mkdir(parents=True,exist_ok=False)
for name in ('summary.json','objects.jsonl','features.npz','logits.npz','frozen_roster.json','model_identity.json'):
    subprocess.run(['scp','94:'+remote+'/'+name,str(out/name)],check=True)
subprocess.run(['scp','-r','94:'+remote+'/run_evidence',str(out/'run_evidence')],check=True)
record={'remote':remote,'local':str(out),'summary':summary,'files':[{ 'name':f.name,'bytes':f.stat().st_size} for f in out.iterdir() if f.is_file()]}
(ROOT/(a.dataset+'_collection_receipt.json')).write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'dataset':a.dataset,'images':summary['images'],'rows':summary['objects_including_background'],'bytes':sum(f['bytes'] for f in record['files'])}))
