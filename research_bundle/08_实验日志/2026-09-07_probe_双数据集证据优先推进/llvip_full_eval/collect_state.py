"""Read-only status; --collect copies only after the diagnostic queue completes."""
from pathlib import Path
import argparse,datetime,json,subprocess
ROOT=Path(__file__).resolve().parent
PY='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
REMOTE='/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_evidence_priority_20260907/llvip_full_eval_attempt2'
CODE=r'''
import json
from pathlib import Path
root=Path(__REMOTE__)
def read(p):return json.loads(p.read_text()) if p.is_file() else None
out=dict(completion=read(root/'queue_attempt1/completion.json'),jobs={})
for model in ['N42','T42']:
    for stage in ['canary','full']:
        p=root/(model+'_'+stage+'_attempt1')
        out['jobs'][model+'_'+stage]=dict(summary=read(p/'summary.json'),failure=read(p/'failure_receipt.json'),status=read(root/'queue_attempt1'/('llvip_full_'+model+'_'+stage+'_status.json')))
out['log_tail']=(root/'campaign.log').read_text()[-2000:]
print(json.dumps(out))
'''.replace('__REMOTE__',repr(REMOTE))

def main():
    p=argparse.ArgumentParser();p.add_argument('--collect',action='store_true');a=p.parse_args()
    r=subprocess.run(['ssh','94',PY,'-'],input=CODE.encode(),capture_output=True,check=True)
    data=json.loads(r.stdout);stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out=ROOT/('status_'+stamp+'.json');out.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(snapshot=str(out),completion=data['completion'],jobs={k:dict(summary=v['summary'],failure=v['failure'],status=v['status']['status'] if v['status'] else None) for k,v in data['jobs'].items()},log_tail=data['log_tail'])))
    if a.collect:
        if not data['completion'] or data['completion']['status']!='COMPLETED':raise ValueError('Queue not complete')
        dest=ROOT/'remote_completed_attempt2'
        if dest.exists():raise FileExistsError('Preserve previous collection')
        subprocess.run(['scp','-r','94:'+REMOTE,str(dest)],check=True)
        (ROOT/'collection_receipt.json').write_text(json.dumps(dict(status='COLLECTED',remote=REMOTE,local=str(dest),snapshot=str(out)),indent=2)+'\n')

if __name__=='__main__':main()
