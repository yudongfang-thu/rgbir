"""Deploy reviewed code or collect a completed diagnostic; never alter training."""
from pathlib import Path
import argparse,datetime,json,shlex,subprocess
ROOT=Path(__file__).resolve().parent
PY='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
REMOTE='/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_evidence_priority_20260907/natural_flow_attempt2'

def main():
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');p.add_argument('--collect',action='store_true');a=p.parse_args()
    if a.launch:
        review=ROOT/'independent_review/ATTEMPT2_IMPORT_REVIEW.json'
        if not review.is_file():raise ValueError('Independent actual code review is required')
        subprocess.run(['ssh','94','mkdir',REMOTE],check=True)
        for name in ('diagnose_natural_flow.py','run_campaign.py','PROTOCOL.md'):
            subprocess.run(['scp',str(ROOT/name),'94:'+REMOTE+'/'+name],check=True)
        subprocess.run(['scp','-r',str(ROOT/'independent_review'),'94:'+REMOTE+'/independent_review'],check=True)
        command=shlex.quote(PY)+' '+shlex.quote(REMOTE+'/run_campaign.py')+' > '+shlex.quote(REMOTE+'/campaign.log')+' 2>&1'
        subprocess.run(['ssh','94','screen -dmS evidence_natural_flow_s20260907 bash -lc '+shlex.quote(command)],check=True)
        with (ROOT/'launch_receipt_attempt2.json').open('x') as f:json.dump(dict(status='DISPATCHED',remote=REMOTE,screen='evidence_natural_flow_s20260907'),f,indent=2)
        return
    code='''
import json
from pathlib import Path
root=Path(__REMOTE__)
def read(p):return json.loads(p.read_text()) if p.is_file() else None
out=dict(completion=read(root/'queue_attempt1/completion.json'),jobs={})
for d in ['llvip','drone']:
 for s in ['canary','full']:
  p=root/(d+'_'+s+'_attempt1');summary=read(p/'summary.json')
  if summary:
   summary={k:summary[k] for k in ['status','dataset','batches','seconds','diagnostic_status','resources']}
  out['jobs'][d+'_'+s]=dict(summary=summary,failure=read(p/'failure_receipt.json'),status=read(root/'queue_attempt1'/('natural_'+d+'_'+s+'_status.json')))
out['log_tail']=(root/'campaign.log').read_text()[-1500:] if (root/'campaign.log').is_file() else None
print(json.dumps(out))
'''.replace('__REMOTE__',repr(REMOTE))
    r=subprocess.run(['ssh','94',PY,'-'],input=code.encode(),capture_output=True,check=True);data=json.loads(r.stdout)
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    (ROOT/('status_'+stamp+'.json')).write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(data))
    if a.collect:
        if not data['completion'] or data['completion']['status']!='COMPLETED':raise ValueError('No completed queue')
        dest=ROOT/'remote_completed_attempt2'
        if dest.exists():raise FileExistsError('Preserve earlier collection')
        subprocess.run(['scp','-r','94:'+REMOTE,str(dest)],check=True)
        (ROOT/'collection_receipt.json').write_text(json.dumps(dict(remote=REMOTE,local=str(dest),status='COLLECTED'),indent=2)+'\n')

if __name__=='__main__':main()
