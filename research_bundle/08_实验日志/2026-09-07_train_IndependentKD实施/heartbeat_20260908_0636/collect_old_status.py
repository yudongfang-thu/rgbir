"""Read only the five existing E200 runs; no checkpoint, GPU call, or hash."""
import csv
import datetime
import json
import math
from pathlib import Path
import subprocess

HERE = Path(__file__).absolute().parent
PREVIOUS = HERE.parent / 'heartbeat_20260908_0536/old_status.json'
PYTHON = '/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
REMOTE = r'''
import csv, datetime, io, json, time
from pathlib import Path
root=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign')
campaign=root/'artifacts/rgbir_independent_kd_v2_20260907/formal_C1_gpu5_attempt2'
runs={'C1_s'+str(s):campaign/'runs'/('C1_seed'+str(s)) for s in (42,0,123)}
runs.update({a:root/'runs/rgbir_task_conditional_c_attribution_20260907'/('full_'+a+'_s42_attempt1') for a in ('c_shuffled','c_same_modal')})
def small_json(path):
    if not path.exists():return None
    if path.stat().st_size>2000000:raise RuntimeError('Unexpected large JSON: '+str(path))
    return json.loads(path.read_text())
rows={}
for name,path in runs.items():
    progress_path=path/'progress.json'
    st=progress_path.stat()
    row=dict(run=str(path),progress=small_json(progress_path),progress_size=st.st_size,
             progress_mtime=st.st_mtime,progress_age_seconds=time.time()-st.st_mtime,
             completion=small_json(path/'completion_receipt.json'),
             failure=small_json(path/'failure_receipt.json'),
             evaluation=small_json(path/'evaluation_val.json'),
             evaluation_receipt=small_json(path/'eval_evidence/run_receipt.json'))
    row['top_level_receipt_names'] = sorted(p.name for p in path.iterdir() if p.is_file() and ('receipt' in p.name or 'failure' in p.name or 'evaluation' in p.name) and p.suffix=='.json')
    metrics=path/'results.csv'
    row['last_completed_epoch']=None
    if metrics.exists():
        if metrics.stat().st_size>2000000:raise RuntimeError('Unexpected large results.csv')
        values=list(csv.DictReader(io.StringIO(metrics.read_text())))
        if values:
            last={k.strip():v.strip() for k,v in values[-1].items() if k is not None and isinstance(v,str)}
            row['last_completed_epoch']=int(float(last['epoch']))
            row['last_completed_elapsed_seconds']=float(last['time'])
    rows[name]=row
print(json.dumps(dict(read_at=datetime.datetime.now().astimezone().isoformat(),runs=rows,
                     remote_mutations=False,new_gpu_tasks=0,new_hash_computed=False,
                     checkpoint_read=False,e8_inspected=False)))
'''

def main():
    prior=json.loads(PREVIOUS.read_text(encoding='utf-8'))
    result=subprocess.run(['ssh','94',PYTHON,'-'],input=REMOTE.encode('utf-8'),capture_output=True,check=True)
    current=json.loads(result.stdout)
    current['previous_snapshot']=str(PREVIOUS)
    current['previous_read_at']=prior['read_at']
    issues=[]
    endpoints=[]
    lines=['**旧五个长训只读健康状态；无新完整端点、评估或失败回执。**', '',
           '采集时间：'+current['read_at']+'；增量基准：'+prior['read_at']+'。', '',
           '|任务|当前 progress.epoch / 200|已完成轮数|成功更新（累计 / 增量）|AMP skips（累计 / 增量）|progress 年龄秒|',
           '|---|---:|---:|---:|---:|---:|']
    for name,row in current['runs'].items():
        p=row['progress']; old=prior['runs'][name]
        same=row['run']==old['run']
        delta=p['optimizer_updates']-old['progress']['optimizer_updates']
        skips=p['amp_skips']-old['progress']['amp_skips']
        row['since_previous']=dict(same_run=same,successful_updates=delta,amp_skips=skips)
        row['amp_accounting_exact']=p['optimizer_updates']+p['amp_skips']==p['update_attempts']
        row['finite_last_kd']=math.isfinite(p['last_kd'])
        if not same or delta<=0 or p['status']!='running' or not row['finite_last_kd'] or not row['amp_accounting_exact'] or row['progress_age_seconds']>180 or row['failure'] is not None:
            issues.append(name)
        if row['completion'] is not None or row['evaluation'] is not None or row['evaluation_receipt'] is not None:endpoints.append(name)
        lines.append('|{}|{}|{}|{} / +{}|{} / +{}|{:.1f}|'.format(name,p['epoch'],row['last_completed_epoch'],p['optimizer_updates'],delta,p['amp_skips'],skips,row['progress_age_seconds']))
    current['status']='READ_ONLY_HEALTHY_NO_NEW_ENDPOINT' if not issues and not endpoints else 'REVIEW_REQUIRED'
    current['health_issues']=issues;current['endpoint_runs']=endpoints
    if issues or endpoints:lines[0]='**存在需复核变化：'+repr(dict(issues=issues,endpoints=endpoints))+'。**'
    lines += ['', '五个任务均按原目录读取 progress、results.csv 及完成/评估/失败小回执；未读取大 checkpoint，未启动 GPU，未查询 E8，未计算新 hash。当前轮数是正在进行的 progress.epoch，已完成轮数单列。健康判断限于状态新鲜、成功更新增长、AMP 计数闭合及最后 KD 有限；不是新 AP 或收敛结论。', '', '[原始状态与远端路径](old_status.json)；[只读采集源码](collect_old_status.py)。']
    with (HERE/'old_status.json').open('x',encoding='utf-8') as stream:json.dump(current,stream,ensure_ascii=False,indent=2)
    with (HERE/'README.md').open('x',encoding='utf-8') as stream:stream.write('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=current['status'],read_at=current['read_at'],issues=issues,endpoints=endpoints,
                         runs={k:dict(epoch=v['progress']['epoch'],updates=v['progress']['optimizer_updates'],amp_skips=v['progress']['amp_skips'],delta=v['since_previous']) for k,v in current['runs'].items()}),ensure_ascii=False))

if __name__=='__main__':main()
