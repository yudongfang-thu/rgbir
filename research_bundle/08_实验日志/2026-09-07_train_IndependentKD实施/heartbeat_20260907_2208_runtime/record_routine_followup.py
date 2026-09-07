"""Record routine snapshots without changing runs or interpreting intermediate AP."""
import argparse
import json
import math
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--current',type=Path,required=True)
p.add_argument('--previous',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
state,previous=read(a.current),read(a.previous)
cards={}
for row in state['project_cuda']:
    card=cards.setdefault(row['gpu'],dict(pids=[],project_vram_mib=0))
    card['pids'].append(row['pid']);card['project_vram_mib']+=row['vram_mib']
for gpu in state['gpus']:
    if gpu['index'] in cards:
        cards[gpu['index']].update(total_mib=gpu['total_mib'],whole_card_free_mib=gpu['free_mib'])
bound={pid for lease in state['leases']['leases'].values() for pid in lease.get('observed_cuda_pids',[])}
checks=dict(three_physical_cards=len(cards)<=3,two_cuda_per_card=all(len(v['pids'])<=2 for v in cards.values()),
    project_vram_under_70pct=all(v['project_vram_mib']<.7*v['total_mib'] for v in cards.values()),
    whole_card_free_over_2gib=all(v['whole_card_free_mib']>=2048 for v in cards.values()),
    rss_under_300gb=state['project_rss_gib']*2**30<300e9,
    recorded_lease_coverage=all(row['pid'] in bound for row in state['project_cuda']))
progress={}
for name,row in state['runs'].items():
    prior=previous['runs'][name]
    now,old=row['progress'],prior['progress']
    progress[name]=dict(epoch=now['epoch'],successful_update_increase=now['optimizer_updates']-old['optimizer_updates'],
        amp_skip_increase=now['amp_skips']-old['amp_skips'],same_run=row['run']==prior['run'],
        complete=bool(row['completion']),failed=bool(row['failure']),evaluation_exists=row['evaluation_exists'],
        progress_status=now['status'],finite_last_kd=math.isfinite(now['last_kd']))
stable=all(checks.values()) and all(r['successful_update_increase']>0 and r['same_run']
    and not r['complete'] and not r['failed'] and not r['evaluation_exists'] and r['finite_last_kd']
    and r['progress_status']=='running' for r in progress.values())
summary=dict(status='NO_ACTIONABLE_CHANGE' if stable else 'REVIEW_REQUIRED',read_at=state['read_at'],
    previous_read_at=previous['read_at'],current_snapshot=str(a.current.resolve()),previous_snapshot=str(a.previous.resolve()),
    progress=progress,resource_checks=checks,per_gpu=cards,project_rss_gib=state['project_rss_gib'],
    scientific_decisions_changed=False,training_mutations=0,github_sync_needed=not stable,
    scope='Routine training/lease/resource observation; no new endpoint or AP interpretation')
a.output.mkdir(exist_ok=False)
(a.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
lines=['# 例行运行核对 '+state['read_at'],'',
       '**'+('原任务均持续更新，资源合规，无新端点或故障，不改变执行计划。' if stable else '发现需要进一步复核的状态，见原始证据。')+'**','',
       '|任务|progress.epoch|新增成功更新|新增AMP skips|','|---|---|---|---|']
for name,r in progress.items():lines.append('|%s|%s|%s|%s|'%(name,r['epoch'],r['successful_update_increase'],r['amp_skip_increase']))
lines+=['','增量相对于 '+previous['read_at']+'。项目RSS %.2fGiB，物理卡%s，各卡项目CUDA任务%s。'%(state['project_rss_gib'],sorted(cards),[len(cards[k]['pids']) for k in sorted(cards)]),
    '','[检查结论](summary.json)；[当前原始快照](../'+a.current.name+')；[前次快照](../'+a.previous.name+')。采集和记录脚本副本在本目录。',
    '','本轮不启动GPU任务，不改变冻结release、lambda、随机流、方法门或端点。已有C0专项复核和L几何证据阻塞保持；已接受逐类接口不重复实施。运行原始大产物仍在94既定campaign，路径见快照。',
    '','无实质阶段变化时仅本地记录，不单独发布GitHub或主动通知；继续等待E200完整端点及独立评价。']
(a.output/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
for name in ('record_routine_followup.py','monitor_running_only.py'):
    (a.output/name).write_bytes((Path(__file__).parent/name).read_bytes())
print(json.dumps(dict(status=summary['status'],checks=checks,progress=progress,output=str(a.output)),ensure_ascii=False))
