"""Add dated scope corrections and safe resource excerpts; preserve all executed outputs."""
from pathlib import Path
import json

root=Path(__file__).parent
workspace=root.parent.parent

def edit_lines(path, transform):
    data=path.read_bytes()
    newline='\r\n' if b'\r\n' in data else '\n'
    lines=transform(data.decode('utf-8').splitlines())
    path.write_bytes((newline.join(lines)+'\n').encode('utf-8'))

status='**状态：两次同批有界推理均已完成，零训练更新。旧 assigned 定义的教师正确/学生错误为 9 个，原生 NMS 后为 1 个且已选中；32 个 selected 中 31 个双方检出。旧低置信候选不能直接解释为真实漏检。详见 [最终报告](FINAL_REPORT.md) 与 [原生四定义读出](witness_analysis_final/README.md)。**'
edit_lines(root/'README.md',lambda ls:[status if l.startswith('**状态：') else l for l in ls])
edit_lines(root/'README.md',lambda ls:[
    '后续[原生检测见证固定计划](WITNESS_FOLLOWUP_PLAN.md)现已完成：80 行旧对象和选择逐项复现，新增的原生匹配读出与结论见[最终报告](FINAL_REPORT.md)。原始首批读出与全部回执保留。'
    if l.startswith('后续[原生检测见证固定计划]') else l for l in ls])

index=workspace/'08_实验日志/README.md'
row='| 2026-09-08 | [probe_同帧选择覆盖](2026-09-08_probe_同帧选择覆盖/README.md) | probe | 两次零更新推理完成；32图80GT的旧候选机会9→原生检出机会1，selected31/32双方已检出；修正低置信解释，转向已有完整dev缓存核对 |'
edit_lines(index,lambda ls:[row if '[probe_同帧选择覆盖](' in l else l for l in ls])

banner='> **同帧检测定义核对已完成（2026-09-08）**：[结果及证据修正](08_实验日志/2026-09-08_probe_同帧选择覆盖/FINAL_REPORT.md)。LLVIP 固定32张训练图、80GT，旧 assigned 低置信机会9个，原生NMS后真实检出差异仅1个且已选中；不代表完整dev。旧候选低置信计数不能直接当漏检空间，下一步优先CPU复核已有完整dev预测。'
def add_banner(ls):
    if any(l.startswith('> **同帧检测定义核对已完成') for l in ls):
        return [banner if l.startswith('> **同帧检测定义核对已完成') else l for l in ls]
    return ls[:2]+[banner,'']+ls[2:]
edit_lines(workspace/'README.md',add_banner)

corrections=[
    (workspace/'07_研究分析/RGBIR数据分析综合报告_20260908.md',
     '> **2026-09-08 后续证据修正**：[同帧原生检测核对](../08_实验日志/2026-09-08_probe_同帧选择覆盖/FINAL_REPORT.md)发现，旧 GT 辅助 assigned anchor 低置信不等于原生 NMS 后漏检。本报告中 raw 对象探针的“修复/低置信”计数须按该定义解读，不能直接当实际漏检或可达 AP；完整 dev AP/TIDE 不受此定义修正影响。下文保留截至9月7日的历史分析，未按新单批比例更改旧数字。'),
    (workspace/'08_实验日志/2026-09-08_probe_快速方向筛选/DIRECTIONS_AND_PROXY_LIMITS.md',
     '> **2026-09-08 后续进展**：本文末尾的同帧探针及原生 NMS 见证现已完成，见[新报告](../2026-09-08_probe_同帧选择覆盖/FINAL_REPORT.md)。本批旧定义9个机会中8个实际已有原生正确检出；原生机会仅1个且入选。本文旧200dev的低置信数字是 assigned 候选状态，不能直接称漏检；不改原数，也不以这批训练图外推dev。下一项改为先复用完整dev预测核对真实错误谱。')]
for path,note in corrections:
    def add_note(ls,note=note):
        if any(l.startswith('> **2026-09-08 后续') for l in ls):return ls
        return ls[:2]+[note,'']+ls[2:]
    edit_lines(path,add_note)

rows=[]
for directory in ['evidence_1255_final','witness_evidence_1315_final']:
    q=root/directory/'queue'
    profile_path=next(q.glob('*_resource_profile.json'))
    a=json.loads(profile_path.read_text(encoding='utf-8'))
    completion=json.loads((q/'completion.json').read_text(encoding='utf-8'))
    admission=json.loads(next(q.glob('*_admission.json')).read_text(encoding='utf-8'))
    sample=a['samples']
    rows.append(dict(source=profile_path.relative_to(root).as_posix(),
       status=a['status'],exit_code=a['exit_code'],monitor_errors=a['monitor_errors'],
       gpu_ids=a['launch']['gpu_ids'],sample_count=len(sample),
       sampled_minimum_free_mib=a['minimum_free_mib'],
       sampled_maximum_project_rss_mib=max(s['project_rss_mib'] for s in sample),
       reservation_vram_mib=a['launch']['expected_vram_mib'],
       reservation_rss_mib=a['launch']['expected_rss_mib'],
       queue_seconds=completion['seconds'],new_resource_pool=False,
       admission_status=admission.get('status'),
       warning='Sampled guard observations, not continuous exact process VRAM/RSS peaks. Full original records remain local and on 94.'))
out=root/'RESOURCE_EXCERPT.json'
if out.exists():raise ValueError('Do not overwrite a resource excerpt')
out.write_text(json.dumps(dict(status='COMPLETED_RESOURCE_EXCERPTS',probes=rows,
    new_hash_computed=False),ensure_ascii=False,indent=2),encoding='utf-8')
print('NARRATIVE_UPDATED_AND_RESOURCE_EXCERPTS_WRITTEN')
