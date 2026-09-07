# -*- coding: utf-8 -*-
"""Bounded descriptive bridge over the existing 200-dev assigned-object cache."""
import collections,importlib.util,json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]/'2026-09-07_probe_Baseline蒸馏机会重诊断'
SOURCE=BASE/'new_probe_analysis/analyze_baseline_probe.py'
INPUT=BASE/'remote_exports/dronevehicle_full_attempt1/objects.jsonl'
STATES=['no_candidate','low_confidence','class_only','localization_only','class_and_localization','correct']
NAMES=['car','freight car','truck','bus','van']
def stat(p):
    s=p.stat();return dict(path=str(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns)
def main():
    out=HERE/'baseline_class_bridge_v1';out.mkdir(exist_ok=False)
    (out/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    copied=out/'analyze_baseline_probe_source.py';copied.write_bytes(SOURCE.read_bytes())
    assert copied.read_bytes()==SOURCE.read_bytes()
    spec=importlib.util.spec_from_file_location('old_baseline',copied);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    allrows=old.read_rows(INPUT);rows=[r for r in allrows if r['split']=='val' and not r.get('is_background',False)]
    assert len(rows)==3084 and len(set(r['image'] for r in rows))==200
    assert (old.CONFIDENCE,old.HIT_IOU,old.PAIR_IOU)==(.25,.5,.5)
    results={};object_rows=[]
    for cls in [None]+list(range(5)):
        selected=[r for r in rows if cls is None or r['class']==cls];key='all' if cls is None else str(cls)
        mask=np.ones(len(selected),bool)
        tq=old.opportunity_summary(selected,mask,teacher_model='T42',require_pair=True)
        n0q=old.opportunity_summary(selected,mask,teacher_model='N0',require_pair=False)
        pair=np.array([r.get('paired_gt_iou') is not None and r['paired_gt_iou']>=old.PAIR_IOU for r in selected])
        n0paired=old.opportunity_summary(selected,pair,teacher_model='N0',require_pair=False)
        states={m:[old.object_view(r,m) for r in selected] for m in ['N42','T42','N0']}
        ncorrect=np.array([r['correct'] for r in states['N42']]);tcorrect=np.array([r['correct'] for r in states['T42']]);n0correct=np.array([r['correct'] for r in states['N0']])
        trepair=pair & ~ncorrect & tcorrect;n0repair=~ncorrect & n0correct
        tharm=pair & ncorrect & ~tcorrect;n0harm=ncorrect & ~n0correct
        assert int(trepair.sum())==tq['repair']['n'] and int(tharm.sum())==tq['harm_if_teacher_replaces_native']['n']
        assert int(n0repair.sum())==n0q['repair']['n'] and int(n0harm.sum())==n0q['harm_if_teacher_replaces_native']['n']
        result=dict(class_name='all' if cls is None else NAMES[cls],gt=len(selected),N_correct=int(ncorrect.sum()),N_errors=int((~ncorrect).sum()),
            N_states={state:sum(x['state']==state for x in states['N42']) for state in STATES},
            T=tq,N0=n0q,N0_on_T_eligible=n0paired,
            common_eligible=dict(gt=int(pair.sum()),N_errors=int((pair&~ncorrect).sum()),N_correct=int((pair&ncorrect).sum()),
                both_repair=int((trepair&n0repair).sum()),T_only_repair=int((trepair&~n0repair).sum()),N0_only_repair=int((pair&n0repair&~trepair).sum())),
            repairs_by_state={model:{state:int(sum(repaired[j] and x['state']==state for j,x in enumerate(states['N42']))) for state in STATES} for model,repaired in [('T',trepair),('N0',n0repair)]})
        assert sum(result['N_states'].values())==result['gt']
        results[key]=result
        if cls is None:
            for j,r in enumerate(selected):object_rows.append(dict(object_id=r['object_id'],image=r['image'],gt_class=r['class'],paired=bool(pair[j]),N_state=states['N42'][j]['state'],T_state=states['T42'][j]['state'],N0_state=states['N0'][j]['state'],T_repair=bool(trepair[j]),T_harm=bool(tharm[j]),N0_repair=bool(n0repair[j]),N0_harm=bool(n0harm[j])))
    a=results['all'];assert (a['N_correct'],a['T']['repair']['n'],a['T']['harm_if_teacher_replaces_native']['n'],a['N0']['repair']['n'],a['N0']['harm_if_teacher_replaces_native']['n'])==(2386,461,193,221,192)
    for k in ['gt','N_correct','N_errors']:assert sum(results[str(c)][k] for c in range(5))==a[k]
    for m in ['T','N0']:
        for k in ['repair','harm_if_teacher_replaces_native']:assert sum(results[str(c)][m][k]['n'] for c in range(5))==a[m][k]['n']
    old.write_json(out/'summary.json',dict(input=stat(INPUT),source=stat(SOURCE),source_copy_byte_identical=True,images=200,settings=dict(confidence=.25,IoU=.5,pair_IoU=.5,view='assigned'),results=results))
    with (out/'object_state_bridge.jsonl').open('w',encoding='utf-8') as f:
        for r in object_rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    lines=['# Drone 200dev对象按全五类桥接','',
        '> 这是同一冻结候选规则下教师/独立RGB互补的对象证据，不能外推完整AP或证明可达KD收益。','',
        '## 设置与完整性','',
        '复用原200dev/3084 GT的assigned对象候选，N42/T42/N0同640方形letterbox。全部五类同规则；conf=.25、同类RGB GT IoU=.50，T要求标签pair IoU≥.50，N0无需IR标签关联。原导出粗候选阈值.05下缺候选仍单列；low_confidence互斥状态优先，其可能同时有类/框问题。未新增推理、读出或阈值。','',
        '原分析函数从源码字节相等副本调用；原总体2386正确、T修复461/损伤193、N0修复221/损伤192完全复现，各类加和闭合。','',
        '## 修复与潜在损伤（括号分母明确）','',
        '|GT类别|GT|N错误|N正确|T eligible GT|T修复/N错误|T损伤/N正确|N0修复/N错误|N0损伤/N正确|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    def cell(n,d):return f'{n}/{d} ({100*n/d:.2f}%)' if d else f'{n}/{d} (NA)'
    for key,r in results.items():
        lines.append('|'+r['class_name']+'|'+ '|'.join(map(str,[r['gt'],r['N_errors'],r['N_correct'],r['T']['comparison_eligible_gt']['n']]))+'|'+ '|'.join([cell(r[m][v]['n'],r[den]) for m,v,den in [('T','repair','N_errors'),('T','harm_if_teacher_replaces_native','N_correct'),('N0','repair','N_errors'),('N0','harm_if_teacher_replaces_native','N_correct')]])+'|')
    lines+=['','T的上述repair/N错误分母含未配对N错误；N0同RGB身份可用全部GT。这些比例不能直接相减。共同eligible比较另列如下，完整JSON还提供本类全部GT比例。','',
        '|类别|共同eligible GT|其中N错误|T修复|N0修复|共同修复|T-only|N0-only|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in results.values():
        q=r['common_eligible'];lines.append('|'+r['class_name']+'|'+'|'.join(map(str,[q['gt'],q['N_errors'],r['T']['repair']['n'],r['N0_on_T_eligible']['repair']['n'],q['both_repair'],q['T_only_repair'],q['N0_only_repair']]))+'|')
    lines+=['','## 固定state分解：N计数 / T修复 / N0修复','',
        '|类别|no_candidate|low_confidence|class_only|localization_only|class_and_localization|','|---|---:|---:|---:|---:|---:|']
    for r in results.values():lines.append('|'+r['class_name']+'|'+'|'.join(f"{r['N_states'][s]} / {r['repairs_by_state']['T'][s]} / {r['repairs_by_state']['N0'][s]}" for s in STATES[:-1])+'|')
    lines+=['','每格三个数依次是本类N该state对象数、其中T可修复数、其中N0可修复数；state发生率分母是本类GT，state修复率分母是首个数。完整JSON还保存重叠的class/confidence/localization维度，避免互斥优先级掩盖复合错误。','',
        '## 解释边界','',
        '- 完整TIDE的macro AP给每类相同权重；本表计数给每个GT同权重，并使用GT关联候选，不包含完整假阳性排序。两种证据的分母和匹配不同。','- 本表T/N0差异仅是不同模型的互补，既有teacher/native recipe差异与单seed限制仍在，不构成纯IR因果贡献。','- T-only候选正确支持存在独立RGB未提供的对象信息，但不能证明其分数可用于完整检测排序，更不证明学生可学得。','- 模型、图像集合、GT辅助、画布和post-NMS合同均保留，不能从200图外推1469图AP。','',
        '产物：`summary.json`全五类、`object_state_bridge.jsonl`全部3084对象复核入口、`analyze_baseline_probe_source.py`原函数副本。协议：`../BRIDGE_PROTOCOL.md`。']
    (out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('Bridge completed',flush=True)
if __name__=='__main__':main()
