"""CPU-only cross-tab on identical cached object rows; never a training join."""
import ast
from collections import Counter
import csv
import json
from pathlib import Path
import shutil
import numpy as np

HERE=Path(__file__).resolve().parent
LOG=HERE.parents[1]
BASE=LOG/'2026-09-07_probe_Baseline蒸馏机会重诊断'
ANALYZER=BASE/'new_probe_analysis/analyze_baseline_probe.py'


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def load_state():
    tree=ast.parse(ANALYZER.read_text(encoding='utf-8'))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='object_view')
    env=dict(np=np,CONFIDENCE=.25,HIT_IOU=.5)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(ANALYZER),'exec'),env)
    return env['object_view']


def counts(rows):
    repair=[r for r in rows if r['teacher_repair']]
    c=dict(gt=len(rows),native_correct=sum(r['native_correct'] for r in rows),native_errors=sum(not r['native_correct'] for r in rows),
        teacher_repairs=len(repair),teacher_potential_harms=sum(r['teacher_potential_harm'] for r in rows),
        repair_states=dict(Counter(r['native_state'] for r in repair)),
        repair_cached_anchor_yes=sum(r['cached_anchor_reference'] for r in repair),
        repair_cached_anchor_no=sum(not r['cached_anchor_reference'] for r in repair),
        repair_common_valid_roi_yes=sum(r['cached_common_valid_region'] for r in repair),
        repair_anchor_and_roi_yes=sum(r['cached_anchor_reference'] and r['cached_common_valid_region'] for r in repair),
        actual_selected_coverage=None)
    c['repair_state_proxy_table']={state:dict(n=len(bucket),anchor_yes=sum(r['cached_anchor_reference'] for r in bucket),
        common_roi_yes=sum(r['cached_common_valid_region'] for r in bucket),
        anchor_and_roi_yes=sum(r['cached_anchor_reference'] and r['cached_common_valid_region'] for r in bucket))
        for state in sorted({r['native_state'] for r in repair}) for bucket in [[r for r in repair if r['native_state']==state]]}
    return c


def run():
    view=load_state();allrows=[];datasets={};sources=[]
    for ds in ('llvip','dronevehicle'):
        path=BASE/'remote_exports'/(ds+'_full_attempt1')/'objects.jsonl'
        raw=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        ids=[r['object_id'] for r in raw]
        if len(ids)!=len(set(ids)):raise ValueError('Duplicate static cache object identity')
        source=path.stat();sources.append(dict(path=str(path),bytes=source.st_size,mtime_ns=source.st_mtime_ns))
        output=[]
        for r in raw:
            if r.get('is_background',False):continue
            if r['split'] not in ('train','val'):raise ValueError('Unexpected split')
            if r['object_id']!=r['image']+'::gt:'+str(r['rgb_gt_local_index']):raise ValueError('Static object ID/GT row differs')
            if r['pair_info']['augmentation']!='centered_letterbox_only':raise ValueError('Unexpected static frame')
            n,t=view(r,'N42'),view(r,'T42')
            pair=r.get('paired_gt_iou') is not None and r['paired_gt_iou']>=.5
            flag=r['anchor_has_reference_candidate']
            if type(flag) is not bool:raise ValueError('Missing boolean static anchor proxy')
            common=any(r['N42']['regions'][level]['valid'] and r['T42']['regions'][level]['valid'] for level in ('P3','P4'))
            output.append(dict(dataset=ds,split=r['split'],object_id=r['object_id'],image=r['image'],
                rgb_gt_local_index=r['rgb_gt_local_index'],gt_class=r['class'],source_group=r['source_group'],
                native_state=n['state'],native_correct=n['correct'],teacher_correct=t['correct'],paired=pair,
                teacher_repair=bool(pair and not n['correct'] and t['correct']),
                teacher_potential_harm=bool(pair and n['correct'] and not t['correct']),
                cached_anchor_reference=flag,cached_common_valid_region=bool(common),actual_selected=None))
        expected=(643,404,146) if ds=='llvip' else (3084,2386,461)
        dev=counts([r for r in output if r['split']=='val'])
        if tuple(dev[k] for k in ('gt','native_correct','teacher_repairs'))!=expected:raise ValueError('Original dev counts did not reproduce')
        datasets[ds]={split:dict(total=counts([r for r in output if r['split']==split]),
            by_class={str(c):counts([r for r in output if r['split']==split and r['gt_class']==c]) for c in sorted({r['gt_class'] for r in output})})
            for split in ('train','val')}
        allrows.extend(output)
    report=dict(status='COMPLETED_STATIC_CACHE_PROXY_AUDIT',datasets=datasets,sources=sources,
        original_state_source=str(ANALYZER),fixed_thresholds=dict(confidence=.25,hit_iou=.5,pair_iou=.5),
        actual_C1_F_selection_join='NOT_IDENTIFIABLE_FROM_CURRENT_CACHES',actual_selected_coverage=None,
        no_training_or_dev_join=True,no_AP_oracle_join=True,proxy_is_gate_bound=False,new_hash_computed=False,gpu_execution=False,
        missing_contract=['Same forward/frame snapshot identity for both detection-error labels and C selection',
            'Augmented global GT row to stable original source GT row mapping',
            'Actual per-object pair/valid/ref_candidate/teacher_correct/q/eligible/selected with per-batch ranking denominator',
            'Actual student error state on same forward, separated from mature fixed reference R',
            'Full predicted FP association if claiming AP-error coverage, rather than one row per GT'],
        model_identity_caveat='LLVIP cached N42 has same checkpoint role as current initial S/R; Drone cached N42 differs from current frozen R. Neither same path nor original GT ID aligns different augmentation frames.')
    with (HERE/'summary.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,ensure_ascii=False,allow_nan=False)
    with (HERE/'objects.csv').open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(allrows[0]));writer.writeheader();writer.writerows(allrows)
    lines=['# 静态对象机会 × 缓存代理（已执行 CPU）','',
        '不能连接为实际 C1/F selected 覆盖；本表只在同一静态缓存行内交叉统计，不跨 train/dev 或增强帧。',
        '', '| 数据集/分割 | GT | N错误 | IR修复 | 修复中旧anchor有/无 | 修复中旧ROI有效 | 两旧代理均有 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for ds,splits in datasets.items():
        for split,data in splits.items():
            v=data['total'];lines.append(f"| {ds}/{split} | {v['gt']} | {v['native_errors']} | {v['teacher_repairs']} | {v['repair_cached_anchor_yes']}/{v['repair_cached_anchor_no']} | {v['repair_common_valid_roi_yes']} | {v['repair_anchor_and_roi_yes']} |")
    lines+=['','逐状态、逐类结果见 summary.json，逐对象可复核表见 objects.csv。Dev原计数严格复现 LLVIP643/404正确/146修复、Drone3084/2386正确/461修复。',
        '', '旧anchor是P3/P4的一对一空间归属，现行C门使用全稠密any-candidate；旧ROI还有公共窗口/padding差异。因此“旧anchor无”不能等同当前gate排除，两旧代理均有也不能等同eligible/selected。这份表不是上界、AP贡献、训练修复或不可学比例。',
        '', '实际selected缺同forward/frame ID、增强GT→原GT映射和每对象q/eligible/selected。现有训练selected使用增强batch全局GT行；静态object_id使用原图局部GT行。只靠同路径或GT行号强连会混淆对象/增强状态。',
        '', '最小后续接口：固定一个既有32图真实批次，在一次S/T/R前向后共同导出每对象原始/增强双GT ID与坐标、S/R/T各自候选和固定状态、原selector的全过滤链/质量/selected/排名分母；记录同一个forward/frame ID。无需梯度或优化，不换阈值。已有低阈值完整dev输出可作独立错误图谱，但缺全稠密窗口不能恢复原selector。',
        '', '本次未推理、未GPU、未计算新hash；不依据训练/开发样本差值推过拟合。']
    (HERE/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    shutil.copyfile(__file__,HERE/'executed_source.py')
    print(json.dumps({ds:{sp:d['total'] for sp,d in v.items()} for ds,v in datasets.items()},ensure_ascii=False))


if __name__=='__main__':run()
