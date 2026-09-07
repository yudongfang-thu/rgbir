"""Curate this authorized stage only; no hashes, inference or remote execution."""
from pathlib import Path
from urllib.parse import unquote
import datetime
import json
import os
import re

WORK=Path('E:/SHARE/光sar')
REPO=Path(__file__).resolve().parents[2]
BUNDLE=REPO/'research_bundle'
ROOTS=[WORK/'08_实验日志'/name for name in (
    '2026-09-08_ops_训练吞吐诊断','2026-09-08_train_分类快速反馈E8','2026-09-07_audit_数据分析综合复盘')]
EXTRAS=[WORK/p for p in ('08_实验日志/README.md','07_研究分析/RGBIR数据分析综合报告_20260908.md',
    '08_实验日志/2026-09-07_train_IndependentKD实施/README.md')]
OMIT_PARTS={'__pycache__','.git','.pytest_cache','runtime_sources'}
ALLOWED={'.py','.md','.json','.jsonl','.yaml','.yml','.csv','.tsv'}
PROCESS_KEYS={'cmdline','command_line','processes','all_processes','ps_output','process_rows','commands'}


def sensitive_shape(value):
    if isinstance(value,dict):
        return any(k.lower() in PROCESS_KEYS or sensitive_shape(v) for k,v in value.items())
    if isinstance(value,list):return any(sensitive_shape(v) for v in value)
    return False


def main():
    selected=[];excluded=[]
    for source in sorted(set([p for root in ROOTS for p in root.rglob('*') if p.is_file()]+EXTRAS)):
        relative=source.relative_to(WORK)
        reason=None
        if any(part in OMIT_PARTS for part in relative.parts):reason='Cache or redundant runtime snapshot'
        elif source.name.startswith('host_profile'):reason='Whole-host process snapshot explicitly excluded'
        elif source.name=='snapshot.json' and any(part.startswith('heartbeat_') for part in relative.parts):reason='Heartbeat raw host/resource snapshot excluded; project status prose/checks retained'
        elif source.suffix.lower() not in ALLOWED and not (source.suffix.lower() in {'.svg','.png'} and '2026-09-07_audit_数据分析综合复盘' in relative.parts):reason='Not required small code/document/numeric evidence; no images, weights, state or logs'
        elif source.stat().st_size>8*2**20:reason='Outside bounded small-artifact publication'
        elif source.suffix.lower()=='.json':
            try:
                if sensitive_shape(json.loads(source.read_text(encoding='utf-8-sig'))):reason='Process/command inventory fields excluded'
            except (UnicodeDecodeError,json.JSONDecodeError):reason='Non-readable JSON excluded'
        if reason:excluded.append(dict(source=str(source),bytes=source.stat().st_size,reason=reason))
        else:selected.append(source)
    mapping={str(p.resolve()).lower():BUNDLE/p.relative_to(WORK) for p in selected}
    records=[]
    for source in selected:
        dest=mapping[str(source.resolve()).lower()]
        dest.parent.mkdir(parents=True,exist_ok=True)
        original=source.read_bytes()
        if source.suffix.lower()=='.md':
            def adapt(match):
                target=unquote(match.group(1).strip().strip('<>')).replace('\\','/')
                if re.match(r'^(https?://|mailto:|#|app://|codex://)',target):return match.group(0)
                name,sep,anchor=target.partition('#');name=re.sub(r':\d+$','',name)
                src=(Path(name) if re.match(r'^[A-Za-z]:/',name) else source.parent/name).resolve()
                dst=mapping.get(str(src).lower())
                if dst is None:
                    try:dst=BUNDLE/src.relative_to(WORK.resolve())
                    except ValueError:return match.group(0)
                if dst.exists() or str(src).lower() in mapping:
                    return ']('+os.path.relpath(dst,dest.parent).replace('\\','/')+('#'+anchor if sep else '')+')'
                return ']（服务器/本地保留，未包含于本阶段发布：'+target+'）'
            payload=re.sub(r'\]\(([^\n)]*)\)',adapt,original.decode('utf-8-sig')).encode('utf-8')
            mode='Markdown navigation adapted; scientific wording retained'
        else:payload=original;mode='byte exact'
        dest.write_bytes(payload)
        assert dest.read_bytes()==payload
        assert source.read_bytes()==original
        records.append(dict(source=str(source),source_bytes=len(original),repository_path=dest.relative_to(REPO).as_posix(),
            bytes=len(payload),verification=mode))
    manifest=dict(status='CURATED_NOT_YET_PUSHED',time=datetime.datetime.now().astimezone().isoformat(),
        scope='Failed block16 and selected-only performance evidence, actual 24-update exact trajectory, training-cadence measurements, E8 launch and prior completed analysis synthesis; no new AP',
        records=records,excluded=excluded,new_file_hashes_computed=False,source_results_modified=False,
        forbidden_payloads_uploaded=False,exclusion_policy='No weights/state/raw .pt, original dataset images, caches, credentials, whole-host process/other-user command inventories; only derived synthesis charts are newly included')
    (REPO/'PERFORMANCE_E8_INCREMENT_20260908.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    note='> **2026-09-08 性能修复与 E8 快速反馈**：原 block16 的真实训练轨迹失败保留；selected-only 在各 24 次成功更新上实现学习损失、梯度、模型/optimizer/EMA 状态逐位一致。正常日志频率仍需约 2.48 秒/C1批，已按先验预算启动共同 E8 的 N/C0/C1 单 seed 队列，尚无新 AP。请读[性能与失败证据](research_bundle/08_实验日志/2026-09-08_ops_训练吞吐诊断/README.md)、[E8 设置与启动](research_bundle/08_实验日志/2026-09-08_train_分类快速反馈E8/README.md)、[既有数据分析综合报告](research_bundle/07_研究分析/RGBIR数据分析综合报告_20260908.md)。旧 E200 继续，本次不声称 E200 等价或方法增益。\n\n'
    for path,prefix in [(REPO/'README.md',note),(BUNDLE/'README.md',note.replace('](research_bundle/',']('))]:
        old=path.read_text(encoding='utf-8')
        if not old.startswith(prefix):path.write_text(prefix+old,encoding='utf-8')
    print(json.dumps(dict(files=len(records),bytes=sum(r['bytes'] for r in records),excluded=len(excluded)),ensure_ascii=False))


if __name__=='__main__':main()
