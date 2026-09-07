"""Prepare a curated GitHub export; does not commit or push automatically."""
from pathlib import Path
import datetime
import json
import os
import re
import shutil
from urllib.parse import unquote

WORKSPACE=Path('E:/SHARE/光sar')
STAGE=Path('\\\\?\\C:\\Users\\MSI-PC\\rgbir_review_worktrees\\evidence-20260906')
BUNDLE=STAGE/'research_bundle'
ALLOWED={'.md','.py','.json','.jsonl','.yaml','.yml','.txt','.csv','.tsv','.png','.jpg','.svg','.pdf','.gz','.log','.sh'}

def main():
    roots=list((WORKSPACE/'08_实验日志').glob('2026-09-07_*'))
    roots += [WORKSPACE/'08_实验日志'/name for name in
              ('2026-09-06_ops_单卡并发与计划澄清','2026-09-06_train_OEv1随机选择对照',
               '2026-09-06_ops_OEv1优先级与对比实验','2026-09-06_audit_实验全景与设置对照')]
    roots += [WORKSPACE/'03_现行工程/SpaceNet6_OTD_official_reproduction/experiments/rgbir_task_conditional_v1', WORKSPACE/'03_现行工程/SpaceNet6_OTD_official_reproduction/experiments/rgbir_independent_kd_v2', WORKSPACE/'refine-logs']
    files=[]
    for root in roots:
        for p in root.rglob('*'):
            if not p.is_file() or p.suffix.lower() not in ALLOWED or '__pycache__' in p.parts:continue
            if p.name.startswith('publication_inventory') or p.name.endswith('.tar.gz'):continue
            # Raw diagnostic JSONL are preserved losslessly as gzip for manageable review diffs.
            if p.suffix=='.jsonl' and Path(str(p)+'.gz').exists():continue
            if p.stat().st_size>20*2**20:continue
            files.append(p)
    files += [WORKSPACE/'08_实验日志/README.md',WORKSPACE/'07_研究分析/RGBIR_Task_Conditional_KD_Codex_Spec_20260907.md']
    files += list((WORKSPACE/'07_研究分析').glob('RGBIR_Independent_Class_Loc_Formal_Spec_20260907*'))
    files += [WORKSPACE/'99_整理回执/20260907_IndependentKD实施新增目录.md']
    files += [WORKSPACE/'08_实验日志/2026-09-06_ops_GitHub完整审计包/README.md',WORKSPACE/'99_整理回执/README.md']
    pairs=[(p,BUNDLE/p.relative_to(WORKSPACE)) for p in sorted(set(files))]
    pairs += [(WORKSPACE/'README.md',BUNDLE/'workspace_context/WORKSPACE_README.md'),
              (WORKSPACE/'MANIFEST.md',BUNDLE/'workspace_context/WORKSPACE_MANIFEST.md'),
              (WORKSPACE/'AGENTS.md',BUNDLE/'workspace_context/WORKSPACE_AGENTS.md')]
    mapping={str(src.resolve()).lower():dst for src,dst in pairs}
    records=[];unresolved=[]
    for src,dst in pairs:
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        if src.suffix=='.md':
            content=src.read_text(encoding='utf-8-sig')
            def replace(match):
                raw=match.group(1);link=unquote(raw.strip().strip('<>')).replace('\\','/')
                if re.match(r'^(https?://|mailto:|#|app://|codex://)',link):return match.group(0)
                link,sep,anchor=link.partition('#');link=re.sub(r':\d+$','',link)
                candidates=[Path(link)] if re.match(r'^[A-Za-z]:/',link) else [src.parent/link,WORKSPACE/link]
                # Frozen nested source copies retain their original relative
                # links; resolve their explicit workspace section when present.
                for section in ('08_实验日志/','07_研究分析/','refine-logs/'):
                    if section in link:candidates.append(WORKSPACE/(section+link.split(section,1)[1]))
                target=None
                for candidate in candidates:
                    resolved=candidate.resolve();target=mapping.get(str(resolved).lower())
                    if target:break
                    try:
                        trial=BUNDLE/resolved.relative_to(WORKSPACE.resolve())
                        if trial.exists():target=trial;break
                        if trial.suffix=='.jsonl' and Path(str(trial)+'.gz').exists():target=Path(str(trial)+'.gz');break
                    except ValueError:pass
                if target:
                    relative=os.path.relpath(target,dst.parent).replace('\\','/')
                    return ']('+relative+('#'+anchor if sep else '')+')'
                unresolved.append({'document':dst.relative_to(STAGE).as_posix(),'target':raw})
                return ']（未导出的工作区路径：'+raw+'）'
            # Derived Markdown uses LF; raw code/evidence retains source bytes.
            dst.write_bytes(re.sub(r'\]\(([^\n)]*)\)',replace,content).encode('utf-8'))
        elif src.read_bytes()!=dst.read_bytes():raise AssertionError('Raw export differs')
        records.append({'source':src.as_posix(),'path':src.relative_to(WORKSPACE).as_posix(),
                        'repository_path':dst.relative_to(STAGE).as_posix(),'bytes':dst.stat().st_size})
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    report={'time':datetime.datetime.now().astimezone().isoformat(),'files':records,'unresolved_links':unresolved,
            'raw_evidence_policy':'Code/JSON/CSV/JSONL/gzip/images byte-preserved; Markdown links adapted; no weights/credentials/full dataset.'}
    (STAGE/('INDEPENDENT_KD_BUNDLE_MANIFEST_'+stamp+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (Path(__file__).parent/('publication_inventory_'+stamp+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'manifest':'INDEPENDENT_KD_BUNDLE_MANIFEST_'+stamp+'.json','files':len(records),'bytes':sum(r['bytes'] for r in records),'unresolved_links':len(unresolved)}))

if __name__=='__main__':main()
