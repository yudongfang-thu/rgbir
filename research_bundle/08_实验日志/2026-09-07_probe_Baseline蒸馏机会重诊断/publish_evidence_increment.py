"""Curate this diagnostic increment; retain raw numeric bytes, adapt only Markdown links."""
from pathlib import Path
import datetime,gzip,json,os,re,shutil
from urllib.parse import unquote
WORK=Path('E:/SHARE/光sar');ROOT=Path(__file__).resolve().parent
REPO=Path('C:/Users/MSI-PC/rgbir_review_worktrees/evidence-20260906');BUNDLE=REPO/'research_bundle'
files=[]
for folder in (ROOT,WORK/'08_实验日志/2026-09-07_audit_定位补救与准入修订'):
    files += [p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='publication_receipt.json']
files += [WORK/'08_实验日志/README.md',WORK/'99_整理回执/20260907_IndependentKD实施新增目录.md']
records=[];excluded=[];mapping={}
for p in files:
    if p.name=='features.npz' or p.stat().st_size>32*2**20 and p.suffix!='.jsonl':
        excluded.append({'source':str(p),'bytes':p.stat().st_size,'reason':'large intermediate array; remote path in collection receipt'});continue
    dest=BUNDLE/p.relative_to(WORK)
    if p.suffix=='.jsonl':dest=dest.with_suffix('.jsonl.gz')
    mapping[str(p.resolve()).lower()]=dest
for p in sorted(files):
    dest=mapping.get(str(p.resolve()).lower())
    if dest is None:continue
    dest.parent.mkdir(parents=True,exist_ok=True)
    if p.suffix=='.jsonl':
        with p.open('rb') as src,gzip.open(dest,'wb',compresslevel=6) as target:shutil.copyfileobj(src,target)
    elif p.suffix=='.md':
        def replace(m):
            raw=unquote(m.group(1).strip().strip('<>')).replace('\\','/')
            if re.match(r'^(https?://|mailto:|#|app://|codex://)',raw):return m.group(0)
            path,sep,anchor=raw.partition('#');path=re.sub(r':\d+$','',path)
            source=Path(path) if re.match(r'^[A-Za-z]:/',path) else p.parent/path
            source=source.resolve();target=mapping.get(str(source).lower())
            if target is None:
                try:target=BUNDLE/source.relative_to(WORK.resolve())
                except ValueError:return ']('+raw+')'
            if target.exists() or str(source).lower() in mapping:
                return ']('+os.path.relpath(target,dest.parent).replace('\\','/')+('#'+anchor if sep else '')+')'
            return ']（服务器/本地中间产物：'+raw+'）'
        dest.write_bytes(re.sub(r'\]\(([^\n)]*)\)',replace,p.read_text(encoding='utf-8-sig')).encode('utf-8'))
    else:dest.write_bytes(p.read_bytes())
    records.append({'source':str(p),'repository_path':str(dest.relative_to(REPO)).replace('\\','/'),'bytes':dest.stat().st_size,'raw_jsonl_compressed':p.suffix=='.jsonl'})
manifest={'time':datetime.datetime.now().astimezone().isoformat(),'records':records,'excluded':excluded,'scope':'2448-image current-baseline diagnostic, no new KD training AP','no_test_access':True}
(REPO/'BASELINE_INFORMATION_INCREMENT_20260907.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
nav='research_bundle/08_实验日志/2026-09-07_probe_Baseline蒸馏机会重诊断'
readme=REPO/'README.md';old=readme.read_text(encoding='utf-8')
note=f'> **2026-09-07 baseline数据重诊断已完成**：两数据集各1024 train＋200 dev，当前Drone N42/IR42/N0与LLVIP baseline。Drone机会以低置信为主、LLVIP定位证据更强；局部IR特征未稳定超过独立RGB，固定ridge欠拟合已单列。请先读[阶段判断]({nav}/STAGE_REPORT.md)及[完整原始复核入口]({nav}/README.md)。这是推理/读出诊断，没有新增KD AP或定位准入。\n\n'
readme.write_bytes((old if old.startswith(note) else note+old).encode('utf-8'))
print(json.dumps({'files':len(records),'bytes':sum(r['bytes'] for r in records),'excluded':len(excluded)}))
