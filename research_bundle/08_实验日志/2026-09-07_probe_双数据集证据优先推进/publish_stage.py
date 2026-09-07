"""Curate evidence without new hashes; raw numeric bytes remain exact."""
from pathlib import Path
import datetime,gzip,json,os,re,shutil
from urllib.parse import unquote
WORK=Path('E:/SHARE/光sar');ROOT=Path(__file__).resolve().parent
REPO=Path('C:/Users/MSI-PC/rgbir_review_worktrees/evidence-20260906');BUNDLE=REPO/'research_bundle'
OMIT={'__pycache__','.git','deps','wheels','.pytest_cache','attempt2_reviewed_sources'}
extra=[WORK/p for p in ['README.md','08_实验日志/README.md','refine-logs/EXPERIMENT_PLAN.md','refine-logs/EXPERIMENT_TRACKER.md',
  '99_整理回执/20260907_IndependentKD实施新增目录.md',
  '08_实验日志/2026-09-07_probe_Baseline蒸馏机会重诊断/STAGE_REPORT.md',
  '08_实验日志/2026-09-07_probe_Baseline蒸馏机会重诊断/README.md',
  '08_实验日志/2026-09-07_train_IndependentKD实施/running_state_20260907_212910.json',
  '08_实验日志/2026-09-07_train_IndependentKD实施/running_state_20260907_215559.json']]
found=[]
for parent,dirs,names in os.walk(ROOT):
    dirs[:]=[d for d in dirs if d not in OMIT]
    found.extend(Path(parent)/name for name in names if name!='publication_receipt.json')
files=sorted(set(found+extra))
def io_path(p):return Path('\\\\?\\'+str(p.absolute()))
records=[];excluded=[];mapping={}
for p in files:
    if p.suffix.lower() in {'.pt','.pth','.ckpt','.pyd','.dll','.whl'} or (p.stat().st_size>32*2**20 and p.suffix!='.jsonl'):
        excluded.append(dict(source=str(p),bytes=p.stat().st_size,reason='Large derived array or runtime binary; source and regeneration inputs retained'))
        continue
    dest=BUNDLE/p.relative_to(WORK)
    if p.suffix=='.jsonl':dest=dest.with_suffix('.jsonl.gz')
    mapping[str(p.resolve()).lower()]=dest
for p in files:
    dest=mapping.get(str(p.resolve()).lower())
    if dest is None:continue
    io_path(dest.parent).mkdir(parents=True,exist_ok=True)
    if p.suffix=='.jsonl':
        with io_path(p).open('rb') as src,gzip.open(io_path(dest),'wb',compresslevel=6) as target:shutil.copyfileobj(src,target)
        assert gzip.decompress(io_path(dest).read_bytes())==io_path(p).read_bytes()
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
        io_path(dest).write_bytes(re.sub(r'\]\(([^\n)]*)\)',replace,io_path(p).read_text(encoding='utf-8-sig')).encode('utf-8'))
    else:
        io_path(dest).write_bytes(io_path(p).read_bytes());assert io_path(dest).read_bytes()==io_path(p).read_bytes()
    records.append(dict(source=str(p),repository_path=str(dest.relative_to(REPO)).replace('\\','/'),bytes=io_path(dest).stat().st_size,
      verification='Markdown links adapted' if p.suffix=='.md' else 'gzip decompression byte exact' if p.suffix=='.jsonl' else 'byte exact'))
manifest=dict(time=datetime.datetime.now().astimezone().isoformat(),records=records,excluded=excluded,
  scope='Full dev baseline AP, class bridge, conditional localization stress, fixed natural training-flow selection diagnostics; no new KD AP',
  no_test_access=True,new_hashes_computed=False,omitted_directories=sorted(OMIT),
  duplicate_review_source_omission='Deep duplicate reviewed source trees omitted; executed run sources, direct diagnostic sources, review stat manifests and full receipts retained')
(REPO/'DUAL_DATASET_EVIDENCE_INCREMENT_20260907.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
nav='research_bundle/08_实验日志/2026-09-07_probe_双数据集证据优先推进'
note=f'> **2026-09-07 双数据集证据推进**：LLVIP完整2406dev旧RGB/IR baseline mAP32.8784/48.8529，定位优先；Drone全1469dev六端点表明少数类混淆贡献macro分类oracle的94.26%，不能将对象计数概括为全AP瓶颈。请先读[新阶段判断]({nav}/STAGE_REPORT.md)与[原始证据/独立审阅入口]({nav}/README.md)。已有C1继续，新定位方法尚未准入。\n\n'
p=REPO/'README.md';old=p.read_text(encoding='utf-8');p.write_bytes((old if old.startswith(note) else note+old).encode('utf-8'))
print(json.dumps(dict(files=len(records),bytes=sum(x['bytes'] for x in records),excluded=len(excluded),byte_verified=sum(x['verification']!='Markdown links adapted' for x in records))))
