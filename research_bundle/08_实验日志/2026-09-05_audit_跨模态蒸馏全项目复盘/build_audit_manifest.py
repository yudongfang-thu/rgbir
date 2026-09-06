"""Hash the audit deliverables and validate local report links; no experiment execution."""
import datetime
import hashlib
import json
import re
from pathlib import Path

D = Path(__file__).resolve().parent
ROOT = D.parents[1]
report = ROOT/'07_研究分析/全项目复盘与研究诊断_20260905.md'
files = sorted(p for p in D.iterdir() if p.is_file() and p.name != 'audit_manifest.json')
files += [report, ROOT/'README.md', ROOT/'08_实验日志/README.md', ROOT/'99_整理回执/20260905_全项目复盘新增产物.md']
records = []
for p in files:
    b = p.read_bytes()
    records.append({'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
targets = re.findall(r'\]\((E:/[^)]+)\)',report.read_text(encoding='utf-8'))
missing = sorted(set(t for t in targets if not Path(t).exists()))
out = {'generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'scope':'audit outputs only, not a training integrity certificate',
       'files':records,'report_local_links':{'count':len(targets),'missing':missing}}
(D/'audit_manifest.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'files':len(files),'report_links':len(targets),'missing_links':missing},ensure_ascii=False))
if missing: raise SystemExit(1)
