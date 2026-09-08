"""Store the small final interpretation beside executed remote artifacts."""
import base64
import json
from pathlib import Path
import subprocess
from prepare_hourly_release import HERE,PY,CAMPAIGN

files=['FINAL_REPORT.md','FAST_SCREEN_PROTOCOL_REVIEW.md','results_1022/summary.json',
       'results_1022/COMPARISON_TABLES.md','results_1022/postflight.json',
       'analysis/analyze_hourly.py','analysis/analyzer_independent_acceptance.json','analysis/ANALYZER_REVIEW.md']
sources={name:base64.b64encode((HERE/name).read_bytes()).decode() for name in files}
code="""import base64,json
from pathlib import Path
root=Path(OUT);root.mkdir(exist_ok=False)
for name,content in FILES.items():
 p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(content))
(root/'README.md').write_text('Final local interpretation of ../ft_screen_attempt1. Full local evidence and working relative links: E:/SHARE/光sar/08_实验日志/2026-09-08_ops_小时级筛选重构/. No new evaluation or checkpoint changes.'+chr(10),encoding='utf-8')
print(json.dumps(dict(status='FINAL_REVIEW_SYNCED',path=str(root),files=list(FILES),new_hash_computed=False)))
""".replace('OUT',repr(CAMPAIGN+'/final_review_1022')).replace('FILES',repr(sources))
r=subprocess.run(['ssh','94',PY,'-'],input=code.encode('utf-8'),capture_output=True,timeout=60)
print(r.stdout.decode());print(r.stderr.decode());r.check_returncode()
with (HERE/'remote_final_review_sync.json').open('x',encoding='utf-8') as f:json.dump(json.loads(r.stdout),f,indent=2)
