"""Record the root's independent rect-only analyzer review and actual CPU tests."""
import datetime
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'object_error_analyzer_rect_v2'
OUT=ROOT/'object_analyzer_rect_independent_review_v2'
OUT.mkdir(exist_ok=False)
result=subprocess.run([sys.executable,'-m','unittest','test_object_error_analysis','-v'],cwd=SOURCE,
    stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
(OUT/'cpu_tests.log').write_text(result.stdout,encoding='utf-8')
if result.returncode:raise RuntimeError(result.stdout)
rows=[]
for name in ('object_error_analysis.py','test_object_error_analysis.py','OBJECT_ERROR_ANALYSIS_RULES.md'):
    path=OUT/name;path.write_bytes((SOURCE/name).read_bytes())
    rows.append(dict(relative=name,accepted_copy=str(path)))
receipt=dict(schema='rgbir-object-error-review-v1',status='ACCEPTED',
    reviewer='/root (independent of /root/review_matrix_spec implementation)',
    reviewed_at=datetime.datetime.now().astimezone().isoformat(),blocking_issues_remaining=0,
    scope='Rect padding input adapter correction; fixed confidence/IoU/area thresholds and repair/damage denominators unchanged.',
    checks=['Independently inspected complete v1-to-v2 code diff',
            '19 known-truth CPU tests rerun, including actual padded-shape binding and immutable boxes',
            'Nominal640 comes from bound config/effective kwargs; actual canvas/original shape comes from bound objects',
            'Actual loader order reconstructs batches; each batch shares one observed canvas',
            'Baseline/candidate original shape, canvas and GT arrays remain exactly compared',
            'input640 scale labels only rename old labels;32^2/96^2 areas and all matching rules unchanged',
            'No new geometry truth, box transformation, source grouping or brightness thresholds inferred',
            'Downstream ERROR_CONTRACT unchanged; output schema records technical v2'],
    actual_C1_results_analyzed=False,actual_legacy_object_analysis_completed=False,
    source_files=rows,test_count=19,tests_exit_code=result.returncode)
(OUT/'review_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(OUT/'REVIEW.md').write_text('# Rect对象分析器独立复核\n\n已检查完整代码差异并实际重跑19项CPU真值测试，全部通过。接受的只是名义640与实际padding画布的接口修正；旧原始输入、失败attempt及阈值均保留。尚未据此运行真实对象分析，不声称新C1有收益。\n',encoding='utf-8')
print(json.dumps(dict(status='ACCEPTED',tests=19,output=str(OUT))))
