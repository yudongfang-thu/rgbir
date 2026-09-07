"""Record the root's independent review of the fixed-point object analyzer."""
import datetime
import json
from pathlib import Path

root = Path(__file__).resolve().parent
out = root / 'object_analyzer_independent_review_v1'
out.mkdir(exist_ok=False)
files = []
for name in ('object_error_analysis.py', 'test_object_error_analysis.py', 'OBJECT_ERROR_ANALYSIS_RULES.md'):
    target = out / name
    target.write_bytes((root / name).read_bytes())
    files.append(dict(relative=name, accepted_copy=str(target)))
receipt = dict(schema='rgbir-object-error-review-v1', status='ACCEPTED',
    reviewer='/root (independent of /root/review_matrix_spec implementation)',
    reviewed_at=datetime.datetime.now().astimezone().isoformat(),
    scope='Fixed conf=.25, same-class IoU=.50 diagnostic only; not COCO AP matching or localization admission.',
    checks=['15 known-truth CPU tests independently rerun and passed',
        'Repair denominator is baseline incorrect GT; damage denominator is baseline correct GT',
        'Background false positives use max IoU over every GT and all images including empty images',
        'Exact GT order and canvas are matched; missing source/brightness stays UNKNOWN',
        'Evaluation raw metric/object/config bytes must be receipt-bound',
        'Unreviewed output cannot acquire the accepted downstream diagnostic contract'],
    blocking_issues_remaining=0, actual_C1_results_analyzed=False, source_files=files)
(out / 'review_receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(out)
