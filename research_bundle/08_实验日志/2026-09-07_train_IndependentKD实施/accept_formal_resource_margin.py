"""Record root review of the final resource-margin-only worker revision."""
import datetime
import json
from pathlib import Path

root=Path(__file__).resolve().parent
out=root/'formal_campaign_independent_review_v3';out.mkdir(exist_ok=False)
files=[]
for name in ('formal_campaign.py','test_formal_campaign.py','FORMAL_CAMPAIGN_WORKER.md'):
    target=out/name;target.write_bytes((root/name).read_bytes())
    files.append(dict(relative=name,accepted_copy=str(target)))
receipt=dict(status='ACCEPTED',reviewer='/root; independent of matrix implementation',
    created_at=datetime.datetime.now().astimezone().isoformat(),blocking_issues_remaining=0,
    scope='Final margin-only difference from independent review v2; all existing lease, source binding, order, dependent evaluation and failure rules unchanged.',
    test_result='15/15 existing CPU tests independently rerun and passed',
    resource_rule=dict(training_vram='ceil(actual C1 canary peak)+2048 MiB',
        evaluation_vram='ceil(actual evaluator peak)+256 MiB',rss='max(8192,ceil(actual process-tree RSS)+4096) MiB'),
    reason='Native assignment allocations vary with batch GT density; first short-run peak is not a proof of the whole-E200 maximum.',
    source_files=files)
(out/'review_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(out)
