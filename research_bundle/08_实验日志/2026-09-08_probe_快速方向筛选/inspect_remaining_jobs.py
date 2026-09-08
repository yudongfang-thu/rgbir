"""Read-only bounded snapshot of existing E200 CSV progress; no AP inference."""
from pathlib import Path
import csv,json,datetime
B=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign')
paths=[B/'artifacts/rgbir_independent_kd_v2_20260907/formal_C1_gpu5_attempt2/runs'/('C1_seed'+str(s)) for s in (42,0,123)]
paths += [B/'runs/rgbir_task_conditional_c_attribution_20260907'/('full_c_'+a+'_s42_attempt1') for a in ('shuffled','same_modal')]
out=[]
for p in paths:
    c=p/'results.csv';row=dict(run=str(p),csv_exists=c.is_file())
    if c.is_file():
        with c.open() as f:rows=list(csv.DictReader(f))
        valid=[r for r in rows if str(r.get('epoch','')).strip().isdigit()]
        row.update(last_logged_epoch=int(valid[-1]['epoch']) if valid else None,csv_mtime_ns=c.stat().st_mtime_ns)
    row['completion_receipt_exists']=(p/'completion_receipt.json').is_file()
    out.append(row)
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='READ_ONLY_OLD_QUEUE_CSV_PROGRESS_NOT_ENDPOINT',runs=out,new_hash_computed=False)))
