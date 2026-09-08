from pathlib import Path
import json
from datetime import datetime,timezone

root=Path(__file__).resolve().parent
source=root.parent.parent/'2026-09-08_probe_DFL真实信息读出'/'evidence_1602_final'/'probe'
dest=root/'input_snapshot'
dest.mkdir(exist_ok=False)
manifest=[]
for name in ('objects.jsonl','anchor_distributions.jsonl','completion_receipt.json','dfl_contract.json'):
    src=source/name
    before=src.read_bytes()
    copy=dest/name
    copy.write_bytes(before)
    same=(before==copy.read_bytes()==src.read_bytes())
    assert same
    manifest.append({'source_path':str(src.resolve()),'snapshot_path':str(copy.resolve()),
                     'bytes':len(before),'mtime_ns':src.stat().st_mtime_ns,
                     'direct_bytes_equal':same,'path_identity':str(src.resolve())})
result={'date':datetime.now(timezone.utc).isoformat(),'identity_method':'resolved paths plus direct full-byte comparison',
        'new_hash_computed':False,'audited_input_hashes':[],
        'hash_omission_reason':'User scope explicitly forbids new hashes; full-byte snapshots and equality checks replace the skill hash default.',
        'inputs':manifest}
(root/'input_snapshot_manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
