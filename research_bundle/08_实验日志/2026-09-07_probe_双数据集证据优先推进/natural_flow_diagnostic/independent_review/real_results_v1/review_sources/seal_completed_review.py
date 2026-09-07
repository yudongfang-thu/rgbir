"""Freeze final review sources and input stats using bytes only, per user no-hash rule."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent;TASK=HERE.parent
out=HERE/'real_results_v1/review_sources';out.mkdir(exist_ok=False)
sources=[]
for name in ('review_real_natural.py','finalize_real_review.py','review_readout_tables.py','seal_completed_review.py','REAL_RESULTS_REVIEW_SCOPE.md'):
    source=HERE/name;target=out/name;before=source.stat();target.write_bytes(source.read_bytes());after=source.stat()
    assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    assert source.read_bytes()==target.read_bytes()
    sources.append(dict(source=str(source),snapshot=str(target),bytes=after.st_size,mtime_ns=after.st_mtime_ns,byte_identity=True))
inputs=[]
for short in ('llvip','drone'):
    for kind in ('canary','full'):
        folder=TASK/'remote_completed_attempt2'/(short+'_'+kind+'_attempt1')
        for name in ('summary.json','selection_batches.jsonl','natural_batches.jsonl','original_natural_batches.jsonl','input_manifest.json','model_identity.json','input_config.yaml'):
            source=folder/name;st=source.stat();inputs.append(dict(path=str(source),bytes=st.st_size,mtime_ns=st.st_mtime_ns))
for folder,pattern in [('governance_group_sources','*'),('completed_readout_attempt2','*.csv')]:
    for source in (TASK/folder).glob(pattern):
        if source.is_file():
            st=source.stat();inputs.append(dict(path=str(source),bytes=st.st_size,mtime_ns=st.st_mtime_ns))
result=dict(status='SEALED',auditor='/root/ap_error',executor='/root/baseline_feature_analysis',model_identity='not independently asserted from role name',
    new_hash_computed=False,sources=sources,inputs=inputs)
with (HERE/'real_results_v1/final_source_and_input_stat_manifest.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps(dict(status='SEALED',sources=len(sources),inputs=len(inputs),new_hash_computed=False)))
