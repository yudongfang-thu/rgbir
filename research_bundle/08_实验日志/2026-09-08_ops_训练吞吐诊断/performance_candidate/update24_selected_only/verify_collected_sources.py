"""CPU byte comparison of collected small source snapshots; no models/hash/GPU."""
import argparse
import json
from pathlib import Path


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def verify(root):
    run=root/'comparison_attempt1'
    old,new=[read(run/name/'source_manifest.json') for name in ('old','selected_only')]
    assert old['new_hash_computed'] is False and new['new_hash_computed'] is False
    assert len(old['files'])==len(new['files'])>0
    rows=[]
    for a,b in zip(old['files'],new['files']):
        assert a['path']==b['path'] and a['byte_identity'] is True and b['byte_identity'] is True
        left=run/'old'/'sources'/Path(a['copy']).name
        right=run/'selected_only'/'sources'/Path(b['copy']).name
        data=left.read_bytes()
        assert data==right.read_bytes() and len(data)==a['bytes']==b['bytes']
        rows.append(dict(remote_source=a['path'],local_old=str(left),local_new=str(right),bytes=len(data),byte_exact=True))
    assert (run/'old'/'sources'/'input_config.yaml').read_bytes()==(run/'selected_only'/'sources'/'input_config.yaml').read_bytes()
    assert old['models']==new['models'] and old['config']==new['config']
    current=Path(__file__).resolve().parent
    bindings=[]
    for name,path in [('compare_24_selected_only.py',current/'compare_24_selected_only.py'),
                      ('selected_only_v1.py',current.parent/'selected_only_v1.py')]:
        match=[r for r in rows if Path(r['remote_source']).name==name]
        assert len(match)==1
        assert path.read_bytes()==Path(match[0]['local_new']).read_bytes()
        bindings.append(dict(name=name,current_local_source=str(path),executed_snapshot_byte_exact=True))
    queue=root/'queue_attempt1'
    profile=read(queue/'c1_selected_update24_diagnostic_resource_profile.json')
    assert profile['status']=='COMPLETED' and profile['exit_code']==0 and profile['monitor_errors']==[]
    samples=profile['samples']
    return dict(status='PASS_COLLECTED_SMALL_SOURCE_INPUT_REVIEW',source_pairs=len(rows),source_rows=rows,
        current_candidate_and_harness_bindings=bindings,config_bytes_exact=True,config=old['config'],models=old['models'],
        gpu_guard=dict(status=profile['status'],exit_code=profile['exit_code'],monitor_errors=profile['monitor_errors'],
            launch=profile['launch'],minimum_free_mib=profile['minimum_free_mib'],
            sampled_project_rss_peak_mib=max(r['project_rss_mib'] for r in samples),sample_count=len(samples)),
        model_contents_opened=False,new_hash_computed=False,state_pt_reopened=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection-root',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    receipt=verify(args.collection_root.resolve())
    args.output.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:receipt[k] for k in ('status','source_pairs','gpu_guard')}))
