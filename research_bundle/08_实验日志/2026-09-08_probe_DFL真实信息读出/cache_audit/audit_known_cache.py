"""Bounded metadata audit of the three known capture/calibration locations only."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess

REMOTE_BASE='/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/'
REMOTE=[REMOTE_BASE+'rgbir_selection_coverage_20260908/attempt1',
        REMOTE_BASE+'rgbir_selection_coverage_20260908/witness_attempt1',
        REMOTE_BASE+'rgbir_direction_screen_20260908/screen_attempt1/calibration/llvip']


def inspect_value(value, keys, shaped):
    if isinstance(value,dict):
        for k,v in value.items():
            if re.search('dfl|logit|probabil',k,re.I): keys[k]+=1
            inspect_value(v,keys,shaped)
    elif isinstance(value,list):
        if len(value)==4 and all(isinstance(x,list) and len(x)==16 and all(isinstance(y,(int,float)) for y in x) for x in value):shaped['4x16']+=1
        if len(value)==64 and all(isinstance(x,(int,float)) for x in value):shaped['flat64']+=1
        for v in value:inspect_value(v,keys,shaped)


def run(workspace,output):
    workspace=Path(workspace);output=Path(output)
    if output.exists():raise FileExistsError(output)
    logs=workspace/'08_实验日志'
    local=[logs/'2026-09-08_probe_同帧选择覆盖/evidence_1255_final/probe',
           logs/'2026-09-08_probe_同帧选择覆盖/witness_evidence_1315_final/probe',
           logs/'2026-09-08_probe_快速方向筛选/results_1203_snapshot/calibration/llvip']
    # Known roots, maxdepth 3 only; no checkpoint/data/global filesystem search.
    command='find '+' '.join(REMOTE)+" -maxdepth 3 -type f -printf '%p\\t%s\\n'"
    result=subprocess.run(['ssh','94',command],capture_output=True,text=True,check=True,encoding='utf-8')
    inventory=[]
    for line in result.stdout.splitlines():
        path,size=line.rsplit('\t',1);inventory.append(dict(path=path,bytes=int(size)))
    candidates=[r for r in inventory if Path(r['path']).suffix.lower() in ('.pt','.pth','.npy','.npz','.safetensors','.bin','.gz','.h5','.hdf5')]
    examined=[]
    for directory in local:
        for file in sorted(directory.iterdir()):
            if file.suffix not in ('.json','.jsonl'):continue
            keys,shapes=Counter(),Counter();text=file.read_text(encoding='utf-8-sig')
            payloads=[json.loads(x) for x in text.splitlines() if x.strip()] if file.suffix=='.jsonl' else [json.loads(text)]
            for value in payloads:inspect_value(value,keys,shapes)
            examined.append(dict(path=str(file),bytes=file.stat().st_size,records=len(payloads),DFL_related_keys=dict(keys),numeric_candidate_shapes=dict(shapes)))
    shapes=sum(sum(x['numeric_candidate_shapes'].values()) for x in examined)
    assert not candidates and shapes==0
    summary=dict(status='CACHE_MISSING_FOR_MATCHED_FIRST32_RAW_DFL',scope='THREE_KNOWN_ARTIFACT_ROOTS_MAXDEPTH3_AND_LOCAL_CAPTURE_JSON',
                 remote_roots=REMOTE,remote_files=len(inventory),remote_tensor_or_compressed_candidate_files=candidates,
                 local_JSON_files=len(examined),local_numeric_4x16_or_flat64_arrays=shapes,
                 checked_keys=examined,
                 limitations=['Not a global filesystem absence claim','Other historical raw exports have different sample/augmentation flow; not substituted',
                              'Source manifests describe Python/code/input paths, not persisted DFL tensors','Scalar DFL loss, xyxy expectations, confidence, witness boxes do not identify DFL probabilities'],
                 missing_fields=['S/R/T per-anchor 4x16 raw DFL logits','Full-bin probabilities with anchor/dtype identity','Same-forward GT-distance/bin semantics for those logits'],
                 SSH_read_only=True,new_GPU=False,new_forward=False,new_hash_computed=False)
    output.mkdir(parents=True)
    (output/'REMOTE_INVENTORY.tsv').write_text(result.stdout,encoding='utf-8')
    (output/'receipt.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ('status','remote_files','local_JSON_files','local_numeric_4x16_or_flat64_arrays')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.workspace,a.output)
