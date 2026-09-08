"""Read-only four-small-file provenance collection from one known completed run."""
import base64,json,pathlib,shlex,subprocess

HERE=pathlib.Path(__file__).resolve().parent
REMOTE='/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_direction_screen_20260908/initial_llvip_recheck_attempt1/evaluation'
code='''import pathlib,json,base64
p=pathlib.Path(%r)
manifest=json.loads((p/'source_manifest.json').read_text())
selected=[r for r in manifest['files'] if pathlib.Path(r['path']).name=='initial_baseline_eval.py']
assert len(selected)==1
files=[p/'initial_evaluation_receipt.json',p/'source_manifest.json',p/'requested_native_config.yaml',pathlib.Path(selected[0]['copy'])]
rows=[]
for f in files:
 s=f.stat();assert s.st_size<200000
 rows.append(dict(path=str(f),bytes=s.st_size,mtime_ns=s.st_mtime_ns,base64=base64.b64encode(f.read_bytes()).decode()))
print(json.dumps(rows))
''' % REMOTE

def run():
    result=subprocess.run(['ssh','94','python3','-c',shlex.quote(code)],capture_output=True,check=True,timeout=45)
    rows=json.loads(result.stdout);target=HERE/'remote_source';target.mkdir(exist_ok=False)
    for row in rows:
        data=base64.b64decode(row.pop('base64'));assert len(data)==row['bytes']
        p=target/pathlib.PurePosixPath(row['path']).name
        with p.open('xb') as f:f.write(data)
        assert p.read_bytes()==data;row['local_copy']=str(p);row['transfer_byte_exact']=True
    with (HERE/'REMOTE_COLLECTION.json').open('x',encoding='utf-8') as f:
        json.dump(dict(status='COMPLETED_READ_ONLY',known_completed_run=REMOTE,files=rows,
            new_GPU=False,new_forward=False,new_weight_read=False,new_hash_computed=False),f,ensure_ascii=False,indent=2)
    print('COLLECTED',len(rows),'known small files')

if __name__=='__main__':run()
