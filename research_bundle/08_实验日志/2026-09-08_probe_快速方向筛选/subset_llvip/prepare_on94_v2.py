# coding: utf-8
"""Authorized small CPU metadata build and byte-exact collection; no GPU or hashes."""
import base64
import datetime
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
PY = '/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
PARENT = '/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_direction_screen_20260908'
OUTPUT = PARENT + '/subset_llvip_v1'


def main():
    receipt_path = HERE / 'remote_build_receipt_v2.json'
    if receipt_path.exists():
        raise FileExistsError(receipt_path)
    files = {n: base64.b64encode((HERE / n).read_bytes()).decode() for n in ['build_llvip_subset_v2.py', 'natural_sampler_source.py']}
    code = '''import base64,json,subprocess,sys
from pathlib import Path
spec=SPEC
p=Path(PARENT)/'subset_builder_llvip_v2'
p.mkdir(parents=True,exist_ok=True)
for name,value in spec.items():
 raw=base64.b64decode(value);target=p/name
 if target.exists():assert target.read_bytes()==raw
 else:target.write_bytes(raw)
 assert target.read_bytes()==raw
subprocess.run([sys.executable,str(p/'build_llvip_subset_v2.py'),'--self-test'],check=True)
subprocess.run([sys.executable,str(p/'build_llvip_subset_v2.py'),'--output',OUTPUT],check=True)
'''.replace('SPEC', repr(files)).replace('PARENT', repr(PARENT)).replace('OUTPUT', repr(OUTPUT))
    run = subprocess.run(['ssh', '94', PY, '-'], input=code.encode(), capture_output=True, timeout=180)
    receipt = {'recorded_at': datetime.datetime.now().astimezone().isoformat(), 'returncode': run.returncode,
               'stdout': run.stdout.decode(), 'stderr': run.stderr.decode(), 'remote_output': OUTPUT,
               'builder_source_bytes': {n: len(base64.b64decode(v)) for n, v in files.items()},
               'source_byte_identity': True, 'gpu_used': False, 'new_hash_computed': False}
    receipt_path.write_bytes((json.dumps(receipt, ensure_ascii=False, indent=2) + '\n').encode())
    run.check_returncode()
    collect = '''from pathlib import Path
import base64,json
p=Path(OUTPUT)
records=[]
for f in sorted(p.rglob('*')):
 if not f.is_file():continue
 assert f.suffix in ['.py','.json','.jsonl','.txt','.yaml','.tsv'] and f.stat().st_size<4*1024*1024
 raw=f.read_bytes();s=f.stat()
 records.append(dict(relative=f.relative_to(p).as_posix(),bytes=len(raw),mtime_ns=s.st_mtime_ns,data=base64.b64encode(raw).decode()))
print(json.dumps(records))
'''.replace('OUTPUT', repr(OUTPUT))
    read = subprocess.run(['ssh', '94', PY, '-'], input=collect.encode(), capture_output=True, check=True, timeout=90)
    records = json.loads(read.stdout)
    dest = HERE / 'remote_subset_llvip_v1'
    dest.mkdir(exist_ok=False)
    for row in records:
        p = dest / row['relative']
        p.parent.mkdir(parents=True, exist_ok=True)
        raw = base64.b64decode(row.pop('data'))
        assert len(raw) == row['bytes']
        p.write_bytes(raw)
        assert p.read_bytes() == raw
        row['byte_exact'] = True
    (HERE / 'collection_receipt.json').write_bytes((json.dumps({'status': 'BYTE_EXACT_COLLECTED',
        'remote': OUTPUT, 'files': records, 'gpu_used': False, 'new_hash_computed': False}, indent=2) + '\n').encode())
    print(run.stdout.decode().strip())
    print(json.dumps({'collected_files': len(records), 'bytes': sum(r['bytes'] for r in records)}))


if __name__ == '__main__':
    main()
