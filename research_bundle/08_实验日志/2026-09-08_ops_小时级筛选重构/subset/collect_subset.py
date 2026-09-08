# coding: utf-8
"""Collect only the newly generated train metadata and lists; byte copies, no hashes."""
import base64
import datetime
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
REMOTE = '/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_hourly_screen_20260908/subset_v1'


def main():
    code = """from pathlib import Path
import base64,json
r=Path(REMOTE)
records=[]
for p in sorted(r.iterdir()):
    assert p.is_file() and p.suffix in {'.json','.jsonl','.yaml','.txt'}
    s=p.stat()
    assert s.st_size<20000000
    b=p.read_bytes()
    records.append(dict(name=p.name,remote=str(p),bytes=len(b),mtime_ns=s.st_mtime_ns,data=base64.b64encode(b).decode()))
print(json.dumps(records))
""".replace('REMOTE', repr(REMOTE))
    run = subprocess.run(['ssh', '94', 'python3', '-'], input=code.encode(), capture_output=True, check=True, timeout=90)
    records = json.loads(run.stdout)
    dest = HERE / 'remote_subset_v1'
    dest.mkdir(exist_ok=False)
    for record in records:
        raw = base64.b64decode(record.pop('data'))
        assert len(raw) == record['bytes']
        path = dest / record['name']
        path.write_bytes(raw)
        assert path.read_bytes() == raw
        record['local'] = str(path)
    receipt = {'status': 'COLLECTED_BYTE_EXACT', 'time': datetime.datetime.now().astimezone().isoformat(),
               'records': records, 'gpu_used': False, 'hashes_computed': False, 'images_or_weights_collected': False}
    (HERE / 'collection_receipt.json').write_bytes((json.dumps(receipt, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
    print(json.dumps({'status': receipt['status'], 'files': len(records), 'bytes': sum(x['bytes'] for x in records)}))


if __name__ == '__main__':
    main()
