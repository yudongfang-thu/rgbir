"""One optional read of the Drone calibration receipt; no waiting, polling or GPU."""
import base64
import datetime
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
PY = '/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
REMOTE = '/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_direction_screen_20260908/screen_attempt1'
CODE = '''from pathlib import Path
import json,base64
p=Path(REMOTE);receipt=p/'calibration/drone/calibration_receipt.json'
records=[]
if receipt.is_file():
 paths=[receipt,p/'calibration/drone/direction_config.yaml']
 for tail in ('_resource_profile.json','_status.json','_job.json'):
  q=p/'queue'/('direction_screen_attempt1_drone_N_calibration'+tail)
  if q.is_file():paths.append(q)
 for f in paths:
  before=f.stat();raw=f.read_bytes();after=f.stat()
  assert len(raw)==before.st_size==after.st_size and before.st_mtime_ns==after.st_mtime_ns
  assert len(raw)<1024*1024
  records.append(dict(path=str(f),relative=f.relative_to(p).as_posix(),bytes=len(raw),mtime_ns=after.st_mtime_ns,data=base64.b64encode(raw).decode()))
print(json.dumps(dict(completed_receipt_present=receipt.is_file(),files=records)))
'''


def main():
    output = HERE / 'drone_calibration_once'
    if output.exists():
        raise FileExistsError(output)
    result = subprocess.run(['ssh', '94', PY, '-'], input=CODE.replace('REMOTE', repr(REMOTE)).encode(),
                            capture_output=True, check=True, timeout=45)
    data = json.loads(result.stdout)
    output.mkdir()
    for r in data['files']:
        p = output / r['relative'];p.parent.mkdir(parents=True, exist_ok=True)
        raw = base64.b64decode(r.pop('data'));assert len(raw) == r['bytes']
        p.write_bytes(raw);assert p.read_bytes() == raw
        r['byte_exact'] = True
    data.update(recorded_at=datetime.datetime.now().astimezone().isoformat(), remote_reads=1, retries=0,
                gpu_started=False, remote_modified=False, new_hash_computed=False)
    (output / 'collection_receipt.json').write_bytes((json.dumps(data, indent=2) + '\n').encode())
    print(json.dumps(data))
    if data['completed_receipt_present']:
        receipt = json.loads((output / 'calibration/drone/calibration_receipt.json').read_text(encoding='utf-8'))
        print(json.dumps(receipt))


if __name__ == '__main__':
    main()
