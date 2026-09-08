"""One read-only snapshot of completed LLVIP calibration/canary small artifacts."""
import base64
import datetime
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
REMOTE = '/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_direction_screen_20260908'
PY = '/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
CODE = r'''
from pathlib import Path
import base64,json
root=Path(REMOTE);out=root/'screen_attempt1';paths=[];missing=[]
calib=out/'calibration/llvip'
assert (calib/'calibration_receipt.json').is_file()
paths += [calib/n for n in ('calibration_receipt.json','calibration_batches.jsonl','direction_config.yaml')]
canary_status={}
for arm in ('N','L2-box','L2-GT'):
 p=out/'canaries/llvip'/arm
 done=(p/'canary.json').is_file();canary_status[arm]='RECEIPT_PRESENT' if done else 'NO_COMPLETED_RECEIPT_AT_SNAPSHOT'
 if done:
  paths += [p/n for n in ('canary.json','initialization_check.json','direction_config.yaml','gradient_checks.jsonl','kd_batches.jsonl','runtime_ready.json') if (p/n).is_file()]
for arm,stage in [('N','calibration'),('N','canary'),('L2-box','canary'),('L2-GT','canary')]:
 name='direction_screen_attempt1_llvip_'+arm+'_'+stage
 for suffix in ('_resource_profile.json','_status.json','_job.json','_admission.json'):
  p=out/'queue'/(name+suffix)
  if p.is_file():paths.append(p)
  else:missing.append(str(p))
for name in ('llvip_coefficient_freeze.json','llvip_canary_checks.json','llvip_completion.json'):
 p=out/'queue'/name
 if p.is_file():paths.append(p)
for name in ('calibrate_direction.py','localization_box_v2.py'):
 paths.append(root/'release_v1'/name)
rows=[]
for p in sorted(set(paths)):
 before=p.stat();raw=p.read_bytes();after=p.stat()
 assert before.st_size==after.st_size==len(raw) and before.st_mtime_ns==after.st_mtime_ns,'Changed during small read'
 assert len(raw)<2*1024*1024
 rows.append(dict(relative=p.relative_to(root).as_posix(),path=str(p),bytes=len(raw),mtime_ns=after.st_mtime_ns,data=base64.b64encode(raw).decode()))
print(json.dumps(dict(files=rows,missing_optional=missing,canary_status=canary_status)))
'''


def main():
    destination = HERE / 'raw_attempt1'
    if destination.exists():
        raise FileExistsError(destination)
    run = subprocess.run(['ssh', '94', PY, '-'], input=CODE.replace('REMOTE', repr(REMOTE)).encode(),
                         capture_output=True, check=True, timeout=90)
    result = json.loads(run.stdout)
    destination.mkdir()
    for row in result['files']:
        p = destination / row['relative'];p.parent.mkdir(parents=True, exist_ok=True)
        raw = base64.b64decode(row.pop('data'));assert len(raw) == row['bytes']
        p.write_bytes(raw);assert p.read_bytes() == raw
        row['local'] = str(p);row['byte_exact'] = True
    result.update(recorded_at=datetime.datetime.now().astimezone().isoformat(), remote_root=REMOTE,
                  status='READ_ONLY_BYTE_EXACT_SNAPSHOT', gpu_started=False, remote_files_modified=False,
                  new_hash_computed=False)
    (HERE / 'collection_receipt.json').write_bytes((json.dumps(result, ensure_ascii=False, indent=2) + '\n').encode())
    print(json.dumps({'files': len(result['files']), 'bytes': sum(r['bytes'] for r in result['files']),
                      'canary_status': result['canary_status'], 'recorded_at': result['recorded_at']}))


if __name__ == '__main__':
    main()
