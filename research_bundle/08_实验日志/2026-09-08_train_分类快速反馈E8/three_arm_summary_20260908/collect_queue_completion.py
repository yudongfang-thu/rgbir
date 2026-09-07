"""Collect only the persisted queue-completion receipt and stat; no GPU or hash."""
import json
import subprocess
from pathlib import Path

REMOTE_PYTHON = '/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
REMOTE_PATH = '/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_short_screen_E8_20260908_attempt1/queue/completion.json'
remote_code = '''import json
from pathlib import Path
p = Path(%r)
s = p.stat()
assert s.st_size < 1000000
print(json.dumps(dict(path=str(p), bytes=s.st_size, mtime_ns=s.st_mtime_ns, text=p.read_text(encoding='utf-8'))))
''' % REMOTE_PATH
result = subprocess.run(['ssh', '94', REMOTE_PYTHON, '-'], input=remote_code,
                        encoding='utf-8', stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        timeout=30, check=True)
collected = json.loads(result.stdout)
value = json.loads(collected['text'])
assert value['status'] == 'SHORT_SCREEN_MATRIX_COMPLETED'
root = Path(__file__).parent
with (root / 'queue_completion.json').open('xb') as f:
    f.write(collected.pop('text').encode('utf-8'))
collected.update(new_hash_computed=False, remote_mutations=False, new_cuda_workloads=0)
with (root / 'queue_completion_collection.json').open('x', encoding='utf-8') as f:
    json.dump(collected, f, indent=2)
print(json.dumps(value, ensure_ascii=False))
