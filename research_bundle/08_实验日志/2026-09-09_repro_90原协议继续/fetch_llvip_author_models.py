import datetime
import json
from pathlib import Path
import time
import urllib.request

root = Path('/mnt/dataX/ydf/projects/RGBT_campaign_90')
out = root/'external_reproductions/llvip_author_baseline'
out.mkdir(parents=True, exist_ok=True)
target = out/'yolov5_trained_model.rar'
assert not target.exists(), 'Preserve existing download; choose another attempt if needed'
url = 'https://drive.usercontent.google.com/download?id=1SPbr0PDiItape602-g-bstkX0P7NZo0q&export=download&authuser=0&confirm=t'
t0 = time.perf_counter()
n = 0
with urllib.request.urlopen(url, timeout=90) as r, target.open('xb') as f:
    headers = dict(r.headers)
    while True:
        block = r.read(2**20)
        if not block:
            break
        if n == 0:
            assert block.startswith(b'Rar!'), 'Response is not expected author RAR archive'
        f.write(block)
        n += len(block)
    expected = int(headers.get('Content-Length', n))
    assert n == expected
receipt = {'status': 'DOWNLOADED', 'author_repository': 'https://github.com/bupt-ai-cz/LLVIP', 'url': url,
           'path': str(target), 'bytes': n, 'elapsed_seconds': time.perf_counter()-t0,
           'completed_at': datetime.datetime.now().astimezone().isoformat(),
           'scope': 'Author archive only; model identity and execution not yet verified', 'new_hash_computed': False}
(out/'download_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt), flush=True)
