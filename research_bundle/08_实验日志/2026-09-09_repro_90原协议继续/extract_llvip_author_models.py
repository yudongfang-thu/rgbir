import json
from pathlib import Path
import time
import libarchive

root = Path('/mnt/dataX/ydf/projects/RGBT_campaign_90/external_reproductions/llvip_author_baseline')
download = json.loads((root/'download_receipt.json').read_text())
archive = root/'yolov5_trained_model.rar'
assert download['status']=='DOWNLOADED' and archive.stat().st_size==download['bytes']
out = root/'extracted_attempt1'
out.mkdir(exist_ok=False)
records=[]
t0=time.perf_counter()
with libarchive.file_reader(str(archive)) as entries:
    for entry in entries:
        rel=Path(entry.pathname.replace('\\','/'))
        dst=(out/rel).resolve()
        assert dst.is_relative_to(out.resolve())
        row={'path':str(rel), 'size':entry.size, 'extracted':False}
        if entry.isfile and rel.suffix.lower() in {'.pt','.pth','.yaml','.txt','.md'}:
            dst.parent.mkdir(parents=True,exist_ok=True)
            n=0
            with dst.open('xb') as f:
                for block in entry.get_blocks():
                    f.write(block)
                    n+=len(block)
            assert n==entry.size
            row['extracted']=True
        records.append(row)
receipt={'status':'EXTRACTED','entries':records,'elapsed_seconds':time.perf_counter()-t0,'no_new_hash':True}
(root/'extraction_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
