import json
from pathlib import Path
import shutil
import time
import yaml

root=Path('/mnt/dataX/ydf/projects/RGBT_campaign_90')
source=root/'data_author_protocol/llvip_attempt1/previous'
out=root/'data_author_protocol/llvip_baseline_attempt1'
out.mkdir(exist_ok=False)
t0=time.perf_counter()
rows=[]
roster=None
for modality in ('visible','infrared'):
    view=out/modality
    (view/'images').mkdir(parents=True)
    for split,expected in [('train',12025),('test',3463)]:
        src_images=source/modality/'images'/split
        files=sorted(src_images.glob('*.jpg'))
        assert len(files)==expected
        (view/'images'/split).symlink_to(src_images, target_is_directory=True)
        labels=view/'labels'/split
        labels.mkdir(parents=True)
        gt=0
        for image in files:
            label=source/modality/'labels'/split/(image.stem+'.txt')
            assert label.is_file()
            shutil.copy2(label,labels/label.name)
            gt+=len([x for x in label.read_text().splitlines() if x.strip()])
        (view/(split+'.txt')).write_text(''.join(str(view/'images'/split/f.name)+'\n' for f in files))
        if split=='test':
            assert gt==7931
            current=[f.stem for f in files]
            assert roster is None or current==roster
            roster=current
        rows.append({'modality':modality,'split':split,'images':expected,'gt':gt})
    (view/'data.yaml').write_text(yaml.safe_dump({'path':str(view),'train':str(view/'train.txt'),
        'val':str(view/'test.txt'),'test':str(view/'test.txt'),'nc':1,'names':['person']},sort_keys=False))
(out/'test_roster.json').write_text(json.dumps({'stems':roster},indent=2)+'\n')
receipt={'status':'PREPARED','source':str(source),'output':str(out),'counts':rows,
    'elapsed_seconds':time.perf_counter()-t0,'scope':'Independent author-baseline cache/labels view; CFT views and official source files unchanged',
    'new_hash_computed':False}
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
