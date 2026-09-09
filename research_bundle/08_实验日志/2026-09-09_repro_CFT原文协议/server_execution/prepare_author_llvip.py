"""Prepare separate official LLVIP splits and both annotation source versions."""
import datetime
import json
from pathlib import Path
import shutil
import time
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path('/mnt/dataX/ydf/projects/RGBT_campaign_90')
OUT = ROOT/'data_author_protocol/llvip_attempt1'
ART = ROOT/'artifacts/cft_author_protocol_20260909_attempt1'

def save(name, value):
    (ART/name).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')

def convert(xmlbytes):
    # Official LLVIP voc_label.py convention: no center -1, width=xmax-xmin.
    doc = ET.fromstring(xmlbytes)
    rows = []
    for obj in doc.findall('object'):
        if obj.findtext('name') != 'person':
            continue
        b = obj.find('bndbox')
        x1,y1,x2,y2 = [float(b.findtext(k)) for k in ('xmin','ymin','xmax','ymax')]
        rows.append('0 %.6f %.6f %.6f %.6f' % ((x1+x2)/2/1280, (y1+y2)/2/1024, (x2-x1)/1280, (y2-y1)/1024))
    return '\n'.join(rows)+'\n' if rows else ''

def main():
    start=time.perf_counter()
    ART.mkdir(parents=True,exist_ok=True)
    OUT.mkdir(parents=True,exist_ok=False)
    oldzip=OUT/'Annotations_previous_version.zip'
    oldurl='https://drive.usercontent.google.com/download?id=1RZqYKHXUVgSOi_eq15EDjfiR2D-tyFVV&export=download&authuser=0&confirm=t'
    with urllib.request.urlopen(oldurl,timeout=45) as response, oldzip.open('xb') as f:
        shutil.copyfileobj(response,f)
    assert zipfile.is_zipfile(oldzip), 'Official previous annotations endpoint did not return a ZIP'
    with zipfile.ZipFile(oldzip) as z:
        legacy={Path(n).stem:z.read(n) for n in z.namelist() if n.lower().endswith('.xml') and not n.startswith('__MACOSX/')}
    save('legacy_annotations_download.json',dict(url=oldurl,bytes=oldzip.stat().st_size,xml_count=len(legacy),status='DOWNLOADED'))
    archive=Path('/mnt/dataY/ydf/dataset/LLVIP.zip')
    with zipfile.ZipFile(archive) as z:
        members=z.namelist()
        stems={}
        for split,expected in [('train',12025),('test',3463)]:
            rgb={Path(n).stem:n for n in members if n.startswith('LLVIP/visible/'+split+'/') and n.endswith('.jpg')}
            ir={Path(n).stem:n for n in members if n.startswith('LLVIP/infrared/'+split+'/') and n.endswith('.jpg')}
            assert len(rgb)==expected and set(rgb)==set(ir)
            stems[split]=sorted(rgb)
            for j,stem in enumerate(stems[split]):
                ann_current=z.read('LLVIP/Annotations/'+stem+'.xml')
                ann_old=legacy[stem]
                for version,xml in [('previous',ann_old),('archive_current',ann_current)]:
                    p=OUT/version/'Annotations'/(stem+'.xml')
                    p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(xml)
                    for modality in ('visible','infrared'):
                        label=OUT/version/modality/'labels'/split/(stem+'.txt')
                        label.parent.mkdir(parents=True,exist_ok=True)
                        label.write_text(convert(xml))
                for modality,mapping in [('visible',rgb),('infrared',ir)]:
                    shared=OUT/'images_raw'/modality/split/(stem+'.jpg')
                    shared.parent.mkdir(parents=True,exist_ok=True)
                    if split=='train':
                        src=ROOT/'data_attempt1/raw/LLVIP_train_only'/modality/'train'/(stem+'.jpg')
                        assert src.is_file();shared.symlink_to(src)
                    else:
                        with z.open(mapping[stem]) as f, shared.open('xb') as out:
                            shutil.copyfileobj(f,out)
                    for version in ('previous','archive_current'):
                        dest=OUT/version/modality/'images'/split/(stem+'.jpg')
                        dest.parent.mkdir(parents=True,exist_ok=True);dest.symlink_to(shared)
                if (j+1)%1000==0:print(split,j+1,flush=True)
        assert not(set(stems['train']) & set(stems['test']))
    for version in ('previous','archive_current'):
        for split in ('train','test'):
            for modality in ('visible','infrared'):
                lst=OUT/version/modality/(split+'.txt')
                lst.write_text(''.join(str(OUT/version/modality/'images'/split/(s+'.jpg'))+'\n' for s in stems[split]))
        y=OUT/version/'LLVIP.yaml'
        y.write_text('train_rgb: '+str(OUT/version/'visible/train.txt')+'\nval_rgb: '+str(OUT/version/'visible/test.txt')+'\ntrain_ir: '+str(OUT/version/'infrared/train.txt')+'\nval_ir: '+str(OUT/version/'infrared/test.txt')+'\nnc: 1\nnames: [person]\n')
    save('data_receipt.json',dict(status='PREPARED',root=str(OUT),official_train=12025,official_test=3463,
        primary_annotations='official previous version, pre-2023 update',secondary_annotations='user uploaded 2023-02-21 archive',
        converter='official LLVIP no-center-shift, fixed1280x1024,6decimals; all person objects',
        purpose='author protocol reproduction only',our_grouped_data_unchanged=True,test_exposure_authorized_by_latest_user=True,
        no_new_digest=True,elapsed_seconds=time.perf_counter()-start,finished=datetime.datetime.now().astimezone().isoformat()))
    print('PREPARED',time.perf_counter()-start,flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:
        import traceback
        ART.mkdir(parents=True,exist_ok=True)
        save('data_failure.json',dict(error=repr(e),traceback=traceback.format_exc()))
        raise
