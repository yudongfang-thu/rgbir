"""Read-only SSH inventory and original Drone labels; never open checkpoint content."""
import json
from pathlib import Path
import subprocess

REMOTE=r'''
import json,os
from pathlib import Path
import yaml
from PIL import Image
B=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign')
run=B/'runs/rgbt_p3_causal_v1/formal_native/dronevehicle/infrared_seed42_native_b32a2'
data=B/'artifacts/rgbt_p3_causal_v1/prepared/dronevehicle/infrared.data.yaml'
cfg=yaml.safe_load(data.read_text());root=Path(cfg['path']);images=sorted((root/cfg['val']).glob('*'))
images=[p for p in images if p.suffix.lower() in {'.jpg','.png','.jpeg','.bmp','.tif','.tiff','.webp'}]
counts=[];classes=[0]*5;firstclasses=[0]*5;labels=[];shapes=set()
for i,p in enumerate(images):
 label=root/'labels/val'/(p.stem+'.txt');rows=[s.split() for s in label.read_text().splitlines() if s.strip()]
 assert all(len(r)==5 and float(r[0]).is_integer() and 0<=int(float(r[0]))<5 for r in rows)
 counts.append(len(rows));labels.append(str(label))
 for row in rows:
  classes[int(float(row[0]))]+=1
  if i<32:firstclasses[int(float(row[0]))]+=1
 with Image.open(p) as im:shapes.add(im.size)
rgb=B/'data/processed/dronevehicle/yolo/hbb_v1/rgb/images/val'
assert {p.name for p in images}=={p.name for p in rgb.glob('*') if p.suffix.lower() in {'.jpg','.png','.jpeg','.bmp','.tif','.tiff','.webp'}}
possible=[];references=[];scanned=0
for base in (B/'artifacts',B/'runs'):
 for directory,dirs,files in os.walk(base,followlinks=False):
  dirs[:]=[d for d in dirs if d not in ('weights','__pycache__','.git','source_snapshot','implementation_snapshot','sources','source_copies')]
  for name in files:
   scanned+=1;p=Path(directory)/name;low=name.lower();size=p.stat().st_size
   if ('drone' in str(p).lower() or 'infrared_seed42' in str(p)) and ('pred' in low or 'capture' in low or low.endswith(('.jsonl','.jsonl.gz','.npz'))):
    possible.append(dict(path=str(p),size=size))
   if low.endswith(('.json','.yaml','.yml','.md')) and size<2000000 and any(k in low for k in ('index','evaluation','receipt','metrics','summary','manifest','contract')):
    try:text=p.read_text(errors='replace')
    except (OSError,UnicodeError):continue
    if str(run/'weights/last.pt') in text:references.append(dict(path=str(p),size=size))
result=dict(status='READ_ONLY_INPUT_AUDIT_COMPLETED',checkpoint_path=str(run/'weights/last.pt'),checkpoint_stat=dict(size=(run/'weights/last.pt').stat().st_size,mtime_ns=(run/'weights/last.pt').stat().st_mtime_ns),
 dataset='dronevehicle',model='T42',source_data=str(data),data=cfg,images=len(images),gt=sum(counts),empty_gt=sum(n==0 for n in counts),class_counts=classes,
 first32=dict(images=[str(p) for p in images[:32]],gt=sum(counts[:32]),class_counts=firstclasses),shape_set=sorted(shapes),
 aliases=[str(p) for p in images],canonical=[str(p.resolve()) for p in images],original_label_files=labels,label_counts=counts,rgb_ir_image_names_exact=True,
 run_file_inventory=[dict(name=p.name,size=p.stat().st_size) for p in run.iterdir() if p.is_file()],run_completion=json.loads((run/'completion_receipt.json').read_text()),
 old_metrics_record=json.loads((run/'metrics_record.json').read_text()),scan_scope='Project artifacts and runs, no directory symlink traversal, excludes weights/source copies; JSON/YAML/MD metadata <=2MB only',
 scanned_files=scanned,possible_prediction_or_capture_files=possible,metadata_references_to_target_checkpoint=references,new_hash_computed=False,GPU_used=False,checkpoint_contents_read=False)
print(json.dumps(result,ensure_ascii=False))
'''

if __name__=='__main__':
    p=subprocess.run(['ssh','94','/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python','-'],input=REMOTE,text=True,encoding='utf-8',capture_output=True,check=True)
    value=json.loads(p.stdout)
    target=Path(__file__).with_name('remote_input_audit.json')
    with target.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in value.items() if k not in ('aliases','canonical','original_label_files','label_counts','possible_prediction_or_capture_files','metadata_references_to_target_checkpoint','run_completion')},ensure_ascii=False))
