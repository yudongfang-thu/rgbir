"""Read-only CPU check of pinned construction paths and worker-dependent augmentation.

Run over ssh stdin with CUDA_VISIBLE_DEVICES empty. No training or remote files.
This is a reconstruction with CURRENT pinned source, not recovery of old initial state.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import contextlib, gc, io, json
from pathlib import Path
from types import SimpleNamespace
import torch, yaml
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data import build_yolo_dataset, build_dataloader
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.torch_utils import init_seeds

torch.set_num_threads(4)
B=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign')
args=yaml.safe_load((B/'runs/rgbir_object_evidence_v1_20260906/full_weight0_s42_attempt1/args.yaml').read_text())
data=check_det_dataset(args['data'], autodownload=False)
result={'scope':'CPU reconstruction only; no historical initial checkpoint recovered; no GPU/training', 'model_path':args['model'], 'data':args['data'], 'construction':[], 'loader':[]}

for seed in (0,42,123):
    original=YOLO(args['model'])
    init_seeds(seed, deterministic=True)
    old=DetectionTrainer.__new__(DetectionTrainer)
    old.data=data;old.args=SimpleNamespace(cls_remap=True)
    old_model=old.get_model(weights=original.model,cfg=original.model.yaml,verbose=False)
    old_state={k:v.clone() for k,v in old_model.state_dict().items()}
    old_rng=torch.get_rng_state().clone()
    init_seeds(seed,deterministic=True)
    new=DetectionTrainer.__new__(DetectionTrainer)
    new.data=data;new.args=SimpleNamespace(cls_remap=True,pretrained=True);new.model=args['model'];new.resume=False
    new.setup_model()
    new_state=new.model.state_dict()
    unequal=[k for k in old_state if not torch.equal(old_state[k],new_state[k])]
    result['construction'].append({'seed':seed,'tensor_count':len(old_state),'unequal_tensors':unequal,'torch_rng_equal':bool(torch.equal(old_rng,torch.get_rng_state()))})
    del original,old,old_model,old_state,new,new_state
    gc.collect()

saved=[]
for workers in (4,8):
    init_seeds(42,deterministic=True)
    config=get_cfg(overrides={k:v for k,v in args.items() if k!='save_dir'})
    ds=build_yolo_dataset(config,data['train'],32,data,mode='train',rect=False,stride=32)
    loader=build_dataloader(ds,batch=32,workers=workers,shuffle=True,rank=-1,device='cpu')
    result.setdefault('loader_runtime',[]).append({'requested_workers':workers,'actual_workers':loader.num_workers,'images':len(ds)})
    for i,batch in enumerate(loader):
        row={'files':list(batch['im_file']),'first_img':batch['img'][0].clone(),'cls':batch['cls'].clone(),'bboxes':batch['bboxes'].clone(),'batch_idx':batch['batch_idx'].clone()}
        if workers==4:saved.append(row)
        else:
            ref=saved[i]
            result['loader'].append({'batch_1based':i+1,'same_file_order':ref['files']==row['files'],'first_image_exact_equal':bool(torch.equal(ref['first_img'],row['first_img'])),'first_image_changed_values':int((ref['first_img']!=row['first_img']).sum()),'labels_exact_equal':all(torch.equal(ref[k],row[k]) for k in ('cls','bboxes','batch_idx')),'first_file':row['files'][0]})
        if i==9:break
    if hasattr(loader,'iterator') and hasattr(loader.iterator,'_shutdown_workers'):loader.iterator._shutdown_workers()
    del loader,ds
    gc.collect()
result['cuda_initialized']=torch.cuda.is_initialized()
print('AUDIT_RESULT_JSON='+json.dumps(result,ensure_ascii=False,allow_nan=False))
