"""Fixed last/EMA endpoint, full development val; never accesses test."""
import argparse
import json
import sys
from pathlib import Path
import torch
import yaml
import ultralytics
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset, IMG_FORMATS

REPO=Path('/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction')
sys.path.insert(0,str(REPO))
from tools.project_resource_guard import require_bound_lease_from_environment
from tools.write_jstars_run_receipt import emit_bound_run_receipt

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    a=p.parse_args()
    cfg=yaml.safe_load(a.config.read_text())
    completed=json.loads((a.run/'completion_receipt.json').read_text())
    if completed['status']!='training_completed': raise ValueError('Full training must complete before endpoint evaluation')
    data=Path(cfg['paths']['student_data_yaml'])
    if 'test' in yaml.safe_load(data.read_text()): raise ValueError('Train/val-only YAML required')
    if len(require_bound_lease_from_environment()['gpus'])!=1: raise ValueError('One GPU only')
    if str(torch.__version__)!=cfg['torch_version'] or ultralytics.__version__!=cfg['ultralytics_version']:
        raise RuntimeError('Pinned environment changed')
    checkpoint=a.run/'weights/last.pt'
    if (a.run/'eval_val').exists(): raise FileExistsError('Do not overwrite an evaluation attempt')
    val_dir=Path(check_det_dataset(str(data),autodownload=False)['val'])
    if not val_dir.is_dir(): raise ValueError('Frozen Drone val must resolve to an image directory')
    roster=a.run/'evaluation_val_roster.txt'
    images=sorted(p.resolve() for p in val_dir.rglob('*') if p.is_file() and p.suffix[1:].lower() in IMG_FORMATS)
    if len(images)!=int(cfg['expected_val_images']): raise ValueError('Frozen validation population changed')
    roster.write_text(''.join(str(p)+'\n' for p in images))
    model=YOLO(str(checkpoint),task='detect')
    metrics=model.val(data=str(data),split='val',imgsz=cfg['imgsz'],batch=cfg['batch'],
        workers=cfg['workers'],device='0',plots=False,save_json=False,verbose=False,
        project=str(a.run),name='eval_val',exist_ok=False)
    result={'AP50':float(metrics.box.map50),'AP75':float(metrics.box.map75),
        'mAP50_95':float(metrics.box.map),'precision':float(metrics.box.mp),'recall':float(metrics.box.mr),
        'checkpoint':str(checkpoint),'endpoint':'fixed_budget_last_ema','split':'val',
        'arm':completed['arm'],'seed':completed['seed'],'single_seed_exploratory':True,
        'official_test_accessed':False,'metric_units':'fraction_0_to_1','method_id':cfg['method_id']}
    target=a.run/'evaluation_val.json'
    target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    emit_bound_run_receipt(run_dir=a.run/'eval_evidence',method_identity=cfg['method_identity'],
        dataset=cfg['dataset'],data_role='development_val',seed=completed['seed'],run_kind='eval',
        trainers=[Path(__file__)],losses=[],configs=[a.config,data],split_rosters=[roster],
        metric_files=[target],environment={'torch':str(torch.__version__),'ultralytics':ultralytics.__version__},
        inputs={'method_id':cfg['method_id'],'arm':completed['arm'],'checkpoint':str(checkpoint),
            'endpoint':'fixed_budget_last_ema','teacher_labels_used_by_kd':True})
    print(json.dumps(result),flush=True)

if __name__=='__main__': main()
