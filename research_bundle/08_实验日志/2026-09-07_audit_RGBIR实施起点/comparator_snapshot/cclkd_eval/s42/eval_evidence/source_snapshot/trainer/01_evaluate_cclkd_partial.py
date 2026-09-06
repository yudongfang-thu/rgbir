"""Re-evaluate three existing partial CCLKD last checkpoints; never train or access test."""
import argparse,csv,json,sys,time
from pathlib import Path
import torch,ultralytics,yaml
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset,IMG_FORMATS
from ultralytics.models.yolo.detect.val import DetectionValidator
REPO=Path('/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction')
ROOT=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign')
sys.path.insert(0,str(REPO))
from tools.project_resource_guard import require_bound_lease_from_environment,bound_lease_resource_record_from_environment
from tools.write_jstars_run_receipt import emit_bound_run_receipt,implementation_files

def dump(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,choices=[0,42,123],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['canary','full'],required=True)
    a=p.parse_args();lease=require_bound_lease_from_environment()
    assert len(lease['gpus'])==1
    assert str(torch.__version__)=='2.10.0+cu128' and ultralytics.__version__=='8.4.115'
    run=ROOT/f'runs/rgbt_cclkd_adapted_v1/cclkd_drone_seed{a.seed}_b32_e200'
    receipt=json.loads((run/'completion_receipt.json').read_text())
    args=yaml.safe_load((run/'args.yaml').read_text())
    assert receipt['status']=='completed' and receipt['seed']==a.seed
    assert receipt['epochs_configured']==200 and receipt['kd_terms_v1']==['LLD','CCL']
    assert args['imgsz']==640 and args['batch']==32 and args['epochs']==200
    assert len(list(csv.DictReader((run/'results.csv').open())))==200
    data=Path(args['data']);original=yaml.safe_load(data.read_text())
    assert 'test' not in original
    val_dir=Path(check_det_dataset(str(data),autodownload=False)['val'])
    images=sorted(x.resolve() for x in val_dir.rglob('*') if x.is_file() and x.suffix[1:].lower() in IMG_FORMATS)
    assert len(images)==1469 and len(set(images))==1469
    a.output.mkdir(parents=True,exist_ok=False)
    full_roster=a.output/'full_val_roster.txt'
    full_roster.write_text(''.join(str(x)+'\n' for x in images))
    used=images;eval_data=data
    if a.mode=='canary':
        def count_objects(image):
            label=Path(str(image).replace('/images/','/labels/')).with_suffix('.txt')
            return len(label.read_text().splitlines()) if label.is_file() else 0
        used=sorted(images,key=lambda x:(-count_objects(x),str(x)))[:64]
        subset=a.output/'canary_64_dense_val.txt'
        subset.write_text(''.join(str(x)+'\n' for x in used))
        cfg=dict(original);cfg['val']=str(subset)
        eval_data=a.output/'canary_data.yaml';eval_data.write_text(yaml.safe_dump(cfg))
    roster=a.output/'evaluation_roster.txt';roster.write_text(''.join(str(x)+'\n' for x in used))
    checkpoint=run/'weights/last.pt'
    dump(a.output/'evaluation_protocol.json',dict(mode=a.mode,seed=a.seed,checkpoint=str(checkpoint),
         data=str(data),evaluated_images=len(used),imgsz=640,batch=32,workers=4,split='val',endpoint='fixed_budget_last_ema',
         method='CCLKD-adapted partial LLD+CCL',teacher_labels_used_by_kd=False,
         historical_training_source_not_fully_bound=True,command=[sys.executable,*sys.argv]))
    torch.cuda.reset_peak_memory_stats();start=time.time()
    model=YOLO(str(checkpoint),task='detect')
    metrics=model.val(data=str(eval_data),split='val',imgsz=640,batch=32,workers=4,device='0',
         plots=False,save_json=False,verbose=False,project=str(a.output),name='val',exist_ok=False)
    torch.cuda.synchronize()
    result=dict(status='canary_completed' if a.mode=='canary' else 'evaluation_completed',
         method_id='CCLKD-ADAPTED-PARTIAL-LLD-CCL-HISTORICAL',arm='cclkd_partial',seed=a.seed,
         checkpoint=str(checkpoint),endpoint='fixed_budget_last_ema',split='val',evaluated_images=len(used),
         official_test_accessed=False,metric_units='fraction_0_to_1',canary=a.mode=='canary',
         AP50=float(metrics.box.map50),AP75=float(metrics.box.map75),mAP50_95=float(metrics.box.map),
         precision=float(metrics.box.mp),recall=float(metrics.box.mr),
         allocated_peak_mib=torch.cuda.max_memory_allocated()/2**20,
         reserved_peak_mib=torch.cuda.max_memory_reserved()/2**20,seconds=time.time()-start,
         resources=bound_lease_resource_record_from_environment())
    target=a.output/('canary_result.json' if a.mode=='canary' else 'evaluation_val.json');dump(target,result)
    emit_bound_run_receipt(run_dir=a.output/'eval_evidence',method_identity='PROTOCOL-ADAPTED',
         dataset='dronevehicle',data_role='development_val',seed=a.seed,run_kind='eval',
         trainers=[Path(__file__),*implementation_files(DetectionValidator,YOLO)],losses=[],
         configs=[data,run/'args.yaml',a.output/'evaluation_protocol.json'],split_rosters=[roster],metric_files=[target],
         environment={'torch':str(torch.__version__),'ultralytics':ultralytics.__version__},
         inputs={'checkpoint':str(checkpoint),'historical_training_receipt':str(run/'completion_receipt.json'),
                 'method_id':result['method_id'],'teacher_labels_used_by_kd':False,
                 'historical_training_source_not_fully_bound':True,'canary':a.mode=='canary'})
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
