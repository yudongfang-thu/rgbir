"""Call the CFT author's LLVIP evaluator, preserving its loader and AP code."""
import argparse
import contextlib
import datetime
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
from types import SimpleNamespace

ROOT=Path('/mnt/dataX/ydf/projects/RGBT_campaign_90')
ART=ROOT/'artifacts/cft_author_protocol_20260909_attempt1'
SOURCE=ROOT/'external_reproductions/cft/author_source'
WEIGHT=ROOT/'external_reproductions/cft/author_llvip.pt'

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['canary','full'],required=True)
    p.add_argument('--batch',type=int,default=64)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,a.output/'executed_wrapper.py')
    start=time.perf_counter()
    def save(name,obj):
        (a.output/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
    os.environ.update(TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',
        MPLCONFIGDIR=str(ROOT/'cache/cft_matplotlib'),TORCH_HOME=str(ROOT/'cache/torch'))
    sys.path.insert(0,str(ROOT))
    from tools.project_resource_guard import require_bound_lease_from_environment
    lease=require_bound_lease_from_environment()
    sys.path.insert(0,str(SOURCE))
    sys.path.insert(1,str(ROOT/'artifacts/reproduction_20260909_attempt1'))
    os.chdir(SOURCE)
    import numpy as np
    np.int=int
    np.float=float
    import torch
    import psutil
    import yaml
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(0.65,0)
    from cft_checkpoint_compat import restore_cft_block_names
    import test as author_test
    original_load=author_test.attempt_load
    def compatible_load(*args,**kwargs):
        model=original_load(*args,**kwargs)
        save('checkpoint_compatibility.json',restore_cft_block_names(model))
        return model
    author_test.attempt_load=compatible_load
    original_loader=author_test.create_dataloader_rgb_ir
    def checked_loader(*args,**kwargs):
        loader,dataset=original_loader(*args,**kwargs)
        rgb=[Path(x).stem for x in dataset.img_files_rgb]
        ir=[Path(x).stem for x in dataset.img_files_ir]
        expected=[Path(x).stem for x in Path(args[0]).read_text().splitlines()]
        assert len(rgb)==3463 and rgb==ir and len(set(rgb))==3463 and set(rgb)==set(expected)
        save('evaluated_roster.json',dict(images=3463,stems=rgb,pair_order_identical=True))
        return loader,dataset
    author_test.create_dataloader_rgb_ir=checked_loader
    original_nms=author_test.non_max_suppression
    def checked_nms(*args,**kwargs):
        class Tee:
            def __init__(self,stream):self.stream=stream;self.messages=[]
            def write(self,value):self.messages.append(value);return self.stream.write(value)
            def flush(self):self.stream.flush()
        tee=Tee(sys.stdout)
        with contextlib.redirect_stdout(tee):result=original_nms(*args,**kwargs)
        if 'NMS time limit' in ''.join(tee.messages):
            raise RuntimeError('Author NMS hit time limit: incomplete batch, result rejected')
        return result
    author_test.non_max_suppression=checked_nms
    yaml_path=ROOT/'data_author_protocol/llvip_attempt1/previous/LLVIP.yaml'
    assert json.loads((ART/'data_receipt.json').read_text())['status']=='PREPARED'
    for rel in ('test.py','utils/datasets.py','utils/general.py','utils/metrics.py'):
        dest=a.output/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(SOURCE/rel,dest)
    shutil.copy2(yaml_path,a.output/'effective_data.yaml')
    plan=dict(status='STARTED',mode=a.mode,batch_size=a.batch,input_size=1024,conf_thres=.001,nms_iou=.5,
        augment=False,rect=True,pad=.5,half_precision=True,model='author CFT LLVIP checkpoint',
        checkpoint=str(WEIGHT),data=str(yaml_path),split='official_test',expected_images=3463,
        annotations='official previous version',identity='PAPER-RECONSTRUCTED',
        purpose='Original-paper protocol checkpoint reevaluation; not our method selection',
        compatibility=['torch2.10 legacy pickle loading','NumPy removed int/float aliases','explicit CFT class rebinding'],
        original_evaluator=str(SOURCE/'test.py'),original_loader=True,original_ap_function=True,
        author_cli_batch_default=64,batch_resource_deviation=(a.batch!=64),lease=lease,
        training_started=False,new_digest_calculated=False,started=datetime.datetime.now().astimezone().isoformat())
    save('plan.json',plan)
    peak={'process_tree_rss_mib':0.0,'nvml_process_mib':0,'nvml_card_used_mib':0,'nvml_min_card_free_mib':999999}
    stop=threading.Event();proc=psutil.Process()
    physical=int(os.environ['CUDA_VISIBLE_DEVICES'].split(',')[0])
    def monitor():
        while not stop.is_set():
            try:
                rss=sum(q.memory_info().rss for q in [proc]+proc.children(recursive=True) if q.is_running())/2**20
                peak['process_tree_rss_mib']=max(peak['process_tree_rss_mib'],rss)
                rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=5)
                for line in rows.splitlines():
                    pid,mem=[x.strip() for x in line.split(',')]
                    if pid==str(os.getpid()) and mem.isdecimal():peak['nvml_process_mib']=max(peak['nvml_process_mib'],int(mem))
                rows=subprocess.check_output(['nvidia-smi','-i',str(physical),'--query-gpu=memory.used,memory.free','--format=csv,noheader,nounits'],text=True,timeout=5)
                used,free=[int(x.strip()) for x in rows.strip().split(',')]
                peak['nvml_card_used_mib']=max(peak['nvml_card_used_mib'],used)
                peak['nvml_min_card_free_mib']=min(peak['nvml_min_card_free_mib'],free)
                if free<2048 or peak['nvml_process_mib']>=24576*.70 or rss>300*1024:
                    save('resource_violation.json',dict(peak));os._exit(86)
            except (psutil.NoSuchProcess,subprocess.SubprocessError,ValueError):pass
            stop.wait(.5)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    try:
        if a.mode=='canary':
            model=compatible_load(str(WEIGHT),map_location=torch.device('cuda:0')).half().eval()
            data=yaml.safe_load(yaml_path.read_text())
            opt=SimpleNamespace(single_cls=False)
            loader,dataset=author_test.create_dataloader_rgb_ir(data['val_rgb'],data['val_ir'],1024,a.batch,
                max(int(model.stride.max()),32),opt,pad=.5,rect=True)
            assert len(dataset)==3463
            items=[]
            for i,(imgs,targets,paths,shapes) in enumerate(loader):
                if i==3:break
                imgs=imgs.cuda().half()/255
                torch.cuda.synchronize();t=time.perf_counter()
                with torch.no_grad():
                    pred,_=model(imgs[:,:3],imgs[:,3:],augment=False)
                    assert torch.isfinite(pred).all()
                    det=author_test.non_max_suppression(pred,.001,.5,multi_label=True)
                torch.cuda.synchronize()
                items.append(dict(batch=i,size=list(imgs.shape),seconds=time.perf_counter()-t,
                                  images=[Path(x).stem for x in paths],detections=sum(len(d) for d in det)))
                print('CANARY_BATCH',i,items[-1]['seconds'],flush=True)
            save('canary_batches.json',items)
            result=dict(status='CANARY_PASS',batches=len(items),images=sum(x['size'][0] for x in items),
                        steady_seconds_per_image=sum(x['seconds'] for x in items[1:])/sum(x['size'][0] for x in items[1:]),ap_computed=False)
        else:
            metric=author_test.ap_per_class
            def capture_metric(tp,conf,pred_cls,target_cls,*args,**kwargs):
                np.savez_compressed(a.output/'author_metric_inputs.npz',tp=tp,conf=conf,pred_cls=pred_cls,target_cls=target_cls)
                return metric(tp,conf,pred_cls,target_cls,*args,**kwargs)
            author_test.ap_per_class=capture_metric
            opt=SimpleNamespace(device='0',project=str(a.output),name='author_output',exist_ok=False,task='test',single_cls=False)
            result_tuple,maps,times=author_test.test(str(yaml_path),weights=str(WEIGHT),batch_size=a.batch,imgsz=1024,
                conf_thres=.001,iou_thres=.5,save_json=False,single_cls=False,augment=False,verbose=True,
                save_txt=True,save_conf=True,save_hybrid=False,plots=False,opt=opt)
            result=dict(status='COMPLETED',metrics=dict(zip(['precision','recall','AP50','AP75','AP50_95'],map(float,result_tuple[:5]))),
                        maps=maps.tolist(),author_timing=list(times),ap_computed=True)
        torch.cuda.synchronize()
        result.update(plan=plan,peak=peak,torch_max_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            torch_max_reserved_mib=torch.cuda.max_memory_reserved()/2**20,elapsed_seconds=time.perf_counter()-start,
            finished=datetime.datetime.now().astimezone().isoformat())
        save('receipt.json',result);print(json.dumps(result),flush=True)
    except Exception as exc:
        save('failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),peak=peak,
            elapsed_seconds=time.perf_counter()-start));raise
    finally:
        stop.set();thread.join(timeout=6)

if __name__=='__main__':main()
