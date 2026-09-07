"""Independent LLVIP full-capture/TIDE/COCO cached CPU verification."""
import contextlib,gzip,importlib.util,json,sys,time
from pathlib import Path,PurePosixPath
import numpy as np
HERE=Path(__file__).resolve().parent;AP=HERE.parent;CAMPAIGN=AP.parent/'llvip_full_eval';DATA=CAMPAIGN/'remote_completed_attempt2';RESULT=AP/'llvip_results_v1'
sys.dont_write_bytecode=True
sys.path[:0]=[str(AP/'deps'),str(AP/'official_tide')]
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tidecv import TIDE
from tidecv.data import Data

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def rows(p):
    with gzip.open(p,'rt',encoding='utf-8') as f:return [json.loads(s) for s in f if s.strip()]
def stat(p):
    p=Path(p);s=p.stat();return {'path':str(p),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns}
def close(a,b):assert abs(float(a)-float(b))<1e-8,(a,b)
def dump(name,x):(HERE/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def pairkey(p,modality):
    root=PurePosixPath('/mnt/dataset/yudongfang/projects/RGBT_campaign/data/processed/llvip/yolo/grouped_v1')/modality/'images'/'dev'
    return str(PurePosixPath(p).relative_to(root))
start=time.time();before={};profilechecks={};captures={};populationchecks={}
core_sources=[CAMPAIGN/'export_full_dev.py',DATA/'export_full_dev.py',CAMPAIGN/'run_campaign.py',DATA/'run_campaign.py',
              AP/'run_tide_audit.py',RESULT/'runner_source.py',RESULT/'summary.json',CAMPAIGN/'verified_pair_manifest.jsonl',CAMPAIGN/'completed_verification_receipt.json']
for p in core_sources:before[str(p)]=stat(p)
assert (CAMPAIGN/'export_full_dev.py').read_bytes()==(DATA/'export_full_dev.py').read_bytes()
assert (CAMPAIGN/'run_campaign.py').read_bytes()==(DATA/'run_campaign.py').read_bytes()
assert (AP/'run_tide_audit.py').read_bytes()==(RESULT/'runner_source.py').read_bytes()
for model,mod in [('N42','visible'),('T42','infrared')]:
    for stage,n,ngt in [('canary',64,279),('full',2406,7879)]:
        folder=DATA/(model+'_'+stage+'_attempt1');summary=read(folder/'summary.json');contract=read(folder/'capture_contract.json');pop=read(folder/'population.json')
        actual=rows(folder/'capture/objects.jsonl.gz');identity=read(folder/'model_identity.json')['capture'];metrics=read(folder/'capture_metrics.json')
        for fname in ['summary.json','capture_contract.json','population.json','model_identity.json','capture_metrics.json','capture/objects.jsonl.gz','sources/0_export_full_dev.py']:
            p=folder/fname;before[str(p)]=stat(p)
        assert summary['status']=='completed' and summary['images']==n and summary['gt_count']==ngt
        assert metrics==summary['metrics'];assert set(r['class_id'] for r in metrics['per_class'])=={0}
        assert len(actual)==len({r['image'] for r in actual})==n
        assert sum(len(r['gt_boxes']) for r in actual)==ngt
        assert all(len(r['gt_boxes'])==len(r['gt_classes']) for r in actual)
        assert all(set(r['gt_classes'])<=set([0.]) and set(r['pred_classes'])<=set([0.]) for r in actual)
        assert all(len(r['pred_boxes'])==len(r['pred_classes'])==len(r['pred_confidence'])<=300 for r in actual)
        alias=contract['alias_to_canonical']
        assert len(alias)==len(set(alias.values()))==n and set(alias.values())==set(contract['roster'])
        assert set(alias)==set(r['image'] for r in actual)==set(pop['evaluated_aliases'])
        assert set(contract['actual_loader_roster'])==set(contract['roster'])
        assert set(pop['evaluated_roster'])==set(contract['roster'])
        assert contract['loader_per_image_labels_exact'] is True
        assert contract['expected_gt']==contract['loader_gt']==contract['captured_gt']==ngt
        assert len(pop['original_label_counts'])==len(pop['original_label_files'])==len(pop['full_roster'])==2406
        assert sum(pop['original_label_counts'])==pop['full_gt']==7879
        assert pop['expected_evaluated_gt']==ngt
        assert all('/'+mod+'/labels/dev/' in p for p in pop['original_label_files'])
        for r in actual:pairkey(r['image'],mod)
        for k in ('official_test_accessed','new_protocol_N','training_modified','physical_alignment_verified'):assert summary[k] is False
        assert identity['scope']=='historical_baseline_not_new_protocol_N'
        assert identity['training_args']['seed']==42 and identity['training_args']['epochs']==200 and identity['training_args']['workers']==8
        assert identity['names']=={'0':'person'} and identity['loaded_source']=='model' and identity['ema_present'] is False and identity['checkpoint_epoch']==-1
        assert identity['path']==summary['checkpoint'] and identity['path'].endswith('/weights/last.pt')
        kwargs=contract['effective_kwargs']
        for k,value in {'imgsz':640,'batch':32,'workers':4,'conf':.001,'iou':.7,'max_det':300,'rect':True,'augment':False,'half':False,'quantize':None}.items():assert kwargs[k]==value
        assert (folder/'sources/0_export_full_dev.py').read_bytes()==(DATA/'export_full_dev.py').read_bytes()
        eq=False
        if stage=='canary':
            assert read(folder/'native_metrics.json')==metrics
            native=read(folder/'native_contract.json')
            assert native['effective_kwargs']==contract['effective_kwargs'] and native['actual_loader_roster']==contract['actual_loader_roster']
            assert native['loader_per_image_labels_exact'] and native['expected_gt']==native['loader_gt']==ngt
            assert summary['native_capture_exact'] is True;eq=True
        else:
            assert summary['native_capture_exact'] is False and not (folder/'native_metrics.json').exists()
            captures[model]=actual
        pr=read(DATA/'queue_attempt1'/('llvip_full_'+model+'_'+stage+'_resource_profile.json'));adm=read(DATA/'queue_attempt1'/('llvip_full_'+model+'_'+stage+'_admission.json'))
        assert pr['status']=='COMPLETED' and pr['exit_code']==0 and not pr['monitor_errors']
        assert pr['launch']['gpu_ids']==[4] and pr['launch']['cuda_processes_per_gpu']==1
        assert len(adm['active_gpus_after'])<=3 and not adm['four_gpu_exception']
        assert pr['minimum_free_mib']>=2048 and max(v['project_rss_mib'] for v in pr['samples'])<=300*1024
        assert summary['resources']['gpu_ids']==[4] and summary['resources']['cuda_pid_counts']=={'4':1}
        if stage=='full':
            canary=read(DATA/'queue_attempt1'/(model+'_canary_acceptance.json'))
            assert canary['source']=='this_path_actual_canary' and canary['vram_measured_mib']==1370
            assert pr['launch']['expected_vram_mib']==canary['reservation_vram_mib']==1882
        key=model+'_'+stage
        populationchecks[key]={'images':n,'gt':ngt,'native_capture_metric_and_kwargs_exact':eq,'full_is_capture_only':stage=='full',
                              'alias_and_roster_bijection':True,'all_original_label_files_under_processed_modality_root':True,
                              'actual_loader_per_image_labels_exact_recorded':True,'capture_gt_and_class_population_exact':True,
                              'old_workers8_model_new_workers4_evaluation':True,'checkpoint':identity['path']}
        profilechecks[key]={'gpu':4,'active_gpus_after':adm['active_gpus_after'],'our_cuda_process_count':1,
                           'minimum_free_mib':pr['minimum_free_mib'],'peak_project_rss_mib':max(v['project_rss_mib'] for v in pr['samples']),
                           'task_peak_nvml_mib':summary['resources']['per_gpu_peak_vram_mib'],'completed_no_monitor_errors':True}

keys={m:{pairkey(r['image'],mod):r for r in captures[m]} for m,mod in [('N42','visible'),('T42','infrared')]}
assert len(keys['N42'])==len(keys['T42'])==2406 and set(keys['N42'])==set(keys['T42'])
manifest=[]
for k in sorted(keys['N42']):
    n,t=keys['N42'][k],keys['T42'][k]
    for f in ['gt_boxes','gt_classes','canvas_shape','original_shape']:assert n[f]==t[f],(k,f)
    manifest.append({'pair_key':k,'rgb_image':n['image'],'ir_image':t['image'],'gt_count':len(n['gt_boxes']),'canvas_shape':n['canvas_shape'],'original_shape':n['original_shape']})
prior=[json.loads(s) for s in (CAMPAIGN/'verified_pair_manifest.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
assert manifest==prior
tidesummary=read(RESULT/'summary.json');assert tidesummary['status']=='COMPLETED' and tidesummary['same_gt_population'] is False

def direct_tide(records,threshold):
    gt,pred=Data('gt',max_dets=300),Data('pred',max_dets=300);gt.add_class(0,'person');pred.add_class(0,'person')
    for i,r in enumerate(records):
        gt.add_image(i,r['image']);pred.add_image(i,r['image'])
        for b in r['gt_boxes']:gt.add_ground_truth(i,0,box=[b[0],b[1],b[2]-b[0],b[3]-b[1]])
        for b,s in zip(r['pred_boxes'],r['pred_confidence']):pred.add_detection(i,0,float(s),box=[b[0],b[1],b[2]-b[0],b[3]-b[1]])
    return TIDE().evaluate(gt,pred,pos_threshold=threshold,background_threshold=.1,mode=TIDE.BOX,use_for_errors=True)

tidechecks={};cocochecks={}
with (HERE/'coco_stdout.log').open('w') as log,contextlib.redirect_stdout(log):
    for model in ('N42','T42'):
        recs=captures[model];expected=read(RESULT/('LLVIP_'+model+'.json'));assert expected==tidesummary['results']['LLVIP_'+model]
        tidechecks[model]={}
        for threshold in (.5,.75):
            run=direct_tide(recs,threshold);e=expected['thresholds'][str(threshold)]
            close(run.ap,e['AP']);main={c.short_name:v for c,v in run.fix_main_errors().items()};special={c.short_name:v for c,v in run.fix_special_errors().items()}
            for k,v in main.items():close(v,e['main_dAP'][k])
            for k,v in special.items():close(v,e['special_dAP'][k])
            assert main['Cls']==main['Both']==0
            assert len(run.ap_data.objs)==1 and run.ap_data.objs[0].num_gt_positives==7879
            points=run.ap_data.objs[0].data_points
            tp=sum(p[1] for p in points.values());fp=len(points)-tp;fn=7879-tp
            original=e['independent_ap']['per_class']['0'];assert (tp,fp,fn)==(original['tp'],original['fp'],original['fn'])
            errors={z.pred['_id']:type(z).short_name for z in run.errors if hasattr(z,'pred')}
            for j,b in enumerate(e['score_diagnostic']['pooled']['bins']):
                pp=[(pid,p) for pid,p in points.items() if b['low']<=p[0]<(b['high'] if j<4 else 1.0000001)]
                assert sum(p[1] for _,p in pp)==b['tp'] and len(pp)-sum(p[1] for _,p in pp)==b['fp']
                assert sum(errors.get(pid)=='Bkg' for pid,_ in pp)==b['error_counts'].get('Bkg',0)
            tidechecks[model][str(threshold)]={'AP':run.ap,'AP_difference_pp':run.ap-e['AP'],'main_dAP':main,'special_dAP':special,'TP':tp,'FP':fp,'FN':fn,'score_bins_exact':True}
        gt=COCO();ims=[];annotations=[];preds=[]
        for i,r in enumerate(recs,1):
            h,w=r['canvas_shape'];ims.append(dict(id=i,height=h,width=w))
            for b in r['gt_boxes']:
                box=[b[0],b[1],b[2]-b[0],b[3]-b[1]];annotations.append(dict(id=len(annotations)+1,image_id=i,category_id=0,bbox=box,area=box[2]*box[3],iscrowd=0))
            for b,s in zip(r['pred_boxes'],r['pred_confidence']):preds.append(dict(image_id=i,category_id=0,bbox=[b[0],b[1],b[2]-b[0],b[3]-b[1]],score=s))
        gt.dataset=dict(images=ims,annotations=annotations,categories=[dict(id=0,name='person')]);gt.createIndex();pd=gt.loadRes(preds)
        ev=COCOeval(gt,pd,'bbox');ev.params.iouThrs=np.array([.5,.75]);ev.params.maxDets=[1,10,300]
        ev.evaluate();ev.accumulate();default=ev.eval['precision'].copy();ev.params.recThrs=np.arange(101)/100;ev.accumulate()
        cocochecks[model]={}
        for j,threshold in enumerate((.5,.75)):
            p=ev.eval['precision'][j,:,:,0,2];actual=float(p[p>-1].mean()*100);target=expected['thresholds'][str(threshold)]['AP'];close(actual,target)
            d=default[j,:,:,0,2];default_ap=float(d[d>-1].mean()*100)
            cocochecks[model][str(threshold)]={'COCO_AP_TIDE_recall_grid':actual,'TIDE_AP':target,'difference_pp':actual-target,'COCO_default_AP':default_ap,'default_difference_pp':default_ap-target}

# Current alias-aware synthetic tests replay only into this review directory.
spec=importlib.util.spec_from_file_location('llvip_tide_adapter_test',AP/'run_tide_audit.py');adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
testout=HERE/'synthetic_rerun';testout.mkdir(exist_ok=False);adapter.validate_synthetic(testout)
assert read(testout/'synthetic_validation.json')==read(RESULT/'synthetic_validation.json')
for p,old in before.items():assert stat(p)==old,('input changed',p)
dump('verification_receipt.json',{'status':'PASS_LIMITED_LLVIP_EVALUATION_SCOPE','auditor':'/root/loc_stress','input_stat_unchanged':list(before.values()),
     'actual_population_and_identity':populationchecks,'resource_receipts':profilechecks,
     'full_pair_keys_and_gt_canvas_original_shape_exact':True,'pair_key_contract':'relative path under registered grouped_v1/<modality>/images/dev, no basename-only pairing',
     'verified_pair_manifest_exact_to_root':True,'image_count':2406,'GT_count':7879,'TIDE_direct_CPU_rerun':tidechecks,
     'independent_COCOeval_AP_crosscheck':cocochecks,'alias_aware_synthetic_tests_exact':True,
     'no_new_hashes':True,'new_GPU_or_SSH':False,'no_original_server_label_bytes_reread':True,'physical_registration_or_KD_gain':False,'seconds':time.time()-start})
print('LLVIP scoped verification PASS; complete captures, TIDE oracle and independent COCO AP, seconds',time.time()-start,flush=True)
