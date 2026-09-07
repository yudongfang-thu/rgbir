"""CPU-only official TIDE adapter. No training import, GPU use, or SSH.

python run_tide_audit.py --output results_v1 (six frozen Drone endpoints)
python run_tide_audit.py --manifest endpoints.json --output llvip_v1
Manifest: list of {name, path, [class_names], [metric_path], [contract_path], [receipt_path]}.
"""
import argparse, collections, copy, csv, datetime, gzip, json, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE/'deps'), str(HERE/'official_tide')]
import numpy as np
from tidecv import TIDE
from tidecv.data import Data

NAMES = ['car', 'freight car', 'truck', 'bus', 'van']
def file_stat(path):
    p=Path(path);s=p.stat()
    return dict(path=str(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns)
def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
def read_rows(path):
    path=Path(path)
    with (gzip.open(path,'rt',encoding='utf-8') if path.suffix=='.gz' else path.open(encoding='utf-8')) as f:
        return [json.loads(x) for x in f if x.strip()]
def xywh(box):
    x,y,r,b=map(float,box)
    assert r>x and b>y, box
    return [x,y,r-x,b-y]
def make_data(rows, names, max_det=300):
    gt, pred=Data('gt',max_dets=max_det),Data('pred',max_dets=max_det)
    for c,n in enumerate(names): gt.add_class(c,n);pred.add_class(c,n)
    for i,r in enumerate(rows):
        gt.add_image(i,r['image']);pred.add_image(i,r['image'])
        for b,c in zip(r['gt_boxes'],r['gt_classes']): gt.add_ground_truth(i,int(c),box=xywh(b))
        for b,c,s in zip(r['pred_boxes'],r['pred_classes'],r['pred_confidence']): pred.add_detection(i,int(c),float(s),box=xywh(b))
    return gt,pred
def iou_matrix(a,b):
    a=np.asarray(a,dtype=np.float64).reshape(-1,4);b=np.asarray(b,dtype=np.float64).reshape(-1,4)
    lt=np.maximum(a[:,None,:2],b[None,:,:2]); rb=np.minimum(a[:,None,2:],b[None,:,2:])
    intersection=np.prod(np.maximum(rb-lt,0),axis=2)
    aa=np.prod(a[:,2:]-a[:,:2],axis=1);bb=np.prod(b[:,2:]-b[:,:2],axis=1)
    return intersection/(aa[:,None]+bb[None,:]-intersection)
def independent_ap(rows, threshold, names, max_det=300):
    """Independent NumPy IoU + score-greedy matcher + thresholdwise AP101.

    Not a call-through of TIDE. Preserves stable input order for score ties.
    """
    by_class=collections.defaultdict(list);ngt=collections.Counter()
    for row in rows:
        ngt.update(map(int,row['gt_classes']))
        order=sorted(range(len(row['pred_boxes'])),key=lambda j:-row['pred_confidence'][j])[:max_det]
        gtcls=np.array(row['gt_classes'],int);used=np.zeros(len(gtcls),bool)
        overlaps=iou_matrix(row['pred_boxes'],row['gt_boxes'])
        for j in order:
            c=int(row['pred_classes'][j]);valid=(gtcls==c)&~used
            k=np.argmax(np.where(valid,overlaps[j],-1)) if len(gtcls) else None
            match=k is not None and valid[k] and overlaps[j,k]>=threshold
            if match: used[k]=True
            by_class[c].append((row['pred_confidence'][j],bool(match)))
    out={}
    for c in sorted(set(ngt)|set(by_class)):
        points=sorted(by_class[c],key=lambda p:-p[0]);truth=np.array([p[1] for p in points],bool)
        tp=np.cumsum(truth);prec=tp/np.arange(1,len(tp)+1)
        recall=tp/ngt[c] if ngt[c] else np.zeros(len(tp))
        # Direct maximum over each recall tail, intentionally unlike TIDE's searchsorted.
        ap=float(np.mean([np.max(prec[recall>=q]) if ngt[c] and np.any(recall>=q) else 0. for q in np.arange(101)/100])*100)
        out[str(c)]=dict(AP=ap,gt=int(ngt[c]),tp=int(truth.sum()),fp=int(len(truth)-truth.sum()),fn=int(ngt[c]-truth.sum()))
    return dict(AP=float(np.mean([r['AP'] for r in out.values()])),per_class=out)
def audit_input(rows, endpoint):
    assert rows and len(set(r['image'] for r in rows))==len(rows)
    counts=collections.Counter();total=0;scores=[]
    for r in rows:
        for prefix in ('gt','pred'):
            assert len(r[prefix+'_boxes'])==len(r[prefix+'_classes'])
            for b,c in zip(r[prefix+'_boxes'],r[prefix+'_classes']):
                assert len(b)==4 and np.isfinite(b).all();xywh(b)
                assert c==int(c) and 0<=int(c)<len(endpoint['class_names'])
        assert len(r['pred_confidence'])==len(r['pred_boxes'])
        assert all(np.isfinite(s) and 0<=s<=1 for s in r['pred_confidence'])
        counts.update(map(int,r['gt_classes']));scores+=r['pred_confidence'];total+=len(r['gt_boxes'])
    contract={}
    if endpoint.get('contract_path'):
        contract=json.loads(Path(endpoint['contract_path']).read_text(encoding='utf-8'))
        assert set(contract['roster'])==set(r['image'] for r in rows)
        assert contract['expected_val_images']==len(rows)
        assert contract['effective_kwargs']['max_det']==300
        assert contract['effective_kwargs']['conf']==.001
    gt_repr=[{k:r[k] for k in ('image','canvas_shape','original_shape','gt_boxes','gt_classes')} for r in rows]
    canonical=sorted(gt_repr,key=lambda r:r['image'])
    return dict(images=len(rows),gt=total,gt_by_class=dict(counts),predictions=len(scores),
        empty_prediction_images=[r['image'] for r in rows if not r['pred_boxes']],
        empty_gt_images=[r['image'] for r in rows if not r['gt_boxes']],
        observed_min_score=min(scores) if scores else None,
        max_predictions_per_image=max(len(r['pred_boxes']) for r in rows),
        images_at_max_det=sum(len(r['pred_boxes'])==300 for r in rows),
        within_class_repeated_score_count=sum(len(v)-len(set(v)) for v in [[s for r in rows for s,c in zip(r['pred_confidence'],r['pred_classes']) if int(c)==k] for k in range(len(endpoint['class_names']))]),
        canvas_shapes=sorted(set(tuple(r['canvas_shape']) for r in rows)),
        effective_kwargs=contract.get('effective_kwargs'),
        source_files={key:file_stat(endpoint[key]) for key in ('path','metric_path','contract_path','receipt_path') if endpoint.get(key)})
def score_diagnostic(run):
    error_by_pred={e.pred['_id']:type(e).short_name for e in run.errors if hasattr(e,'pred')}
    bins=[.001,.01,.05,.25,.5,1.0000001]
    result={}
    sets=[('pooled',list(run.preds.annotations))]+[(str(c),[p for p in run.preds.annotations if p['class']==c]) for c in run.ap_data.objs]
    for group, ps in sets:
        rows=[]
        for lo,hi in zip(bins[:-1],bins[1:]):
            selected=[p for p in ps if lo<=p['score']<hi and 'used' in p]
            tp=sum(p['used'] is True for p in selected)
            rows.append(dict(low=lo,high=min(hi,1.),tp=tp,fp=len(selected)-tp,
                error_counts=dict(collections.Counter(error_by_pred.get(p['_id'],'TP') for p in selected))))
        quant={}
        for label,status in [('TP',True),('FP',False)]:
            s=[p['score'] for p in ps if p.get('used') is status]
            quant[label]=dict(count=len(s),quantiles=np.quantile(s,[0,.1,.25,.5,.75,.9,1]).tolist() if s else None)
        result[group]=dict(bins=rows,score_quantiles=quant)
    return result
def evaluate(rows,names,threshold,errors=True):
    gt,pred=make_data(rows,names)
    return TIDE().evaluate(gt,pred,pos_threshold=threshold,background_threshold=.1,mode=TIDE.BOX,use_for_errors=errors)
def summarize_run(run, rows, names):
    independent=independent_ap(rows,run.pos_thresh,names)
    assert abs(independent['AP']-run.ap)<1e-9,(independent['AP'],run.ap)
    for c,a in run.ap_data.objs.items():
        ref=independent['per_class'][str(c)]
        assert abs(ref['AP']-a.get_ap())<1e-9
        assert ref['gt']==a.num_gt_positives
        assert ref['tp']==sum(p[1] for p in a.data_points.values())
        assert ref['fn']==len(a.false_negatives)
    return dict(AP=run.ap,main_dAP={t.short_name:v for t,v in run.fix_main_errors().items()},
        special_dAP={t.short_name:v for t,v in run.fix_special_errors().items()},
        error_counts={t.short_name:v for t,v in run.count_errors().items()},
        independent_ap=independent,score_diagnostic=score_diagnostic(run))
def synthetic_rows():
    def r(name,gt,pred):
        return dict(image=name,canvas_shape=[100,100],original_shape=[100,100],gt_boxes=[v[0] for v in gt],gt_classes=[v[1] for v in gt],pred_boxes=[v[0] for v in pred],pred_classes=[v[1] for v in pred],pred_confidence=[v[2] for v in pred])
    box=[0,0,10,10];shift=[6,0,16,10]
    return [r('correct',[(box,0)],[(box,0,.9)]),r('class',[(box,0)],[(box,1,.8)]),
        r('loc',[(box,0)],[(shift,0,.7)]),r('both',[(box,0)],[(shift,1,.6)]),
        r('duplicate',[(box,0)],[(box,0,.95),(box,0,.65)]),
        r('background',[],[([50,50,60,60],0,.5)]),r('miss_empty_prediction',[(box,0)],[]),
        r('class1_correct',[(box,1)],[(box,1,.99)]),r('empty_image',[],[])]
def validate_synthetic(out):
    rows=synthetic_rows();report={}
    for t in (.5,.75):
        run=evaluate(rows,['a','b'],t);summary=summarize_run(run,rows,['a','b'])
        observed={run.preds.images[e.pred['image']]['name']:type(e).short_name for e in run.errors if hasattr(e,'pred')}
        expected={'class':'Cls','loc':'Loc','both':'Both','duplicate':'Dupe','background':'Bkg'}
        assert observed==expected,(observed,expected)
        missed={run.gt.images[e.gt['image']]['name'] for e in run.errors if type(e).short_name=='Miss'}
        assert missed=={'both','miss_empty_prediction'},missed
        # Positive monotonic transformation preserves all ties and order.
        transformed=copy.deepcopy(rows)
        for row in transformed: row['pred_confidence']=[x*x for x in row['pred_confidence']]
        assert abs(evaluate(transformed,['a','b'],t,False).ap-run.ap)<1e-9
        report[str(t)]=dict(AP=run.ap,expected_prediction_error_labels=expected,observed_prediction_error_labels=observed,missed=sorted(missed),independent_AP=summary['independent_ap']['AP'],monotonic_invariance=True)
    assert xywh([2,3,12,23])==[2,3,10,20]
    # Score ties and a predicted class absent from GT: preserve official macro-class behavior.
    extra=copy.deepcopy(rows[:1]);extra[0]['pred_boxes']+=[[50,50,60,60]];extra[0]['pred_classes']+=[2];extra[0]['pred_confidence']+=[.9]
    a=evaluate(extra,['a','b','absent'],.5,False).ap;b=independent_ap(extra,.5,['a','b','absent'])['AP']
    assert abs(a-b)<1e-9 and a==50.
    report['absent_gt_class_and_score_tie']=dict(AP=a,independent_AP=b,expected_AP=50.)
    write_json(out/'synthetic_truth.json',rows);write_json(out/'synthetic_validation.json',report)
def default_endpoints():
    root=HERE.parents[1]/'2026-09-07_train_IndependentKD实施'/'legacy_diagnostics_snapshot_20260907_161459'
    return [dict(name=f'{arm}_s{seed}',path=str(root/f'{arm}_s{seed}_attempt1'/'predictions/objects.jsonl.gz'),
        metric_path=str(root/f'{arm}_s{seed}_attempt1'/'evaluation_val.json'),
        contract_path=str(root/f'{arm}_s{seed}_attempt1'/'evaluation_contract.json'),
        receipt_path=str(root/f'{arm}_s{seed}_attempt1'/'reevaluation_receipt.json'),class_names=NAMES,arm=arm,seed=seed)
        for seed in (0,42,123) for arm in ('N','C0')]
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',type=Path);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--tests-only',action='store_true');args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    (out/'PROTOCOL.md').write_bytes((HERE/'PROTOCOL.md').read_bytes())
    (out/'PROTOCOL_ADDENDUM.md').write_bytes((HERE/'PROTOCOL_ADDENDUM.md').read_bytes())
    write_json(out/'execution_environment.json',dict(python=sys.version,executable=sys.executable,numpy=np.__version__,tide_source_receipt=file_stat(HERE/'official_source_receipt.json'),script=file_stat(__file__),protocol=file_stat(HERE/'PROTOCOL.md'),started_at=datetime.datetime.now().astimezone().isoformat(),cpu_only=True,new_hashes_computed=False))
    validate_synthetic(out)
    if args.tests_only: print('synthetic tests accepted');return
    endpoints=json.loads(args.manifest.read_text(encoding='utf-8')) if args.manifest else default_endpoints()
    write_json(out/'input_manifest.json',endpoints)
    results={};first_gt=None;identical_gt=True
    for endpoint in endpoints:
        started=time.time();name=endpoint['name'];names=endpoint.get('class_names',NAMES);endpoint['class_names']=names
        rows=read_rows(endpoint['path']);inventory=audit_input(rows,endpoint)
        gt_rows=sorted([{k:r[k] for k in ('image','canvas_shape','original_shape','gt_boxes','gt_classes')} for r in rows],key=lambda r:r['image'])
        if first_gt is None: first_gt=gt_rows
        else: identical_gt=identical_gt and gt_rows==first_gt
        result=dict(endpoint=endpoint,input_audit=inventory,pinned_metrics=None,thresholds={})
        if endpoint.get('metric_path'):
            metrics=json.loads(Path(endpoint['metric_path']).read_text(encoding='utf-8'))
            result['pinned_metrics']={k:metrics[k] for k in ('AP50','AP75','mAP50_95','per_class') if k in metrics}
        for threshold in (.5,.75):
            run=evaluate(rows,names,threshold)
            result['thresholds'][str(threshold)]=summarize_run(run,rows,names)
        result['seconds']=time.time()-started;write_json(out/(name+'.json'),result);results[name]=result
        print(name,'completed',round(result['seconds'],2),flush=True)
    if not args.manifest: assert identical_gt,'Six-endpoint GT mismatch'
    write_json(out/'summary.json',dict(status='COMPLETED',same_gt_population=identical_gt,gt_comparison='direct exact equality of sorted image/canvas/shape/GT/class records',results=results))
    fields=['endpoint','threshold','AP','Cls','Loc','Both','Dupe','Bkg','Miss','FalsePos','FalseNeg']
    with (out/'tide_endpoints.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for name,result in results.items():
            for threshold,rec in result['thresholds'].items():writer.writerow(dict(endpoint=name,threshold=threshold,AP=rec['AP'],**rec['main_dAP'],**rec['special_dAP']))
    print('COMPLETED',flush=True)
if __name__=='__main__': main()
