"""Independent COCOeval AP crosscheck of saved complete prediction caches."""
import contextlib,gzip,json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'deps'))
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def main():
    out=HERE/'root_coco_review_v1';out.mkdir(exist_ok=False)
    results=HERE/'drone_results_v1'
    manifest=json.loads((results/'input_manifest.json').read_text())
    records=[]
    with (out/'coco_stdout.log').open('w') as log,contextlib.redirect_stdout(log):
        for e in manifest:
            with gzip.open(e['path'],'rt',encoding='utf-8') as f:rows=[json.loads(x) for x in f if x.strip()]
            gt=COCO();images=[];annotations=[];detections=[]
            for i,row in enumerate(rows,1):
                h,w=row['canvas_shape'];images.append(dict(id=i,width=w,height=h))
                for box,c in zip(row['gt_boxes'],row['gt_classes']):
                    x1,y1,x2,y2=box;b=[x1,y1,x2-x1,y2-y1]
                    annotations.append(dict(id=len(annotations)+1,image_id=i,category_id=int(c),bbox=b,area=b[2]*b[3],iscrowd=0))
                for box,c,s in zip(row['pred_boxes'],row['pred_classes'],row['pred_confidence']):
                    x1,y1,x2,y2=box
                    detections.append(dict(image_id=i,category_id=int(c),bbox=[x1,y1,x2-x1,y2-y1],score=float(s)))
            gt.dataset=dict(images=images,annotations=annotations,categories=[dict(id=i,name=n) for i,n in enumerate(e['class_names'])])
            gt.createIndex();pred=gt.loadRes(detections)
            ev=COCOeval(gt,pred,'bbox');ev.params.iouThrs=np.array([.5,.75]);ev.params.maxDets=[1,10,300]
            ev.evaluate();ev.accumulate()
            expected=json.loads((results/(e['name']+'.json')).read_text())
            values={}
            for index,threshold in enumerate((.5,.75)):
                p=ev.eval['precision'][index,:,:,0,2];actual=float(p[p>-1].mean()*100)
                tide=expected['thresholds'][str(threshold)]['AP'];difference=actual-tide
                if abs(difference)>1e-8:raise ValueError((e['name'],threshold,actual,tide))
                values[str(threshold)]=dict(coco_AP=actual,tide_AP=tide,difference_pp=difference)
            records.append(dict(endpoint=e['name'],images=len(rows),gt=len(annotations),predictions=len(detections),thresholds=values))
    summary=dict(status='PASS',independent_evaluator='pycocotools 2.0.7 COCOeval',checks=records,
        matching_scope='AP50/AP75 full caches; maxDets300 all area; no ignore/crowd metadata',
        oracle_scope='AP crosscheck does not independently validate causal attainability of TIDE oracle dAP',
        source=str(Path(__file__).resolve()),official_test_accessed=False,new_gpu_tasks=0)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',endpoints=len(records),max_abs_AP_difference_pp=max(abs(v['difference_pp']) for r in records for v in r['thresholds'].values()))))

if __name__=='__main__':main()
