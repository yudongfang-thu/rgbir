"""Post-result descriptive follow-up requested by root: exact official per-class Cls oracle.
No new metric thresholds, selection, training, or hash computation.
"""
from pathlib import Path
import collections,json,time
import numpy as np
import run_tide_audit as core
from tidecv.errors.main_errors import ClassError
HERE=Path(__file__).resolve().parent
def main():
    out=HERE/'class_oracle_v1';out.mkdir(exist_ok=False)
    (out/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    results={}
    for endpoint in core.default_endpoints():
        name=endpoint['name'];rows=core.read_rows(endpoint['path']);record={}
        for t in (.5,.75):
            run=core.evaluate(rows,endpoint['class_names'],t)
            fixed=run.fix_errors(lambda error:isinstance(error,ClassError))
            per_class={}
            for c,original in run.ap_data.objs.items():
                after=fixed.objs[c]
                assert original.num_gt_positives==after.num_gt_positives
                per_class[str(c)]=dict(name=endpoint['class_names'][c],gt=original.num_gt_positives,
                    AP=original.get_ap(),Cls_oracle_AP=after.get_ap(),Cls_oracle_dAP=after.get_ap()-original.get_ap())
            mean_delta=np.mean([r['Cls_oracle_dAP'] for r in per_class.values()])
            official=run.fix_main_errors(error_types=[ClassError])[ClassError]
            assert abs(max(mean_delta,0)-official)<1e-9
            matrices={}
            for label,lo,hi in [('all',.001,1.0000001),('ge25',.25,1.0000001),('ge50',.5,1.0000001)]:
                matrix=np.zeros((len(endpoint['class_names']),len(endpoint['class_names'])),int)
                for e in run.error_dict[ClassError]:
                    if lo<=e.pred['score']<hi:matrix[e.pred['class'],e.gt['class']]+=1
                matrices[label]=matrix.tolist()
            record[str(t)]=dict(per_class=per_class,macro_Cls_dAP=official,confusion_counts_predicted_row_true_column=matrices)
        results[name]=record;core.write_json(out/(name+'.json'),record);print(name,'complete',flush=True)
    core.write_json(out/'summary.json',results)
    lines=['# Official TIDE class-oracle follow-up','',
        'Descriptive follow-up requested after the frozen six-endpoint result exposed a large macro Cls contribution. No thresholds, model selection or training were changed. Each row is the same official independent Cls oracle decomposed into per-class AP differences; these are not achievable KD gains.','',
        '|IoU|Arm|Class|GT|AP mean±SD|Cls oracle dAP mean±SD|C0−N dAP seeds 0/42/123|',
        '|---|---|---|---:|---:|---:|---|']
    for t in ('0.5','0.75'):
        for arm in ('N','C0'):
            for c,name in enumerate(core.NAMES):
                rr=[results[f'{arm}_s{s}'][t]['per_class'][str(c)] for s in (0,42,123)]
                ap=[r['AP'] for r in rr];dp=[r['Cls_oracle_dAP'] for r in rr]
                delta=[results[f'C0_s{s}'][t]['per_class'][str(c)]['Cls_oracle_dAP']-results[f'N_s{s}'][t]['per_class'][str(c)]['Cls_oracle_dAP'] for s in (0,42,123)]
                lines.append(f'|{t}|{arm}|{name}|{rr[0]["gt"]}|{np.mean(ap):.4f} ± {np.std(ap,ddof=1):.4f}|{np.mean(dp):.4f} ± {np.std(dp,ddof=1):.4f}|'+', '.join(f'{x:+.4f}' for x in delta)+'|')
    lines+=['','Matrices in JSON: predicted class is row, associated true class is column. The matrices describe TIDE Cls error predictions, including extra cross-class post-NMS detections; they are not a single-label classification confusion matrix of GT objects. Cls oracle may remove an erroneous detection if its GT is already detected and may repair one best erroneous prediction otherwise.','',
        'A macro AP bottleneck establishes where errors affect this evaluator. It does not establish that the IR teacher supplies correct transferable knowledge, or that the RGB student can learn the oracle.']
    (out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
if __name__=='__main__':main()
