"""Frozen, receipt-only single-seed FT3 direction readout. No model loading."""
import argparse
import json
import math
from pathlib import Path
import shutil

SCOPE = 'OBJECT_DFL_FT3_BNFROZEN'
ENDPOINT = 'OBJECT_DFL_FT3_BNFROZEN_LAST_EMA'
ARMS = {'llvip': ('N','L3-DFL','L3-GT')}
POPULATIONS = {'llvip': (2406,7879,1)}
AP = ('mAP50_95','AP50','AP75')
METRICS = AP + ('precision','recall')
CONTRASTS = {'llvip': (('L3-DFL','N'),('L3-GT','N'),('L3-DFL','L3-GT'))}


def require(value, message):
    if not value: raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def number(value, lower=0., upper=1.):
    return type(value) in (int,float) and math.isfinite(value) and lower <= value <= upper


def validate(receipt, dataset, arm):
    images, objects, nc = POPULATIONS[dataset]
    fixed = dict(status='OBJECT_DFL_EVALUATION_COMPLETED', scope=SCOPE, endpoint=ENDPOINT, method_identity=SCOPE,
        dataset=dataset, arm=arm, seed=42, single_seed=True, epochs=3, independent_lr_horizon=3,
        full_dev_images=images, full_dev_gt_objects=objects, observed_images=images,
        gt_objects_captured=objects, metric_units='fraction_0_to_1',
        formal_e200_complete=False, formal_paper_gain_claim=False, accepted_endpoint_claim=False,
        official_test_accessed=False, new_hash_computed=False)
    for k,v in fixed.items():
        require(receipt.get(k)==v and (type(v) is not bool or receipt.get(k) is v), 'Receipt identity differs: '+k)
    require(all(number(receipt.get(k)) for k in METRICS), 'Invalid overall metric fraction')
    rows=receipt.get('per_class',[])
    require(len(rows)==nc and all(type(r.get('class_id')) is int for r in rows), 'Invalid class count/ID')
    require(sorted(r['class_id'] for r in rows)==list(range(nc)), 'Missing/duplicate class ID')
    require(all(isinstance(r.get('name'),str) and r['name'] for r in rows), 'Missing class name')
    require(all(number(r.get(k)) for r in rows for k in AP), 'Invalid class metric fraction')
    for k in AP:
        require(math.isclose(sum(r[k] for r in rows)/nc,receipt[k],rel_tol=0,abs_tol=1e-12), 'Class macro AP differs: '+k)
    require(number(receipt.get('kd_coefficient')), 'Invalid calibrated coefficient')
    require((receipt['kd_coefficient']==0) if arm=='N' else (receipt['kd_coefficient']>0), 'Native/KD dose identity differs')
    require(receipt.get('bn_training_evidence',{}).get('bn_running_buffers_unchanged') is True, 'Missing BN freeze evidence')
    require(number(receipt.get('seconds'),upper=float('inf')), 'Invalid evaluation duration')
    require(isinstance(receipt.get('training_model'),str) and receipt['training_model'], 'Missing shared initialization identity')
    p=receipt.get('evaluation_identity_projection',{})
    require(p.get('dataset')==dataset and p.get('subset_dev_roster_exact_to_full') is True, 'Missing full-dev projection')
    require(p.get('expected_dev_images')==images and p.get('expected_dev_gt_objects')==objects, 'Projection population differs')
    require(isinstance(p.get('actual_evaluation_data_yaml'),str) and p['actual_evaluation_data_yaml'], 'Missing full-dev YAML identity')
    return receipt


def compatible(a,b):
    require(a['training_model']==b['training_model'], 'Control initialization differs')
    pa,pb=a['evaluation_identity_projection'],b['evaluation_identity_projection']
    require(pa['actual_evaluation_data_yaml']==pb['actual_evaluation_data_yaml'], 'Control full-dev YAML differs')
    an={x['class_id']:x['name'] for x in a['per_class']}
    bn={x['class_id']:x['name'] for x in b['per_class']}
    require(an==bn, 'Control class mapping differs')


def difference(a,b):
    compatible(a,b)
    pp={k:(a[k]-b[k])*100. for k in METRICS}
    names={1:'positive',0:'zero',-1:'negative'}
    direction={k:names[1 if pp[k]>0 else -1 if pp[k]<0 else 0] for k in METRICS}
    aa={r['class_id']:r for r in a['per_class']};bb={r['class_id']:r for r in b['per_class']}
    perclass=[dict(class_id=i,name=aa[i]['name'],delta_pp={k:(aa[i][k]-bb[i][k])*100. for k in AP}) for i in sorted(aa)]
    return dict(delta_pp=pp,direction=direction,per_class=perclass)


def summarize(receipts, unavailable=None):
    """Keys are (dataset, arm); callers may supply explicit missing/failed states."""
    unavailable=unavailable or {}; datasets={}
    require(not (set(receipts)&set(unavailable)), 'An arm cannot have both completed and unavailable input')
    expected={(ds,arm) for ds,arms in ARMS.items() for arm in arms}
    require(set(receipts).issubset(expected) and set(unavailable).issubset(expected), 'Unknown matrix input')
    for ds,arms in ARMS.items():
        valid={arm:validate(receipts[(ds,arm)],ds,arm) for arm in arms if (ds,arm) in receipts}
        if ds=='llvip' and 'L3-DFL' in valid and 'L3-GT' in valid:
            require(valid['L3-DFL']['kd_coefficient']==valid['L3-GT']['kd_coefficient'],
                    'LLVIP L3-DFL/L3-GT must share the fixed teacher-mask coefficient')
        # All collected arms must agree on the comparison population, even when N is absent.
        values=list(valid.values())
        for v in values[1:]:compatible(values[0],v)
        rows=[]
        for arm in arms:
            if arm not in valid:
                state=unavailable.get((ds,arm),dict(status='MISSING',reason='No collected terminal receipt'))
                require(state.get('status') in ('MISSING','FAILED','CONFLICT'), 'Invalid unavailable status')
                rows.append(dict(arm=arm,**state,raw_fraction=None,display_percent=None,per_class=None))
                continue
            r=valid[arm]
            rows.append(dict(arm=arm,status='COMPLETED',raw_fraction={k:r[k] for k in METRICS},
                display_percent={k:r[k]*100. for k in METRICS},
                per_class=[dict(class_id=x['class_id'],name=x['name'],raw_fraction={k:x[k] for k in AP},
                    display_percent={k:x[k]*100. for k in AP}) for x in sorted(r['per_class'],key=lambda x:x['class_id'])],
                kd_coefficient=r['kd_coefficient'],evaluation_seconds=r['seconds'],training_model=r['training_model']))
        comparisons=[]
        for a,b in CONTRASTS[ds]:
            item=dict(contrast=a+' - '+b,arm=a,control=b)
            if a not in valid or b not in valid:
                item.update(status='UNAVAILABLE',missing_or_failed_arms=[x for x in (a,b) if x not in valid],
                            delta_pp=None,direction=None,per_class=None)
            else:item.update(status='COMPUTED',**difference(valid[a],valid[b]))
            comparisons.append(item)
        datasets[ds]=dict(images=POPULATIONS[ds][0],gt_objects=POPULATIONS[ds][1],arms=rows,comparisons=comparisons)
    complete=len(receipts)==len(expected)
    return dict(status='COMPLETE_OBJECT_DFL_READOUT' if complete else 'PARTIAL_OBJECT_DFL_READOUT',scope=SCOPE,
        endpoint=ENDPOINT,seed=42,n_seeds=1,standard_deviation=None,raw_metric_units='fraction_0_to_1',
        display_metric_units='percent',difference_units='percentage_points',datasets=datasets,
        decision_policy='Display signed raw control contrasts only; no AP threshold or best-arm selection',
        formal_e200_complete=False,formal_paper_gain_claim=False,causal_modality_claim=False,
        automatically_extend_matrix=False,automatically_admit_e200=False,new_hash_computed=False,
        scope_limit='Receipt-only FT3 screening; no checkpoint/GT-box re-audit, SD, significance or cross-dataset pooling. Root interprets gradient and coverage separately.')


def markdown(result):
    out=['# 固定三轮方向筛选回执汇总','',
         '仅 seed 42、BN running buffers 冻结的 FT3 筛选。表中 AP 为百分数，差值为百分点（pp）。正负号只描述观测方向；不计算 SD、不选择最好臂、不自动扩展矩阵或进入 E200。',
         '', '本汇总检查完成回执与完整 dev 计数；不重新证明 GT 框版本、参数来源或模态因果。梯度/覆盖率由根报告结合解释。']
    for ds,data in result['datasets'].items():
        out+=['','## '+ds,'',f"完整 dev：{data['images']} 图，{data['gt_objects']} GT。",'',
              '| 臂 | 状态 | mAP50–95 (%) | AP50 (%) | AP75 (%) | eval 秒 |','|---|---|---:|---:|---:|---:|']
        for row in data['arms']:
            values=[f"{row['display_percent'][k]:.6f}" for k in AP] if row['status']=='COMPLETED' else ['—']*3
            seconds=f"{row['evaluation_seconds']:.3f}" if row['status']=='COMPLETED' else '—'
            out.append('| '+' | '.join([row['arm'],row['status']]+values+[seconds])+' |')
        out+=['','| 固定对照差 | 状态 | ΔmAP50–95 (pp) | ΔAP50 (pp) | ΔAP75 (pp) |',
               '|---|---|---:|---:|---:|']
        for row in data['comparisons']:
            values=[f"{row['delta_pp'][k]:+.6f}" for k in AP] if row['status']=='COMPUTED' else ['—']*3
            out.append('| '+' | '.join([row['contrast'],row['status']]+values)+' |')
        out+=['','逐类原值与全部固定对照差保存在 summary.json；类别按 class_id 配对。缺失、失败或冲突臂均不填零。']
    return '\n'.join(out)+'\n'


def collect(campaign):
    receipts={};unavailable={};sources=[]
    for ds,arms in ARMS.items():
        for arm in arms:
            directory=campaign/'evaluations'/ds/arm
            p=directory/'direction_evaluation_receipt.json'; f=directory/'direction_evaluation_failure.json'
            present=[x for x in (p,f) if x.is_file()]
            for x in present:
                st=x.stat();sources.append(dict(path=str(x.absolute()),bytes=st.st_size,mtime_ns=st.st_mtime_ns))
            if p.is_file() and f.is_file():
                unavailable[(ds,arm)]=dict(status='CONFLICT',reason='Completed and failure receipts coexist; no result selected')
            elif f.is_file():
                fail=read(f)
                require(fail.get('status')=='OBJECT_DFL_EVALUATION_FAILED' and fail.get('scope')==SCOPE, 'Unexpected failure receipt scope')
                unavailable[(ds,arm)]=dict(status='FAILED',reason=str(fail.get('error','Evaluation failed')))
            elif p.is_file():receipts[(ds,arm)]=read(p)
    result=summarize(receipts,unavailable);result['input_sources']=sources
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();require(args.campaign.is_dir(),'Collected campaign directory not found')
    require(not args.output.exists(),'New output directory required')
    result=collect(args.campaign)
    args.output.mkdir(parents=True,exist_ok=False)
    with (args.output/'summary.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False)
    (args.output/'README.md').write_text(markdown(result),encoding='utf-8')
    shutil.copyfile(__file__,args.output/'analyze_object_dfl_source.py')
    require(Path(__file__).read_bytes()==(args.output/'analyze_object_dfl_source.py').read_bytes(),'Analyzer source copy differs')
    print(result['status'])


if __name__=='__main__':main()
