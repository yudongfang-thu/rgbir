"""Full original-cache recomputation; no producer imports, GPU, or content hashes."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import json, math, sys
sys.dont_write_bytecode=True
from independent_oracle import reference

HERE=Path(__file__).resolve().parent
ENTRY=HERE.parent
INPUT=ENTRY.parent/'2026-09-08_probe_DFL真实信息读出'/'evidence_1602_final'/'probe'
OUTPUT=ENTRY/'transport'/'output_attempt1'

def jread(path):return json.loads(path.read_text(encoding='utf-8'))
def lines(path):return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
def dump(path,data):
    with path.open('x',encoding='utf-8') as f:json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)
def close(a,b,tol=1e-12):return abs(a-b)<=tol*max(1,abs(a),abs(b))

def audit():
    evidence=[];checks={};problems=[];maxima=Counter()
    def check(name,ok,detail):
        checks[name]={'pass':bool(ok),'detail':detail}
        if not ok:problems.append(name)
    manifest=jread(HERE/'input_snapshot_manifest.json')
    for rec in manifest['inputs']:
        src=Path(rec['source_path']);snapshot=Path(rec['snapshot_path'])
        check('input_bytes_'+src.name,src.read_bytes()==snapshot.read_bytes(),'Full-byte comparison, not size-only or hash.')
        evidence.append({'path':str(src.resolve()),'bytes':src.stat().st_size,'mtime_ns':src.stat().st_mtime_ns,'snapshot':str(snapshot.resolve())})
    objects=lines(INPUT/'objects.jsonl');distributions=lines(INPUT/'anchor_distributions.jsonl')
    receipt=jread(INPUT/'completion_receipt.json');contract=jread(INPUT/'dfl_contract.json')
    byid={d['distribution_id']:d for d in distributions}
    check('fixed_population',len(objects)==80 and len(distributions)==len(byid)==317 and len({o['stable_rgb_gt_id'] for o in objects})==80,'80 distinct RGB GT objects and 317 distinct distribution IDs.')
    check('contract_identity',receipt['current_forward_id']==contract['current_forward_id'] and all(x['current_forward_id']==receipt['current_forward_id'] for x in objects+distributions) and (contract['frames'],contract['objects_n'],contract['unique_distributions'])==(32,80,317),'Four declared input files refer to the same completed original forward.')
    summary=jread(OUTPUT/'summary.json');rows=lines(OUTPUT/'objects.jsonl')
    check('result_executed',summary['status']=='OBJECT_COORDINATE_TRANSPORT_CPU_COMPLETED' and len(rows)==160,'Completed summary and 160 materialized per-object/path results exist.')
    row_index={(r['stable_rgb_gt_id'],r['variant']):r for r in rows}
    check('result_unique',len(row_index)==160,'Exactly one result for every one of 80 objects and two frozen interfaces.')
    variants={'same_R_primary':'same_historical_R_index','native_T_pressure':'own_native_iou50'}
    counts={k:Counter() for k in variants};identity_counts=Counter();available=Counter();audit_rows=[]
    compare_count=0
    for obj in objects:
        dest=obj['roles']['S']['historical_R_candidate']
        for variant,role_name in variants.items():
            actual=row_index[(obj['stable_rgb_gt_id'],variant)];teach=obj['roles']['T'][role_name]
            errors=[]
            def require(ok,label):
                if not ok:errors.append(label)
            for k in ('stable_rgb_gt_id','stable_ir_gt_id','image_index','frame_id','current_forward_id'):
                require(actual[k]==obj[k],'object_identity_'+k)
            require(actual['destination_role']==dest and actual['teacher_role']==teach and actual['teacher_role_name']==role_name,'fixed_role_identity')
            if dest is None or teach is None:
                expected_status='MISSING_ROLE'
                require(actual['status']==expected_status and actual['result'] is None,'missing_explicit_no_target')
                require(actual['missing_student']==(dest is None) and actual['missing_teacher']==(teach is None),'missing_reason')
            else:
                available[variant]+=1
                s=byid[dest['distribution_id']];t=byid[teach['distribution_id']]
                require(s['model']=='S' and t['model']=='T' and s['image_index']==t['image_index']==obj['image_index'],'distribution_role_image_identity')
                require(s['anchor_index']==dest['anchor_index'] and t['anchor_index']==teach['anchor_index'],'anchor_identity')
                require(dest['own_gt_id']==obj['stable_rgb_gt_id'] and teach['own_gt_id']==obj['stable_ir_gt_id'],'GT_identity')
                for original,field in ((s,'destination_geometry'),(t,'source_geometry')):
                    require(all(actual[field][k]==original[k] for k in actual[field]),field)
                probs=[]
                for row in t['raw_logits']:
                    weights=[math.exp(v-max(row)) for v in row];denom=math.fsum(weights)
                    probs.append([w/denom for w in weights])
                oracle=reference(probs,obj['ir_gt_xyxy'],obj['rgb_gt_xyxy'],t['center_xy'],s['center_xy'],t['stride'],s['stride'])
                expected_status='SUPPORTED' if oracle['accepted'] else 'OUTSIDE_STUDENT_SUPPORT'
                result=actual['result'];require(actual['status']==result['status']==expected_status,'support_status')
                require(result['source_logits']==t['raw_logits'],'source_logits')
                require(actual['cached_source_probabilities_fp32']==t['probabilities_fp32'] and actual['cached_source_probabilities_native']==t['probabilities_native'],'cached_FP32_native_retained')
                fp32diff=max(abs(p-q) for pr,qr in zip(probs,t['probabilities_fp32']) for p,q in zip(pr,qr))
                maxima['FP64_source_vs_cached_FP32']=max(maxima['FP64_source_vs_cached_FP32'],fp32diff)
                require(close(actual['recomputed_FP64_vs_cached_FP32_max_abs'],fp32diff),'FP64_FP32_disclosure')
                require(result['temperature']==1 and all(result[k] is False for k in ('clamped','truncated','renormalized')),'fixed_temperature_no_repair')
                isidentity=all(a==1 and b==0 for a,b in oracle['affine'])
                identity_counts[variant]+=isidentity
                require(result['algebraic_identity']==isidentity,'algebraic_identity')
                for side,side_actual in enumerate(result['sides']):
                    p=probs[side];d=oracle['distances'][side]
                    perr=max(abs(a-b) for a,b in zip(result['source_probabilities'][side],p))
                    derr=max(abs(a-b) for a,b in zip(side_actual['mapped_distance_bins'],d))
                    maxima['source_probability_error']=max(maxima['source_probability_error'],perr)
                    maxima['mapped_distance_error']=max(maxima['mapped_distance_error'],derr)
                    require(perr<1e-13 and derr<1e-12,'full_source_and_unclamped_distance')
                    out=oracle['outside'][side];bins=[x['source_bin'] for x in out]
                    require(side_actual['rejected_source_bins']==bins,'all_positive_mass_support')
                    require(close(side_actual['rejected_positive_mass'],math.fsum(x['mass'] for x in out)),'rejected_mass')
                    mass=math.fsum(p);moment=math.fsum(v*weight for v,weight in zip(d,p))
                    require(close(side_actual['source_mass'],mass) and close(side_actual['mapped_expectation_bin'],moment),'mass_and_first_moment')
                    require(abs(mass-1)<1e-14 and min(p)>0,'source_full_positive_normalized')
                    a,b=oracle['affine'][side]
                    require(close(side_actual['affine_a']['numerator']/side_actual['affine_a']['denominator'],a) and close(side_actual['affine_b']['numerator']/side_actual['affine_b']['denominator'],b),'affine_coefficients')
                    axis=side%2;sign=-1 if side<2 else 1
                    gt_t=obj['ir_gt_xyxy'];gt_s=obj['rgb_gt_xyxy'];scale=(gt_s[axis+2]-gt_s[axis])/(gt_t[axis+2]-gt_t[axis])
                    edge=math.fsum(prob*(gt_s[axis]+(t['center_xy'][axis]+sign*j*t['stride']-gt_t[axis])*scale) for j,prob in enumerate(p))
                    edgeerr=abs(side_actual['mapped_edge_coordinate']-edge)
                    maxima['mapped_edge_error']=max(maxima['mapped_edge_error'],edgeerr)
                    require(edgeerr<1e-10,'edge_coordinate_expectation')
                    require(close(side_actual['edge_from_mapped_expectation'],s['center_xy'][axis]+sign*s['stride']*moment),'student_decode_coordinate')
                    if not bins:
                        q=side_actual['target_probabilities']
                        expected=[math.fsum(p[j]*max(0,1-abs(d[j]-k)) for j in range(16)) for k in range(16)]
                        terr=max(abs(x-y) for x,y in zip(q,expected))
                        maxima['target_probability_error']=max(maxima['target_probability_error'],terr)
                        require(len(q)==16 and min(q)>=0 and terr<1e-12,'full_target_hat_basis')
                        require(abs(math.fsum(q)-mass)<1e-12 and abs(math.fsum(k*v for k,v in enumerate(q))-moment)<1e-12,'target_mass_expectation_closure')
                    else:require(side_actual['target_probabilities'] is None,'unsupported_side_has_no_target')
                    compare_count+=1
                if oracle['accepted']:
                    require(result['target_probabilities']==[x['target_probabilities'] for x in result['sides']],'whole_object_target')
                else:require(result['target_probabilities'] is None,'rejected_object_no_partial_target')
            counts[variant][expected_status]+=1
            audit_rows.append({'stable_rgb_gt_id':obj['stable_rgb_gt_id'],'variant':variant,'expected_status':expected_status,'passed':not errors,'errors':errors})
    check('full_recomputation',all(x['passed'] for x in audit_rows),f'80 objects x 2 paths; {compare_count} available side vectors, every bin, full independent hat-basis recomputation.')
    for variant in variants:
        v=summary['variants'][variant]
        check('summary_'+variant,v['status_counts']==dict(counts[variant]) and v['GT_objects']==80 and v['available_pairs']==available[variant] and v['supported_targets']==counts[variant]['SUPPORTED'] and v['algebraic_identity_objects']==identity_counts[variant],dict(counts[variant]))
    for name in ('probability_transport.py','run_cached_transport.py','test_transport_cpu.py','PLAN.md'):
        path=ENTRY/'transport'/name;snapshot=HERE/'source_snapshot_v2'/name;executed=OUTPUT/'sources'/name
        check('source_bytes_'+name,path.read_bytes()==snapshot.read_bytes()==executed.read_bytes(),'Current, pre-execution reviewed snapshot, and executed source copy match byte for byte.')
        evidence.append({'path':str(path.resolve()),'bytes':path.stat().st_size,'snapshot':str(snapshot.resolve()),'executed_copy':str(executed.resolve())})
    check('scope_guards',all(summary[k] is False for k in ('physical_registration_claim','old_L1_geometry_unblocked','training_admission','GPU_used','new_forward','new_training','new_hash_computed','clamping','truncation','renormalization','anchor_reselection')),'Completed CPU target interface only; no training or old physical geometry admission.')
    dump(HERE/'recomputation_rows.json',audit_rows)
    recompute={'date':datetime.now(timezone.utc).isoformat(),'passed':not problems,'checks':checks,'problems':problems,'max_errors':dict(maxima),'available_side_vectors':compare_count,'variant_counts':{k:dict(v) for k,v in counts.items()},'identity_counts':dict(identity_counts),'input_source_evidence':evidence}
    dump(HERE/'recomputation_summary.json',recompute)
    result_snapshot=HERE/'result_snapshot';result_snapshot.mkdir(exist_ok=False)
    for name in ('objects.jsonl','summary.json','README.md'):
        path=OUTPUT/name;raw=path.read_bytes();(result_snapshot/name).write_bytes(raw)
        assert path.read_bytes()==(result_snapshot/name).read_bytes()
    print(json.dumps({k:recompute[k] for k in ('passed','problems','max_errors','available_side_vectors','variant_counts','identity_counts')},indent=2))
    return recompute

if __name__=='__main__':
    audit()
