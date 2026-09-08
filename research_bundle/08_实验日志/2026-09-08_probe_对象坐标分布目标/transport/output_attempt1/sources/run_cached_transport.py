"""Fixed 80-object/317-distribution CPU runner; run only after source review."""
import argparse,collections,json,math,shutil,time
from pathlib import Path
from probability_transport import transport_logits,stable_softmax,SIDES

SCOPE='OBJECT_COORDINATE_FULL_PROBABILITY_TRANSPORT_CPU'
VARIANTS={'same_R_primary':'same_historical_R_index','native_T_pressure':'own_native_iou50'}

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def lines(path):return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]
def require(ok,message):
    if not ok:raise ValueError(message)
def stat(path):
    p=Path(path);s=p.stat();return dict(path=str(p.absolute()),bytes=s.st_size,mtime_ns=s.st_mtime_ns)
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)

def validate_inputs(objects,distributions,receipt,contract):
    require(receipt['status']=='RAW_DFL_SINGLE_BATCH_COMPLETED' and receipt['scope']=='RAW_DFL_SINGLE_BATCH','Wrong input completion')
    require(receipt['dataset']=='llvip' and receipt['seed']==42,'Wrong dataset/seed')
    require((receipt['frames'],receipt['objects_n'],receipt['unique_distributions'],receipt['batches'])==(32,80,317,1),'Wrong fixed population')
    require(contract['scope']=='RAW_DFL_SINGLE_BATCH' and contract['current_forward_id']==receipt['current_forward_id'],'Contract forward/scope differs')
    require((contract['frames'],contract['objects_n'],contract['unique_distributions'])==(32,80,317),'Contract population differs')
    require(contract['full_probability_vectors'] is True and contract['GT_distances_clamped'] is False,'Contract probability/distance scope differs')
    for key in ('optimizer_updates','backward','training'):require(receipt[key]==0,'Input is not zero-update')
    for key in ('identity_exact','first_batch_stream_exact','student_full_state_unchanged','auxiliary_full_state_unchanged'):require(receipt[key] is True,'Input identity failed: '+key)
    require(len(objects)==80 and len(distributions)==317,'Wrong cache size')
    require(len({x['stable_rgb_gt_id'] for x in objects})==80 and len({x['stable_ir_gt_id'] for x in objects})==80,'Duplicate GT identity')
    index={x['distribution_id']:x for x in distributions};require(len(index)==317,'Duplicate distribution identity')
    forward=receipt['current_forward_id']
    for d in distributions:
        require(d['current_forward_id']==forward and d['model'] in ('S','R','T'),'Distribution forward/model identity differs')
        require(d['bins']==16 and d['side_order']==list(SIDES) and len(d['raw_logits'])==4,'Frozen logits layout differs')
        for logits in d['raw_logits']:stable_softmax(logits)
    for obj in objects:
        require(obj['current_forward_id']==forward,'Object forward identity differs')
        for name,roles in obj['roles'].items():
            for role in roles.values():
                if role is None:continue
                d=index[role['distribution_id']]
                require((d['model'],d['image_index'],d['anchor_index'])==(name,obj['image_index'],role['anchor_index']),'Role/anchor identity differs')
                own='ir' if name=='T' else 'rgb'
                require(role['own_gt_id']==obj['stable_'+own+'_gt_id'] and role['own_gt_xyxy']==obj[own+'_gt_xyxy'],'Own GT role differs')
    return index

def analyze(objects,index):
    rows=[]
    for obj in objects:
        destination=obj['roles']['S']['historical_R_candidate']
        for variant,teacher_role in VARIANTS.items():
            teacher=obj['roles']['T'][teacher_role]
            row=dict(variant=variant,stable_rgb_gt_id=obj['stable_rgb_gt_id'],stable_ir_gt_id=obj['stable_ir_gt_id'],
                image_index=obj['image_index'],frame_id=obj['frame_id'],current_forward_id=obj['current_forward_id'],
                original_bucket=obj['bucket'],destination_role=destination,teacher_role_name=teacher_role,teacher_role=teacher)
            if destination is None or teacher is None:
                row.update(status='MISSING_ROLE',missing_student=destination is None,missing_teacher=teacher is None,result=None)
            else:
                s=index[destination['distribution_id']];t=index[teacher['distribution_id']]
                result=transport_logits(t['raw_logits'],obj['ir_gt_xyxy'],obj['rgb_gt_xyxy'],t['center_xy'],s['center_xy'],t['stride'],s['stride'])
                row.update(status=result['status'],result=result,
                    source_geometry={k:t[k] for k in ('distribution_id','anchor_index','center_xy','stride','level','raw_dtype')},
                    destination_geometry={k:s[k] for k in ('distribution_id','anchor_index','center_xy','stride','level')},
                    cached_source_probabilities_fp32=t['probabilities_fp32'],cached_source_probabilities_native=t['probabilities_native'])
                if 'source_probabilities' in result:
                    row['recomputed_FP64_vs_cached_FP32_max_abs']=max(abs(a-b) for p,q in zip(result['source_probabilities'],t['probabilities_fp32']) for a,b in zip(p,q))
            rows.append(row)
    return rows

def summarize(rows):
    output={}
    for variant in VARIANTS:
        group=[x for x in rows if x['variant']==variant];counts=dict(collections.Counter(r['status'] for r in group))
        available=sum(r['status']!='MISSING_ROLE' for r in group);supported=counts.get('SUPPORTED',0)
        sides=[s for r in group if r['result'] is not None for s in r['result'].get('sides',[])]
        maxima={k:max([abs(s[k]) for s in sides if k in s],default=None) for k in ('mass_closure_error','expectation_closure_error_bin','edge_closure_error')}
        output[variant]=dict(GT_objects=len(group),available_pairs=available,status_counts=counts,
            algebraic_identity_objects=sum(bool(r['result'] and r['result'].get('algebraic_identity')) for r in group),
            support_status='NO_SUPPORTED_TARGET' if supported==0 else ('ALL_AVAILABLE_SUPPORTED' if supported==available else 'PARTIAL_SUPPORT'),
            supported_targets=supported,maxima=maxima,
            recomputed_FP64_vs_cached_FP32_max_abs=max([r['recomputed_FP64_vs_cached_FP32_max_abs'] for r in group if 'recomputed_FP64_vs_cached_FP32_max_abs' in r],default=None))
    return output

def run(args):
    started=time.perf_counter();require(not args.output.exists(),'Output already exists')
    paths=[args.input/n for n in ('objects.jsonl','anchor_distributions.jsonl','completion_receipt.json','dfl_contract.json')]
    before=[stat(p) for p in paths]
    objects,distributions,receipt=lines(paths[0]),lines(paths[1]),read(paths[2])
    index=validate_inputs(objects,distributions,receipt,read(paths[3]));rows=analyze(objects,index)
    require(before==[stat(p) for p in paths],'Input stat changed')
    summary=dict(status='OBJECT_COORDINATE_TRANSPORT_CPU_COMPLETED',scope=SCOPE,current_forward_id=receipt['current_forward_id'],
        objects=80,cached_distributions=317,interface_rows=len(rows),variants=summarize(rows),inputs=before,
        source_probability='Float64 stable T=1 softmax recomputed from saved raw logits; cached FP32/native retained separately',
        geometry='Exact rational affine on saved binary scalars; annotation correspondence only',
        support_boundary='Any strictly positive mass outside [0,15] rejects whole object; no tolerance',
        closure_tolerance=dict(mass_abs=1e-12,expectation_abs=1e-12,expectation_rel=1e-12,edge_abs=1e-10,edge_rel=1e-12),
        temperature=1,clamping=False,truncation=False,renormalization=False,anchor_reselection=False,
        physical_registration_claim=False,old_L1_geometry_unblocked=False,training_admission=False,
        GPU_used=False,new_forward=False,new_training=False,new_hash_computed=False,seconds=time.perf_counter()-started)
    args.output.mkdir(parents=True,exist_ok=False)
    with (args.output/'objects.jsonl').open('x',encoding='utf-8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    write(args.output/'summary.json',summary)
    source=args.output/'sources';source.mkdir()
    for name in ('probability_transport.py','run_cached_transport.py','test_transport_cpu.py','PLAN.md'):
        shutil.copyfile(Path(__file__).with_name(name),source/name)
    text=['# Fixed cached object-coordinate probability transport','',
        'CPU execution completed; no training admission or physical registration claim. Old L1 geometry remains blocked.','',
        '|Prelisted input|All GT|Available|Supported|Identity|Support state|','|---|---:|---:|---:|---:|---|']
    for name,v in summary['variants'].items():text.append('|%s|%s|%s|%s|%s|%s|'%(name,v['GT_objects'],v['available_pairs'],v['supported_targets'],v['algebraic_identity_objects'],v['support_status']))
    text+=['','Missing and rejected objects remain explicit in objects.jsonl. No CE/AP-based choice between these inputs. Source probabilities are newly computed float64 T=1 softmax from saved logits; cached FP32/native vectors are retained, not silently equated. Integer/support geometry is exact rational; mass/expectation closure is floating-point with the fixed recorded tolerances.','',
        'Only all-four-side supported rows expose an object target. Partial side diagnostics are not a usable partial object target. Runtime seconds: '+str(summary['seconds'])+'.']
    with (args.output/'README.md').open('x',encoding='utf-8') as f:f.write('\n'.join(text)+'\n')
    print(json.dumps(dict(status=summary['status'],seconds=summary['seconds'],variants=summary['variants'])))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
