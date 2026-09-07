"""Read only both completed 64-batch attempts; reject partials and canaries."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASETS = {'llvip': 'llvip', 'drone': 'dronevehicle'}
L_CHAIN = ('rgb_gt_count', 'common_count', 'pair_iou_count', 'geometry_count', 'inside_count',
           'support_count', 'unique_owner_count', 'reference_candidate_count', 'base_count',
           'reference_reliable_count', 'both_reliable_count', 'reference_localization_gap_count',
           'teacher_rgb_quality_count', 'teacher_own_quality_count', 'eligible_count', 'selected_count')
UNKNOWN = '<UNKNOWN_SOURCE_GROUP>'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def fraction(num, den):
    return num / den if den else None


def require_completed(summary, dataset):
    checks = [summary['status']=='COMPLETED', summary['dataset']==dataset, summary['batches']==64,
              summary['canary_only'] is False, summary['seed']==20260907,
              summary['diagnostic_status']=='UNVERIFIED_GEOMETRY_DIAGNOSTIC',
              summary['trace_exact_all_recorded_fields'] is True,
              summary['classification']['C0_C1_selection_identical'] is True,
              summary['localization']['geometry_verified'] is False,
              summary['localization']['formal_L1_admitted'] is False]
    checks += [summary[k] is False for k in ('backward_executed','training_executed','calibration_executed','validation_or_test_accessed','new_hash_computed')]
    if not all(checks):
        raise ValueError('Only completed fixed 64-batch, non-canary, unverified-geometry diagnostic accepted')


def concentration(histogram):
    ranked = sorted(histogram.items(), key=lambda x: (-x[1], str(x[0])))
    total = sum(histogram.values())
    return {'objects': total, 'unique_units': len(histogram), 'top1_objects': sum(v for _,v in ranked[:1]),
            'top5_objects': sum(v for _,v in ranked[:5]), 'top1_fraction': fraction(sum(v for _,v in ranked[:1]),total),
            'top5_fraction': fraction(sum(v for _,v in ranked[:5]),total), 'top5_units': [str(k) for k,_ in ranked[:5]]}


def gate_rows(dataset, counts, images, groups, batches):
    rows=[]; previous=None
    for gate in L_CHAIN:
        count=counts[gate]
        if previous is not None and count>previous:
            raise ValueError('Localization cumulative gate increases: '+gate)
        loss=None if previous is None else previous-count
        rows.append({'dataset':dataset,'gate':gate,'objects':count,'previous_objects':previous,
                     'lost_objects':loss,'loss_fraction_previous':fraction(loss,previous) if previous is not None else None,
                     'retention_fraction_rgb_gt':fraction(count,counts['rgb_gt_count']),
                     'unique_images':len(images[gate]),'known_source_groups':len(groups[gate]),
                     'batches_with_any':batches[gate]})
        previous=count
    return rows


def load_dataset(root, short, dataset):
    folder=root/(short+'_full_attempt1')
    required=('summary.json','selection_batches.jsonl','natural_batches.jsonl','original_natural_batches.jsonl','model_identity.json')
    if not folder.is_dir() or not all((folder/name).is_file() for name in required):
        raise ValueError('Full completed input files missing: '+str(folder))
    summary=read_json(folder/'summary.json');require_completed(summary,dataset)
    selections=read_jsonl(folder/'selection_batches.jsonl')
    natural=read_jsonl(folder/'natural_batches.jsonl')
    original=read_jsonl(folder/'original_natural_batches.jsonl')
    if len(selections)!=64 or len(natural)!=64 or len(original)!=64 or natural!=original:
        raise ValueError('Incomplete or changed original 64-batch trace: '+dataset)
    if [r['batch'] for r in selections]!=list(range(64)) or [r['batch'] for r in natural]!=list(range(64)):
        raise ValueError('Batch order or completeness differs')
    identity=read_json(folder/'model_identity.json')
    names={str(k):v for k,v in identity['reference']['names'].items()}
    if names!={str(k):v for k,v in identity['teacher']['names'].items()}:
        raise ValueError('Teacher/reference class identity differs')
    c_total=Counter();l_total=Counter();l_image=defaultdict(set);l_group=defaultdict(set);l_batch=Counter()
    c_class={k:Counter() for k in names};c_class_batches=defaultdict(set)
    c_levels=defaultdict(Counter);l_strides=defaultdict(Counter)
    hist=defaultdict(Counter);c_selected_images=set();c_selected_groups=set();c_batches=0
    l_class={k:Counter() for k in names};observed_images=set()
    for index,(entry,trace) in enumerate(zip(selections,natural)):
        if entry['actual_B']!=32 or entry['trace_exact'] is not True or entry['geometry_status']!='UNVERIFIED_GEOMETRY_DIAGNOSTIC':
            raise ValueError('Batch contract differs')
        if len(entry['images'])!=32 or len(trace['images'])!=32:
            raise ValueError('Short image batch')
        for a,b in zip(entry['images'],trace['images']):
            if a['image']!=b['image'] or a['rgb_source']!=b['rgb_source']:
                raise ValueError('Selection identity differs from natural source')
            observed_images.add(a['rgb_source'])
        c=entry['classification'];l=entry['localization']
        if c['c0_c1_selected_exact'] is not True or l['per_image_replay_counts_and_ids_exact'] is not True:
            raise ValueError('Missing actual selector equivalence evidence')
        if l['geometry_verified'] or l['geometry_mask_supplied']:
            raise ValueError('Unexpected verified geometry in diagnostic')
        records=c['base_records'];c_total.update({k:int(v) for k,v in c['counts'].items() if k.endswith('_count')})
        actual_c={'base_count':len(records),'eligible_count':sum(r['eligible'] for r in records),'selected_count':sum(r['selected'] for r in records)}
        if any(c['counts'][k]!=v for k,v in actual_c.items()):raise ValueError('C record counts differ')
        c_batches+=int(actual_c['selected_count']>0)
        local_classes=defaultdict(Counter)
        for r in records:
            label=str(r['class'])
            if label not in names:raise ValueError('Unknown class')
            position=int(r['batch_index']);image=entry['images'][position]
            if r['image']!=image['image'] or r['governed_group']!=image['governed_group']:
                raise ValueError('C source record mismatch')
            levels=r['valid_levels']
            if len(levels)!=2 or not any(levels):raise ValueError('Invalid C base effective levels')
            for phase,take in [('base',True),('eligible',r['eligible']),('selected',r['selected'])]:
                if not take:continue
                c_class[label][phase]+=1;local_classes[label][phase]+=1
                c_class_batches[(label,phase)].add(index)
                c_levels[(label,phase)]['objects']+=1
                for level,valid in zip(('P3','P4'),levels):c_levels[(label,phase)][level]+=int(valid)
                c_levels[(label,phase)]['both']+=int(all(levels))
                hist[('C0_C1',phase,'image')][image['rgb_source']]+=1
                hist[('C0_C1',phase,'group')][image['governed_group'] or UNKNOWN]+=1
            if r['selected']:
                if not r['eligible']:raise ValueError('Selected C object not eligible')
                c_selected_images.add(image['rgb_source'])
                if image['governed_group'] is not None:c_selected_groups.add(image['governed_group'])
        for label,counts in c['class_counts'].items():
            if any(int(counts[p])!=local_classes[label][p] for p in ('base','eligible','selected')):raise ValueError('C per-class count differs')
        per_image=entry['localization_per_image'];per_counts=Counter()
        if len(per_image)!=32:raise ValueError('Missing per-image L replay')
        for position,row in enumerate(per_image):
            im=entry['images'][position]
            if row['position']!=position or row['image']!=im['image'] or row['rgb_source']!=im['rgb_source'] or row['governed_group']!=im['governed_group']:
                raise ValueError('L replay image identity mismatch')
            per_counts.update(row['gate_counts'])
            for gate,count in row['gate_counts'].items():
                if count:
                    l_image[gate].add(im['rgb_source'])
                    if im['governed_group'] is not None:l_group[gate].add(im['governed_group'])
        for gate,count in per_counts.items():
            if l[gate]!=count:raise ValueError('Whole/per-image gate count differs')
            l_total[gate]+=count;l_batch[gate]+=int(count>0)
        lrecords=l['base_records']
        if len(lrecords)!=l['base_count'] or sum(r['selected'] for r in lrecords)!=l['selected_count'] or l['normalizer']!=max(1,len(lrecords)):
            raise ValueError('L base/selected denominator differs')
        for r in lrecords:
            im=entry['images'][int(r['batch_index'])];label=str(r['class'])
            if r['image_id']!=im['image'] or label not in names:raise ValueError('L object image/class mismatch')
            if r['selected']!=r['quality_gate']:raise ValueError('Teacher-mode L gate/selection differ')
            if not all(0<=x<=14.99 for side in ('rgb_distances','ir_distances') for x in r[side]):raise ValueError('Unsupported L base distances')
            for phase,take in [('base',True),('selected',r['selected'])]:
                if take:
                    l_class[label][phase]+=1;l_strides[phase][str(int(r['stride']))]+=1
                    hist[('L_UNVERIFIED',phase,'image')][im['rgb_source']]+=1
                    hist[('L_UNVERIFIED',phase,'group')][im['governed_group'] or UNKNOWN]+=1
    if len(observed_images)!=2048:raise ValueError('Natural without-replacement source count differs')
    if dict(c_total)!=summary['classification']['counts'] or dict(l_total)!=summary['localization']['counts']:
        raise ValueError('Batch counts do not reproduce original summary')
    if c_batches!=summary['classification']['selected_batches'] or sorted(c_selected_images)!=summary['classification']['selected_images'] or sorted(c_selected_groups)!=summary['classification']['selected_source_groups']:
        raise ValueError('C selected batch/image/group summary differs')
    if l_batch['selected_count']!=summary['localization']['selected_batches']:raise ValueError('L selected batches differ')
    for gate,row in summary['localization']['each_gate'].items():
        if row['objects']!=l_total[gate] or row['images']!=sorted(l_image[gate]) or row['source_groups']!=sorted(l_group[gate]) or row['unique_images']!=len(l_image[gate]):
            raise ValueError('L gate population summary differs')
    for label in names:
        expected=summary['classification']['class_counts'].get(label,{})
        if any(c_class[label][phase]!=expected.get(phase,0) for phase in ('base','eligible','selected')):raise ValueError('C class summary differs')
    classes=[];levels=[];strides=[];concentrations=[];histograms=[]
    for label,name in names.items():
        count=c_class[label];base=count['base'];eligible=count['eligible'];selected=count['selected']
        classes.append({'dataset':dataset,'class_id':label,'class_name':name,'base':base,'eligible':eligible,'selected':selected,
            'base_fraction_all_base':fraction(base,c_total['base_count']),'eligible_fraction_base':fraction(eligible,base),
            'selected_fraction_base':fraction(selected,base),'selected_fraction_eligible':fraction(selected,eligible),
            'selected_fraction_all_selected':fraction(selected,c_total['selected_count']),
            'batches_with_base':len(c_class_batches[(label,'base')]),'batches_with_selected':len(c_class_batches[(label,'selected')]),
            'L_unverified_base':l_class[label]['base'],'L_unverified_selected':l_class[label]['selected']})
        for phase in ('base','eligible','selected'):
            count=c_levels[(label,phase)];n=count['objects']
            levels.append({'dataset':dataset,'class_id':label,'class_name':name,'phase':phase,'objects':n,
                'valid_P3_objects':count['P3'],'valid_P4_objects':count['P4'],'valid_both_objects':count['both'],
                'valid_P3_fraction':fraction(count['P3'],n),'valid_P4_fraction':fraction(count['P4'],n),
                'mean_effective_levels':fraction(count['P3']+count['P4'],n)})
    for phase in ('base','selected'):
        counts=l_strides[phase]
        for stride in ('8','16'):
            strides.append({'dataset':dataset,'phase':phase,'stride':int(stride),'objects':counts[stride],
                            'fraction':fraction(counts[stride],sum(counts.values()))})
    for (family,phase,unit),counts in sorted(hist.items()):
        row={'dataset':dataset,'family':family,'phase':phase,'unit':unit,**concentration(counts),
             'unknown_group_objects':counts.get(UNKNOWN,0) if unit=='group' else 0,'denominator':'objects_in_this_family_and_phase'}
        concentrations.append(row)
        histograms += [{'dataset':dataset,'family':family,'phase':phase,'unit':unit,'identity':key,'objects':value}
                       for key,value in sorted(counts.items(),key=lambda x:(-x[1],str(x[0])))]
    # Ensure empty selected sets are still represented with a zero denominator.
    for family in ('C0_C1','L_UNVERIFIED'):
        for unit in ('image','group'):
            if not any(r['family']==family and r['phase']=='selected' and r['unit']==unit for r in concentrations):
                concentrations.append({'dataset':dataset,'family':family,'phase':'selected','unit':unit,**concentration({}),
                    'unknown_group_objects':0,'denominator':'objects_in_this_family_and_phase'})
    overview={'dataset':dataset,'batches':64,'unique_natural_images':2048,'C_base':c_total['base_count'],
        'C_eligible':c_total['eligible_count'],'C_selected':c_total['selected_count'],'C_selected_batches':c_batches,
        'C_selected_images':len(c_selected_images),'C_selected_groups':len(c_selected_groups),
        'L_base':l_total['base_count'],'L_selected':l_total['selected_count'],'L_selected_batches':l_batch['selected_count'],
        'L_selected_images':len(l_image['selected_count']),'L_selected_groups':len(l_group['selected_count'])}
    manifest={str(folder/name):{'bytes':(folder/name).stat().st_size,'mtime_ns':(folder/name).stat().st_mtime_ns} for name in required}
    return {'overview':overview,'gates':gate_rows(dataset,l_total,l_image,l_group,l_batch),'classes':classes,'levels':levels,
            'strides':strides,'concentrations':concentrations,'histograms':histograms,'input_manifest':manifest,
            'model_identity':identity,'source_group_method':summary['source_group_method']}


def write_csv(path, rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for row in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})


def table(rows, fields):
    def cell(v):return '' if v is None else str(v).replace('|','\\|')
    return '\n'.join(['|'+'|'.join(fields)+'|','|'+'|'.join(['---']*len(fields))+'|']+
                     ['|'+'|'.join(cell(row.get(k)) for k in fields)+'|' for row in rows])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=HERE/'remote_completed_attempt2')
    parser.add_argument('--output',type=Path,default=HERE/'completed_readout_attempt2')
    args=parser.parse_args()
    # Both must validate before creating any output or declaring completion.
    results=[load_dataset(args.input,short,dataset) for short,dataset in DATASETS.items()]
    args.output.mkdir(parents=True,exist_ok=False)
    for key in ('gates','classes','levels','strides','concentrations','histograms'):
        write_csv(args.output/(key+'.csv'),[row for r in results for row in r[key]])
    overview=[r['overview'] for r in results];write_csv(args.output/'overview.csv',overview)
    inputs={key:value for r in results for key,value in r['input_manifest'].items()}
    receipt={'status':'COMPLETED_CPU_READOUT_PENDING_INDEPENDENT_REVIEW','datasets':[r['overview'] for r in results],
        'input_manifest':inputs,'source_group_methods':{r['overview']['dataset']:r['source_group_method'] for r in results},
        'inputs_only_full_64':True,'canary_included':False,'original_jsonl_unchanged':True,'hash_computed':False,
        'geometry_verified':False,'training_admitted':False,'model_identity':{r['overview']['dataset']:r['model_identity'] for r in results}}
    for path,before in inputs.items():
        p=Path(path)
        if p.stat().st_size!=before['bytes'] or p.stat().st_mtime_ns!=before['mtime_ns']:raise ValueError('Input changed during readout')
    (args.output/'summary.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines=['# 完整自然64批选择诊断汇总','',
        '**仅使用两组完成的64批full；原JSONL保持不变，所有计数由逐对象/逐图记录回算并核对summary。等待独立结果复核。**','',
        '状态为 UNVERIFIED_GEOMETRY_DIAGNOSTIC。C0/C1共享同一选择集；L为未验证几何的全放行诊断，不是已准入L1或严格数学上界。没有backward、校准、模型更新或AP。','',
        table(overview,['dataset','C_base','C_eligible','C_selected','C_selected_batches','C_selected_images','C_selected_groups','L_base','L_selected','L_selected_batches','L_selected_images','L_selected_groups']),'',
        '每组固定2048个自然源图；selected_batches指存在至少一个选中对象的批，不是非零真实共享梯度批。以下集中度以该选择集合的对象贡献为分母；未知来源单列，来源组不是标定组。','',
        table([row for r in results for row in r['concentrations'] if row['phase']=='selected'],['dataset','family','unit','objects','unique_units','top1_fraction','top5_fraction','unknown_group_objects']),'',
        '详细表：gates.csv为定位累计gate的对象损失/前级分母、图组与有对象批；classes.csv为全类别C base/eligible/selected比例及L类别计数；levels.csv为C的有效P3/P4共同层（两层都有效可同时计数，并非互斥物体尺寸档）；strides.csv为L主选择anchor的stride；histograms.csv保留全部图/组贡献便于核算top1/top5。','',
        '计数的样本来自固定自然增强窗口，且使用cfg原冻结T/R。Drone reference是旧formal_native RGB42，不能和近期新N42读出混用。自然流完整复现不证明独立几何，也不保证已选对象提供非零/有益梯度。正式几何、校准和对照要求保持不变。']
    (args.output/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'status':receipt['status'],'output':str(args.output),'overview':overview},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
