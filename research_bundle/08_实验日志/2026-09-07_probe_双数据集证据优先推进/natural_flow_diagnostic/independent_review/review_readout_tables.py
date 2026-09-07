"""Independent regeneration of every published CSV cell from original full JSONL."""
import collections,csv,json,math
from pathlib import Path
HERE=Path(__file__).resolve().parent;TASK=HERE.parent;RAW=TASK/'remote_completed_attempt2';TABLES=TASK/'completed_readout_attempt2'
CHAIN=('rgb_gt_count','common_count','pair_iou_count','geometry_count','inside_count','support_count','unique_owner_count',
 'reference_candidate_count','base_count','reference_reliable_count','both_reliable_count','reference_localization_gap_count',
 'teacher_rgb_quality_count','teacher_own_quality_count','eligible_count','selected_count')
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s]
def fraction(n,d):return n/d if d else None
def check(value,expected):
    if expected is None:return value==''
    if isinstance(expected,list):return json.loads(value)==expected
    if isinstance(expected,(float,int)):return math.isfinite(float(value)) and abs(float(value)-expected)<=1e-12
    return value==str(expected)
def main():
    expected=collections.defaultdict(list)
    for short,dataset in [('llvip','llvip'),('drone','dronevehicle')]:
        run=RAW/(short+'_full_attempt1');s=read(run/'summary.json');batches=rows(run/'selection_batches.jsonl')
        names=read(run/'model_identity.json')['reference']['names']
        stats=collections.Counter();counts=collections.defaultdict(collections.Counter);class_batches=collections.defaultdict(set)
        levels=collections.defaultdict(collections.Counter);strides=collections.defaultdict(collections.Counter)
        hist=collections.defaultdict(collections.Counter);gate_images=collections.defaultdict(set);gate_groups=collections.defaultdict(set);gate_batches=collections.Counter()
        for index,b in enumerate(batches):
            for family,key,phases in [('C0_C1','classification',('base','eligible','selected')),('L_UNVERIFIED','localization',('base','selected'))]:
                for record in b[key]['base_records']:
                    image=b['images'][record['batch_index']];label=str(record['class'])
                    for phase in phases:
                        if phase!='base' and not record[phase]:continue
                        counts[(family,label)][phase]+=1;class_batches[(family,label,phase)].add(index)
                        hist[(family,phase,'image')][image['rgb_source']]+=1
                        hist[(family,phase,'group')][image['governed_group'] or '<UNKNOWN_SOURCE_GROUP>']+=1
                        if family=='C0_C1':
                            lv=levels[(label,phase)];lv['objects']+=1;lv['P3']+=int(record['valid_levels'][0]);lv['P4']+=int(record['valid_levels'][1]);lv['both']+=int(all(record['valid_levels']))
                        else:strides[phase][int(record['stride'])]+=1
            for g in CHAIN:
                stats[g]+=b['localization'][g];gate_batches[g]+=int(b['localization'][g]>0)
            for image in b['localization_per_image']:
                for g,n in image['gate_counts'].items():
                    if n:
                        gate_images[g].add(image['rgb_source'])
                        if image['governed_group'] is not None:gate_groups[g].add(image['governed_group'])
        previous=None
        for g in CHAIN:
            n=stats[g];loss=previous-n if previous is not None else None
            expected['gates'].append(dict(dataset=dataset,gate=g,objects=n,previous_objects=previous,lost_objects=loss,
                loss_fraction_previous=fraction(loss,previous) if previous is not None else None,retention_fraction_rgb_gt=fraction(n,stats['rgb_gt_count']),
                unique_images=len(gate_images[g]),known_source_groups=len(gate_groups[g]),batches_with_any=gate_batches[g]));previous=n
        for label,name in names.items():
            c=counts[('C0_C1',label)];l=counts[('L_UNVERIFIED',label)]
            expected['classes'].append(dict(dataset=dataset,class_id=label,class_name=name,base=c['base'],eligible=c['eligible'],selected=c['selected'],
                base_fraction_all_base=fraction(c['base'],s['classification']['counts']['base_count']),eligible_fraction_base=fraction(c['eligible'],c['base']),
                selected_fraction_base=fraction(c['selected'],c['base']),selected_fraction_eligible=fraction(c['selected'],c['eligible']),
                selected_fraction_all_selected=fraction(c['selected'],s['classification']['counts']['selected_count']),
                batches_with_base=len(class_batches[('C0_C1',label,'base')]),batches_with_selected=len(class_batches[('C0_C1',label,'selected')]),
                L_unverified_base=l['base'],L_unverified_selected=l['selected']))
            for phase in ('base','eligible','selected'):
                lv=levels[(label,phase)];n=lv['objects']
                expected['levels'].append(dict(dataset=dataset,class_id=label,class_name=name,phase=phase,objects=n,valid_P3_objects=lv['P3'],
                    valid_P4_objects=lv['P4'],valid_both_objects=lv['both'],valid_P3_fraction=fraction(lv['P3'],n),valid_P4_fraction=fraction(lv['P4'],n),
                    mean_effective_levels=fraction(lv['P3']+lv['P4'],n)))
        for phase in ('base','selected'):
            for stride in (8,16):expected['strides'].append(dict(dataset=dataset,phase=phase,stride=stride,objects=strides[phase][stride],fraction=fraction(strides[phase][stride],sum(strides[phase].values()))))
        for (family,phase,unit),values in hist.items():
            ranked=sorted(values.items(),key=lambda pair:(-pair[1],str(pair[0])));n=sum(values.values());top1=sum(v for _,v in ranked[:1]);top5=sum(v for _,v in ranked[:5])
            expected['concentrations'].append(dict(dataset=dataset,family=family,phase=phase,unit=unit,objects=n,unique_units=len(values),top1_objects=top1,
                top5_objects=top5,top1_fraction=fraction(top1,n),top5_fraction=fraction(top5,n),top5_units=[str(k) for k,v in ranked[:5]],
                unknown_group_objects=values.get('<UNKNOWN_SOURCE_GROUP>',0) if unit=='group' else 0,denominator='objects_in_this_family_and_phase'))
            for identity,n in values.items():expected['histograms'].append(dict(dataset=dataset,family=family,phase=phase,unit=unit,identity=identity,objects=n))
        c=s['classification'];l=s['localization'];expected['overview'].append(dict(dataset=dataset,batches=64,unique_natural_images=2048,
            C_base=c['counts']['base_count'],C_eligible=c['counts']['eligible_count'],C_selected=c['counts']['selected_count'],C_selected_batches=c['selected_batches'],
            C_selected_images=c['selected_unique_images'],C_selected_groups=len(c['selected_source_groups']),L_base=l['counts']['base_count'],L_selected=l['counts']['selected_count'],
            L_selected_batches=l['selected_batches'],L_selected_images=l['each_gate']['selected_count']['unique_images'],L_selected_groups=len(l['each_gate']['selected_count']['source_groups'])))
    keys=dict(overview=['dataset'],gates=['dataset','gate'],classes=['dataset','class_id'],levels=['dataset','class_id','phase'],strides=['dataset','phase','stride'],
        concentrations=['dataset','family','phase','unit'],histograms=['dataset','family','phase','unit','identity'])
    checks=[]
    for name,want in expected.items():
        with (TABLES/(name+'.csv')).open(encoding='utf-8-sig',newline='') as f:actual=list(csv.DictReader(f))
        key=lambda r:tuple(str(r[k]) for k in keys[name]);lookup={key(r):r for r in actual}
        if len(actual)!=len(want) or len(lookup)!=len(actual):raise AssertionError('row count/duplicate '+name)
        cells=0
        for row in want:
            seen=lookup[key(row)]
            if set(row)!=set(seen):raise AssertionError('column coverage '+name)
            for k,v in row.items():
                if not check(seen[k],v):raise AssertionError('CSV value '+name+str(key(row))+':'+k+' actual='+str(seen[k])+' expected='+str(v))
                cells+=1
        checks.append(dict(file=name+'.csv',rows=len(actual),cells=cells,status='PASS'))
    report=dict(status='PASS',date='2026-09-07',auditor='/root/ap_error',tables=checks,total_cells=sum(r['cells'] for r in checks),
        source='Independent loops over original full JSONL; no author summarizer function called',new_hash_computed=False)
    out=HERE/'real_results_v1/readout_tables_review.json'
    with out.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
