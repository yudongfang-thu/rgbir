"""Independently recompute complete diagnostic records and draw CPU-only figures."""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path
import statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
KEYS=('drone_train','drone_val','llvip_train','llvip_val','llvip_verified_train')
FILTERS=('rgb_gt_count','common_count','pair_iou_count','geometry_count','inside_count','support_count',
         'unique_owner_count','reference_candidate_count','base_count','reference_reliable_count',
         'both_reliable_count','reference_localization_gap_count','teacher_rgb_quality_count',
         'teacher_own_quality_count','selected_count')
GROUPS=('class','scale_bin','luminance_proxy','source_group','brightness_proxy_bin')
LABELS={'drone_train':'Drone train','drone_val':'Drone dev','llvip_train':'LLVIP train','llvip_val':'LLVIP dev',
        'llvip_verified_train':'LLVIP verified frame'}


def read_jsonl(path):
    text = path.read_text(encoding='utf-8') if path.is_file() else gzip.decompress(path.with_suffix(path.suffix+'.gz').read_bytes()).decode('utf-8')
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def describe(values):
    if not values:return {'n':0}
    array=np.asarray(values,float)
    return {'n':len(values),'mean':float(array.mean()),'median':float(np.median(array)),
            'q25':float(np.quantile(array,.25)),'q75':float(np.quantile(array,.75)),
            'min':float(array.min()),'max':float(array.max()),
            'sample_sd':statistics.stdev(values) if len(values)>1 else None}


def fraction(n,d):return n/d if d else None


def record_id(row):return row['image'],row['rgb_gt_local_index']


def load_and_analyze(key):
    folder=HERE/key
    summary=json.loads((folder/'summary.json').read_text())
    receipt=json.loads((folder/'run_evidence/run_receipt.json').read_text())
    if summary.get('status')!='completed' or receipt.get('terminal_status')!='COMPLETED':
        raise ValueError('Not a complete endpoint: '+key)
    bound=[json.loads((folder/'run_evidence'/name).read_text()) for name in receipt['metric_snapshots']]
    if summary not in bound:raise ValueError('Bound summary mismatch: '+key)
    d1=read_jsonl(folder/'d1_objects.jsonl')
    d2=read_jsonl(folder/'d2_anchors.jsonl')
    images=read_jsonl(folder/'images.jsonl')
    selected=[row for row in d2 if row['selected']]
    by_id={record_id(row):row for row in d1}
    if len(by_id)!=len(d1):raise ValueError('Duplicate D1 object identity')
    if len({record_id(row) for row in d2})!=len(d2):raise ValueError('Duplicate D2 base object identity')
    if not all(record_id(row) in by_id for row in d2):raise ValueError('D2 object is absent in D1')
    actual={'rgb_gt_count':len(d1),'base_count':len(d2),'selected_count':len(selected)}
    if not all(summary['totals'][name]==value for name,value in actual.items()):raise ValueError('Record/summary count differs')
    if len(images)!=summary['images']:raise ValueError('Image record count differs')
    states=Counter(row['reference_state'] for row in d1)
    for name,count in states.items():
        if summary['d1_totals'].get(name,0)!=count:raise ValueError('D1 state summary differs')
    opportunity=sum(row['object_level_localization_opportunity'] for row in d1)
    if opportunity!=summary['d1_totals']['object_level_localization_opportunity']:raise ValueError('D1 opportunity count differs')
    delta_ce=[r['teacher_gt_dfl_ce']-r['reference_gt_dfl_ce'] for r in selected]
    cosine=[r['reference_logit_kd_native_cosine'] for r in selected if r['reference_logit_kd_native_cosine'] is not None]
    selected_states=Counter(by_id[record_id(r)]['reference_state'] for r in selected)
    d1_opportunity_ids={record_id(r) for r in d1 if r['object_level_localization_opportunity']}
    selected_ids={record_id(r) for r in selected}
    groups=[]
    for field in GROUPS:
        population=defaultdict(lambda:{'rgb_gt':0,'d1_opportunity':0,'base':0,'selected':0,'ce':[],'cosine':[]})
        for row in d1:
            item=population[str(row[field])];item['rgb_gt']+=1;item['d1_opportunity']+=int(row['object_level_localization_opportunity'])
        for row in d2:
            item=population[str(row[field])];item['base']+=1;item['selected']+=int(row['selected'])
            if row['selected']:
                item['ce'].append(row['teacher_gt_dfl_ce']-row['reference_gt_dfl_ce'])
                if row['reference_logit_kd_native_cosine'] is not None:item['cosine'].append(row['reference_logit_kd_native_cosine'])
        for group,counts in sorted(population.items()):
            original1=summary['d1_groups'].get(field+':'+group,{})
            original2=summary['d2_groups'].get(field+':'+group,{})
            for name,mine,theirs in [('GT',counts['rgb_gt'],original1.get('rgb_gt_count',0)),('base',counts['base'],original2.get('base_count',0)),('selected',counts['selected'],original2.get('selected_count',0))]:
                if mine!=theirs:raise ValueError(f'Group {field}/{group}/{name} count differs')
            ce,cos=counts.pop('ce'),counts.pop('cosine')
            groups.append({'run':key,'group_type':field,'group':group,**counts,
              'selected_per_rgb_gt':fraction(counts['selected'],counts['rgb_gt']),
              'selected_per_base':fraction(counts['selected'],counts['base']),
              'teacher_ce_worse_count':sum(value>0 for value in ce),
              'teacher_ce_delta_mean':statistics.mean(ce) if ce else None,
              'negative_logit_cosine_count':sum(value<0 for value in cos)})
    return {'key':key,'dataset':summary['dataset'],'split':summary['split'],'images':len(images),
            'geometry_verified':summary['geometry_verified'],'geometry_status':summary['geometry_status'],
            'filters':{name:summary['totals'][name] for name in FILTERS},'reference_states':dict(states),
            'd1_opportunity':opportunity,'d1_opportunity_per_gt':fraction(opportunity,len(d1)),
            'd2_base':len(d2),'d2_selected':len(selected),'d2_selected_per_gt':fraction(len(selected),len(d1)),
            'd2_selected_per_base':fraction(len(selected),len(d2)),
            'selected_teacher_minus_reference_dfl_ce':describe(delta_ce),
            'teacher_ce_worse_count':sum(x>0 for x in delta_ce),'teacher_ce_worse_fraction':fraction(sum(x>0 for x in delta_ce),len(delta_ce)),
            'selected_reference_logit_cosine':describe(cosine),
            'negative_logit_cosine_count':sum(x<0 for x in cosine),'negative_logit_cosine_fraction':fraction(sum(x<0 for x in cosine),len(cosine)),
            'selected_d1_reference_states':dict(selected_states),
            'd1_d2_overlap':{'both':len(d1_opportunity_ids&selected_ids),'d1_only':len(d1_opportunity_ids-selected_ids),'d2_only':len(selected_ids-d1_opportunity_ids)},
            'groups':groups,'receipt_metric_and_record_counts_verified':True},selected


def plots(summaries, selected_by_run, output):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'figure.dpi':120,'savefig.dpi':240,'font.family':'DejaVu Sans'})
    keys=[key for key in KEYS[:4] if key in summaries]
    fig,axes=plt.subplots(1,2,figsize=(12.5,4.6),gridspec_kw={'width_ratios':[1,1.35]})
    x=np.arange(len(keys));width=.34
    axes[0].bar(x-width/2,[100*summaries[k]['d1_opportunity_per_gt'] for k in keys],width,label='D1 object-level opportunity',color='#4C78A8')
    bars=axes[0].bar(x+width/2,[100*summaries[k]['d2_selected_per_gt'] for k in keys],width,label='D2 selected R-anchor',color='#F58518')
    for bar,key in zip(bars,keys):
        s=summaries[key];axes[0].text(bar.get_x()+width/2,bar.get_height()+.08,f"{s['d2_selected']}/{s['filters']['rgb_gt_count']}",ha='center',fontsize=8)
    axes[0].set_xticks(x,[LABELS[k].replace(' ','\n',1) for k in keys]);axes[0].set_ylabel('Percent of all RGB GT objects')
    axes[0].set_title('D1 and D2 use different predictions\nThey are not nested subsets');axes[0].legend(fontsize=8)
    stages=('rgb_gt_count','pair_iou_count','base_count','both_reliable_count','reference_localization_gap_count','selected_count')
    stage_labels=('All RGB GT','Label pair\nIoU >= .8','Base E','Both\nreliable','R IoU\n< .70','Selected')
    for key,color in zip(keys,('#4C78A8','#72B7B2','#F58518','#E45756')):
        s=summaries[key];den=s['filters']['rgb_gt_count']
        axes[1].plot(range(len(stages)),[100*s['filters'][st]/den for st in stages],'-o',label=LABELS[key],color=color)
    axes[1].set_xticks(range(len(stages)),stage_labels);axes[1].set_ylabel('Percent of all RGB GT objects');axes[1].set_ylim(0,105)
    axes[1].set_title('D2 cumulative filters (geometry NOT verified)');axes[1].legend(fontsize=8)
    fig.suptitle('Diagnostic opportunity only — UNVERIFIED geometry',fontweight='bold')
    fig.text(.5,.015,'Independent verified LLVIP frame: 2 GT -> 0 geometry-eligible -> 0 base -> 0 selected.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.055,1,.93])
    for extension in ('png','pdf'):fig.savefig(output/f'opportunity_and_filters.{extension}')
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,8),sharex=True,sharey=True)
    for ax,key in zip(axes.flat,keys):
        rows=selected_by_run[key];s=summaries[key]
        ce=[r['teacher_gt_dfl_ce']-r['reference_gt_dfl_ce'] for r in rows]
        gain=[r['teacher_rgb_iou']-r['reference_iou'] for r in rows]
        conflict=[r['reference_logit_kd_native_cosine'] is not None and r['reference_logit_kd_native_cosine']<0 for r in rows]
        for label,flag,color in [('nonnegative cosine',False,'#4C78A8'),('negative cosine',True,'#D1495B')]:
            ids=[i for i,value in enumerate(conflict) if value==flag]
            ax.scatter(np.asarray(gain)[ids],np.asarray(ce)[ids],s=12,alpha=.65,color=color,label=label,edgecolors='none')
        ax.axhline(0,color='black',lw=.9);ax.set_title(f"{LABELS[key]}: selected n={len(rows)}")
        ax.text(.98,.96,f"Teacher DFL CE worse: {s['teacher_ce_worse_count']}/{len(rows)}\nNegative logit cosine: {s['negative_logit_cosine_count']}/{len(rows)}",transform=ax.transAxes,ha='right',va='top',fontsize=9)
        ax.set_xlabel('Teacher minus R mean-box IoU (RGB GT)');ax.set_ylabel('Teacher minus R GT-DFL CE\n(positive = worse teacher target)')
    axes[0,0].legend(loc='lower left',fontsize=8)
    fig.suptitle('Selected mean boxes and full DFL targets can disagree\nUNVERIFIED geometry; cosine is a frozen-R logit proxy',fontweight='bold')
    fig.tight_layout(rect=[0,0,1,.94])
    for extension in ('png','pdf'):fig.savefig(output/f'selected_dfl_content.{extension}')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    summaries={};selected={}
    for key in KEYS:summaries[key],selected[key]=load_and_analyze(key)
    (args.output/'summary_recomputed.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    group_rows=[row for result in summaries.values() for row in result['groups']]
    with (args.output/'group_comparison.csv').open('w',newline='',encoding='utf-8') as fh:
        writer=csv.DictWriter(fh,fieldnames=list(group_rows[0]));writer.writeheader();writer.writerows(group_rows)
    with (args.output/'filter_counts.csv').open('w',newline='',encoding='utf-8') as fh:
        writer=csv.DictWriter(fh,fieldnames=['run','geometry_verified',*FILTERS]);writer.writeheader()
        writer.writerows({'run':key,'geometry_verified':s['geometry_verified'],**s['filters']} for key,s in summaries.items())
    plots(summaries,selected,args.output)
    print(json.dumps({key:{k:v for k,v in s.items() if k not in ('groups','filters')} for key,s in summaries.items()},indent=2))


if __name__=='__main__':main()
