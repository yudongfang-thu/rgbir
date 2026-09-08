"""Prepare a separate adaptation of the accepted LLVIP native capture; no execution."""
from pathlib import Path
import json, shutil

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'2026-09-07_probe_双数据集证据优先推进/llvip_full_eval/export_full_dev.py'
OUT=HERE/'release'
OUT.mkdir(exist_ok=False)
shutil.copyfile(OLD,HERE/'accepted_llvip_export_full_dev_source.py')
s=OLD.read_text(encoding='utf-8')
def replace(old,new):
    global s
    assert s.count(old)==1, (old,s.count(old))
    s=s.replace(old,new)
replace('historical LLVIP baselines','historical Drone IR42 teacher')
replace('import argparse, gc, gzip, inspect, json, random, shutil, sys, time, traceback','import argparse, gc, gzip, inspect, json, math, random, shutil, sys, time, traceback')
replace("This frozen LLVIP val source must be a directory","This frozen Drone IR val source must be a directory")
replace('def run(a):', '''SCOPE='DRONE_IR42_FULL_DEV_CAPTURE'
CLASS_NAMES=['car','freight car','truck','bus','van']

def label_array(lines):
    import numpy as np
    if any(len(row)!=5 for row in lines):raise ValueError('YOLO label requires five fields')
    array=np.asarray(lines,dtype=np.float32).reshape(-1,5)
    if (not np.isfinite(array).all() or (array[:,0]!=np.floor(array[:,0])).any()
        or (array[:,0]<0).any() or (array[:,0]>=5).any()
        or (array[:,1:]<0).any() or (array[:,1:]>1).any() or (array[:,3:]<=0).any()):
        raise ValueError('Invalid Drone IR normalized labels')
    return array

def validate_metrics(result,expected_class_ids):
    if [r['class_id'] for r in result['per_class']]!=expected_class_ids:
        raise ValueError('Missing, reordered or unexpected observed-class metrics')
    for key in ('AP50','AP75','mAP50_95','precision','recall'):
        if not math.isfinite(result[key]) or not 0<=result[key]<=1:raise ValueError('Invalid metric: '+key)
    for row in result['per_class']:
        if row['name']!=CLASS_NAMES[row['class_id']]:raise ValueError('Metric class name differs')
        for key in ('AP50','AP75','mAP50_95'):
            if not math.isfinite(row[key]) or not 0<=row[key]<=1:raise ValueError('Invalid per-class metric')

def run(a):''')
replace("if len(roster)!=2406:raise ValueError('Expected complete LLVIP dev2406')", "if len(roster)!=1469:raise ValueError('Expected complete Drone IR dev1469')")
replace("spec=json.loads(a.spec.read_text())['models'][a.model]", "frozen=json.loads(a.spec.read_text());spec=frozen['models'][a.model]\n    if frozen['scope']!=SCOPE:raise ValueError('Wrong capture scope')")
replace("data=yaml.safe_load(data_source.read_text())", "data=yaml.safe_load(data_source.read_text())\n    if state(checkpoint)!=spec['checkpoint_stat']:raise ValueError('Frozen teacher checkpoint stat differs')\n    if {int(k):v for k,v in data['names'].items()}!=dict(enumerate(CLASS_NAMES)):raise ValueError('Wrong IR class order')")
replace("all_labels=img2label_paths(aliases);label_counts=[];expected_labels={}", "if aliases!=spec['aliases'] or roster!=spec['canonical']:raise ValueError('Frozen full dev roster differs')\n    all_labels=img2label_paths(aliases);label_counts=[];expected_labels={}")
replace("        if any(len(row)!=5 or float(row[0])!=0 for row in lines):raise ValueError('Unexpected LLVIP label content')\n        array=np.asarray(lines,dtype=np.float32).reshape(-1,5)\n        if not np.isfinite(array).all() or (array[:,1:]<0).any() or (array[:,1:]>1).any() or (array[:,3:]<=0).any():\n            raise ValueError('Nonfinite or invalid normalized LLVIP label coordinates')", "        array=label_array(lines)")
replace("if sum(label_counts)<=0:raise ValueError('No GT in original dev labels')", "if label_counts!=spec['label_counts'] or sum(label_counts)!=24490:raise ValueError('Frozen per-image IR GT counts differ')\n    full_class_counts=np.bincount(np.concatenate(list(expected_labels.values()))[:,0].astype(int),minlength=5).tolist()\n    if full_class_counts!=spec['class_counts']:raise ValueError('Frozen full IR class counts differ')")
replace("if len(shapes)!=1:raise ValueError('Canary requires full-roster uniform image dimensions')", "if shapes!={(640,512)}:raise ValueError('Frozen original image dimensions differ')")
replace("actual_aliases=aliases[:64] if a.canary else aliases", "actual_aliases=aliases[:32] if a.canary else aliases")
replace("expected_gt=sum(label_counts[:64] if a.canary else label_counts)", "expected_gt=sum(label_counts[:32] if a.canary else label_counts)\n        class_counts=np.bincount(np.concatenate([expected_labels[p] for p in actual_aliases])[:,0].astype(int),minlength=5).tolist()\n        if a.canary and (expected_gt!=514 or class_counts!=[447,24,30,13,0]):raise ValueError('Frozen first32 GT/classes differ')\n        expected_class_ids=[i for i,n in enumerate(class_counts) if n>0]")
replace("expected_evaluated_gt=expected_gt,full_gt=sum(label_counts),", "expected_evaluated_gt=expected_gt,full_gt=sum(label_counts),class_counts=class_counts,\n            full_class_counts=full_class_counts,expected_metric_class_ids=expected_class_ids,\n            empty_gt_images=sum(n==0 for n in label_counts),independent_ir_labels=True,")
replace("if len(model.names)!=1:raise ValueError('Expected one LLVIP class')", "if len(model.names)!=5:raise ValueError('Expected five Drone classes')")
replace("scope='historical_baseline_not_new_protocol_N'", "scope='historical_native_infrared_teacher_seed42_E200'")
replace("contract['loader_per_image_labels_exact']=True", "contract['loader_per_image_labels_exact']=True\n                contract['scope']=SCOPE;contract['independent_ir_labels']=True\n                contract['expected_metric_class_ids']=expected_class_ids")
replace("agnostic_nms=False,single_cls=False,quantize=None)", "agnostic_nms=False,single_cls=False,quantize=None,half=False)")
replace("if len(result['per_class'])!=1 or result['per_class'][0]['class_id']!=0:raise ValueError('Missing expected person class metrics')", "validate_metrics(result,expected_class_ids)")
replace("dict(status='completed',model=a.model,dataset='llvip',", "dict(status='completed',scope=SCOPE,model=a.model,dataset='dronevehicle',")
replace("images=len(actual),full_population=2406,canary=a.canary,native_capture_exact=a.canary,", "images=len(actual),full_population=1469,canary=a.canary,native_capture_exact=a.canary,")
replace("gt_count=expected_gt,", "gt_count=expected_gt,full_gt=24490,class_counts=class_counts,class_names=CLASS_NAMES,")
replace("training_modified=False,new_protocol_N=False,physical_alignment_verified=False", "training_modified=False,new_protocol_N=False,physical_alignment_verified=False,\n            new_hash_computed=False,optimizer_updates=0,independent_ir_labels=True")
replace("choices=['N42','T42']", "choices=['T42']")
(OUT/'export_drone_teacher.py').write_text(s,encoding='utf-8')
a=json.loads((HERE/'remote_input_audit.json').read_text(encoding='utf-8'))
spec=dict(scope='DRONE_IR42_FULL_DEV_CAPTURE',dataset='dronevehicle',canary_images=32,
          models={'T42':dict(checkpoint=a['checkpoint_path'],checkpoint_stat=a['checkpoint_stat'],data=a['source_data'],
                            aliases=a['aliases'],canonical=a['canonical'],label_counts=a['label_counts'],class_counts=a['class_counts'])})
(OUT/'drone_teacher_spec.json').write_text(json.dumps(spec,indent=2,ensure_ascii=False),encoding='utf-8')
print('Prepared',OUT)
