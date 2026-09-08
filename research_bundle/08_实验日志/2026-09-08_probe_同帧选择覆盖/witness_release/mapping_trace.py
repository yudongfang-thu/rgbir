"""Metadata-only original-GT identity tap for the pinned paired HBB pipeline.

Install after PairDataset construction and BEFORE constructing its loader.
No second image read, transform, RNG draw, iterator or floating nearest match.
Original GT means the verified native dataset.labels row before augmentation;
it is deliberately not an inferred XML object number or text-file line number.
"""
from __future__ import annotations
import ast
import copy
import functools
import inspect
import json
from pathlib import Path
import textwrap
import numpy as np
import torch
from torch.utils.data import Dataset


class MappingIdentityError(RuntimeError):pass


def canonical(path):return str(Path(str(path)).resolve())


def array(value):
    return value.detach().cpu().numpy() if isinstance(value,torch.Tensor) else np.asarray(value)


def count_cls(labels):
    a=array(labels['cls'])
    if a.ndim!=2 or a.shape[1]!=1 or not np.isfinite(a).all():
        raise MappingIdentityError('Only finite native [N,1] HBB classes supported')
    return len(a)


def _native_filter_contract(member):
    """Fail if the observed boolean is not the exact native final row selector."""
    try:tree=ast.parse(textwrap.dedent(inspect.getsource(type(member).apply_instances)))
    except (OSError,TypeError,SyntaxError) as e:raise MappingIdentityError('Cannot identify native filter source') from e
    fn=tree.body[0]
    assignments=[n for n in ast.walk(fn) if isinstance(n,ast.Assign)]
    def dump(code):return ast.dump(ast.parse(code).body[0],include_attributes=False)
    required=[dump('cls = labels["cls"]'),dump('labels["instances"] = new_instances[i]'),dump('labels["cls"] = cls[i]')]
    seen=[ast.dump(n,include_attributes=False) for n in assignments]
    if any(seen.count(x)!=1 for x in required):raise MappingIdentityError('Unknown native row-selection semantics')
    calls=[n for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='box_candidates']
    if len(calls)!=1 or not isinstance(calls[0].func.value,ast.Name) or calls[0].func.value.id!='self':
        raise MappingIdentityError('Expected exactly one native box_candidates call')
    tail=[ast.dump(n,include_attributes=False) for n in fn.body[-3:-1]]
    if tail!=required[1:] or not isinstance(fn.body[-1],ast.Return):
        raise MappingIdentityError('Filter must directly select instances/classes at function end')
    mask=[n for n in assignments if any(isinstance(t,ast.Name) and t.id=='i' for t in n.targets)]
    if len(mask)!=1 or mask[0].value is not calls[0]:raise MappingIdentityError('Unknown filter mask reassignment')


class CandidateTap:
    def __init__(self,original,recorder):self.original,self.recorder=original,recorder

    def __call__(self,*args,**kwargs):
        result=self.original(*args,**kwargs)
        active=self.recorder.active
        if active is None:raise MappingIdentityError('Native filtering outside observed transform')
        keep=np.asarray(result)
        if keep.dtype!=np.bool_ or keep.shape!=(len(active['kept_rows']),):
            raise MappingIdentityError('Unrecognized native candidate boolean mask')
        box2=kwargs.get('box2',args[1] if len(args)>1 else None)
        if np.asarray(box2).shape!=(4,len(keep)):
            raise MappingIdentityError('Native filter input cardinality differs')
        before=list(active['kept_rows'])
        active['filter_events'].append(dict(kind='RandomPerspective.box_candidates',
            source_rows_before=before,keep_mask=keep.tolist(),source_rows_after=[r for r,k in zip(before,keep) if k]))
        active['kept_rows']=[r for r,k in zip(before,keep) if k]
        return result


def _members(transform):
    if type(transform).__name__=='Compose':
        for child in transform.transforms:yield from _members(child)
    else:yield transform


class IdentityRecorder:
    def __init__(self,pair):
        self.active=None;self.records={};self.sources={}
        for modality,dataset in [('rgb',pair.base),('ir',pair.teacher_base)]:
            if dataset is None:raise MappingIdentityError('Separate paired RGB/IR label datasets required')
            if len(dataset.labels)!=len(dataset.im_files):raise MappingIdentityError('Original label roster is incomplete')
            for label,path in zip(dataset.labels,dataset.im_files):
                source=canonical(path)
                if source in self.sources:raise MappingIdentityError('Ambiguous duplicated modality/source path')
                if canonical(label['im_file'])!=source:raise MappingIdentityError('Native cached label/image roster differs')
                cls=np.asarray(label['cls']);boxes=np.asarray(label['bboxes'])
                if label.get('bbox_format')!='xywh' or label.get('normalized') is not True:
                    raise MappingIdentityError('Original labels must be normalized xywh')
                if cls.shape!=(len(boxes),1) or boxes.shape!=(len(cls),4) or not np.isfinite(cls).all() or not np.isfinite(boxes).all():
                    raise MappingIdentityError('Malformed original HBB labels')
                if np.any(boxes[:,2:]<=0):raise MappingIdentityError('Original box width/height must be positive')
                if len(label.get('segments',[])) or label.get('keypoints') is not None:
                    raise MappingIdentityError('Only unambiguous HBB labels supported')
                self.sources[source]=dict(modality=modality,classes=cls.copy(),boxes=boxes.copy())

    def begin(self,labels):
        if self.active is not None:raise MappingIdentityError('Nested/recursive transform identity is unsupported')
        source=canonical(labels['im_file']);original=self.sources.get(source)
        if original is None:raise MappingIdentityError('Transform source absent from original label roster')
        if source in self.records:raise MappingIdentityError('Source transformed twice in one paired sample')
        instances=labels.get('instances')
        if instances is None or not np.array_equal(array(labels['cls']),original['classes']) or not np.array_equal(array(instances.bboxes),original['boxes']):
            raise MappingIdentityError('Pre-augmentation GT rows differ from original verified label table')
        if getattr(getattr(instances,'_bboxes',None),'format',None)!='xywh' or getattr(instances,'normalized',None) is not True:
            raise MappingIdentityError('Unknown input Instances coordinate representation')
        self.active=dict(source=source,kept_rows=list(range(len(original['classes']))),filter_events=[])

    def finish(self,output):
        active=self.active
        if active is None:raise MappingIdentityError('Missing active identity transform')
        original=self.sources[active['source']];ids=active['kept_rows'];n=count_cls(output)
        if len(active['filter_events'])!=1:raise MappingIdentityError('Expected exactly one affine candidate-filter event')
        if n!=len(ids) or not np.array_equal(array(output['cls']),original['classes'][ids]):
            raise MappingIdentityError('Native output classes/order do not close with actual filter')
        boxes=array(output['bboxes'])
        if boxes.shape!=(n,4) or not np.isfinite(boxes).all():raise MappingIdentityError('Malformed transformed boxes')
        source=active['source'];aug=[]
        for j,i in enumerate(ids):
            aug.append(dict(source_gt_row=i,stable_gt_id=source+'::native_gt:'+str(i),
                source_class=float(original['classes'][i,0]),source_bbox_normalized_xywh=original['boxes'][i].tolist(),
                augmented_gt_row=j,augmented_class=float(array(output['cls'])[j,0]),
                augmented_bbox_normalized_xywh=boxes[j].tolist()))
        self.records[source]=dict(image_path=source,modality=original['modality'],
            original_gt_identity_basis='native verified dataset.labels row before augmentation; not raw annotation file line',
            original_gt_count=len(original['classes']),augmented_gt_count=n,
            original_rows=[dict(source_gt_row=i,stable_gt_id=source+'::native_gt:'+str(i),
                source_class=float(c[0]),source_bbox_normalized_xywh=b.tolist())
                for i,(c,b) in enumerate(zip(original['classes'],original['boxes']))],
            kept_source_gt_rows=ids,dropped_source_gt_rows=[i for i in range(len(original['classes'])) if i not in set(ids)],
            rows=aug,filter_events=copy.deepcopy(active['filter_events']))
        self.active=None


class TransformTap:
    def __init__(self,original,recorder):self.original,self.recorder=original,recorder
    def __call__(self,labels):
        self.recorder.begin(labels)
        try:
            result=self.original(labels)
            self.recorder.finish(result)
            return result
        except BaseException:
            self.recorder.active=None
            raise


class MappingTraceDataset(Dataset):
    def __init__(self,pair):
        if getattr(pair,'same_modal',None) is not False:raise MappingIdentityError('Independent paired mode required')
        self.pair=pair;self.recorder=IdentityRecorder(pair)
        if isinstance(pair.base.transforms,TransformTap):raise MappingIdentityError('Mapping tap installed twice')
        members=list(_members(pair.base.transforms))
        allowed={'RandomPerspective','LetterBox','RandomFlip','Format','RandomHSV','Albumentations','Mosaic','MixUp','CutMix','CopyPaste'}
        if any(type(m).__name__ not in allowed for m in members):raise MappingIdentityError('Unknown transform identity semantics')
        perspective=[m for m in members if type(m).__name__=='RandomPerspective']
        if len(perspective)!=1:raise MappingIdentityError('Exactly one native affine filter required')
        formats=[m for m in members if type(m).__name__=='Format']
        if len(formats)!=1 or members[-1] is not formats[0]:raise MappingIdentityError('One final native Format required')
        # The underlying pinned PairDataset already validates no mixing, HSV,
        # masks, keypoints, OBB or Albumentations. Recheck mutation-prone flags.
        for m in members:
            name=type(m).__name__
            if name in {'Mosaic','MixUp','CutMix','CopyPaste'} and float(m.p)!=0.:raise MappingIdentityError('Mixing is unsupported')
            if name=='Albumentations' and m.transform is not None:raise MappingIdentityError('Albumentations is unsupported')
            if name=='Format' and any(bool(getattr(m,k,False)) for k in ('return_mask','return_keypoint','return_obb')):
                raise MappingIdentityError('Only HBB Format is supported')
        for m in perspective:
            _native_filter_contract(m)
            if isinstance(m.box_candidates,CandidateTap):raise MappingIdentityError('Candidate tap installed twice')
            m.box_candidates=CandidateTap(m.box_candidates,self.recorder)
        pair.base.transforms=TransformTap(pair.base.transforms,self.recorder)

    def __len__(self):return len(self.pair)
    def __getattr__(self,name):
        if name in {'pair','recorder'}:raise AttributeError(name)
        return getattr(self.pair,name)
    def __getitem__(self,index):
        self.recorder.records.clear()
        result=self.pair[index]
        info=dict(result['pair_info']);rgb=canonical(info['weak_source']);ir=canonical(info['strong_source'])
        if set(self.recorder.records)!={rgb,ir}:raise MappingIdentityError('Missing or ambiguous independent modality trace')
        info['gt_identity_trace']=dict(status='STABLE_GT_IDENTITY_VERIFIED',dataset_index=int(index),
            rgb=copy.deepcopy(self.recorder.records[rgb]),ir=copy.deepcopy(self.recorder.records[ir]))
        return dict(result,pair_info=info)
    def collate_fn(self,rows):return self.pair.collate_fn(rows)


def _construct(original_type,*args,**kwargs):return MappingTraceDataset(original_type(*args,**kwargs))


def make_traced_dataset(original_type):
    """Inject returned callable as legacy builder's DualLabelRGBIRDataset."""
    return functools.partial(_construct,original_type)


def batch_identity_contract(batch,frame_id):
    if not isinstance(frame_id,str) or not frame_id:raise MappingIdentityError('Explicit nonempty frame ID required')
    info=batch.get('pair_info')
    if not isinstance(info,list) or len(info)!=len(batch['im_file']):raise MappingIdentityError('Per-image trace missing')
    teacher=batch.get('teacher_batch')
    if teacher is None:
        teacher={k:batch['strong_'+k] for k in ('cls','bboxes','batch_idx')}
    result=dict(status='STABLE_GT_IDENTITY_VERIFIED',frame_id=frame_id,rgb_rows=[],ir_rows=[],frames=[],
        identity_basis='native verified pre-augmentation label rows',nearest_neighbor_matching=False,
        second_data_stream=False,new_hash_computed=False)
    for modality,target in [('rgb',batch),('ir',teacher)]:
        ids=array(target['batch_idx']).reshape(-1);classes=array(target['cls']);boxes=array(target['bboxes'])
        rows=[]
        for image_index,pair in enumerate(info):
            trace=pair.get('gt_identity_trace',{})
            if trace.get('status')!='STABLE_GT_IDENTITY_VERIFIED':raise MappingIdentityError('Unverified image identity')
            entry=trace[modality];indices=np.flatnonzero(ids==image_index)
            if len(indices)!=entry['augmented_gt_count']:raise MappingIdentityError('Batch target count differs from trace')
            if modality=='rgb' and canonical(batch['im_file'][image_index])!=entry['image_path']:
                raise MappingIdentityError('Batch image order differs from trace')
            for index,row in zip(indices,entry['rows']):
                if float(classes[index,0])!=row['augmented_class'] or not np.array_equal(boxes[index],np.asarray(row['augmented_bbox_normalized_xywh'],dtype=boxes.dtype)):
                    raise MappingIdentityError('Actual collated labels differ from traced row')
                rows.append(dict(row,global_gt_row=int(index),image_index=image_index,
                    image_path=entry['image_path'],frame_id=frame_id))
            if modality=='rgb':result['frames'].append(dict(image_index=image_index,dataset_index=trace['dataset_index'],rgb=trace['rgb'],ir=trace['ir']))
        rows.sort(key=lambda r:r['global_gt_row'])
        if [r['global_gt_row'] for r in rows]!=list(range(len(classes))):raise MappingIdentityError('Some batch targets lack stable identity')
        result[modality+'_rows']=rows
    return result


def export_batch_trace(batch,path,frame_id):
    result=batch_identity_contract(batch,frame_id)
    result['batch_image_shapes']={k:list(batch[k].shape) for k in ('img','strong_img')}
    with Path(path).open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    return result
