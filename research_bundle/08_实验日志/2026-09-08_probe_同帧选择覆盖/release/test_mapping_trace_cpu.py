"""CPU native-filter and real paired-wrapper identity/RNG checks."""
import argparse
import ast
import __future__
import copy
import importlib.util
import json
from pathlib import Path
import random
import sys
import types
import unittest
import numpy as np
import torch
import mapping_trace as m

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OPTIONS=argparse.ArgumentParser(add_help=False)
OPTIONS.add_argument('--reference-dir',type=Path)
OPTIONS.add_argument('--native-augment',type=Path)
OPTIONS.add_argument('--receipt',type=Path)
CLI,_=OPTIONS.parse_known_args()
PINNED=(CLI.reference_dir/'task_conditional_reference' if CLI.reference_dir else
    ROOT/'03_现行工程/SpaceNet6_OTD_official_reproduction/experiments/rgbir_independent_kd_v2/task_conditional_reference')
sys.path[:0]=[str(PINNED),str(PINNED/'legacy_oev1')]
from tracked_pair_data import TrackedDualLabelRGBIRDataset


class Instances:
    def __init__(self,bboxes,segments=None,keypoints=None,bbox_format='xywh',normalized=True):
        self.bboxes=np.asarray(bboxes,dtype=np.float32).copy();self._bboxes=types.SimpleNamespace(format=bbox_format)
        self.normalized=normalized;self.segments=np.empty((0,0,2)) if segments is None else segments;self.keypoints=keypoints
    def __len__(self):return len(self.bboxes)
    def convert_bbox(self,format):
        if format==self._bboxes.format:return
        b=self.bboxes.copy()
        if format=='xyxy':self.bboxes=np.concatenate((b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2),axis=1)
        elif format=='xywh':self.bboxes=np.concatenate(((b[:,:2]+b[:,2:])/2,b[:,2:]-b[:,:2]),axis=1)
        else:raise ValueError(format)
        self._bboxes.format=format
    def denormalize(self,w,h):
        if self.normalized:self.bboxes*=np.array([w,h,w,h],np.float32);self.normalized=False
    def scale(self,scale_w,scale_h,bbox_only=True):self.bboxes*=np.array([scale_w,scale_h,scale_w,scale_h])
    def clip(self,w,h):self.bboxes[:,[0,2]]=self.bboxes[:,[0,2]].clip(0,w);self.bboxes[:,[1,3]]=self.bboxes[:,[1,3]].clip(0,h)
    def __getitem__(self,index):return Instances(self.bboxes[index],self.segments,self.keypoints,self._bboxes.format,self.normalized)


# Execute the exact already-captured native apply_instances and box_candidates
# function bodies, with a tiny HBB Instances implementation and synthetic images.
SPEC=importlib.util.find_spec('ultralytics')
NATIVE=(CLI.native_augment if CLI.native_augment else
    Path(list(SPEC.submodule_search_locations)[0])/'data/augment.py' if SPEC else
    ROOT/'08_实验日志/2026-09-06_ops_GitHub完整审计包/remote_snapshot_20260906/framework_snapshot/ultralytics/data/augment.py')
tree=ast.parse(NATIVE.read_text(encoding='utf-8'))
native_cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='RandomPerspective')
ns=dict(np=np,Instances=Instances,Any=object)
for name in ('apply_instances','box_candidates'):
    fn=copy.deepcopy(next(n for n in native_cls.body if isinstance(n,ast.FunctionDef) and n.name==name));fn.decorator_list=[]
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(NATIVE),'exec',flags=__future__.annotations.compiler_flag),ns)


class RandomPerspective:
    degrees=shear=perspective=0.
    apply_instances=ns['apply_instances']
    box_candidates=staticmethod(ns['box_candidates'])
    def get_params(self,labels):
        # Consume each real paired-replay RNG stream exactly once per call.
        x=random.random();np.random.random();torch.rand(1)
        matrix=np.eye(3);matrix[0,2]=x
        return dict(M=matrix,scale=1.,orig_shape=labels['img'].shape[:2],size=(32,32))
    def apply_bboxes(self,boxes,matrix):
        out=boxes.copy();out[:,[0,2]]+=matrix[0,2];return out
    def __call__(self,labels):return self.apply_instances(labels,self.get_params(labels))


class RandomFlip:
    direction='horizontal'
    def get_params(self,labels):return dict(flip=random.random()<.5,direction=self.direction)
    def __call__(self,labels):
        if self.get_params(labels)['flip']:
            labels['img']=np.ascontiguousarray(labels['img'][:,::-1]);b=labels['instances'].bboxes.copy()
            labels['instances'].bboxes[:,0]=32-b[:,2];labels['instances'].bboxes[:,2]=32-b[:,0]
        return labels


class Format:
    bgr=0.;return_mask=return_keypoint=return_obb=False
    def __call__(self,labels):
        instances=labels.pop('instances');instances.convert_bbox('xywh')
        labels['bboxes']=torch.from_numpy(instances.bboxes/32)
        labels['cls']=torch.from_numpy(labels['cls']);labels['batch_idx']=torch.zeros(len(instances))
        labels['img']=torch.from_numpy(np.ascontiguousarray(labels['img'].transpose(2,0,1)))
        return labels


class Compose:
    def __init__(self,transforms):self.transforms=transforms
    def __call__(self,labels):
        for t in self.transforms:labels=t(labels)
        return labels


class TinyDataset:
    def __init__(self,modality,empty=False):
        self.im_files=[str((HERE/'synthetic'/modality/(str(i)+'.png')).resolve()) for i in range(2)]
        self.labels=[]
        for p in self.im_files:
            boxes=np.empty((0,4),np.float32) if empty else np.array([[.3,.4,.2,.2],[1.4,.5,.1,.1],[.7,.8,.2,.2]],np.float32)
            self.labels.append(dict(im_file=p,cls=np.zeros((len(boxes),1),np.float32),bboxes=boxes,
                bbox_format='xywh',normalized=True,segments=[],keypoints=None))
        self.rect=False;self.cache=False;self.imgsz=32;self.augment=True;self.ims=[None]*2
        self.data={'names':{0:'person'}};self.transforms=Compose([RandomPerspective(),RandomFlip(),Format()]);self.calls=0
    def __len__(self):return len(self.labels)
    def get_image_and_label(self,index):
        self.calls+=1;random.random();np.random.random();torch.rand(1)
        label=copy.deepcopy(self.labels[index]);label['instances']=Instances(label.pop('bboxes'))
        label.update(img=np.arange(32*32*3,dtype=np.uint8).reshape(32,32,3),ori_shape=(32,32))
        return label
    @staticmethod
    def collate_fn(rows):
        out={k:[r[k] for r in rows] for k in ('im_file','ori_shape')}
        out['img']=torch.stack([r['img'] for r in rows])
        for k in ('cls','bboxes'):out[k]=torch.cat([r[k] for r in rows])
        out['batch_idx']=torch.cat([torch.full_like(r['batch_idx'],i) for i,r in enumerate(rows)])
        return out


def pair(trace=False,empty=False):
    rgb,ir=TinyDataset('visible',empty),TinyDataset('infrared',empty)
    cls=m.make_traced_dataset(TrackedDualLabelRGBIRDataset) if trace else TrackedDualLabelRGBIRDataset
    return cls(rgb,ir,dict(zip(rgb.im_files,ir.im_files)))


def seed():random.seed(42);np.random.seed(42);torch.manual_seed(42)
def draws():return random.random(),float(np.random.random()),torch.rand(5)


class Checks(unittest.TestCase):
    def test_actual_native_filter_source_recognized(self):m._native_filter_contract(RandomPerspective())
    def test_pair_payload_rng_and_source_calls_exact(self):
        plain,traced=pair(),pair(True)
        seed();a=plain[0];ra=draws()
        seed();b=traced[0];rb=draws()
        for key in ('img','cls','bboxes','batch_idx','strong_img','strong_cls','strong_bboxes','strong_batch_idx'):
            self.assertTrue(torch.equal(a[key],b[key]),key)
        self.assertEqual(ra[:2],rb[:2]);self.assertTrue(torch.equal(ra[2],rb[2]))
        self.assertEqual(plain.base.calls,traced.base.calls);self.assertEqual(plain.teacher_base.calls,traced.teacher_base.calls)
    def test_exact_ids_skip_filtered_middle_same_class_gt(self):
        p=pair(True);seed();batch=p.collate_fn([p[0],p[1]]);r=m.batch_identity_contract(batch,'same-frame')
        self.assertEqual([x['source_gt_row'] for x in r['rgb_rows']],[0,2,0,2])
        self.assertEqual([x['global_gt_row'] for x in r['rgb_rows']],[0,1,2,3])
        self.assertEqual([f['rgb']['dropped_source_gt_rows'] for f in r['frames']],[[1],[1]])
        self.assertNotEqual(r['rgb_rows'][0]['stable_gt_id'],r['ir_rows'][0]['stable_gt_id'])
    def test_augmented_label_change_fails(self):
        p=pair(True);seed();batch=p.collate_fn([p[0]])
        batch['bboxes'][0,0]+=.01
        with self.assertRaises(m.MappingIdentityError):m.batch_identity_contract(batch,'f')
    def test_original_label_disagreement_fails(self):
        p=pair(True);p.base.labels[0]['bboxes'][0,0]+=.01
        with self.assertRaises(m.MappingIdentityError):p[0]
    def test_empty_labels_have_empty_but_verified_identity(self):
        p=pair(True,True);seed();batch=p.collate_fn([p[0]])
        r=m.batch_identity_contract(batch,'empty');self.assertEqual(r['rgb_rows'],[]);self.assertEqual(r['ir_rows'],[])
    def test_instrumentation_does_not_consume_loader_generator(self):
        results=[]
        for traced in (False,True):
            seed();p=pair(traced);g=torch.Generator().manual_seed(123)
            loader=torch.utils.data.DataLoader(p,batch_size=2,shuffle=True,num_workers=0,generator=g,collate_fn=p.collate_fn)
            batch=next(iter(loader));results.append((batch,g.get_state().clone(),draws()))
        a,b=results
        self.assertEqual(a[0]['im_file'],b[0]['im_file'])
        for key in ('img','cls','bboxes','strong_img','strong_cls','strong_bboxes'):self.assertTrue(torch.equal(a[0][key],b[0][key]))
        self.assertTrue(torch.equal(a[1],b[1]));self.assertEqual(a[2][:2],b[2][:2]);self.assertTrue(torch.equal(a[2][2],b[2][2]))
    def test_unverified_or_invalid_transform_fails(self):
        p=pair();p.base.transforms.transforms.append(Format())
        with self.assertRaises(m.MappingIdentityError):m.MappingTraceDataset(p)
        with self.assertRaises(m.MappingIdentityError):m.batch_identity_contract({'im_file':['x']},'f')


if __name__=='__main__':
    p=argparse.ArgumentParser(parents=[OPTIONS]);args=p.parse_args()
    if args.receipt is None:p.error('--receipt required')
    torch.set_num_threads(1)
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    with args.receipt.open('x',encoding='utf-8') as f:json.dump(dict(status='PASS' if r.wasSuccessful() else 'FAIL',
        tests=r.testsRun,failures=len(r.failures),errors=len(r.errors),native_filter_source=str(NATIVE),
        real_pinned_pair_and_tracked_class=True,synthetic_images=True,GPU_used=False,new_hash_computed=False),f,indent=2)
    raise SystemExit(0 if r.wasSuccessful() else 1)
